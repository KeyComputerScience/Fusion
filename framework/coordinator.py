"""Joint retraining/inference coordination with per-node resource constraints."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from fusion import FusionSnapshot, clip, finite


@dataclass
class Decision:
    slot: int
    training_profile: str | None
    inference_profile: str | None
    objective: float | None
    solver: str
    status: str
    training_allowed: bool
    rho: float
    recovery_state: float
    fusion_completed_slot: int
    gamma: float
    uncertainty: float
    coverage: float
    evaluated_pairs: int
    trajectory: list[float]
    coordinate_converged: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Coordinator:
    def __init__(self, config: dict, profiles: dict) -> None:
        self.config = config
        self.profiles = profiles
        self.training = profiles["training"]
        self.inference = profiles["inference"]
        self.base = profiles["base_training_profile"]
        self.H = float(config["initial_recovery"])
        self.trigger_active = False
        self.warm_start: tuple[str, str] | None = None
        self.quality = {p["id"]: dict(p["quality_prior"]) for p in self.inference}
        self.training_by_id = {p["id"]: p for p in self.training}
        self.inference_by_id = {p["id"]: p for p in self.inference}
        self._validate_profiles()

    def _validate_profiles(self) -> None:
        if len(self.training_by_id) != len(self.training) or len(self.inference_by_id) != len(self.inference):
            raise ValueError("Profile IDs must be unique within each group")
        if self.base not in self.training_by_id:
            raise ValueError("The base training profile is required")
        base = self.training_by_id[self.base]
        if any(base[name] != 0 for name in ("gain", "cost", "intensity")):
            raise ValueError("Base profile gain/cost/intensity must be zero")
        for profile in self.training + self.inference:
            matrix = self._matrix(profile["resources"], self.config["nodes"])
            if any(value < 0 or not finite(value) for row in matrix for value in row):
                raise ValueError("Resource demands must be finite and nonnegative")
        if any(value != 0 for row in self._matrix(base["resources"], self.config["nodes"]) for value in row):
            raise ValueError("Base training profile resources must be zero")
        for profile in self.training:
            if any(not finite(profile[name]) or profile[name] < 0 for name in ("gain", "cost", "intensity")):
                raise ValueError("Training calibration values must be nonnegative")
            if not isinstance(profile["deployment_delay"], int) or profile["deployment_delay"] < 1:
                raise ValueError("Deployment delay must be an integer of at least one slot")
        for profile in self.inference:
            if not 0.0 <= profile["beta"] <= 1.0:
                raise ValueError("Nominal quality must lie in [0,1]")
            if set(profile["quality_prior"]) != set(self.config["quality_weights"]):
                raise ValueError("Quality priors must match quality weights")
            if any(not finite(v) or not 0 <= v <= 1 for v in profile["quality_prior"].values()):
                raise ValueError("Quality priors must be finite and in [0,1]")

    @staticmethod
    def _matrix(value: list, nodes: int) -> list[list[float]]:
        if len(value) == 3 and all(finite(v) for v in value):
            return [list(map(float, value)) for _ in range(nodes)]
        if len(value) != nodes or any(len(row) != 3 for row in value):
            raise ValueError("Resources must be a three-vector or a nodes-by-three matrix")
        return [list(map(float, row)) for row in value]

    def retention(self, snapshot: FusionSnapshot) -> float:
        return clip(1.0 - self.config["eta_gamma"] * snapshot.gamma
                    - self.config["eta_uncertainty"] * snapshot.uncertainty,
                    self.config["rho_min"], 1.0)

    def _allowed(self, snapshot: FusionSnapshot) -> bool:
        trigger = self.config["trigger"]
        if snapshot.gamma >= trigger["on"]:
            self.trigger_active = True
        elif snapshot.gamma < trigger["off"]:
            self.trigger_active = False
        return (self.trigger_active and not snapshot.fallback
                and snapshot.coverage >= trigger["minimum_coverage"]
                and snapshot.uncertainty <= trigger["maximum_uncertainty"])

    def feasible(self, i: dict, j: dict, training_load: float, inference_load: float,
                 capacities: list[list[float]]) -> bool:
        a = self._matrix(i["resources"], self.config["nodes"])
        b = self._matrix(j["resources"], self.config["nodes"])
        return all(training_load * a[m][r] + inference_load * b[m][r] <= capacities[m][r] + 1e-12
                   for m in range(self.config["nodes"]) for r in range(3))

    def service_quality(self, j: dict) -> float:
        return j["beta"] * sum(weight * self.quality[j["id"]][name]
                               for name, weight in self.config["quality_weights"].items())

    def value(self, i: dict, j: dict, training_load: float, rho: float,
              snapshot: FusionSnapshot, remaining_slots: int) -> float:
        horizon = min(self.config["lookahead"], remaining_slots)
        attenuation = math.exp(-self.config["xi"] * snapshot.gamma)
        current = attenuation * self.service_quality(j) * (-math.expm1(-self.H))
        # Tangent is evaluated at the decayed no-update baseline, not at H(t).
        delay = i["deployment_delay"]
        marginal = attenuation * sum(
            math.exp(-(rho ** h) * self.H) * rho ** (h - delay)
            for h in range(delay, horizon + 1))
        intensity = training_load * i["intensity"]
        return (current + marginal * training_load * i["gain"]
                - self.config["lambda_cost"] * training_load * i["cost"]
                - self.config["omega_intensity"] * intensity ** 2
                - self.config["kappa_uncertainty"] * snapshot.uncertainty * intensity)

    def choose(self, slot: int, training_load: float, inference_load: float, capacities: list,
               snapshot: FusionSnapshot, remaining_slots: int, solver: str = "auto") -> Decision:
        if remaining_slots < 0 or not isinstance(remaining_slots, int):
            raise ValueError("Remaining horizon must be a nonnegative integer")
        if any(not finite(v) or v < 0 for v in (training_load, inference_load)):
            raise ValueError("Loads must be finite and nonnegative")
        capacities = self._matrix(capacities, self.config["nodes"])
        if any(not finite(v) or v < 0 for row in capacities for v in row):
            raise ValueError("Capacities must be finite and nonnegative")
        rho = self.retention(snapshot)
        allowed = self._allowed(snapshot)
        candidates_i = self.training if allowed else [self.training_by_id[self.base]]
        candidates_j = self.inference
        if solver == "auto":
            solver = "exact" if len(self.training) * len(self.inference) <= self.config["exact_pair_limit"] else "ao"
        if solver not in {"exact", "ao"}:
            raise ValueError("Solver must be auto, exact or ao")

        evaluated = 0
        cache: dict[tuple[str, str], float | None] = {}

        def evaluate(i: dict, j: dict) -> float | None:
            nonlocal evaluated
            key = (i["id"], j["id"])
            if key not in cache:
                evaluated += 1
                cache[key] = (self.value(i, j, training_load, rho, snapshot, remaining_slots)
                              if self.feasible(i, j, training_load, inference_load, capacities) else None)
            return cache[key]

        def best(options: list[tuple[dict, dict]], incumbent: tuple[dict, dict] | None = None):
            selected = incumbent
            score = evaluate(*incumbent) if incumbent is not None else None
            for pair in options:
                candidate = evaluate(*pair)
                # Retain incumbent at ties to avoid equal-value AO cycles.
                if candidate is not None and (score is None or candidate > score + 1e-12):
                    selected, score = pair, candidate
            return selected, score

        trajectory: list[float] = []
        converged = False
        if solver == "exact":
            pair, objective = best([(i, j) for i in candidates_i for j in candidates_j])
            if objective is not None:
                trajectory = [objective]
        else:
            # Feasible anchors: inference priority, recovery priority, prior decision.
            anchors: list[tuple[dict, dict]] = []
            ordered_i = sorted(candidates_i, key=lambda p: (p["intensity"], p["id"]))
            ordered_j = sorted(candidates_j, key=lambda p: (-self.service_quality(p), p["id"]))
            for i in ordered_i:
                anchor = next(((i, j) for j in ordered_j if evaluate(i, j) is not None), None)
                if anchor is not None:
                    anchors.append(anchor)
                    break
            for i in sorted(candidates_i, key=lambda p: (-p["gain"], p["id"])):
                anchor, _ = best([(i, j) for j in ordered_j])
                if anchor is not None:
                    anchors.append(anchor)
                    break
            if self.warm_start and any(i["id"] == self.warm_start[0] for i in candidates_i):
                warm = (self.training_by_id[self.warm_start[0]], self.inference_by_id[self.warm_start[1]])
                if evaluate(*warm) is not None:
                    anchors.append(warm)
            unique = {(i["id"], j["id"]): (i, j) for i, j in anchors}
            pair, objective = None, None
            for anchor in unique.values():
                local = anchor
                history = [evaluate(*local)]
                local_converged = False
                for _ in range(self.config["ao_max_sweeps"]):
                    previous_pair = local
                    local, _ = best([(i, local[1]) for i in candidates_i], incumbent=local)
                    local, score = best([(local[0], j) for j in candidates_j], incumbent=local)
                    history.append(score)
                    if local == previous_pair:
                        local_converged = True
                        break
                score = evaluate(*local)
                if objective is None or score > objective + 1e-12:
                    pair, objective, trajectory, converged = local, score, history, local_converged
        if pair is None:
            return Decision(slot, None, None, None, solver, "infeasible_admission_required", allowed,
                            rho, self.H, snapshot.completed_slot, snapshot.gamma, snapshot.uncertainty,
                            snapshot.coverage, evaluated, [])
        self.warm_start = pair[0]["id"], pair[1]["id"]
        return Decision(slot, pair[0]["id"], pair[1]["id"], objective, solver, "feasible", allowed,
                        rho, self.H, snapshot.completed_slot, snapshot.gamma, snapshot.uncertainty,
                        snapshot.coverage, evaluated, trajectory, converged)

    def observe(self, feedback: dict, rho_used: float) -> dict:
        """Use actual logged deployment/quality, never the recommended update."""
        gain = feedback.get("deployed_gain")
        gain_kind = "observed"
        if gain is None:
            completed = feedback.get("completed_training_profile")
            completed_load = feedback.get("completed_training_load")
            if completed is not None and finite(completed_load):
                if completed not in self.training_by_id or completed_load < 0:
                    raise ValueError("Invalid completed training feedback")
                gain = self.training_by_id[completed]["gain"] * completed_load
                gain_kind = "profile_estimate_on_observed_completion"
            else:
                gain, gain_kind = 0.0, "unobserved_gain_zero_assumption"
        if not finite(gain) or gain < 0:
            raise ValueError("Deployed gain must be finite and nonnegative")
        self.H = rho_used * self.H + gain

        profile_id = feedback.get("logged_inference")
        if profile_id not in self.inference_by_id:
            raise ValueError("Unknown logged inference profile")
        if feedback.get("logged_training") not in self.training_by_id:
            raise ValueError("Unknown logged training profile")
        metrics = {"completion": feedback.get("completion_ratio"),
                   "deadline_compliance": (math.exp(-max(0.0, feedback["deadline_loss"]))
                                           if finite(feedback.get("deadline_loss")) else None),
                   "availability": feedback.get("availability"),
                   "throughput": feedback.get("throughput_ratio")}
        rate = self.config["quality_update_rate"]
        for name, value in metrics.items():
            if finite(value):
                self.quality[profile_id][name] = ((1.0 - rate) * self.quality[profile_id][name]
                                                + rate * clip(value))
        return {"recovery_after_feedback": self.H, "deployed_gain_used": gain, "gain_kind": gain_kind}
