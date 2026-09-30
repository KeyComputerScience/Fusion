"""Backend contract for executing the coordinator in an actual edge/DRL system.

This module does not invent a trained policy, DRL rewards, or a measured gain.
Implement the protocol with the target backend. See README.md for field timing.
"""
from dataclasses import dataclass
from typing import Protocol

from coordinator import Decision
from pipeline import FusionPipeline


@dataclass
class SlotObservation:
    slot: int
    training_load: float
    inference_load: float
    capacities: list[list[float]]
    future_retentions: list[float] | None = None
    pending_gain_forecast: dict[int,float] | None = None


class ServiceBackend(Protocol):
    def begin_slot(self, slot: int) -> SlotObservation:
        """Current data, before any outcome from this slot is observed."""
        ...

    def dispatch_and_observe(self, decision: Decision) -> dict:
        """Execute profiles; return actual post-slot feedback in input_schema.json.

        Gains enter feedback only on actual completed deployments. Logging IDs
        describe actual execution. Capacities already account for running jobs.
        """
        ...

    def handle_infeasible_and_observe(self, decision: Decision) -> dict:
        """Apply service admission/deferral; no infeasible profile dispatch.

        If no profile is executed, set execution_status='no_dispatch', logged
        IDs to None, and supply any real pending-deployment feedback separately.
        """
        ...


def execute_one_slot(pipeline: FusionPipeline, backend: ServiceBackend):
    observation = backend.begin_slot(pipeline.next_slot)
    decision = pipeline.decide(observation.slot, observation.training_load,
        observation.inference_load, observation.capacities,
        observation.future_retentions, observation.pending_gain_forecast)
    if decision.status == "feasible":
        feedback = backend.dispatch_and_observe(decision)
    else:
        feedback = backend.handle_infeasible_and_observe(decision)
    if feedback["slot"] != observation.slot:
        raise ValueError("Backend feedback does not match this slot")
    # Pre-slot evidence fields and post-slot outcome fields form one aligned row.
    feedback.update(training_load=observation.training_load,
                    inference_load=observation.inference_load,capacities=observation.capacities)
    observed = pipeline.observe(feedback)
    return decision, observed


def run_live(pipeline: FusionPipeline, backend: ServiceBackend):
    """Yield bounded-memory decisions and feedback; callers choose what to log."""
    for _ in range(pipeline.config["evaluation_horizon"]):
        yield execute_one_slot(pipeline, backend)
