"""Execute a declared synthetic, three-view classification adaptation appendix.

This changes the task, views and encoders from the original QMF/PDF benchmark.
It MUST NOT be presented as reproduction of the paper's Table 4 or as proof
of general SOTA superiority. All labels/data originate from the declared toy
classification generator; no service outcomes or Alibaba records are invented.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from official_sota_adapters import ScalarBranchFusion
from revised_fusion import solve_weights


DESIGN = {
    "scope": "independent_controlled_synthetic_multiview_classification_adaptation",
    "original_benchmark_reproduction": False,
    "operational_trace_or_service_replay": False,
    "seeds": [10, 20, 30, 40, 50],
    "train_samples": 1024, "validation_samples": 256, "test_samples": 1536,
    "classes": 3, "views": 3, "features_per_view": 4, "hidden_per_view": 16,
    "training_epochs": 100, "learning_rate": 0.001, "weight_decay": 0.0,
    "training_batches": "full chronological training prefix per optimisation step",
    "model_selection": "lowest validation joint cross-entropy, never test labels",
    "generator": "uniform latent class; independently drawn fixed class prototypes per view plus Gaussian noise",
    "prototype_scale": 1.5, "train_validation_noise": [0.6, 0.6, 0.6],
    "test_noise_phases": [[0.6, 0.6, 0.6], [2.0, 0.6, 0.6], [2.0, 2.0, 0.6]],
    "noise_metadata_exposed_to_models": False,
    "test_phase_samples": 512, "evaluation_cohort": 64,
    "protocols": ["frozen", "prequential_delayed_labels"],
    "label_arrival": "cohort labels arrive after all predictions for that cohort",
    "online_training": "one equal-budget gradient step per revealed cohort for each model",
    "revised_risk": {"temperature": 1.0, "residual_penalty": 1.0, "inertia": 0.1,
                     "cap": 0.7, "decay": 0.9, "consistency_scale": 0.2},
    "revised_risk_limitation": "no support/freshness/observation-quality metadata; delayed vector risk only, not full original RF",
    "methods": ["EF_CLASSIFIER", "QMF_SCALAR", "PDF_SCALAR", "REVISED_RISK_SCALAR"],
    "keep_negative_results": True,
}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_data(seed):
    rng = np.random.default_rng(seed + 911)
    prototype = rng.normal(0.0, DESIGN["prototype_scale"], (3, 3, 4))
    arrays = {}
    for split, count in (("train", 1024), ("validation", 256), ("test", 1536)):
        y = rng.integers(0, 3, count)
        x = np.stack([prototype[view, y] for view in range(3)], axis=1)
        noise = np.full((count, 3, 1), 0.6)
        if split == "test":
            for phase, values in enumerate(DESIGN["test_noise_phases"]):
                noise[phase * 512:(phase + 1) * 512] = np.asarray(values)[None, :, None]
        x = x + rng.normal(size=x.shape) * noise
        arrays[f"{split}_x"] = x.astype(np.float32)
        arrays[f"{split}_y"] = y.astype(np.int64)
    return arrays


def tensor_arrays(data):
    return {key: torch.from_numpy(value) for key, value in data.items()}


def fit_model(method, data, seed):
    torch.manual_seed(seed)
    model = ScalarBranchFusion(method, modalities=3, features=4, hidden=16, classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=DESIGN["learning_rate"], weight_decay=0.0)
    history = torch.zeros(len(data["train_y"]), 3)
    best, best_loss, best_epoch = None, float("inf"), 0
    trajectory = []
    for epoch in range(DESIGN["training_epochs"]):
        model.train()
        loss, branch_loss = model.training_loss(data["train_x"], data["train_y"], history)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history += branch_loss
        model.eval()
        with torch.no_grad():
            validation = float(F.cross_entropy(model(data["validation_x"])[0], data["validation_y"]))
        trajectory.append({"epoch": epoch + 1, "train_loss": float(loss.detach()), "validation_ce": validation})
        if validation < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), validation, epoch + 1
    model.load_state_dict(best)
    return model, {"selected_epoch": best_epoch, "validation_ce": best_loss,
                   "total_parameters": sum(p.numel() for p in model.parameters()),
                   "classification_backbone_parameters": sum(p.numel() for module in (model.encoders, model.classifiers) for p in module.parameters()),
                   "trajectory": trajectory}


class DelayedVectorRisk:
    """RF classification adaptation with ONLY previously revealed vector risk.

    No corruption flags/noise levels/true labels enter prediction. R is PSD:
    each update is a weighted average of vector residual Gram matrices.
    """
    def __init__(self):
        self.matrix = np.eye(3) * 0.05
        self.previous = np.ones(3) / 3
        self.update_count = 0

    def predict(self, branch_probabilities):
        setting = DESIGN["revised_risk"]
        prior = np.exp(-np.diag(self.matrix) / setting["consistency_scale"])
        weights, audit = solve_weights(prior, [True] * 3, cap=setting["cap"],
                                      residual_matrix=self.matrix, previous=self.previous,
                                      entropy_temperature=setting["temperature"],
                                      residual_penalty=setting["residual_penalty"], inertia=setting["inertia"])
        self.previous = weights
        probability = np.einsum("m,bmc->bc", weights, branch_probabilities)
        return probability, weights, audit

    def observe(self, saved_branch_probabilities, revealed_labels):
        one_hot = np.eye(3)[revealed_labels]
        residual = saved_branch_probabilities - one_hot[:, None, :]
        gram = np.einsum("bmc,bnc->mn", residual, residual) / (2.0 * len(residual))
        decay = DESIGN["revised_risk"]["decay"]
        self.matrix = decay * self.matrix + (1.0 - decay) * gram
        self.update_count += 1


def metrics(probability, labels):
    predicted = probability.argmax(axis=1)
    accuracy = float(np.mean(predicted == labels))
    nll = float(-np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-12)).mean())
    confidence = probability.max(axis=1)
    ece = 0.0
    for index in range(10):
        selected = (confidence >= index / 10) & ((confidence < (index + 1) / 10) if index < 9 else confidence <= 1)
        if selected.any():
            ece += float(selected.mean() * abs((predicted[selected] == labels[selected]).mean() - confidence[selected].mean()))
    return {"accuracy": accuracy, "nll": nll, "ece_10": ece}


def evaluate(method, original, data, protocol):
    model = copy.deepcopy(original)
    optimizer = torch.optim.Adam(model.parameters(), lr=DESIGN["learning_rate"])
    risk = DelayedVectorRisk() if method == "REVISED_RISK_SCALAR" else None
    if risk is not None:
        # Validation labels are before test time and are visible to model selection
        # for every comparator. Their use for this risk fit is explicitly disclosed.
        model.eval()
        with torch.no_grad():
            validation_probabilities = model(data["validation_x"])[1].softmax(-1).numpy()
        for start in range(0, len(validation_probabilities), 64):
            risk.observe(validation_probabilities[start:start+64], data["validation_y"][start:start+64].numpy())
    records, all_probabilities = [], []
    for start in range(0, len(data["test_y"]), 64):
        x, y = data["test_x"][start:start+64], data["test_y"][start:start+64]
        model.eval()
        with torch.no_grad():
            fused, branches, confidence, diagnostics = model(x)
            branch_probabilities = branches.softmax(-1).numpy()
            if risk is not None:
                probabilities, weights, audit = risk.predict(branch_probabilities)
            else:
                probabilities = fused.softmax(-1).numpy()
                weights = diagnostics.get("weights", diagnostics.get("coefficients")).numpy()
                audit = None
        all_probabilities.append(probabilities)
        records.append({"cohort_start": start, "noise_phase": start // 512,
                        "predicted_before_label_arrival": True,
                        "mean_coefficients_or_weights": np.asarray(weights).mean(axis=0).tolist() if np.asarray(weights).ndim == 2 else np.asarray(weights).tolist(),
                        "simplex_weights": method != "QMF_SCALAR", "solver": audit,
                        **metrics(probabilities, y.numpy())})
        if protocol == "prequential_delayed_labels":
            if risk is not None:
                risk.observe(branch_probabilities, y.numpy())
            model.train()
            loss, _ = model.training_loss(x, y, torch.zeros(len(y), 3))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    probabilities = np.concatenate(all_probabilities)
    labels = data["test_y"].numpy()
    result = {"method": method, "protocol": protocol, "test_samples": len(labels),
              "gradient_updates_after_revealed_labels": 24 if protocol == "prequential_delayed_labels" else 0,
              **metrics(probabilities, labels)}
    for phase in range(3):
        result[f"phase_{phase}_accuracy"] = metrics(probabilities[phase*512:(phase+1)*512], labels[phase*512:(phase+1)*512])["accuracy"]
    return result, records, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "design.json", DESIGN)
    results, provenance = [], []
    for seed in DESIGN["seeds"]:
        arrays = build_data(seed)
        data_path = args.outdir / f"data_seed_{seed}.npz"
        np.savez_compressed(data_path, **arrays)
        provenance.append({"seed": seed, "data_path": data_path.name, "sha256": digest(data_path),
                           "training_class_counts": np.bincount(arrays["train_y"], minlength=3).tolist()})
        data = tensor_arrays(arrays)
        models = {}
        for method in ("EF_CLASSIFIER", "QMF_SCALAR", "PDF_SCALAR"):
            model, audit = fit_model(method, data, seed)
            models[method] = model
            torch.save(model.state_dict(), args.outdir / f"{method}_seed_{seed}.pt")
            write_json(args.outdir / f"{method}_seed_{seed}_training.json", audit)
        for protocol in DESIGN["protocols"]:
            for method in DESIGN["methods"]:
                model = models["EF_CLASSIFIER" if method == "REVISED_RISK_SCALAR" else method]
                result, cohorts, probabilities = evaluate(method, model, data, protocol)
                result["seed"] = seed
                results.append(result)
                write_json(args.outdir / f"{method}_{protocol}_seed_{seed}.json", {"result": result, "cohorts": cohorts})
                np.save(args.outdir / f"{method}_{protocol}_seed_{seed}_probabilities.npy", probabilities)
        print(json.dumps({"seed_completed": seed, "protocol_results": 8}), flush=True)
    fields = list(results[0])
    with (args.outdir / "seed_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    summary = []
    for protocol in DESIGN["protocols"]:
        for method in DESIGN["methods"]:
            subset = [row for row in results if row["method"] == method and row["protocol"] == protocol]
            row = {"method": method, "protocol": protocol, "seeds": len(subset)}
            for key in ("accuracy", "nll", "ece_10", "phase_0_accuracy", "phase_1_accuracy", "phase_2_accuracy"):
                values = [r[key] for r in subset]; average = statistics.mean(values); std = statistics.stdev(values)
                row.update({f"{key}_mean": average, f"{key}_std": std,
                            f"{key}_ci95_lower": average - 2.776 * std / math.sqrt(5),
                            f"{key}_ci95_upper": average + 2.776 * std / math.sqrt(5)})
            summary.append(row)
    write_json(args.outdir / "summary.json", summary)
    write_json(args.outdir / "provenance.json", {"data": provenance, "torch": torch.__version__,
               "numpy": np.__version__, "entrypoint_sha256": digest(__file__),
               "adapter_sha256": digest(Path(__file__).with_name("official_sota_adapters.py")),
               "solver_sha256": digest(Path(__file__).with_name("revised_fusion.py")),
               "status": "executed_complete", "runs": len(results)})


if __name__ == "__main__":
    main()
