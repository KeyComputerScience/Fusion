"""Portable revision entry point: verify supplied evidence, or run in a fresh copy.

Default verification never reruns expensive experiments or edits result files.
Full recomputation requires a new destination and retains every protocol output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    expected = {"torch": "2.2.2", "numpy": "1.26.4", "scipy": "1.14.1",
                "scikit-learn": "1.5.2", "matplotlib": "3.9.4", "pytest": "9.1.1"}
    actual, missing = {}, []
    for name in expected:
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual[name] = None
            missing.append(name)
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing) + ". Install requirements.txt first.")
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": actual,
            "recommended_packages": expected, "recommended_python": "3.11.x",
            "exact_environment_match": sys.version_info[:2] == (3, 11)
            and all(actual[name].split("+")[0] == version for name, version in expected.items())}


def invoke(bundle, arguments, *, capture=False):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "information-fusion-revision-matplotlib-cache"))
    command = [sys.executable, *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=bundle, env=env, text=True, check=True,
                            capture_output=capture)
    if capture:
        print(result.stdout + result.stderr, end="", flush=True)
    return result


def verify(bundle, *, skip_tests=False):
    started = time.perf_counter()
    report = {"environment": environment(), "existing_results_read_only": True,
              "expensive_experiments_rerun": False, "checks": {}}
    manifest = read(bundle / "data/original_synthetic/manifest.json")
    inputs = []
    for entry in manifest["streams"]:
        source = bundle / "data/original_synthetic" / Path(entry["path"]).name
        if digest(source) != entry["sha256"]:
            raise ValueError("Original paired synthetic input hash mismatch: " + str(source))
        inputs.append({"seed": entry["seed"], "path": str(source.relative_to(bundle)), "sha256": entry["sha256"]})
    report["checks"]["original_paired_input_hashes"] = inputs
    required = {
        "original": ("seed_results.json", 30),
        "quality": ("seed_results.json", 700),
        "source_ablation": ("seed_results.json", 775),
        "service": ("seed_results.json", 160),
        "update_necessity": ("seed_results.json", 60),
        "official_scalar": ("seed_results.csv", 40),
    }
    for name, (filename, expected) in required.items():
        path = bundle / "results" / name / filename
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                count = len(list(csv.DictReader(stream)))
        else:
            count = len(read(path))
        if count != expected:
            raise ValueError(f"Incomplete {name}: {count} result rows, expected {expected}")
        report["checks"][name] = {"rows": count, "expected": expected, "sha256": digest(path)}
    update = read(bundle / "results/update_necessity/manifest.json")
    if update.get("status") != "complete" or update.get("completed_runs") != 60:
        raise ValueError("The update-necessity diagnostic has not completed")
    official = read(bundle / "results/official_scalar/provenance.json")
    if official.get("status") != "executed_complete" or official.get("runs") != 40:
        raise ValueError("The scalar official-kernel experiment has not completed")
    budget_path = bundle / "results/update_necessity/budget_sensitivity/manifest.json"
    if budget_path.is_file():
        budget = read(budget_path)
        if budget.get("status") != "complete" or budget.get("completed_new_runs") != 80:
            raise ValueError("The budget sensitivity extension has not completed")
        report["checks"]["post_hoc_budget_sensitivity"] = {"new_runs": 80, "manifest_sha256": digest(budget_path), "post_hoc": True}
        for extension in ("png", "pdf", "svg"):
            path = bundle / "figures" / f"dose_response.{extension}"
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError("Missing dose-response export: " + str(path))
        report["checks"]["dose_response_figures"] = "Three nonempty PNG/PDF/SVG exports present"
    else:
        report["checks"]["post_hoc_budget_sensitivity"] = {"present": False, "required": False}
    for filename in ("quality_interventions", "component_ablations"):
        for extension in ("png", "pdf", "svg"):
            path = bundle / "figures" / f"{filename}.{extension}"
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError("Missing exported figure: " + str(path))
    report["checks"]["quality_figures"] = "Six nonempty PNG/PDF/SVG exports present"
    invoke(bundle, ["code/audit_quality_results.py", "--check-only"], capture=True)
    report["checks"]["causal_quality_audit"] = "passed; result files were not changed"
    if not skip_tests:
        test = invoke(bundle, ["-m", "pytest", "code", "-q", "--disable-warnings",
                               "--junitxml", "verification/pytest.xml"], capture=True)
        report["checks"]["pytest"] = {"exit_code": test.returncode, "report": "verification/pytest.xml"}
    else:
        report["checks"]["pytest"] = {"skipped_by_option": True}
    report.update(all_passed=True, elapsed_seconds=time.perf_counter() - started)
    write(bundle / "verification/bundle_verification.json", report)
    print("Verified supplied evidence. Existing results were read only.", flush=True)
    return report


def prepare_destination(destination):
    destination = destination.resolve()
    for source in (HERE.resolve(), (BUNDLE / "data/original_synthetic").resolve()):
        if destination == source or source in destination.parents:
            raise ValueError("Destination must be outside the source code and source data trees")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Full recomputation requires a new or empty destination; existing evidence is never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE, destination / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    shutil.copytree(BUNDLE / "data/original_synthetic", destination / "data/original_synthetic")
    for name in ("requirements.txt", "README.md", "environment_notes.json", "experimental_protocol.md",
                 "baseline_review.md", "quality_findings.md", "upstream_provenance.json",
                 "performance_template.tex", "sota_section.tex", "update_section.tex", "conclusion.tex"):
        if (BUNDLE / name).is_file():
            shutil.copy2(BUNDLE / name, destination / name)
    return destination


def plots(bundle):
    for path in sorted((bundle / "code").glob("plot_*.py")):
        if path.name == "plot_update_dose.py" and not (bundle / "results/update_necessity/budget_sensitivity/analysis.json").is_file():
            print("Skipping dose-response plot: the post-hoc budget extension was omitted.", flush=True)
            continue
        invoke(bundle, [path.relative_to(bundle)])


def full_run(destination, workers, *, include_budget=True):
    bundle = prepare_destination(destination)
    report = {"environment": environment(), "fresh_destination": str(bundle), "completed_stages": []}
    invoke(bundle, ["-m", "pytest", "code", "-q", "--disable-warnings", "--junitxml", "verification/pytest.xml"])
    stages = [
        ("original_execution", ["code/run_original.py", "--workers", workers]),
        ("quality_replay", ["code/quality_benchmark.py"]),
        ("quality_audit", ["code/audit_quality_results.py"]),
        ("source_removal", ["code/source_ablation.py"]),
        ("independent_service_execution", ["code/run_service_revision.py", "--workers", workers]),
        ("update_necessity", ["code/update_necessity.py"]),
    ]
    if include_budget:
        stages.append(("post_hoc_budget_sensitivity", ["code/update_budget_sensitivity.py"]))
    stages.append(("official_kernel_scalar_adaptations", ["code/run_official_sota.py", "--outdir", "results/official_scalar"]))
    for name, arguments in stages:
        invoke(bundle, arguments)
        report["completed_stages"].append(name)
        write(bundle / "verification/recompute_progress.json", report)
    analysis = bundle / "code/analyze_revision.py"
    budget_analysis = bundle / "results/update_necessity/budget_sensitivity/analysis.json"
    if analysis.is_file() and budget_analysis.is_file():
        invoke(bundle, ["code/analyze_revision.py"])
        report["completed_stages"].append("table_analysis")
        if (bundle / "code/build_manuscript.py").is_file():
            invoke(bundle, ["code/build_manuscript.py"])
            report["completed_stages"].append("performance_latex_assembly")
        if (bundle / "code/export_table_data.py").is_file():
            invoke(bundle, ["code/export_table_data.py"])
            report["completed_stages"].append("full_precision_table_csv_exports")
    else:
        report["optional_analysis"] = "Complete Performance tables require the declared budget extension and analyze_revision.py; these were omitted. Raw initial-protocol results are complete."
        print(report["optional_analysis"], flush=True)
    plots(bundle)
    report["completed_stages"].append("plots")
    verify(bundle, skip_tests=True)
    report["all_passed"] = True
    write(bundle / "verification/recompute_progress.json", report)
    print("Full recomputation complete:", bundle, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Check supplied evidence; default, no expensive reruns")
    mode.add_argument("--run", action="store_true", help="Recompute all protocols into a new destination")
    mode.add_argument("--plots-only", action="store_true", help="Regenerate plots from saved results only")
    parser.add_argument("--destination", type=Path, help="New/empty output directory required by --run")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-tests", action="store_true", help="For packaging verification after a separately recorded pytest run")
    parser.add_argument("--skip-budget-sensitivity", action="store_true", help="Omit the declared post hoc extension in a fresh run")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.run:
        if args.destination is None:
            parser.error("--run requires --destination to protect the supplied results")
        full_run(args.destination, args.workers, include_budget=not args.skip_budget_sensitivity)
    elif args.plots_only:
        plots(BUNDLE)
    else:
        verify(BUNDLE, skip_tests=args.skip_tests)


if __name__ == "__main__":
    main()
