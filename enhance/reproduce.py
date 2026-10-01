"""One entry point for tests, executed experiments, data, statistics and LaTeX."""
import argparse
import os
import re
import subprocess
import sys

from data_pipeline import ROOT, write_json


def invoke(*arguments, capture=False):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True,
                            capture_output=capture, text=True)
    if capture:
        print(result.stdout + result.stderr, end="", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check supplied logs and rerun two full references")
    parser.add_argument("--smoke", action="store_true", help="Run small execution checks; do not generate manuscript results")
    args = parser.parse_args()
    test_reports = []
    for directory in ("core/tests", "tests"):
        result = invoke("-m", "unittest", "discover", "-s", directory, "-v", capture=True)
        matched = re.search(r"Ran (\d+) tests in ([0-9.]+)s", result.stdout + result.stderr)
        if matched is None:
            raise ValueError("Cannot record the executed test count")
        test_reports.append({"suite": directory, "tests": int(matched[1]),
                             "seconds": float(matched[2]), "return_code": result.returncode, "all_passed": True})
    write_json(ROOT / ("smoke_results" if args.smoke else "results") / "test_results.json", test_reports)
    if args.smoke:
        invoke("run_experiments.py", "--quick")
        return
    if not args.verify_only:
        invoke("data_pipeline.py", "generate")
        invoke("run_experiments.py")
        invoke("fusion_benchmarks.py")
    invoke("analyze_results.py")
    invoke("verify_reexecution.py")
    invoke("export_data.py")
    invoke("build_performance.py")
    print("Complete: validated executed results, data provenance, forecasts and performance.tex", flush=True)


if __name__ == "__main__":
    main()
