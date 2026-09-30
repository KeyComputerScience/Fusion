"""Validate mathematical/causal invariants and the generated diagnostic outputs."""
import json
from pathlib import Path
import sys
import unittest

from logio import read_json


def validate():
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = [{"test":str(test),"traceback":detail} for test,detail in result.failures+result.errors]
    audit = read_json(root / "solver_audit.json")
    diagnostics = read_json(root / "experiment_results/diagnostics.json")["results"]
    checks = [*audit["diagnostics"].values(),*diagnostics]
    names = ("resource_violations","future_information_violations","weight_or_mask_violations")
    totals = {name:sum(row[name] for row in checks) for name in names}
    source_map = read_json(root / "methodology_map.json")
    tex = (root / "core_methodology.tex").read_text(encoding="utf-8")
    label_errors = [row["label"] for row in source_map["equation_to_code"]
                    if "\\label{"+row["label"]+"}" not in tex]
    passed = result.wasSuccessful() and not any(totals.values()) and not label_errors
    report = {
        "status":"passed" if passed else "failed", "tests_run":result.testsRun,
        "unit_test_failures":failures, "diagnostic_runs":len(checks),
        "diagnostic_slots":sum(row["slots"] for row in checks),
        "invariant_violation_totals":totals, "missing_equation_labels":label_errors,
        "covered_cases":["source masking and outages","capped-simplex projection",
            "delayed target availability","no future suffix leakage",
            "future-baseline tangent bound","known pending deployments",
            "geometric zero/one/near-one limits","coupled resource feasibility",
            "pruning preserves proxy optimum","AO monotonicity and a nonzero-gap counterexample",
            "recommendation is not deployment","bounded live history","no-dispatch feedback"],
        "provenance":"synthetic functional verification",
        "establishes_real_DRL_improvement":False,
        "pdf_layout_verified_separately":"See pdf_qa.json when included."
    }
    (root / "validation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"tests_run":result.testsRun,
                      "diagnostic_runs":len(checks),"violation_totals":totals},indent=2))
    return passed


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
