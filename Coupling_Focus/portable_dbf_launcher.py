"""Portable execution of unchanged secondary DBF/EAvg benchmark files.

Install this file at the delivered package root beside run_reproduction.py.
Only the earlier launcher's filesystem rebasing is used; all source bytes and
numerical function/class bodies remain unchanged. A fresh output is required.
"""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent
DBF_REL = Path("work/fusion_coupling_focus_20261005/dbf")
PHYSICAL_REL = Path("work/fusion_cj_next_20261004/physical")
TASKS = {"rss": "rss348", "harth": "harth779"}
PHASES = ("verify", "calibrate-rss", "calibrate-harth", "locktest", "test-rss", "test-harth",
          "analyze-rss", "analyze-harth", "readiness-freeze", "readiness-verify",
          "readiness-calibrate",
          "readiness-test-rss", "readiness-test-harth", "readiness-analyze-rss", "readiness-analyze-harth")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    answer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(answer)
    return answer


def source_relative(path, original_project):
    path = Path(path)
    if not path.is_absolute():
        if ".." in path.parts:
            raise ValueError("Closure paths must stay within project")
        return path
    return path.relative_to(original_project)


def closure(archive, original_project, readiness_protocol=None):
    """Collect frozen, exact-byte code/data/comparison records, not new outputs."""
    dbf = archive / DBF_REL
    protocol = load(dbf / "protocol.json")
    marker = load(dbf / "pre_calibration_freeze.json")
    assert sha(dbf / "protocol.json") == marker["protocol_sha256"]
    assert sha(dbf / "operator_checks.json") == marker["operator_checks_sha256"]
    wanted = {}

    def add(relative, expected=None):
        relative = Path(relative)
        source = archive / relative
        assert source.is_file(), str(relative)
        actual = sha(source)
        if expected is not None:
            assert actual == expected, str(relative)
        if str(relative) in wanted:
            assert wanted[str(relative)] == actual
        wanted[str(relative)] = actual

    def add_hashes(record):
        for field in ("source_hashes", "dataset_hashes", "comparison_hashes"):
            for path, expected in record.get(field, {}).items():
                add(source_relative(path, original_project), expected)
        for path, expected in record.get("inputs", {}).items():
            add(source_relative(path, original_project), expected)

    add_hashes(protocol)
    frozen = load(archive / PHYSICAL_REL / "algorithm_freeze.json")
    add_hashes(frozen)
    for relative in (PHYSICAL_REL / "algorithm_freeze.json", PHYSICAL_REL / "acquisition_authorization.json",
                     PHYSICAL_REL / "acquisition_record.json", DBF_REL / "protocol.json",
                     DBF_REL / "pre_calibration_freeze.json", DBF_REL / "operator_checks.json",
                     DBF_REL / "analysis_pretest_freeze.json", DBF_REL / "analyze_dbf.py"):
        add(relative)
    analysis_freeze = load(dbf / "analysis_pretest_freeze.json")
    assert sha(dbf / "analyze_dbf.py") == analysis_freeze["analysis_sha256"]
    authorization = load(archive / PHYSICAL_REL / "acquisition_authorization.json")
    assert sha(archive / PHYSICAL_REL / "algorithm_freeze.json") == authorization["algorithm_freeze_sha256"]
    assert sha(archive / PHYSICAL_REL / "protocol.json") == authorization["physical_protocol_sha256"]
    assert sha(archive / PHYSICAL_REL / "adapter.py") == authorization["adapter_sha256"]
    if readiness_protocol:
        relative = DBF_REL / readiness_protocol
        record = load(archive / relative)
        add_hashes(record)
        add(relative)
        marker_relative = relative.parent / "pre_intervention_freeze.json"
        readiness_marker = load(archive / marker_relative)
        assert sha(archive / relative) == readiness_marker["protocol_sha256"]
        assert record["source_sha256"] == sha(dbf / "readiness_intervention.py")
        add(marker_relative)
    return wanted


def copy_checked(archive, project, relative, expected=None):
    source, target = archive / relative, project / relative
    assert source.is_file(), str(source)
    digest = sha(source)
    if expected is not None:
        assert digest == expected
    if target.exists():
        assert sha(target) == digest, ("Existing replay file differs", str(relative))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    assert sha(target) == digest


def copy_selections(archive, project):
    """Explicit archived-test mode only; never called for fresh calibration."""
    for task in TASKS.values():
        for suffix in ("_selection.json", "_selection_freeze.json"):
            copy_checked(archive, project, DBF_REL / (task + suffix))
    copy_checked(archive, project, DBF_REL / "all_selections_before_test.json")


def copy_analysis_records(archive, project, task):
    copy_selections(archive, project)
    folder = archive / DBF_REL / task
    paths = [folder / "completion.json", folder / "guarded_all.json.gz", folder / "issued_all.json.gz"]
    paths += sorted(folder.glob("*_results.json.gz"))
    assert len(paths) > 3, "Archived task outcome files missing"
    for path in paths:
        copy_checked(archive, project, path.relative_to(archive))
    assert not (project / DBF_REL / task / "summary.json").exists(), "Use a fresh analysis output"


def copy_readiness_analysis_records(archive, project, task):
    """The locked-score diagnostic consumes byte-identical original outcomes."""
    relative = DBF_REL / "readiness_intervention"
    copy_checked(archive, project, DBF_REL / task / "summary.json")
    copy_checked(archive, project, relative / "calibration_summary.json")
    folder = archive / relative / task
    paths = [folder / "completion.json", folder / "guarded_all.json.gz"]
    paths += sorted(folder.glob("*_results.json.gz"))
    for path in paths:
        copy_checked(archive, project, path.relative_to(archive))
    assert not (project / relative / task / "summary.json").exists(), "Use a fresh readiness analysis output"


def prepare(args, rebasing, archive, original_project):
    output = args.out.resolve()
    if output == ROOT or ROOT in output.parents or output == archive or archive in output.parents:
        raise ValueError("Output must be outside the delivered package and archived project")
    output.mkdir(parents=True, exist_ok=True)
    state = output / "DBF_REPLAY_STATE.json"
    project = output / "project"
    wanted = closure(archive, original_project, args.readiness_protocol)
    if state.exists():
        record = load(state)
        assert record["protocol_sha256"] == sha(archive / DBF_REL / "protocol.json")
        assert record["launcher_sha256"] == sha(__file__)
        assert record["archive_project"] == str(archive)
        assert project.is_dir()
    else:
        assert not list(output.iterdir()), "Initial output directory must be empty"
        project.mkdir()
        dump(state, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            protocol_sha256=sha(archive / DBF_REL / "protocol.json"), launcher_sha256=sha(__file__),
            archive_project=str(archive), copied_frozen_records_preserve_bytes=True,
            fresh_replay_is_not_preregistration=True, fresh_calibration_contains_archived_secondary_selections=False))
    for relative, digest in wanted.items():
        if args.phase == "readiness-freeze" and Path(relative) in (
                DBF_REL / args.readiness_protocol,
                DBF_REL / Path(args.readiness_protocol).parent / "pre_intervention_freeze.json"):
            continue
        copy_checked(archive, project, Path(relative), digest)
    if args.phase.startswith("readiness-"):
        relative = DBF_REL / args.entry_module
        copy_checked(archive, project, relative)
        # The readiness module's own frozen protocol should list any further
        # code dependency. For its freeze phase, include top-level source files
        # only; no readiness selection/outcome files are imported here.
        for path in (archive / DBF_REL).glob("*.py"):
            copy_checked(archive, project, path.relative_to(archive))
    if args.archived_selections:
        if args.phase not in ("test-rss", "test-harth"):
            raise ValueError("--archived-selections applies only to original DBF/EAvg test phases")
        copy_selections(archive, project)
    if args.phase in ("analyze-rss", "analyze-harth"):
        copy_analysis_records(archive, project, TASKS[args.phase.split("-")[-1]])
    if args.phase in ("readiness-analyze-rss", "readiness-analyze-harth"):
        copy_readiness_analysis_records(archive, project, TASKS[args.phase.split("-")[-1]])
    installed = rebasing.install_paths(project)
    return output, project, installed


def compare_json(actual, archived):
    result = dict(numeric_leaves=0, maximum_absolute_difference=0.0, all_decoded_values_exact=True)
    def compare(a, b, path=""):
        if isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), path
            for key in a:
                compare(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + "/" + str(i))
        elif isinstance(a, bool) or not isinstance(a, (int, float)):
            assert a == b, (path, a, b)
        else:
            assert isinstance(b, (int, float)) and not isinstance(b, bool), path
            error = abs(a-b)
            result["numeric_leaves"] += 1
            result["maximum_absolute_difference"] = max(result["maximum_absolute_difference"], error)
            result["all_decoded_values_exact"] &= a == b
            assert error <= 1e-10 + 1e-12*abs(b), (path, a, b)
    compare(actual, archived)
    return dict(passed=True, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--out", type=Path, help="empty fresh replay root outside the archive/package")
    parser.add_argument("--archive-project", type=Path, default=ROOT / "project")
    parser.add_argument("--reproduction-launcher", type=Path, default=ROOT / "run_reproduction.py")
    parser.add_argument("--archived-selections", action="store_true",
                        help="test archived locked configurations in fresh output; no secondary outcomes copied")
    parser.add_argument("--entry-module", default="readiness_intervention.py",
                        help="readiness source filename relative to the DBF directory")
    parser.add_argument("--entry-function", help="optional explicit readiness callable name")
    parser.add_argument("--entry-args", help="JSON list passed verbatim to the readiness callable")
    parser.add_argument("--readiness-protocol", help="readiness frozen protocol filename, when available")
    args = parser.parse_args()
    if args.phase.startswith("readiness-") and args.readiness_protocol is None:
        args.readiness_protocol = "readiness_intervention/protocol.json"
    archive = args.archive_project.resolve()
    rebasing = module("dbf_portable_filesystem_rebasing", args.reproduction_launcher.resolve())
    original = Path(load(rebasing.ROOT / "SOURCE_IDENTITIES.json")["original_project"])
    wanted = closure(archive, original, args.readiness_protocol)
    if args.phase in ("verify", "readiness-verify") and args.out is None:
        result = dict(passed=True, phase=args.phase, frozen_closure_files=len(wanted),
            frozen_protocol_sha256=sha(archive / DBF_REL / "protocol.json"),
            all_frozen_file_bytes_checked=True, source_rewriting=False,
            no_model_fitting_or_outcome_replay=True)
        print(json.dumps(result, indent=2))
        return
    assert args.out is not None, "--out is required for execution phases"
    # Archive comparisons are decoded before original-workspace reads are
    # denied. Scientific entrypoints subsequently access the copied project.
    expected_summary = None
    if args.phase in ("analyze-rss", "analyze-harth"):
        task = TASKS[args.phase.split("-")[-1]]
        expected_summary = load(archive / DBF_REL / task / "summary.json")
    elif args.phase in ("readiness-analyze-rss", "readiness-analyze-harth"):
        task = TASKS[args.phase.split("-")[-1]]
        expected_summary = load(archive / DBF_REL / "readiness_intervention" / task / "summary.json")
    output, project, installed = prepare(args, rebasing, archive, original)
    dbf = project / DBF_REL
    comparison = None
    if args.phase.startswith("readiness-"):
        runner = module("portable_dbf_readiness", dbf / args.entry_module)
        words = args.phase.split("-")[1:]
        function = args.entry_function or words[0]
        call_args = json.loads(args.entry_args) if args.entry_args is not None else (
            [TASKS[words[-1]]] if words[-1] in TASKS else [])
        assert isinstance(call_args, list), "--entry-args must decode to a list"
        getattr(runner, function)(*call_args)
        if function == "analyze":
            task = call_args[0]
            comparison = compare_json(load(dbf / "readiness_intervention" / task / "summary.json"), expected_summary)
    elif args.phase.startswith("analyze-"):
        analyzer = module("portable_dbf_archived_analysis", dbf / "analyze_dbf.py")
        task = TASKS[args.phase.split("-")[-1]]
        analyzer.run(task)
        comparison = compare_json(load(dbf / task / "summary.json"), expected_summary)
    else:
        runner = module("portable_secondary_dbf", dbf / "run_dbf.py")
        runner.verify()
        if args.phase == "locktest":
            runner.locktest()
        else:
            method, short_task = args.phase.split("-")
            getattr(runner, method)(TASKS[short_task])
    result = dict(passed=True, phase=args.phase, filesystem_rebasing=installed,
        archive_protocol_preserved=True, numerical_source_bytes_unchanged=True,
        function_and_class_bodies_unchanged=True, original_workspace_reads_denied=True,
        output=str(output), comparison=comparison,
        fresh_replay_is_not_preregistration=True)
    dump(output / (args.phase + "_portable_report.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
