"""Cheap safeguards for the portable runner; no experiments are executed here."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce_revision as runner


def source_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "source"
    code = bundle / "code"
    data = bundle / "data/original_synthetic"
    code.mkdir(parents=True)
    data.mkdir(parents=True)
    (code / "portable.py").write_text("print('portable')\n")
    (data / "manifest.json").write_text("{}")
    for name in ("performance_template.tex", "sota_section.tex", "update_section.tex"):
        (bundle / name).write_text(name)
    monkeypatch.setattr(runner, "HERE", code)
    monkeypatch.setattr(runner, "BUNDLE", bundle)
    return bundle


@pytest.mark.parametrize("kind", ["nonempty", "file", "code_descendant", "data_descendant"])
def test_recomputation_refuses_unsafe_destination(tmp_path, monkeypatch, kind):
    bundle = source_bundle(tmp_path, monkeypatch)
    destination = tmp_path / "destination"
    if kind == "nonempty":
        destination.mkdir()
        (destination / "existing_evidence").write_text("preserve")
    elif kind == "file":
        destination.write_text("preserve")
    elif kind == "code_descendant":
        destination = bundle / "code/recursive_copy"
    else:
        destination = bundle / "data/original_synthetic/recursive_copy"
    with pytest.raises(ValueError):
        runner.prepare_destination(destination)


def test_fresh_copy_contains_data_and_manuscript_inputs(tmp_path, monkeypatch):
    source_bundle(tmp_path, monkeypatch)
    destination = runner.prepare_destination(tmp_path / "fresh")
    assert (destination / "code/portable.py").is_file()
    assert (destination / "data/original_synthetic/manifest.json").is_file()
    assert all((destination / name).is_file() for name in
               ("performance_template.tex", "sota_section.tex", "update_section.tex"))
    assert not (destination / "results").exists()


def test_full_stage_order_assembles_after_all_evidence(tmp_path, monkeypatch):
    bundle = tmp_path / "fresh"
    (bundle / "code").mkdir(parents=True)
    for name in ("analyze_revision.py", "build_manuscript.py", "export_table_data.py"):
        (bundle / "code" / name).touch()
    budget = bundle / "results/update_necessity/budget_sensitivity"
    budget.mkdir(parents=True)
    (budget / "analysis.json").write_text("{}")
    commands = []
    monkeypatch.setattr(runner, "prepare_destination", lambda destination: bundle)
    monkeypatch.setattr(runner, "environment", lambda: {})
    monkeypatch.setattr(runner, "invoke", lambda bundle, arguments, **kwargs: commands.append(arguments))
    monkeypatch.setattr(runner, "plots", lambda bundle: commands.append(["plots"]))
    monkeypatch.setattr(runner, "verify", lambda bundle, **kwargs: commands.append(["verify"]))
    runner.full_run(bundle, 2)
    stages = [str(command[0]) for command in commands]
    assert stages.index("code/run_original.py") < stages.index("code/quality_benchmark.py")
    assert stages.index("code/quality_benchmark.py") < stages.index("code/source_ablation.py")
    assert stages.index("code/update_necessity.py") < stages.index("code/update_budget_sensitivity.py")
    assert stages.index("code/run_official_sota.py") < stages.index("code/analyze_revision.py")
    assert stages.index("code/analyze_revision.py") < stages.index("code/build_manuscript.py")
    assert stages.index("code/build_manuscript.py") < stages.index("code/export_table_data.py")
    assert stages.index("code/export_table_data.py") < stages.index("plots") < stages.index("verify")


def test_dose_plot_requires_retained_budget_data(tmp_path, monkeypatch, capsys):
    (tmp_path / "code").mkdir()
    (tmp_path / "code/plot_quality.py").touch()
    (tmp_path / "code/plot_update_dose.py").touch()
    commands = []
    monkeypatch.setattr(runner, "invoke", lambda bundle, arguments: commands.append(arguments))
    runner.plots(tmp_path)
    assert commands == [[Path("code/plot_quality.py")]]
    assert "Skipping dose-response plot" in capsys.readouterr().out
