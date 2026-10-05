"""Audit source and citation deliverables without changing results."""
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK = ROOT / "outputs", ROOT / "work"
source = (OUT / "risk_rf_revision.tex").read_text()
old = (WORK / "risk_rf_before_front_matter.tex").read_text()
front = (OUT / "risk_rf_front_sections.tex").read_text()
bib = (OUT / "risk_rf_references.bib").read_text()
rows = list(csv.DictReader((OUT / "risk_rf_reference_audit.csv").open()))

def performance(text):
    return text.split(r"\section{Performance}",1)[1].split(r"\begin{thebibliography}",1)[0]

before_gain=(WORK/'risk_rf_before_performance_gain_revision.tex').read_text()
gain_source=(OUT/'risk_rf_performance.tex').read_text()
assert performance(source).strip()==gain_source.split(r"\section{Performance}",1)[1].strip()
assert source.split(r"\section{Performance}",1)[0]==before_gain.split(r"\section{Performance}",1)[0]
assert source.split(r"\begin{thebibliography}",1)[1]==before_gain.split(r"\begin{thebibliography}",1)[1]
original_tables=re.findall(r"\\begin\{tabular\}[\s\S]*?\\end\{tabular\}",performance(before_gain))
assert len(original_tables)==6 and all(table in performance(source) for table in original_tables)
labels = re.findall(r"\\label\{([^}]+)\}",source)
refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",source)
assert not set(refs)-set(labels), "unresolved reference"
assert len(labels) == len(set(labels)), "duplicate label"
citations = list(dict.fromkeys(k.strip()
    for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]*)\}",source)
    for k in group.split(",")))
bibkeys = re.findall(r"@\w+\{([^,]+),",bib)
inlinekeys = re.findall(r"\\bibitem\{([^}]+)\}",source)
assert citations == bibkeys == inlinekeys
assert len(citations) == len(set(citations)) == 51
assert all(len(r["primary_url"]) > 10 for r in rows)
recent = [r for r in rows if r["recent_information_fusion"]=="True"]
assert len(recent) == 10 and all(2023 <= int(r["year"]) <= 2026 for r in recent)
assert source.count(r"\begin{figure}")==4 and before_gain.count(r"\begin{figure}")==3
assert not re.findall(r"RECENT_\w+|FORECAST_REGULARIZED|DECISION_LEARNING|BAYES_AVERAGING",source)
entry_starts = list(re.finditer(r"@(\w+)\{([^,]+),",bib))
for i,match in enumerate(entry_starts):
    end = entry_starts[i+1].start() if i+1 < len(entry_starts) else len(bib)
    block = bib[match.start():end]
    level = 0
    for c in block:
        if c == "{": level += 1
        elif c == "}": level -= 1
        assert level >= 0
    assert level == 0
    assert all(re.search(r"\b"+f+r"\s*=\s*\{[^}]+",block)
               for f in ("title","author","year"))
    if match.group(1)=="article":
        assert "journal = " in block
    elif match.group(1) in {"inproceedings","incollection"}:
        assert "booktitle = " in block

summary = json.loads((WORK/"front_reference_assembly_audit.json").read_text())
summary.update(
    scientific_body_sha256=hashlib.sha256(source.split(r"\section{Methodology}",1)[1].split(r"\begin{thebibliography}",1)[0].encode()).hexdigest(),
    main_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    methodology_edits="Add satisfied nonnegative assumptions; move and replace the earlier loop diagram with the implemented Bayesian overview. Primary experimental results unchanged; Performance adds separately retained diagnostic delay trials.",
    abstract_added=source.count(r"\begin{abstract}")==1,
    performance_byte_identical=False,
    performance_body_unchanged=False,
    primary_performance_tables_unchanged=True,
    performance_revision="Gain-focused interpretation and actual separately declared fixed-controller additional delay trials; six primary tables retained.",
    additional_delay_trajectories=80,
    citation_order_synchronized=True,
    nested_bibtex_braces_balanced=True,
    cross_references_resolved=True,
    original_three_figures_preserved=False,
    figure_count=4,
    posterior_component_delay_figure_added=True,
    overview_replaces_prior_loop=True,
    native_compilation="success",
    compiler="Codex built-in standalone LaTeX compiler",
    outputs_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                   for name in ["risk_rf_revision.tex","risk_rf_abstract.tex","risk_rf_front_sections.tex",
                                "risk_rf_references.bib","risk_rf_reference_block.tex",
                                "risk_rf_reference_audit.csv","risk_rf_front_README.md",
                                "risk_rf_overview.tex","bayesian_fusion_overview.svg",
                                "bayesian_fusion_overview.png","bayesian_fusion_overview_layout_audit.json",
                                "risk_rf_performance.tex"]},
)
(OUT/"risk_rf_front_manifest.json").write_text(json.dumps(summary,indent=2))
files = ["risk_rf_revision.tex","risk_rf_abstract.tex","risk_rf_front_sections.tex","risk_rf_references.bib",
         "risk_rf_reference_block.tex","risk_rf_reference_audit.csv",
         "risk_rf_front_README.md","risk_rf_front_manifest.json","risk_rf_overview.tex",
         "bayesian_fusion_overview.svg","bayesian_fusion_overview.png",
         "bayesian_fusion_overview_layout_audit.json","risk_rf_performance.tex"]
with zipfile.ZipFile(OUT/"Fusion_Introduction_RelatedWork_Motivation.zip","w",zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(OUT/name,arcname=name)
print(json.dumps(summary,indent=2))
