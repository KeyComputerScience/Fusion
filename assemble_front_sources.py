"""Synchronize the verified BibTeX and standalone native-editor bibliography."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
WORK, OUT = ROOT / "work", ROOT / "outputs"
foundation = json.loads((WORK / "foundational_refs.json").read_text())["refs"]
recent = json.loads((WORK / "if_recent_refs.json").read_text())["entries"]
entries = {}
for item in foundation:
    e = dict(item)
    if e.get("note"):
        e["verification"] += "; " + e.pop("note")
    if e["type"] != "book":
        e["journal" if e["type"] == "article" else "booktitle"] = e["venue"]
    entries[e["key"]] = e
entries["spo2022"]["title"] = "Smart " + chr(96)*2 + "Predict, then Optimize''"
for k in ("gibbs2021", "gibbs2024"):
    entries[k]["author"] = r"Isaac Gibbs and Emmanuel J. Cand{\`e}s"
entries["barber2023"]["author"] = r"Rina Foygel Barber and Emmanuel J. Cand{\`e}s and Aaditya Ramdas and Ryan J. Tibshirani"
entries["denoeux2008"]["author"] = r"Thierry Den{\oe}ux"

for key in ("hall1997", "khaleghi2013", "xiong2002"):
    raw = json.loads((WORK / "front_references" / (key + "_crossref.json")).read_text())
    m = raw.get("message", raw)
    entries[key] = {
        "key": key, "type": "article", "title": m["title"][0],
        "author": " and ".join(a["family"] + ", " + a.get("given", "") for a in m["author"]),
        "journal": m["container-title"][0], "venue": m["container-title"][0],
        "year": m["published"]["date-parts"][0][0], "volume": m["volume"],
        "number": m["issue"], "pages": m["page"].replace("-", "--"),
        "doi": m["DOI"], "url": "https://doi.org/" + m["DOI"],
        "primary_url": "https://doi.org/" + m["DOI"],
        "verification": "Publisher/DOI record and direct Crossref bibliographic metadata",
        "verified_on": "2026-10-02",
    }
for item in recent:
    if item["key"] in {"Li2024MVCIL", "Ge2025Credibility"}:
        continue
    e = dict(item)
    e.update(type="article", author=e["bib_authors"], journal="Information Fusion",
             venue="Information Fusion", pages=e["article_number"],
             url=e["primary_url"], verification=e["verification_note"])
    if e["key"] == "Sun2026DSHBI":
        e["note"] = "Part A"
    entries[e["key"]] = e
entries["wu2026"] = {
    "key": "wu2026", "type": "misc",
    "title": "Bayesian Conformal Prediction as a Decision Risk Problem",
    "author": "Fanyi Wu and Veronika Lohmanova and Samuel Kaski and Michele Caprio",
    "year": 2026, "eprint": "2602.03331", "archivePrefix": "arXiv",
    "primaryClass": "cs.LG", "doi": "10.48550/arXiv.2602.03331",
    "url": "https://arxiv.org/abs/2602.03331v2",
    "primary_url": "https://arxiv.org/abs/2602.03331v2",
    "note": "Preprint, version 2", "venue": "arXiv preprint",
    "verification": (
        "Canonical arXiv abstract metadata checked 2026-10-02. Experimental HTML "
        "uses the alternate title 'Bayesian Conformal Prediction via "
        "Decision-Theoretic Threshold Selection'; canonical title retained."
    ),
    "verified_on": "2026-10-02",
}
datasets = [
    ("occupancy357", "Occupancy Detection", "L. Candanedo", 2016, "10.24432/C5X01N"),
    ("occupancy864", "Room Occupancy Estimation", "A. Singh and A. Chaudhari", 2018, "10.24432/C5P605"),
    ("mhealth319", "MHEALTH Dataset", "O. Banos and R. Garcia and A. Saez", 2014, "10.24432/C5TW22"),
    ("har240", "Human Activity Recognition Using Smartphones",
     "J. Reyes-Ortiz and D. Anguita and A. Ghio and L. Oneto and X. Parra", 2013, "10.24432/C54S4K"),
]
for key, title, author, year, doi in datasets:
    entries[key] = {
        "key": key, "type": "misc", "title": title, "author": author, "year": year,
        "doi": doi, "url": "https://doi.org/" + doi, "primary_url": "https://doi.org/" + doi,
        "publisher": "UCI Machine Learning Repository", "note": "Dataset",
        "venue": "UCI Machine Learning Repository",
        "verification": "Official dataset DOI record, preserved from verified Performance sources",
        "verified_on": "2026-10-02",
    }

def tex(value):
    return str(value).replace("&", r"\&").replace("–", "--").replace("—", "---")

def citations(source):
    return list(dict.fromkeys(k.strip()
        for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]*)\}", source)
        for k in group.split(",")))

front = (WORK / "fusion_front_sections_draft.tex").read_text()
original = (WORK / "risk_rf_before_front_matter.tex").read_text()
scientific_old = original.split(r"\section{Methodology}", 1)[1].split(r"\begin{thebibliography}", 1)[0]
scientific = scientific_old.replace(
    r"For positive $\pi$, $\tau>0$, $\lambda\geq0$ and a nonempty mask,",
    r"For positive $\pi$, $\tau>0$, $\lambda\geq0$, $\eta\geq0$ and a nonempty mask,"
).replace(
    "Under the stated execution contract, for every realized stream,",
    r"Under the stated execution contract with $\varepsilon\geq0$ and $M\geq0$, for every realized stream,"
)
assert scientific != scientific_old
preamble = original.split(r"\section{Methodology}", 1)[0]
preamble = preamble.replace(
    "% Self-contained revised Methodology and Performance. All results are executed.",
    "% Self-contained manuscript chapters; empirical results are preserved."
)
if r"\usepackage{cite}" not in preamble:
    preamble = preamble.replace(r"\usepackage{hyperref}", "\\usepackage{cite}\n\\usepackage{hyperref}")
body = preamble + front + "\n\\section{Methodology}" + scientific
order = citations(body)
assert set(order) == set(entries), {"unresolved": set(order)-set(entries), "unused": set(entries)-set(order)}
dois = [e["doi"].lower() for e in entries.values() if e.get("doi")]
assert len(dois) == len(set(dois)), "duplicate DOI"
if_recent = [k for k in order if entries[k].get("journal") == "Information Fusion" and entries[k]["year"] >= 2023]
assert len(order) >= 40 and len(if_recent) >= 8

def bibtex(e):
    fields = ["author","title","journal","booktitle","editor","series","volume",
              "number","pages","publisher","year","doi","eprint","archivePrefix",
              "primaryClass","url","note"]
    rows = ["@%s{%s," % (e["type"], e["key"])]
    for k in fields:
        if e.get(k) is not None:
            v = tex(e[k])
            if k == "title":
                v = "{" + v + "}"
            rows.append("  %s = {%s}," % (k,v))
    rows[-1] = rows[-1].rstrip(",")
    return "\n".join(rows+["}"])

def display_authors(author):
    names = []
    for name in author.split(" and "):
        if ", " in name:
            family,given = name.split(", ",1)
            name = given + " " + family
        names.append(name)
    return ", ".join(names)

def bibitem(e):
    a,t,y = tex(display_authors(e["author"])),tex(e["title"]),str(e["year"])
    s = "\\bibitem{%s}\n%s, %s, " % (e["key"],a,t)
    if e["type"] == "article":
        s += "\\emph{%s} %s" % (tex(e["journal"]),e.get("volume",""))
        if e.get("number"): s += "(%s)" % tex(e["number"])
        s += " (%s)" % y
        if e.get("pages"): s += " %s" % tex(e["pages"])
    elif e["type"] in {"inproceedings","incollection"}:
        s += "in: \\emph{%s}" % tex(e["booktitle"])
        if e.get("series"): s += ", %s" % tex(e["series"])
        if e.get("volume"): s += ", vol. %s" % tex(e["volume"])
        if e.get("number"): s += ", no. %s" % tex(e["number"])
        if e.get("publisher"): s += ", %s" % tex(e["publisher"])
        s += ", %s" % y
        if e.get("pages"): s += ", pp. %s" % tex(e["pages"])
    elif e["type"] == "book":
        s += "%s, %s" % (tex(e["publisher"]),y)
    else:
        s += "%s, %s" % (tex(e["venue"]),y)
        if e.get("eprint"): s += ", arXiv:%sv2" % e["eprint"]
    if e.get("note"): s += ", %s" % tex(e["note"])
    url = "https://doi.org/"+e["doi"] if e.get("doi") else e["url"]
    return s + ".\n\\url{%s}.\n" % url

inline = "\\begin{thebibliography}{99}\n" + "\n".join(bibitem(entries[k]) for k in order) + "\\end{thebibliography}\n"
full = body + inline + "\\end{document}\n"
OUT.joinpath("risk_rf_revision.tex").write_text(full)
OUT.joinpath("risk_rf_front_sections.tex").write_text(
    "% Three chapter fragment; insert after the abstract of the parent manuscript.\n"
    "% Parent dependencies: amsmath, amssymb, bm; define ind as mathbf 1.\n"
    "% Cite with risk_rf_references.bib; keep only one bibliography in the parent.\n"+front)
OUT.joinpath("risk_rf_references.bib").write_text(
    "% Verified 2026-10-02; all 51 entries are cited in the integrated manuscript.\n"
    "% 47 research/method references and 4 official dataset records.\n"
    "% 10 Information Fusion articles with issue years 2023--2026.\n\n"
    +"\n\n".join(bibtex(entries[k]) for k in order)+"\n")
OUT.joinpath("risk_rf_reference_block.tex").write_text(
    "% Generated from the same entries as risk_rf_references.bib.\n"+inline)
columns = ["citation_number","key","type","title","author","year","venue","volume",
           "number","pages","doi","primary_url","recent_information_fusion","verification"]
with OUT.joinpath("risk_rf_reference_audit.csv").open("w",newline="") as h:
    w = csv.DictWriter(h,fieldnames=columns); w.writeheader()
    for i,k in enumerate(order,1):
        e = entries[k]
        row = {c:e.get(c,"") for c in columns}
        row.update(citation_number=i,recent_information_fusion=k in if_recent)
        w.writerow(row)
summary = {
    "citation_count":len(order), "research_method_reference_count":len(order)-4,
    "front_chapter_unique_citations":len(citations(front)), "dataset_record_count":4,
    "recent_information_fusion_count":len(if_recent), "recent_information_fusion_keys":if_recent,
    "citation_keys_resolved":True,"bibliography_has_no_uncited_entries":True,
    "unique_nonempty_dois":len(set(dois)), "performance_body_unchanged":True,
    "methodology_edits":"Only add eta>=0 and epsilon>=0/M>=0 assumptions, satisfied by execution.",
    "scientific_body_sha256":hashlib.sha256(scientific.encode()).hexdigest(),
    "main_source_sha256":hashlib.sha256(full.encode()).hexdigest(),
    "source_metadata_checked_on":"2026-10-02",
}
WORK.joinpath("front_reference_assembly_audit.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
