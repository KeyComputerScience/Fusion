"""Assemble the verified front-section bibliography and single-file preview.

The three-section prose is edited in front_sections.tex. Candidate catalogs
contain verified metadata and claim boundaries; this script validates coverage
and exports manuscript BibTeX without installing or invoking TeX.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "dubois2016fusion": "dubois2016principles",
    "cuzzolin2025uncertainty": "cuzzolin2025measures",
    "ddef2024": "shao2024ddef",
    "huang2025contextual": "huang2025reliability",
    "mtmsa2024": "liu2024missing",
    "Bregman1967": "bregman1967relaxation",
    "Boyd2004": "boyd2004convex",
    "BeckTeboulle2003": "beck2003mirror",
    "CesaBianchiLugosi2006": "cesabianchi2006prediction",
    "Wintenberger2017": "boa",
    "Joulani2013": "joulani2013delayed",
    "WangStrong1996": "wang1996quality",
    "Lin1991": "lin1991jsd",
    "Kunsch1989": "kunsch1989bootstrap",
    "Dempster1967": "dempster1967bounds",
    "Shafer1976": "shafer1976evidence",
    "BatesGranger1969": "bates1969combination",
    "qmf2023": "qmf",
    "pdf2024": "pdf",
    "tmc2021": "han2021trusted",
    "saef2026": "liu2026saef",
    "ekya2022": "bhardwaj2022ekya",
    "recl2023": "khani2023recl",
    "orric2024": "cai2024orric",
    "zhouedge2019": "zhou2019edge",
    "gama2014": "gama2014drift",
    "lu2019": "lu2019drift",
}
ACCENTS = {
    "é": r"{\'e}", "É": r"{\'E}", "á": r"{\'a}",
    "í": r"{\'i}", "ó": r"{\'o}", "ú": r"{\'u}",
    "ü": r'{\"u}', "ö": r'{\"o}', "ò": r"{\`o}",
    "à": r"{\`a}", "ñ": r"{\~n}", "ã": r"{\~a}",
    "š": r"{\v s}", "Ž": r"{\v Z}", "ė": r"{\.e}",
    "ı": r"{\i}",
}


def tex_accents(text: str) -> str:
    return "".join(ACCENTS.get(character, character) for character in text)


def read_catalog() -> list[dict]:
    records = []
    for filename in ("root_candidates.json", "if_candidates.json",
                     "method_candidates.json", "modern_candidates.json"):
        for record in json.loads((ROOT / filename).read_text()):
            record = dict(record)
            record["bibkey"] = ALIASES.get(record["bibkey"], record["bibkey"])
            record["fields"] = {
                key: tex_accents(str(value)) for key, value in record["fields"].items()
            }
            if "url" not in record["fields"]:
                record["fields"]["url"] = record["source_url"]
            record["source_url"] = record.get(
                "source_url", record.get("sourceurl", record.get("sources", [None])[0])
            )
            record["support_scope"] = record.get(
                "support_scope", record.get("support", "")
            )
            records.append(record)
    return records


def citation_order(source: str) -> list[str]:
    keys = []
    uncommented = re.sub(r"(?m)(?<!\\)%.*$", "", source)
    for match in re.finditer(r"\\cite(?:p|t)?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", uncommented):
        for key in match[1].split(","):
            key = key.strip()
            if key not in keys:
                keys.append(key)
    return keys


def bibtex(record: dict) -> str:
    lines = ["@" + record["type"] + "{" + record["bibkey"] + ","]
    for key, value in record["fields"].items():
        # Protect full titles; metadata capitalization is retained by numeric styles.
        value = "{" + value + "}" if key == "title" else value
        lines.append("  " + key + " = {" + value + "},")
    lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines + ["}"])


def author_text(text: str) -> str:
    authors = []
    for author in text.split(" and "):
        if "," in author:
            family, given = author.split(",", 1)
            authors.append(given.strip() + " " + family.strip())
        else:
            authors.append(author)
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + ", and " + authors[-1]


def bibitem(record: dict) -> str:
    fields = record["fields"]
    lines = [r"\bibitem{" + record["bibkey"] + "}",
             author_text(fields["author"]) + ". " + fields["title"] + "."]
    venue = fields.get("journal", fields.get("booktitle", fields.get("publisher", "")))
    details = r"\emph{" + venue + "}"
    if record["type"] == "inproceedings" and "series" in fields:
        details += ", " + fields["series"]
    if "volume" in fields:
        details += ", " + fields["volume"]
    if "number" in fields:
        details += "(" + fields["number"] + ")"
    if "pages" in fields:
        details += ", " + fields["pages"]
    details += ", " + fields["year"] + "."
    lines.append(details)
    if "doi" in fields:
        lines.append(r"\href{https://doi.org/" + fields["doi"] + "}{doi: " + fields["doi"] + "}.")
    else:
        lines.append(r"\url{" + fields["url"] + "}.")
    return "\n".join(lines)


def main() -> None:
    source_path = ROOT / "front_sections.tex"
    source = source_path.read_text()
    # Normalize one-time draft aliases and leave an idempotent manuscript source.
    for old, new in ALIASES.items():
        source = re.sub(r"(?<=[{,])" + re.escape(old) + r"(?=[,}])", new, source)
    source_path.write_text(source)
    catalog = read_catalog()
    by_key = {record["bibkey"]: record for record in catalog}
    keys = citation_order(source)
    if len(by_key) != len(catalog):
        raise ValueError("Duplicate bibliography keys")
    if set(keys) != set(by_key):
        raise ValueError({"missing": sorted(set(keys) - set(by_key)),
                          "unused": sorted(set(by_key) - set(keys))})
    if len(keys) <= 30:
        raise ValueError("More than thirty cited references are required")
    records = [by_key[key] for key in keys]
    dois = [record["fields"]["doi"].lower() for record in records if "doi" in record["fields"]]
    if len(dois) != len(set(dois)):
        raise ValueError("Duplicate reference DOI")
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source)
    if len(labels) != len(set(labels)) or set(refs) - set(labels):
        raise ValueError("Duplicate or unresolved internal labels")
    bibliography = "% Verified primary-source metadata, checked 2026-10-01.\n"
    bibliography += "% Exactly the references cited by front_sections.tex; no padding entries.\n\n"
    bibliography += "\n\n".join(bibtex(record) for record in records) + "\n"
    # Simple delimiter validation in addition to exact catalog-to-source checks.
    depth = 0
    for character in bibliography:
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
        if depth < 0:
            raise ValueError("Unbalanced BibTeX delimiters")
    if depth:
        raise ValueError("Unbalanced BibTeX delimiters")
    (ROOT / "front_sections.bib").write_text(bibliography)

    preamble = r"""% Single-file preview: bibliography is embedded for the native compiler.
% Use front_sections.tex + front_sections.bib in the complete journal manuscript.
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,microtype}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\title{Quality-Conditioned Reliability Fusion for Adaptive Edge AI Services}
\author{}
\date{}
\begin{document}
\maketitle
"""
    preview = preamble + source + "\n" + r"\begin{thebibliography}{99}" + "\n"
    preview += "\n\n".join(bibitem(record) for record in records)
    preview += "\n" + r"\end{thebibliography}" + "\n" + r"\end{document}" + "\n"
    (ROOT / "front_sections_preview.tex").write_text(preview)
    if_count = sum(record["fields"].get("journal") == "Information Fusion" for record in records)
    report = {
        "checked_date": "2026-10-01", "cited_references": len(keys),
        "information_fusion_articles": if_count, "unique_dois": len(set(dois)),
        "unresolved_citations": [], "uncited_bibliography_entries": [],
        "unresolved_internal_labels": [], "duplicate_labels": [],
        "sections": re.findall(r"\\section\{([^}]+)\}", source),
        "citation_order": keys, "native_compilation": "pending",
        "scope": "Metadata/abstract verification and correspondence to executed revision; no claim of reproducing uncoded methods.",
    }
    (ROOT / "front_sections_verification.json").write_text(json.dumps(report, indent=2))
    (ROOT / "front_sections_reference_catalog.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    with (ROOT / "front_sections_reference_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["number", "bibkey", "title", "year", "venue", "doi", "primary_source", "supported_scope"])
        writer.writeheader()
        for number, record in enumerate(records, 1):
            fields = record["fields"]
            writer.writerow({"number": number, "bibkey": record["bibkey"],
                             "title": fields["title"], "year": fields["year"],
                             "venue": fields.get("journal", fields.get("booktitle", fields.get("publisher", ""))),
                             "doi": fields.get("doi", ""), "primary_source": record["source_url"],
                             "supported_scope": record["support_scope"]})
    print(json.dumps({"references": len(keys), "information_fusion_articles": if_count,
                      "sections": report["sections"]}))


if __name__ == "__main__":
    main()
