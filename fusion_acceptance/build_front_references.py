"""Build verified front-matter references and a single-source preview block."""
from pathlib import Path
import re
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def entries(text):
    result = {}
    for match in re.finditer(r"@\w+\s*\{([^,]+),", text):
        start = match.start()
        level = 0
        for pos in range(text.index("{", start), len(text)):
            if text[pos] == "{":
                level += 1
            elif text[pos] == "}":
                level -= 1
                if level == 0:
                    result[match.group(1).strip()] = text[start:pos + 1]
                    break
    return result

catalog = {}
for filename in ("foundational_refs.bib", "if_recent_refs.bib"):
    catalog.update(entries((ROOT / "work" / filename).read_text()))

manual = r"""
@article{khaleghi2013,
 author = {Khaleghi, Bahador and Khamis, Alaa and Karray, Fakhreddine O. and Razavi, Saiedeh N.},
 title = {Multisensor data fusion: A review of the state-of-the-art},
 journal = {Information Fusion}, year = {2013}, volume = {14}, number = {1}, pages = {28--44},
 doi = {10.1016/j.inffus.2011.08.001},
 url = {https://www.sciencedirect.com/science/article/abs/pii/S1566253511000558}
}
@article{Li2023Mixtures,
 author = {Li, Tiancheng and Liang, Haozhe and Xiao, Bing and Pan, Quan and He, You},
 title = {Finite mixture modeling in time series: A survey of Bayesian filters and fusion approaches},
 journal = {Information Fusion}, year = {2023}, volume = {98}, pages = {101827},
 doi = {10.1016/j.inffus.2023.101827},
 url = {https://www.sciencedirect.com/science/article/pii/S1566253523001434}
}
@article{blackwell1953,
 author = {Blackwell, David}, title = {Equivalent comparisons of experiments},
 journal = {The Annals of Mathematical Statistics}, year = {1953}, volume = {24}, number = {2}, pages = {265--272},
 doi = {10.1214/aoms/1177729032},
 url = {https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full}
}
@article{rockafellar2002,
 author = {Rockafellar, R. Tyrrell and Uryasev, Stanislav},
 title = {Conditional value-at-risk for general loss distributions},
 journal = {Journal of Banking \& Finance}, year = {2002}, volume = {26}, number = {7}, pages = {1443--1471},
 doi = {10.1016/S0378-4266(02)00271-6},
 url = {https://www.sciencedirect.com/science/article/pii/S0378426602002716}
}
@inproceedings{dbf2025,
 author = {Bezirganyan, Grigor and Sellami, Sana and Berti-Equille, Laure and Fournier, S{\'e}bastien},
 title = {Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion},
 booktitle = {Proceedings of the 28th International Conference on Artificial Intelligence and Statistics},
 series = {Proceedings of Machine Learning Research}, volume = {258}, pages = {3142--3150}, year = {2025}, publisher = {PMLR},
 url = {https://proceedings.mlr.press/v258/bezirganyan25a.html}
}
@misc{rss348,
 author = {Bacciu, Davide and Barsocchi, Paolo and Chessa, Stefano and Gallicchio, Claudio and Micheli, Alessio},
 title = {Indoor User Movement Prediction from RSS Data}, year = {2014},
 howpublished = {UCI Machine Learning Repository}, doi = {10.24432/C5761H},
 url = {https://archive.ics.uci.edu/dataset/348/indoor+user+movement+prediction+from+rss+data}
}
@misc{daphnet245,
 author = {Roggen, Daniel and Plotnik, Meir and Hausdorff, Jeffrey},
 title = {Daphnet Freezing of Gait}, year = {2010},
 howpublished = {UCI Machine Learning Repository}, doi = {10.24432/C56K78},
 url = {https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait}
}
@article{howard2021,
 author = {Howard, Steven R. and Ramdas, Aaditya and McAuliffe, Jon and Sekhon, Jasjeet},
 title = {Time-uniform, nonparametric, nonasymptotic confidence sequences},
 journal = {The Annals of Statistics}, year = {2021}, volume = {49}, number = {2}, pages = {1055--1080},
 doi = {10.1214/20-AOS1991}, url = {https://arxiv.org/abs/1810.08240}
}
@article{jaynes1957,
 author = {Jaynes, E. T.}, title = {Information Theory and Statistical Mechanics},
 journal = {Physical Review}, year = {1957}, volume = {106}, number = {4}, pages = {620--630},
 doi = {10.1103/PhysRev.106.620}, url = {https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620}
}
@book{nelsen2006,
 author = {Nelsen, Roger B.}, title = {An Introduction to Copulas},
 edition = {2}, publisher = {Springer New York}, year = {2006},
 doi = {10.1007/0-387-28678-0}, url = {https://link.springer.com/book/10.1007/0-387-28678-0}
}
@article{dempster1977,
 author = {Dempster, A. P. and Laird, N. M. and Rubin, D. B.},
 title = {Maximum Likelihood from Incomplete Data Via the {EM} Algorithm},
 journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
 year = {1977}, volume = {39}, number = {1}, pages = {1--22},
 doi = {10.1111/j.2517-6161.1977.tb01600.x}, url = {https://academic.oup.com/jrsssb/article/39/1/1/7027539}
}
@article{rubin1976,
 author = {Rubin, Donald B.}, title = {Inference and missing data},
 journal = {Biometrika}, year = {1976}, volume = {63}, number = {3}, pages = {581--592},
 doi = {10.1093/biomet/63.3.581}, url = {https://academic.oup.com/biomet/article-abstract/63/3/581/270932}
}
@book{little2019,
 author = {Little, Roderick J. A. and Rubin, Donald B.}, title = {Statistical Analysis with Missing Data},
 edition = {3}, publisher = {Wiley}, year = {2019},
 doi = {10.1002/9781119482260}, url = {https://onlinelibrary.wiley.com/doi/book/10.1002/9781119482260}
}
@article{ledoit2004,
 author = {Ledoit, Olivier and Wolf, Michael}, title = {A well-conditioned estimator for large-dimensional covariance matrices},
 journal = {Journal of Multivariate Analysis}, year = {2004}, volume = {88}, number = {2}, pages = {365--411},
 doi = {10.1016/S0047-259X(03)00096-4}, url = {https://www.sciencedirect.com/science/article/pii/S0047259X03000964}
}
@article{white1980,
 author = {White, Halbert},
 title = {A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity},
 journal = {Econometrica}, year = {1980}, volume = {48}, number = {4}, pages = {817--838},
 doi = {10.2307/1912934}, url = {https://www.jstor.org/stable/1912934}
}
@inproceedings{guo2017,
 author = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
 title = {On Calibration of Modern Neural Networks},
 booktitle = {Proceedings of the 34th International Conference on Machine Learning},
 series = {Proceedings of Machine Learning Research}, year = {2017}, volume = {70}, pages = {1321--1330}, publisher = {PMLR},
 url = {https://proceedings.mlr.press/v70/guo17a.html}
}
@inproceedings{ovadia2019,
 author = {Ovadia, Yaniv and Fertig, Emily and Ren, Jie and Nado, Zachary and Sculley, D. and Nowozin, Sebastian and Dillon, Joshua and Lakshminarayanan, Balaji and Snoek, Jasper},
 title = {Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift},
 booktitle = {Advances in Neural Information Processing Systems}, year = {2019}, volume = {32},
 url = {https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html}
}
@article{brier1950,
 author = {Brier, Glenn W.}, title = {Verification of forecasts expressed in terms of probability},
 journal = {Monthly Weather Review}, year = {1950}, volume = {78}, number = {1}, pages = {1--3},
 doi = {10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2},
 url = {https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2}
}
@article{parzen1962,
 author = {Parzen, Emanuel}, title = {On Estimation of a Probability Density Function and Mode},
 journal = {The Annals of Mathematical Statistics}, year = {1962}, volume = {33}, number = {3}, pages = {1065--1076},
 doi = {10.1214/aoms/1177704472},
 url = {https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-33/issue-3/On-Estimation-of-a-Probability-Density-Function-and-Mode/10.1214/aoms/1177704472.full}
}
@article{rosenblatt1956,
 author = {Rosenblatt, Murray}, title = {Remarks on Some Nonparametric Estimates of a Density Function},
 journal = {The Annals of Mathematical Statistics}, year = {1956}, volume = {27}, number = {3}, pages = {832--837},
 doi = {10.1214/aoms/1177728190},
 url = {https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-3/Remarks-on-Some-Nonparametric-Estimates-of-a-Density-Function/10.1214/aoms/1177728190.full}
}
@article{gneiting2013,
 author = {Gneiting, Tilmann and Ranjan, Roopesh}, title = {Combining predictive distributions},
 journal = {Electronic Journal of Statistics}, year = {2013}, volume = {7}, pages = {1747--1782},
 doi = {10.1214/13-EJS823}, url = {https://arxiv.org/abs/1106.1638}
}
"""
catalog.update(entries(manual))
keys = [
    "khaleghi2013", "baltrusaitis2019", "qmf2023", "pdf2024", "dbf2025",
    "Han2026QPBayes", "uhlmann2003", "Li2023Mixtures", "blackwell1953",
    "Zhang2024CommonInfo", "rockafellar2002", "gneiting2007", "gibbs2021",
    "gibbs2024", "joulani2013", "rss348", "daphnet245", "han2021",
    "Shao2024DDEF", "Huang2025Evidential", "Folgado2023Explainability",
    "ma2022", "Liu2024MTMSA", "donti2017", "spo2022", "howard2021",
    "jaynes1957", "nelsen2006", "dempster1977", "rubin1976", "little2019",
    "ledoit2004", "white1980", "guo2017", "ovadia2019", "brier1950",
    "parzen1962", "rosenblatt1956", "gneiting2013", "Hu2024DelayedFusion",
]
assert len(keys) == len(set(keys))

def field(entry, name):
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*\{", entry, re.I)
    if not match:
        return ""
    start = match.end()
    level = 1
    for pos in range(start, len(entry)):
        if entry[pos] == "{":
            level += 1
        elif entry[pos] == "}":
            level -= 1
            if level == 0:
                return entry[start:pos]
    raise ValueError(name)

def author_text(value):
    names = []
    for author in value.split(" and "):
        if "," in author:
            family, given = author.split(",", 1)
            names.append(given.strip() + " " + family.strip())
        else:
            names.append(author)
    return ", ".join(names)

block = [r"\begin{thebibliography}{99}"]
for key in keys:
    item = catalog[key]
    authors = author_text(field(item, "author"))
    title = field(item, "title")
    venue = field(item, "journal") or field(item, "booktitle") or field(item, "howpublished") or field(item, "publisher")
    edition = field(item, "edition")
    if edition:
        venue += ", edition " + edition
    volume, number, pages, year = (field(item, f) for f in ("volume", "number", "pages", "year"))
    locus = venue
    if volume:
        locus += " " + volume
        if number:
            locus += "(" + number + ")"
    locus += " (" + year + ")"
    if pages:
        locus += " " + pages
    block += [r"\bibitem{" + key + "}", authors + ", " + title + ", " + locus + "."]
    doi = field(item, "doi")
    url = "https://doi.org/" + quote(doi, safe="/()-.;:") if doi else field(item, "url")
    block += [r"\url{" + url + "}."]
block += [r"\end{thebibliography}"]

(OUT / "references.bib").write_text("% Verified primary sources for CJRT front matter.\n\n" + "\n\n".join(catalog[k] for k in keys) + "\n")
(OUT / "bibliography_block.tex").write_text("\n".join(block) + "\n")
front = (OUT / "front.tex").read_text()
used = set()
for match in re.finditer(r"\\cite\{([^}]+)\}", front):
    used.update(match.group(1).split(","))
assert used <= set(keys), used - set(keys)
assert set(keys) <= used, set(keys) - used
recent = sum(field(catalog[k], "journal") == "Information Fusion" and int(field(catalog[k], "year")) >= 2023 for k in keys)
print({"references": len(keys), "cited_keys": len(used), "recent_information_fusion": recent})
