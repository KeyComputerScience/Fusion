from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
target=root/"outputs/risk_rf_revision.tex"
source=target.read_text()
(root/"work/risk_rf_before_overview.tex").write_text(source)
fragment=(root/"work/bayesian_overview_section.tex").read_text()
pattern=r"\\begin\{figure\}\[t\]\n\\centering\n\\begin\{tikzpicture\}\[>=stealth, every node/.style=\{font=\\small\},.*?\\label\{fig:loop\}\n\\end\{figure\}\n"
source,n=re.subn(pattern,"",source,flags=re.S)
assert n==1, "Prior loop figure must occur exactly once"
assert r"\section{Overview}" not in source
source=source.replace(r"\section{Methodology}",fragment+"\n"+r"\section{Methodology}",1)
assert source.count(r"\begin{figure}")==3
assert source.count(r"\label{fig:overview}")==1
target.write_text(source)
(root/"outputs/risk_rf_overview.tex").write_text(
    "% Overview chapter and self-contained TikZ figure.\n"
    "% Parent dependencies: tikz, amsmath, amssymb; ind = mathbf 1.\n"
    "% Corresponding Methodology/Performance labels are defined in risk_rf_revision.tex.\n"
    +fragment
)
print("Inserted Overview and replaced prior loop; three total figures retained.")
