"""Assemble the two editable section fragments into one drop-in LaTeX body."""
from pathlib import Path


def assemble():
    root = Path(__file__).resolve().parent
    header = ("% Complete drop-in methodology body. Required packages: amsmath,\n"
              "% amssymb, amsthm, graphicx, natbib. Define a proposition environment.\n"
              "% Edit the two section fragments and run assemble_methodology.py.\n")
    body = "\n".join((root / name).read_text(encoding="utf-8")
                      for name in ("system_fusion_model.tex","core_coordination.tex"))
    destination = root / "core_methodology.tex"
    destination.write_text(header + body,encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(assemble())
