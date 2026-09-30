"""Compile the standalone methods PDF using XeLaTeX and BibTeX."""
import argparse
from pathlib import Path
import shutil
import subprocess
from assemble_methodology import assemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", help="Folder of licensed Times New Roman TTFs, if not installed")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    assemble()
    build = root / "build"
    build.mkdir(exist_ok=True)
    source = "main.tex"
    if args.font_dir:
        folder = str(Path(args.font_dir).resolve()).replace("\\", "/") + "/"
        if any(character in folder for character in "{}%#\n\r"):
            raise ValueError("Use a font directory without TeX-special characters")
        for filename in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
            if not (Path(folder) / filename).is_file():
                raise FileNotFoundError(Path(folder) / filename)
        source = "\\def\\TimesFontPath{" + folder + "}\\input{main.tex}"
    command = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               "-jobname=methods_preview", "-output-directory=build", source]
    for number, executable in enumerate((command, ["bibtex", "build/methods_preview"], command, command), 1):
        result = subprocess.run(executable, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (build / f"compile_{number}.txt").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            print(result.stdout[-7000:])
            raise SystemExit(result.returncode)
    shutil.copyfile(build / "methods_preview.pdf", root / "methods_preview.pdf")
    print(root / "methods_preview.pdf")


if __name__ == "__main__":
    main()
