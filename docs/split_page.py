"""
Split a User Guide source page into the Python and R notebooks that are executed to produce its outputs.

A source page (``docs/source/{name}.md``) holds the prose and the code of both languages in the MyST text notebook
format. Markdown cells are separated by ``+++`` lines, which may carry cell metadata as JSON
(``+++ {"tags": ["r-only"]}``). Code cells are ``{code-cell}`` fences whose argument names the language, ``python`` or
``r``, optionally followed by ``:tags:`` lines::

    ```{code-cell} r
    :tags: [remove-cell]
    options(repr.plot.width = 7, repr.plot.height = 3.4)
    ```

The Python notebook receives every markdown cell except those tagged ``r-only`` and every ``python`` code cell; the R
notebook receives every markdown cell except those tagged ``python-only`` and every ``r`` code cell. Both notebooks
therefore share the same prose cells, which ``docs/merge_notebooks.py`` pairs into language tabs after execution. Each
notebook opens with a hidden setup cell common to all pages, which renders figures as PDF and as PNG at the given
resolution.

The Snakemake rule ``split_page`` writes ``results/docs/Python/{name}.ipynb`` and ``results/docs/R/{name}.ipynb``,
with the resolution ``DOCS_FIGURE_DPI`` of the Snakefile. Run directly, ``python docs/split_page.py <dpi> <name> ...``
writes the same files for each page name, with figures rendered at ``dpi`` dots per inch.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

KERNELS = {
    "python": dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                   language_info=dict(name="python")),
    "r": dict(kernelspec=dict(name="ir-fastdfe", display_name="R (fastdfe)", language="R"),
              language_info=dict(name="R")),
}

# source of the hidden setup cell that opens every notebook of a language, formatted with the figure resolution
SETUP = {
    "python": (
        "# the PDF renderings of the figures are written to docs/outputs by docs/extract_outputs.py\n"
        "%config InlineBackend.figure_formats = ['png', 'pdf']\n"
        "# pad the saved figure, as its tight bounding box leaves out the axis labels of 3D plots\n"
        "%config InlineBackend.print_figure_kwargs = {{'bbox_inches': 'tight', 'pad_inches': 0.3, 'dpi': {dpi}}}\n"
        "%precision %.7g\n"
        "# numpy floats define their own repr, which the precision magic does not reach\n"
        "import numpy\n"
        "_formatter = get_ipython().display_formatter.formatters['text/plain']\n"
        "for _type in (numpy.float16, numpy.float32, numpy.float64, numpy.longdouble):\n"
        "    _formatter.for_type(_type, lambda x, p, cycle: p.text('%.7g' % x))"
    ),
    "r": (
        "options(repr.plot.res = {dpi})\n"
        "# the PDF renderings of the figures are written to docs/outputs by docs/extract_outputs.py\n"
        "options(jupyter.plot_mimetypes = c('text/plain', 'image/png', 'application/pdf'))\n"
        "# the ggplot2 theme of the R figures, padded like the Python figures\n"
        "ggplot2::theme_set(ggplot2::theme_bw() + ggplot2::theme(plot.margin = ggplot2::margin(12, 12, 12, 12)))"
    ),
}

ONLY_TAG = {"python": "python-only", "r": "r-only"}

CODE_CELL = re.compile(r"^(`{3,})\{code-cell\}\s+(python|r)\s*$")
SEPARATOR = re.compile(r"^\+\+\+\s*(\{.*\})?\s*$")
OPTION = re.compile(r"^:(\w+):\s*(.*)$")


def parse(text: str) -> list[dict]:
    """
    Parse a source page into cells.

    :return: Cells as ``{"type": "markdown" | "code", "language": "python" | "r" | None, "tags": [...],
        "source": str}``, in page order.
    """
    cells = []
    markdown, metadata = [], {}
    lines = text.split("\n")
    i = 0

    def flush():
        source = "\n".join(markdown).strip("\n")
        if source.strip():
            cells.append(dict(type="markdown", language=None, tags=list(metadata.get("tags", [])), source=source))

    while i < len(lines):
        line = lines[i]

        if m := SEPARATOR.match(line):
            flush()
            markdown, metadata = [], json.loads(m.group(1)) if m.group(1) else {}
            i += 1
            continue

        if m := CODE_CELL.match(line):
            flush()
            markdown, metadata = [], {}
            fence, language = m.group(1), m.group(2)
            body, tags = [], []
            i += 1

            while i < len(lines) and (o := OPTION.match(lines[i])):
                if o.group(1) == "tags":
                    tags = [t.strip() for t in o.group(2).strip("[] ").split(",") if t.strip()]
                i += 1

            while i < len(lines) and lines[i] != fence:
                body.append(lines[i])
                i += 1

            if i == len(lines):
                raise ValueError(f"unterminated {language} code cell")

            cells.append(dict(type="code", language=language, tags=tags, source="\n".join(body)))
            i += 1
            continue

        markdown.append(line)
        i += 1

    flush()

    return cells


def lines_of(source: str) -> list[str]:
    parts = source.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def code_cell(source: str, tags: list) -> dict:
    return dict(cell_type="code", execution_count=None, metadata=dict(tags=tags) if tags else {}, outputs=[],
                source=lines_of(source))


def notebook(cells: list[dict], language: str, dpi: int) -> dict:
    """Build the notebook of one language from the parsed cells of a page, rendering figures at ``dpi``."""
    other = ONLY_TAG["r" if language == "python" else "python"]
    out = [code_cell(SETUP[language].format(dpi=dpi), ["remove-cell"])]

    for cell in cells:
        if cell["type"] == "markdown" and other not in cell["tags"]:
            out.append(dict(cell_type="markdown", metadata=dict(tags=cell["tags"]) if cell["tags"] else {},
                            source=lines_of(cell["source"])))
        elif cell["type"] == "code" and cell["language"] == language:
            out.append(code_cell(cell["source"], cell["tags"]))

    return dict(nbformat=4, nbformat_minor=4, metadata=KERNELS[language], cells=out)


def split(source: Path, python: Path, r: Path, dpi: int):
    """Write the Python and R notebooks of the source page at ``source``, rendering figures at ``dpi``."""
    cells = parse(source.read_text())

    for language, path in (("python", python), ("r", r)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook(cells, language, dpi), indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        jobs = [(Path(snakemake.input[0]), Path(snakemake.output.python), Path(snakemake.output.r))]
        dpi = snakemake.params.dpi
    except NameError:
        dpi = int(sys.argv[1])
        jobs = [(ROOT / "docs" / "source" / f"{name}.md", ROOT / "results" / "docs" / "Python" / f"{name}.ipynb",
                 ROOT / "results" / "docs" / "R" / f"{name}.ipynb") for name in sys.argv[2:]]

    for source, python, r in jobs:
        split(source, python, r, dpi)
