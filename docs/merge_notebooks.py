"""
Merge a paired Python and R documentation notebook into one page whose code is shown in synchronised language tabs.

Both notebooks hold the same sequence of shared markdown cells. The page takes its prose from the Python notebook, and
the code cells between two shared markdown cells become one tab per language, with the executed outputs pasted by
myst-nb glue from hidden carrier cells in the merged notebook. A markdown cell tagged ``r-only`` in the R notebook (or
``python-only`` in the Python notebook) opens a segment whose prose and code are rendered inside that language's tab,
after the shared segment it follows. The language-only segments of both notebooks at one position share a tab set, so
their markdown must not contain section headings.

Figures are shown at their nominal size of 100 CSS pixels per inch at the resolution they were rendered at, scaled by
``SINGLE_FIGURE_SCALE`` unless their code cell is tagged ``full-width``, as cells drawing side-by-side panels are.

The notebooks are split from the page source ``docs/source/{name}.md`` by ``docs/split_page.py`` and executed
before merging. The Snakemake rule ``merge_page`` writes ``docs/reference/{name}.ipynb`` from
``results/docs/Python/{name}.executed.ipynb`` and ``results/docs/R/{name}.executed.ipynb``, with the resolution
``DOCS_FIGURE_DPI`` of the Snakefile. Run directly, ``python docs/merge_notebooks.py <dpi> <name> ...`` does the same
for each page name, with ``dpi`` the resolution the figures were rendered at.

"""
import base64
import copy
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_outputs import displayed_outputs, mask_text, tags

ROOT = Path(__file__).parent.parent

GLUE_PREFIX = "application/papermill.record/"

LANGUAGES = {
    "Python": dict(tab="{fab}`python` Python", sync="python", lexer="python", only="python-only"),
    "R": dict(tab="{fab}`r-project` R", sync="r", lexer="r", only="r-only"),
}

# class of the language tab sets, styled in docs/_static/custom.css (docutils strips classes beginning with "language-")
TAB_SET_CLASS = "code-tabs"

# tag of the code cells whose figures are shown at their full nominal width
FULL_WIDTH_TAG = "full-width"

# display scale of the figures of code cells not tagged FULL_WIDTH_TAG
SINGLE_FIGURE_SCALE = 0.8


def text(cell: dict) -> str:
    return "".join(cell["source"])


def split_segments(nb: dict, only_tag: str) -> tuple[list, list]:
    """
    Split a notebook into shared segments and language-only segments.

    :return: ``shared``, a list of ``{"md": cell, "code": [cells]}``, and ``only``, a list of
        ``(shared_index, segment)`` for segments opened by a markdown cell tagged ``only_tag``.
    """
    shared, only = [], []
    current = None

    for cell in nb["cells"]:
        if "remove-cell" in tags(cell):
            continue

        if cell["cell_type"] == "markdown":
            current = {"md": cell, "code": []}

            if only_tag in tags(cell):
                only.append((len(shared) - 1, current))
            else:
                shared.append(current)

        elif cell["cell_type"] == "code":
            if current is None:
                current = {"md": None, "code": []}
                shared.append(current)

            current["code"].append(cell)

    return shared, only


def display_metadata(data: dict, metadata: dict, dpi: int, full_width: bool) -> dict:
    """
    Display metadata of an output. A PNG image is shown at its nominal size of 100 CSS pixels per inch rendered at
    ``dpi``, scaled by ``SINGLE_FIGURE_SCALE`` unless ``full_width``. Only the width is given, as myst-nb writes a given
    height as an inline style that holds while the column scales the width down.
    """
    metadata = copy.deepcopy(metadata)

    if "image/png" in data:
        pixels = struct.unpack(">I", base64.b64decode(data["image/png"])[16:20])[0]
        width = round(pixels * 100 / dpi)

        if not full_width:
            width = round(width * SINGLE_FIGURE_SCALE)

        metadata["image/png"] = dict(width=width)

    return metadata


def carrier_outputs(cell: dict, key_prefix: str, dpi: int) -> tuple[list, list]:
    """
    Convert the displayed outputs of a code cell into hidden glue outputs.

    :return: The glue outputs and their keys, in display order.
    """
    outputs, keys = [], []

    for data, metadata in displayed_outputs(cell):
        # the PDF rendering of a figure is written to docs/outputs only
        data = {k: v for k, v in data.items() if k != "application/pdf"}

        # the carrier is committed with the page, so the timings of this run must not reach it
        if "text/plain" in data:
            data = dict(data, **{"text/plain": mask_text("".join(data["text/plain"]))})

        key = f"{key_prefix}-{len(keys)}"
        keys.append(key)
        outputs.append(dict(
            output_type="display_data",
            data={GLUE_PREFIX + k: v for k, v in data.items()},
            metadata=dict(
                display_metadata(data, metadata, dpi, FULL_WIDTH_TAG in tags(cell)),
                scrapbook=dict(name=key, mime_prefix=GLUE_PREFIX),
            ),
        ))

    return outputs, keys


def fence(n: int, directive: str, argument: str, options: str, body: str) -> str:
    # tilde fences, as the info string of a backtick fence cannot contain the backticks of an icon role
    tildes = "~" * n
    return f"{tildes}{{{directive}}} {argument}\n{options}\n\n{body}\n{tildes}\n"


def tab_block(code_by_language: dict, carriers: list, segment_id: str, dpi: int) -> str:
    """Render one segment's code as a tab set, appending the hidden glue carrier cells to ``carriers``."""
    items = []

    for language, cells in code_by_language.items():
        spec = LANGUAGES[language]
        parts = []

        for i, cell in enumerate(cells):
            if cell["cell_type"] == "markdown":
                parts.append(text(cell).rstrip() + "\n")
                continue

            if "remove-input" not in tags(cell) and text(cell).strip():
                parts.append(f"```{spec['lexer']}\n{text(cell).rstrip()}\n```\n")

            outputs, keys = carrier_outputs(cell, f"{spec['sync']}-{segment_id}-{i}", dpi)
            for key, output in zip(keys, outputs):
                glue = f"```{{glue}} {key}\n```\n"
                # tables take the theme's notebook table style, which applies inside a div of class cell_output (raw
                # HTML, as docutils turns the underscore of a container class into a hyphen)
                if GLUE_PREFIX + "text/html" in output["data"]:
                    glue = f'<div class="cell_output">\n\n{glue}\n</div>\n'
                parts.append(glue)

            if outputs:
                carriers.append(dict(cell_type="code", execution_count=None, source=[],
                                     metadata=dict(tags=["remove-cell"]), outputs=outputs))

        if parts:
            items.append(fence(5, "tab-item", spec["tab"], f":sync: {spec['sync']}", "\n".join(parts)))

    if not items:
        return ""

    return fence(6, "tab-set", "", f":sync-group: language\n:class: {TAB_SET_CLASS}", "\n".join(items))


def markdown_cell(source: str) -> dict:
    lines = source.split("\n")
    return dict(cell_type="markdown", metadata={}, source=[l + "\n" for l in lines[:-1]] + [lines[-1]])


def merge(python_path: Path, r_path: Path, out: Path, dpi: int):
    """
    Merge the Python notebook at ``python_path`` and the R notebook at ``r_path`` into ``out``.

    :param dpi: Resolution the figures of both notebooks were rendered at, in dots per inch.
    :raises ValueError: If the notebooks hold different numbers of shared segments.
    """
    python = json.load(open(python_path))
    r = json.load(open(r_path))
    page = out.stem

    py_shared, py_only = split_segments(python, LANGUAGES["Python"]["only"])
    r_shared, r_only = split_segments(r, LANGUAGES["R"]["only"])

    if len(py_shared) != len(r_shared):
        raise ValueError(f"{page}: {len(py_shared)} shared Python segments but {len(r_shared)} shared R segments")

    cells, carriers = [], []
    extra = {k: [] for k in range(-1, len(py_shared))}
    for language, only in (("Python", py_only), ("R", r_only)):
        for k, segment in only:
            extra[k].append((language, segment))

    def emit(md, code_by_language, segment_id):
        if md is not None:
            cells.append(markdown_cell(text(md)))
        block = tab_block(code_by_language, carriers, segment_id, dpi)
        if block:
            cells.append(markdown_cell(block))

    def emit_only(k):
        tabs = {}
        for language, segment in extra[k]:
            tabs.setdefault(language, []).extend(([segment["md"]] if segment["md"] else []) + segment["code"])
        emit(None, tabs, f"s{k}-only")

    emit_only(-1)
    for k, (a, b) in enumerate(zip(py_shared, r_shared)):
        emit(a["md"], {"Python": a["code"], "R": b["code"]}, f"s{k}")
        emit_only(k)

    merged = dict(nbformat=4, nbformat_minor=4, metadata=python["metadata"], cells=cells + carriers)
    out.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        jobs = [(Path(snakemake.input.python), Path(snakemake.input.r), Path(snakemake.output[0]))]
        dpi = snakemake.params.dpi
    except NameError:
        dpi = int(sys.argv[1])
        jobs = [(ROOT / "results" / "docs" / "Python" / f"{name}.executed.ipynb",
                 ROOT / "results" / "docs" / "R" / f"{name}.executed.ipynb",
                 ROOT / "docs" / "reference" / f"{name}.ipynb") for name in sys.argv[2:]]

    for python, r, out in jobs:
        merge(python, r, out, dpi)
