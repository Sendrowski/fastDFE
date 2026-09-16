"""
Write the outputs an executed User Guide notebook displays to ``docs/outputs/{page}/``, so that changes to them show up
in version control.

Each section with outputs has a directory ``<number>-<heading>``, numbered in page order from ``00`` for the part before
the first subheading, holding the ``python-<n>`` or ``r-<n>`` files of the notebook's language in display order: figures
as PDF, or PNG without a PDF rendering, tables as HTML and other output as text. Since the notebooks of both languages
hold the same headings, their files share the section directories, and extracting one language keeps the files of the
other. The metadata of a PDF that differs between renderings of the same figure, the timings of progress bars and the
paths of temporary files are replaced by fixed values.

The Snakemake rule ``extract_page_outputs`` writes the outputs of ``results/docs/{language}/{page}.executed.ipynb``.
Run directly, ``python docs/extract_outputs.py <notebook> ...`` does the same for each executed notebook.
"""
import base64
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_outputs import displayed_outputs, mask_text, tags

ROOT = Path(__file__).parent.parent

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

FENCED_CODE = re.compile(r"^(`{3,}|~{3,}).*?^\1", re.MULTILINE | re.DOTALL)

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# elapsed time, and remaining time where the total is known, of a progress bar
PROGRESS_TIME = re.compile(r"(?<=\[)\d+(?::\d+)+(?:<(?:\d+(?::\d+)+|\?))?(?=,)")

# rate of a progress bar, with the spaces it is padded with. tqdm writes <unit>/s above one iteration
# per second and s/<unit> below, so the direction depends on how fast the run happened to be. The
# unit names what the bar counts and is kept
PROGRESS_RATE = re.compile(r"[ \t]*[\d.?]+ ?(?:([a-z]+)/s|s/ ?([a-z]+))\b")

# temporary directory of this machine, which executes the notebooks, and names of temporary files created in it
TEMP_DIR = re.compile(re.escape(tempfile.gettempdir()))
TEMP_FILE = re.compile(r"(?<=<tmp>/)tmp[a-z0-9_]{8}")

# spaces a progress bar pads its redrawn line with
TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)

# creation and modification dates and file identifier of a PDF
PDF_DATE = re.compile(rb"D:\d{14}")
PDF_ID = re.compile(rb"(/ID\s*\[\s*<)([0-9a-fA-F]+)(>\s*<)([0-9a-fA-F]+)(>)")


def slug(title: str) -> str:
    """Directory name of a section heading: its lower-case words joined by hyphens, without MyST roles and literals."""
    title = re.sub(r"\{[^}]*\}`([^`]*)`", r"\1", title)

    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def output_file(data: dict) -> tuple[str, bytes]:
    """
    File extension and content of a displayed output: the PDF of a figure, the PNG of a figure without a PDF rendering,
    the HTML of a table, or the text without ANSI colour codes, trailing spaces, progress bar timings and the paths of
    temporary files.
    """
    if "application/pdf" in data:
        pdf = base64.b64decode("".join(data["application/pdf"]))
        # fixed values of the same length keep the cross-reference offsets of the PDF valid
        pdf = PDF_DATE.sub(b"D:19700101000000", pdf)
        return "pdf", PDF_ID.sub(lambda m: m[1] + b"0" * len(m[2]) + m[3] + b"0" * len(m[4]) + m[5], pdf)

    if "image/png" in data:
        return "png", base64.b64decode("".join(data["image/png"]))

    if "text/html" in data:
        return "html", "".join(data["text/html"]).encode()

    text = mask_text("".join(data["text/plain"]))

    # a stream ends with a line break or not depending on when it was flushed
    return "txt", (text.rstrip("\n") + "\n").encode()


def sections(nb: dict, page: str) -> list[tuple[str, list]]:
    """
    Split the displayed outputs of a notebook by section.

    :return: The directory name and the displayed output data of each section, in page order.
    """
    titles, outputs = [page], [[]]

    for cell in nb["cells"]:
        if "remove-cell" in tags(cell):
            continue

        if cell["cell_type"] == "markdown":
            for level, title in HEADING.findall(FENCED_CODE.sub("", "".join(cell["source"]))):
                if len(level) == 1 and len(titles) == 1:
                    titles[0] = title
                else:
                    titles.append(title)
                    outputs.append([])

        elif cell["cell_type"] == "code":
            outputs[-1] += [data for data, _ in displayed_outputs(cell)]

    return [(f"{number:02d}-{slug(title)}", data) for number, (title, data) in enumerate(zip(titles, outputs))]


def extract(notebook: Path, directory: Path, language: str):
    """
    Write the displayed outputs of the executed ``notebook`` to ``directory``, replacing the files of ``language``.

    :param language: The file name prefix of the notebook's language, ``python`` or ``r``.
    """
    page_sections = sections(json.loads(notebook.read_text()), directory.name)

    for path in directory.glob(f"*/{language}-*"):
        path.unlink()

    for name, outputs in page_sections:
        for n, data in enumerate(outputs, start=1):
            extension, content = output_file(data)
            path = directory / name / f"{language}-{n}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    # directories of sections the page no longer has, once neither language has files in them
    current = {name for name, _ in page_sections}
    for section in directory.glob("*/"):
        if section.name not in current and not any(section.iterdir()):
            section.rmdir()


if __name__ == "__main__":
    try:
        notebooks = [Path(snakemake.input[0])]
    except NameError:
        notebooks = [Path(arg) for arg in sys.argv[1:]]

    for notebook in notebooks:
        page = notebook.name.removesuffix(".executed.ipynb")
        extract(notebook, ROOT / "docs" / "outputs" / page, notebook.parent.name.lower())
