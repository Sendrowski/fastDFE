"""
The outputs of executed User Guide notebook cells as the pages display them, shared by ``docs/merge_notebooks.py``,
which pastes them into the pages, and ``docs/extract_outputs.py``, which writes them to ``docs/outputs``.
"""


import re
import tempfile

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


def mask_text(text: str) -> str:
    """
    Plain-text output with the parts that differ between runs of the same code replaced: ANSI colour codes, trailing
    spaces of redrawn progress bars, progress-bar timings and rates, and the paths of temporary files.

    :param text: The text of an output.
    :return: The text with run-specific parts masked.
    """
    text = TRAILING_SPACE.sub("", ANSI_ESCAPE.sub("", text))
    text = TEMP_FILE.sub("tmp--------", TEMP_DIR.sub("<tmp>", text))
    text = PROGRESS_TIME.sub(lambda m: re.sub(r"[\d?]+", "--", m[0]), text)

    return PROGRESS_RATE.sub(lambda m: f" --{m[1] or m[2]}/s", text)


def tags(cell: dict) -> list:
    return cell.get("metadata", {}).get("tags", [])


def displayed_outputs(cell: dict) -> list[tuple[dict, dict]]:
    """
    Outputs a page displays for a code cell, in display order: non-empty streams as plain text, tables as HTML, other
    values as plain text without their Markdown and LaTeX renderings, and figures with all their renderings.

    :return: The data and metadata of each displayed output.
    """
    if "remove-cell" in tags(cell) or "remove-output" in tags(cell):
        return []

    displayed = []

    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            text = "".join(output["text"])
            if text.strip():
                displayed.append(({"text/plain": text}, {}))

        elif output["output_type"] in ("display_data", "execute_result"):
            data = output["data"]
            if "text/plain" in data and not any(k.startswith("image/") for k in data):
                html = "".join(data.get("text/html", ""))
                data = {"text/html": html} if "<table" in html else {"text/plain": data["text/plain"]}
            displayed.append((data, output.get("metadata", {})))

    return displayed
