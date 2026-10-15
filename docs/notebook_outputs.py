"""
The outputs of executed User Guide notebook cells as the pages display them, shared by ``docs/merge_notebooks.py``,
which pastes them into the pages, and ``docs/extract_outputs.py``, which writes them to ``docs/outputs``.
"""


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
