"""
Coalesce the stored stream outputs of executed documentation notebooks into one clean stream per code cell.

``jupyter nbconvert`` stores each tqdm carriage-return update as a separate stream output, so a progress bar renders
split across several output blocks and inflates the stored notebook. This merges consecutive same-name stream outputs
and keeps only the final carriage-return state of each line. Progress bars stay enabled; only the stored output changes.

The Snakemake rules ``execute_python_page`` and ``execute_r_page`` run this after executing a notebook.
Run directly as ``python docs/coalesce_streams.py <notebook> ...``. Already coalesced notebooks are left unchanged.
"""
import json
import sys


def collapse_carriage_returns(text: str) -> str:
    """Keep the text after the last carriage return on each line, the final state of a progress bar."""
    return "\n".join(line.split("\r")[-1] if "\r" in line else line for line in text.split("\n"))


def as_str(text) -> str:
    return "".join(text) if isinstance(text, list) else text


def coalesce(nb: dict) -> bool:
    """
    Coalesce the stream outputs of every code cell in place.

    :return: Whether the notebook changed.
    """
    changed = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        merged = []
        for out in cell.get("outputs", []):
            if (out.get("output_type") == "stream" and merged
                    and merged[-1].get("output_type") == "stream"
                    and merged[-1].get("name") == out.get("name")):
                merged[-1]["text"] = as_str(merged[-1]["text"]) + as_str(out.get("text", ""))
                changed = True
            else:
                out = dict(out)
                if out.get("output_type") == "stream":
                    out["text"] = as_str(out.get("text", ""))
                merged.append(out)

        for out in merged:
            if out.get("output_type") == "stream":
                collapsed = collapse_carriage_returns(out["text"])
                if collapsed != out["text"]:
                    out["text"] = collapsed
                    changed = True

        cell["outputs"] = merged

    return changed


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path) as fh:
            nb = json.load(fh)

        if coalesce(nb):
            with open(path, "w") as fh:
                json.dump(nb, fh, indent=1, ensure_ascii=False)
                fh.write("\n")
            print("coalesced", path)
        else:
            print("unchanged", path)
