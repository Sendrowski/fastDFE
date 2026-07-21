"""Coalesce tqdm progress-bar stderr into a single clean stream per code cell.

``jupyter nbconvert`` stores each tqdm carriage-return update as a separate
stderr *stream* output, so a progress bar renders split across several output
blocks in the built docs. This merges consecutive same-name stream outputs and
collapses carriage returns to the final state, so a progress bar renders as a
single tidy line. Progress bars stay enabled; only the stored output is tidied.

The docs build does not execute notebooks (``nb_execution_mode = 'off'``), so run
this once after re-executing the reference notebooks, e.g.::

    jupyter nbconvert --to notebook --execute --inplace docs/reference/Python/parser.ipynb
    python docs/coalesce_streams.py docs/reference/Python/parser.ipynb

For the R (IRkernel) notebooks the capture is driven by docs/execute_r_notebook.py,
which should be followed by this script in the same way.

It is idempotent: already-tidy notebooks are left unchanged.
"""
import json
import sys


def _collapse_cr(text: str) -> str:
    # keep only the text after the last carriage return on each line (final tqdm state)
    return '\n'.join(line.split('\r')[-1] if '\r' in line else line for line in text.split('\n'))


def _as_str(text) -> str:
    return ''.join(text) if isinstance(text, list) else text


def coalesce(nb: dict) -> bool:
    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        merged = []
        for out in cell.get('outputs', []):
            if (out.get('output_type') == 'stream' and merged
                    and merged[-1].get('output_type') == 'stream'
                    and merged[-1].get('name') == out.get('name')):
                merged[-1]['text'] = _as_str(merged[-1]['text']) + _as_str(out.get('text', ''))
                changed = True
            else:
                out = dict(out)
                if out.get('output_type') == 'stream':
                    out['text'] = _as_str(out.get('text', ''))
                merged.append(out)
        for out in merged:
            if out.get('output_type') == 'stream':
                collapsed = _collapse_cr(out['text'])
                if collapsed != out['text']:
                    out['text'] = collapsed
                    changed = True
        cell['outputs'] = merged
    return changed


def main(paths) -> None:
    for path in paths:
        with open(path) as fh:
            nb = json.load(fh)
        if coalesce(nb):
            with open(path, 'w') as fh:
                json.dump(nb, fh, indent=1, ensure_ascii=False)
                fh.write('\n')
            print('coalesced', path)
        else:
            print('unchanged', path)


if __name__ == '__main__':
    main(sys.argv[1:])
