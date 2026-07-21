"""
Execute an R (IRkernel) reference notebook in place so that Python-side logging emitted through
reticulate becomes visible in the rendered cell outputs.

reticulate does not passively forward the embedded Python interpreter's stdout/stderr to IRkernel
cell output under nbconvert, so ``fastdfe`` (and ``sfsutils``, whose parser fastDFE re-exports) log
lines would otherwise be lost. Redirecting the Python streams does not help here: under IRkernel
reticulate routes Python stderr to the process's real file descriptor 2, which neither
``reticulate::py_capture_output`` (a Python ``sys.stderr`` swap) nor an ``os.dup2`` redirect reliably
intercepts. This driver therefore captures at the logging layer instead, which is independent of
stream routing: it attaches a buffered ``logging.StreamHandler`` to both the ``fastdfe`` and
``sfsutils`` loggers for the duration of each cell and redirects the tqdm progress bar into the same
buffer, then emits the buffered text to R's stderr (via ``message()``, the only R stream nbconvert
captures here) and tears the capture down. The buffered progress bar is coalesced afterwards like the
Python notebooks (docs/coalesce_streams.py). Inference and parsing are run serially
(``Settings.parallelize = False``) so that every log record is emitted in the parent process where
the handler sees it (records from multiprocessing workers would not propagate to it); the same lines
are logged either way, and the internal parallelism is immaterial for the small example inputs.

fastDFE's spectrum plots (``Spectra.plot()`` and friends) are Python matplotlib figures. reticulate's
inline-figure hook only fires in an interactive Jupyter session, not under nbconvert, so here the
kernel is pinned to the Agg backend (no GUI window can open) and each figure the cell leaves open is
saved explicitly and shown via ``IRdisplay::display_png``. It is saved at the R ``repr.plot.*`` size
(width/height inches at the plot dpi), matching what reticulate's inline rendering produces, so these
plots come out the same size as the notebook's R-side (ggplot) figures rather than at matplotlib's
native size. R-side ggplot/base plots are captured natively by IRkernel during evaluation.

The original (clean) source is restored before the notebook is written back, so the captured output
lands in the cell's outputs while the persisted source stays unwrapped. Cells that bind the Python
interpreter (``use_condaenv``, ``load_fastdfe``, ``library(reticulate)``) are left unwrapped:
calling into Python there would initialise it before the intended conda environment is selected.

Usage:  python execute_r_notebook.py <notebook.ipynb> [kernel_name] [timeout_seconds]
"""

import os
import sys

# Force a non-interactive matplotlib backend for the spawned kernel. reticulate's own inline-figure
# hook does not fire under nbconvert (it works in an interactive Jupyter/JupyterLab session, which is
# how these notebooks were first rendered), so fastdfe's Spectra.plot() would otherwise reach
# plt.show() and open a GUI window. With Agg no window opens and the figures are captured explicitly
# (see _FIG_CAPTURE) at the same size reticulate would have used.
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# cells binding the Python interpreter must run before any capture call
_SKIP_MARKERS = ("use_condaenv", "load_fastdfe", "library(reticulate)")

# fastDFE parsing is delegated to sfsutils, so both loggers must be captured
_LOGGERS = ("fastdfe", "sfsutils")

# attach a buffered handler to the fastdfe/sfsutils loggers (matching their own colored format) and
# redirect the tqdm progress bar into the same buffer, so both land in the captured output
_LOG_OPEN = (
    "import io, logging, fastdfe, tqdm\n"
    "_fd_buf = io.StringIO()\n"
    "_fd_h = logging.StreamHandler(_fd_buf)\n"
    "_fd_h.setFormatter(fastdfe.ColoredFormatter('%(levelname)s:%(name)s: %(message)s'))\n"
    f"for _fd_name in {_LOGGERS!r}:\n"
    "    logging.getLogger(_fd_name).addHandler(_fd_h)\n"
    "_fd_tqdm_init = tqdm.std.tqdm.__init__\n"
    "def _fd_patched_init(self, *a, **k):\n"
    "    k['file'] = _fd_buf\n"
    # force unicode block glyphs; tqdm falls back to ASCII '#' when writing to a StringIO
    "    k.setdefault('ascii', False)\n"
    "    _fd_tqdm_init(self, *a, **k)\n"
    "tqdm.std.tqdm.__init__ = _fd_patched_init\n"
)

# restore tqdm, detach the handler, and read back what was buffered
_LOG_CLOSE = (
    "tqdm.std.tqdm.__init__ = _fd_tqdm_init\n"
    f"for _fd_name in {_LOGGERS!r}:\n"
    "    logging.getLogger(_fd_name).removeHandler(_fd_h)\n"
    "_fd_h.flush()\n"
    "_fd_captured = _fd_buf.getvalue()\n"
    "_fd_buf.close()\n"
)

# Save any matplotlib figures the cell left open (fastdfe's Spectra.plot() etc. call plt.show(), a
# no-op under Agg, and leave the figure open) so they can be displayed inline via IRdisplay. The
# figure is sized to the R ``repr.plot.*`` options (``_fd_w`` x ``_fd_h`` inches at ``_fd_res`` dpi),
# exactly as reticulate's inline rendering would have, so these plots match the notebook's other
# figures instead of coming out at matplotlib's native size. R-side ggplot/base plots are captured
# natively by IRkernel during evaluation and never reach here.
_FIG_CAPTURE = (
    "import matplotlib.pyplot as _plt, tempfile as _tf, os as _os\n"
    "_fd_fig_paths = []\n"
    "for _num in _plt.get_fignums():\n"
    "    _fig = _plt.figure(_num)\n"
    "    _fig.set_size_inches(_fd_w, _fd_h)\n"
    "    _fd_fd, _fd_path = _tf.mkstemp(suffix='.png')\n"
    "    _os.close(_fd_fd)\n"
    "    _fig.savefig(_fd_path, dpi=_fd_res)\n"
    "    _fd_fig_paths.append(_fd_path)\n"
    "_plt.close('all')\n"
)


def _wrap(src: str) -> str:
    """
    Wrap a cell body so the ``fastdfe``/``sfsutils`` log records emitted during its evaluation are
    captured via a temporary logging handler and emitted to R's stderr, while preserving the block's
    own (visible) value for auto-printing. Inference and parsing are switched to serial execution so
    all their log records stay in the parent process where the handler sees them.

    :param src: The original R cell source.
    :return: The wrapped source.
    """
    return (
        '.fdmod <- reticulate::import("fastdfe")\n'
        ".fdmod$Settings$parallelize <- FALSE\n"
        f'reticulate::py_run_string("{_LOG_OPEN}")\n'
        ".res <- withVisible({\n"
        f"{src}\n"
        "})\n"
        f'reticulate::py_run_string("{_LOG_CLOSE}")\n'
        # hand the R plot-size options to Python, then save any open matplotlib figures at that size
        'reticulate::py_run_string(sprintf("_fd_w, _fd_h, _fd_res = %s, %s, %s",'
        ' getOption("repr.plot.width", 7), getOption("repr.plot.height", 7),'
        ' getOption("repr.plot.res", 120)))\n'
        f'reticulate::py_run_string("{_FIG_CAPTURE}")\n'
        ".pyout <- reticulate::py$`_fd_captured`\n"
        # message() reaches the IRkernel stderr stream; cat(file=stderr()) does not under nbconvert
        "if (!is.null(.pyout) && nzchar(.pyout)) message(.pyout, appendLF = FALSE)\n"
        # display matplotlib figures the cell left open; R-side plots were already drawn during eval
        'for (.p in reticulate::py$`_fd_fig_paths`) IRdisplay::display_png(file = .p)\n'
        "if (.res$visible) .res$value else invisible(.res$value)\n"
    )


class _CapturingExecutePreprocessor(ExecutePreprocessor):
    """
    An :class:`ExecutePreprocessor` that captures reticulate's Python output per cell and restores
    the original cell source afterwards.
    """

    def preprocess_cell(self, cell, resources, index):
        original = None

        if cell.cell_type == "code" and cell.source.strip() \
                and not any(m in cell.source for m in _SKIP_MARKERS):
            original = cell.source
            cell.source = _wrap(original)

        cell, resources = super().preprocess_cell(cell, resources, index)

        if original is not None:
            cell.source = original

        return cell, resources


def main(path: str, kernel: str = "ir", timeout: int = 1200) -> None:
    """
    Execute the notebook in place with per-cell Python-output capture.

    :param path: Path to the notebook.
    :param kernel: Jupyter kernel name.
    :param timeout: Per-cell execution timeout in seconds.
    """
    nb = nbformat.read(path, as_version=4)

    ep = _CapturingExecutePreprocessor(timeout=int(timeout), kernel_name=kernel)
    ep.preprocess(nb, {"metadata": {"path": "."}})

    nbformat.write(nb, path)


if __name__ == "__main__":
    main(*sys.argv[1:])
