"""Timestamped milestones and progress for terminal and log output."""

from contextlib import contextmanager
from datetime import datetime
import sys
from time import perf_counter

from tqdm import tqdm


def message(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm.write(f"[{timestamp}] {text}", file=sys.stderr)
    sys.stderr.flush()


@contextmanager
def timed(label):
    start = perf_counter()
    message(label)
    try:
        yield
    except Exception:
        message(f"{label} — failed after {perf_counter() - start:.2f}s")
        raise
    else:
        message(f"{label} — done in {perf_counter() - start:.2f}s")


class _LogProgress(tqdm):
    """Write infrequent newline-delimited updates when stderr is redirected."""

    def display(self, msg=None, pos=None):
        if msg != "":
            message_text = str(self) if msg is None else msg
            self.fp.write(message_text + "\n")
            self.fp.flush()
        return True


def progress(iterable=None, *, label, unit, total=None):
    terminal = sys.stderr.isatty()
    bar = tqdm if terminal else _LogProgress
    return bar(
        iterable, total=total, desc=label, unit=unit, file=sys.stderr,
        dynamic_ncols=terminal, mininterval=0.25 if terminal else 30,
        miniters=1, disable=False,
        bar_format=None if terminal else "{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
