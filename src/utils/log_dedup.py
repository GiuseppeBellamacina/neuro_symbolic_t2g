"""Deduplicate Python logging output on cluster jobs.

HF libraries (``huggingface_hub``, ``datasets``, ``transformers``, ``peft``,
``trl``) attach their own ``StreamHandler`` to their library logger AND leave
``propagate=True``.  Every library record is therefore emitted twice on our
jobs: once by the library handler (bare ``%(message)s``) and once by the root
handler (unsloth's ``[name|LEVEL]`` format in train jobs, the entrypoint's
``basicConfig`` format in eval jobs).  In slurm-eval-7077/7078 this doubled
every DNS-retry warning.

The fix: strip the library-owned handlers and let records flow to the root
logger exactly once (root always has a handler — the entrypoint's
``basicConfig``, or unsloth's when it configures the root first).  This
changes only the *duplicate* copy, never the message level or content.
"""

from __future__ import annotations

import logging

# Loggers that ship their own StreamHandler alongside propagate=True.
# Keep in sync with observed duplicates in slurm-train-7073 / slurm-eval-7077.
_LIB_LOGGERS = (
    "huggingface_hub",
    "datasets",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "urllib3",
    "httpx",
    "httpcore",
    "filelock",
    "fsspec",
)


def dedupe_library_loggers() -> None:
    """Remove library-owned handlers so each record prints exactly once.

    Idempotent and safe to call at any point after the entrypoint's
    ``logging.basicConfig`` (records then reach the root handler only).
    Loggers without handlers are left untouched.
    """
    for name in _LIB_LOGGERS:
        lg = logging.getLogger(name)
        if lg.handlers:
            lg.handlers.clear()
