"""Lightweight distributed-training helpers.

These utilities work with any launcher (``torchrun``, ``accelerate launch``,
``deepspeed``) because they rely on the standard ``LOCAL_RANK`` environment
variable that all launchers set.
"""

from __future__ import annotations

import os


def is_main_process() -> bool:
    """Return ``True`` when running on rank-0 (or in single-GPU / CPU mode)."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0
