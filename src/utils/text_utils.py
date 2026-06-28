"""
Shared text utilities for T2G — gloss extraction, prompt parsing.

Extracted from ``src/rewards/t2g_rewards.py`` and ``src/utils/metrics.py``
to eliminate duplication (DRY).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Gloss text extraction
# ---------------------------------------------------------------------------


def extract_gloss_text(completion: str) -> str:
    """Extract clean gloss tokens from a model completion.

    Strips thinking tags, code fences, and extra whitespace.

    Args:
        completion: Raw model completion string.

    Returns:
        Cleaned gloss token string.
    """
    # Strip <think>...</think> blocks (if thinking mode was on)
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()

    # Strip fenced code blocks
    m = re.search(r"```(?:gloss)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    return text


# ---------------------------------------------------------------------------
# Prompt → user instruction extraction
# ---------------------------------------------------------------------------


def extract_user_text(prompt: Any) -> str:
    """Extract the user message text from a prompt in any format.

    TRL's GRPOTrainer may pass prompts as formatted chat strings,
    stringified lists, or raw strings depending on the backend.
    This function extracts the user instruction (English sentence
    to translate) regardless of format.

    Args:
        prompt: The prompt in whatever format the trainer provides.

    Returns:
        The user instruction string, or ``""`` if no user content
        could be extracted.
    """
    if prompt is None:
        return ""

    # Format 1: list of chat messages (most common from TRL)
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))

    # Format 2: plain string
    text = str(prompt)
    if not text:
        return ""

    # Try Qwen/ChatML format: <|im_start|>user\nTEXT<|im_end|>
    m = re.search(r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try "user: TEXT" or "user\nTEXT" pattern
    m = re.search(r"(?:^|\n)user[:\s]\n?(.*?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: return the whole string
    return text
