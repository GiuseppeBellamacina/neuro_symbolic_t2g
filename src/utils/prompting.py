"""Centralized T2G prompt builder.

Ensures training, evaluation, and ad-hoc generation produce identical
prompt byte streams regardless of the calling context.

Usage:
    from src.utils.prompting import build_t2g_prompt

    prompt = build_t2g_prompt("The man walks into the house.", tokenizer)
"""

from __future__ import annotations

from typing import Any

#: System prompt used across all T2G interactions.
SYSTEM_PROMPT = (
    "You are an English-to-ASL-gloss translator. "
    "Translate the following English sentence into a sequence of "
    "ASL glosses. Output ONLY the gloss tokens separated by spaces. "
    "Do not include explanations or extra text."
)


def build_t2g_prompt(text: str, tokenizer: Any) -> str:
    """Build a formatted T2G prompt from an English sentence.

    Uses the tokenizer's built-in ``apply_chat_template`` if available
    (preferred — produces the exact format the model was trained with),
    falling back to a Qwen-compatible manual format for tokenizers
    without a chat template.

    Args:
        text: The English sentence to translate.
        tokenizer: A Hugging Face tokenizer.

    Returns:
        The formatted prompt string, ready for ``tokenizer()`` or
        ``model.generate()``.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    if (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback: Qwen/ChatML-compatible manual format
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
