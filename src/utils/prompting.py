"""Centralized T2G prompt builder.

Ensures training, evaluation, and ad-hoc generation produce identical
prompt byte streams regardless of the calling context.

The builder supports optional few-shot demonstration blocks retrieved from
the TRAIN split (see ``src/training/retrieval_setup.py``).  With
``examples=None`` (the default) the produced prompt is **byte-identical**
to the original zero-shot format, so SFT — which builds its prompts via
``SYSTEM_PROMPT`` + raw user text and never passes examples — is
unaffected.

Few-shot user-content format (``examples`` set, e.g. 2 demonstrations)::

    Translate the following English sentence into ASL gloss.

    Examples:
    English: The cat sleeps on the sofa
    ASL gloss: CAT SLEEP SOFA

    English: A dog runs in the park
    ASL gloss: DOG RUN PARK

    Now translate:
    English: The man walks into the house.

Usage:
    from src.utils.prompting import build_t2g_prompt

    prompt = build_t2g_prompt("The man walks into the house.", tokenizer)
    few_shot = build_t2g_prompt(
        "The man walks into the house.",
        tokenizer,
        examples=[...],
    )
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

#: Instruction framing the few-shot user content (see module docstring).
_FEW_SHOT_HEADER = "Translate the following English sentence into ASL gloss."


def format_few_shot_examples(examples: list[Any]) -> str:
    """Render few-shot ``(text, gloss)`` examples as a text block.

    Accepts any iterable of objects exposing ``.text`` and ``.gloss``
    attributes (e.g. :class:`src.retrieval.RetrievedExample`) or plain
    dicts with ``"text"``/``"gloss"`` keys.

    Produces a block like::

        Examples:
        English: The cat sleeps on the sofa
        ASL gloss: CAT SLEEP SOFA

        English: A dog runs in the park
        ASL gloss: DOG RUN PARK

    Args:
        examples: Few-shot demonstrations — each a ``RetrievedExample``-like
            object or a ``{"text", "gloss"}`` dict.

    Returns:
        The formatted block (``""`` for an empty input).
    """
    if not examples:
        return ""
    blocks = [
        f"English: {_example_text(ex)}\nASL gloss: {_example_gloss(ex)}"
        for ex in examples
    ]
    return "Examples:\n" + "\n\n".join(blocks)


def _example_text(ex: Any) -> str:
    """Return the source English text of a ``RetrievedExample``-like object."""
    return ex["text"] if isinstance(ex, dict) else ex.text


def _example_gloss(ex: Any) -> str:
    """Return the gold gloss of a ``RetrievedExample``-like object."""
    return ex["gloss"] if isinstance(ex, dict) else ex.gloss


def build_t2g_prompt(
    text: str,
    tokenizer: Any,
    *,
    examples: list[Any] | None = None,
) -> str:
    """Build a formatted T2G prompt from an English sentence.

    Uses the tokenizer's built-in ``apply_chat_template`` if available
    (preferred — produces the exact format the model was trained with),
    falling back to a Qwen-compatible manual format for tokenizers
    without a chat template.

    With ``examples=None`` (or an empty list) the user content is the raw
    sentence — byte-identical to the legacy zero-shot prompt.  With
    ``examples`` the user content is framed as a few-shot task (see the
    module docstring for the exact format).

    Args:
        text: The English sentence to translate.
        tokenizer: A Hugging Face tokenizer.
        examples: Optional few-shot ``(text, gloss)`` demonstrations
            (``RetrievedExample``-like or ``{"text", "gloss"}`` dicts).
            ``None``/empty ⇒ zero-shot prompt.

    Returns:
        The formatted prompt string, ready for ``tokenizer()`` or
        ``model.generate()``.
    """
    if examples:
        user_content = (
            f"{_FEW_SHOT_HEADER}\n\n"
            f"{format_few_shot_examples(examples)}\n\n"
            f"Now translate:\nEnglish: {text}"
        )
    else:
        user_content = text

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
