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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptProfile:
    """Language pair of a T2G corpus, as it appears in the prompt.

    Attributes:
        name: Profile id (value of ``dataset.prompt_profile``).
        system_prompt: System message.
        few_shot_header: Instruction framing the few-shot user content.
        source_label: Label of the source sentence in few-shot blocks.
        gloss_label: Label of the gloss sequence in few-shot blocks.
    """

    name: str
    system_prompt: str
    few_shot_header: str
    source_label: str
    gloss_label: str


#: Profili per coppia di lingue. ``en-asl`` è il prompt storico (ASLG-PC12),
#: byte-identico a quello sotto cui sono stati prodotti tutti i risultati e
#: calcolati i fingerprint (adapter SFT, cache della baseline).
PROMPT_PROFILES: dict[str, PromptProfile] = {
    "en-asl": PromptProfile(
        name="en-asl",
        system_prompt=(
            "You are an English-to-ASL-gloss translator. "
            "Translate the following English sentence into a sequence of "
            "ASL glosses. Output ONLY the gloss tokens separated by spaces. "
            "Do not include explanations or extra text."
        ),
        few_shot_header="Translate the following English sentence into ASL gloss.",
        source_label="English",
        gloss_label="ASL gloss",
    ),
    # PHOENIX-2014T: tedesco (meteo) → glosse DGS. Istruzioni in inglese come
    # per ASLG-PC12, così il fattore che cambia fra i due dataset è la coppia
    # di lingue del task, non la lingua delle istruzioni.
    "de-dgs": PromptProfile(
        name="de-dgs",
        system_prompt=(
            "You are a German-to-DGS-gloss translator. "
            "Translate the following German sentence into a sequence of "
            "DGS (German Sign Language) glosses. Output ONLY the gloss "
            "tokens separated by spaces. "
            "Do not include explanations or extra text."
        ),
        few_shot_header="Translate the following German sentence into DGS gloss.",
        source_label="German",
        gloss_label="DGS gloss",
    ),
}

DEFAULT_PROMPT_PROFILE = "en-asl"

#: System prompt of the default (ASLG-PC12) profile.
SYSTEM_PROMPT = PROMPT_PROFILES[DEFAULT_PROMPT_PROFILE].system_prompt

#: Instruction framing the few-shot user content (see module docstring).
_FEW_SHOT_HEADER = PROMPT_PROFILES[DEFAULT_PROMPT_PROFILE].few_shot_header


def get_prompt_profile(profile: str | PromptProfile | None = None) -> PromptProfile:
    """Resolve a profile id (or ``None`` → default ``en-asl``).

    Raises:
        ValueError: for an unknown profile id.
    """
    if isinstance(profile, PromptProfile):
        return profile
    name = profile or DEFAULT_PROMPT_PROFILE
    if name not in PROMPT_PROFILES:
        raise ValueError(
            f"prompt profile sconosciuto: {name!r}. Noti: {sorted(PROMPT_PROFILES)}"
        )
    return PROMPT_PROFILES[name]


def prompt_profile_for_config(config: Mapping[str, Any] | None) -> PromptProfile:
    """Profile declared by ``dataset.prompt_profile`` (default ``en-asl``)."""
    ds_cfg = (config or {}).get("dataset") or {}
    return get_prompt_profile(ds_cfg.get("prompt_profile"))


def system_prompt_for_config(config: Mapping[str, Any] | None) -> str:
    """System prompt of the config's profile (see :func:`prompt_profile_for_config`)."""
    return prompt_profile_for_config(config).system_prompt


def format_few_shot_examples(
    examples: list[Any], profile: str | PromptProfile | None = None
) -> str:
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
        profile: Prompt profile (labels); ``None`` → ``en-asl``.

    Returns:
        The formatted block (``""`` for an empty input).
    """
    if not examples:
        return ""
    prof = get_prompt_profile(profile)
    blocks = [
        f"{prof.source_label}: {_example_text(ex)}\n"
        f"{prof.gloss_label}: {_example_gloss(ex)}"
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
    glossary_block: str | None = None,
    profile: str | PromptProfile | None = None,
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
        glossary_block: Optional pre-rendered rare-word glossary block (see
            ``src/utils/glossary.py::format_glossary_block``), prepended to
            the user content. ``None``/empty ⇒ byte-identical to the prompt
            without this argument. TRAIN-TIME ONLY: ``eval_t2g.py`` never
            passes this — see ``src/utils/glossary.py`` for why evaluation
            must stay glossary-free.
        profile: Prompt profile (language pair, see :data:`PROMPT_PROFILES`);
            ``None`` → ``en-asl``, byte-identical to the historical prompt.

    Returns:
        The formatted prompt string, ready for ``tokenizer()`` or
        ``model.generate()``.
    """
    prof = get_prompt_profile(profile)
    if examples:
        user_content = (
            f"{prof.few_shot_header}\n\n"
            f"{format_few_shot_examples(examples, prof)}\n\n"
            f"Now translate:\n{prof.source_label}: {text}"
        )
    else:
        user_content = text

    if glossary_block:
        user_content = f"{glossary_block}\n\n{user_content}"

    messages = [
        {"role": "system", "content": prof.system_prompt},
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
        f"<|im_start|>system\n{prof.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
