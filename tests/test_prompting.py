"""Tests for the centralized T2G prompt builder (zero-shot + few-shot).

No network, no models: a fake tokenizer drives both the manual ChatML
fallback and the ``apply_chat_template`` path.

The few-shot format is a *contract* — any change to it (headers, markers,
separators) must be intentional and reviewed here first.
"""

from __future__ import annotations

from src.retrieval import RetrievedExample
from src.utils.prompting import (
    SYSTEM_PROMPT,
    build_t2g_prompt,
    format_few_shot_examples,
)


class _ManualTokenizer:
    """Tokenizer without a chat template → manual ChatML fallback path."""


class _ChatTokenizer:
    """Tokenizer exposing ``apply_chat_template`` (returns a canned string)."""

    chat_template = "qwen"

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=True
    ):
        return "CHAT|" + repr(messages) + "|"


def _legacy_manual_prompt(user_content: str) -> str:
    """Reproduce the pre-few-shot fallback output byte-for-byte."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _sample_examples() -> list[dict[str, str]]:
    """Two dict-based few-shot examples used across the tests."""
    return [
        {"text": "The cat sleeps on the sofa", "gloss": "CAT SLEEP SOFA"},
        {"text": "A dog runs in the park", "gloss": "DOG RUN PARK"},
    ]


# ---------------------------------------------------------------------------
# Zero-shot regression: byte-identical to the legacy prompt
# ---------------------------------------------------------------------------


def test_zero_shot_manual_fallback_byte_identical():
    """``examples=None`` ⇒ byte-identical to the legacy zero-shot prompt."""
    text = "The man walks into the house."
    prompt = build_t2g_prompt(text, _ManualTokenizer())
    assert prompt == _legacy_manual_prompt(text)


def test_zero_shot_empty_examples_byte_identical():
    """An empty examples list must NOT change the prompt."""
    text = "The man walks into the house."
    prompt = build_t2g_prompt(text, _ManualTokenizer(), examples=[])
    assert prompt == _legacy_manual_prompt(text)


def test_zero_shot_chat_template_path_unchanged():
    """The ``apply_chat_template`` path still receives system+user messages."""
    tok = _ChatTokenizer()
    text = "The man walks into the house."
    prompt = build_t2g_prompt(text, tok)
    expected = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt == expected


# ---------------------------------------------------------------------------
# format_few_shot_examples
# ---------------------------------------------------------------------------


def test_format_few_shot_examples_with_dicts():
    """Dicts with ``text``/``gloss`` keys render in order, blank-separated."""
    block = format_few_shot_examples(_sample_examples())
    assert block == (
        "Examples:\n"
        "English: The cat sleeps on the sofa\n"
        "ASL gloss: CAT SLEEP SOFA\n"
        "\n"
        "English: A dog runs in the park\n"
        "ASL gloss: DOG RUN PARK"
    )


def test_format_few_shot_examples_with_retrieved_objects():
    """``RetrievedExample`` objects (``.text``/``.gloss``) render identically."""
    examples = [
        RetrievedExample(
            text="The cat sleeps on the sofa",
            gloss="CAT SLEEP SOFA",
            score=0.81,
            index=0,
        ),
        RetrievedExample(
            text="A dog runs in the park", gloss="DOG RUN PARK", score=0.72, index=1
        ),
    ]
    assert format_few_shot_examples(examples) == format_few_shot_examples(
        _sample_examples()
    )


def test_format_few_shot_examples_empty():
    """Empty input renders an empty block."""
    assert format_few_shot_examples([]) == ""


# ---------------------------------------------------------------------------
# Few-shot prompt: exact format contract
# ---------------------------------------------------------------------------


def test_few_shot_prompt_exact_format():
    """The full few-shot prompt matches the documented format exactly."""
    text = "The man walks into the house."
    prompt = build_t2g_prompt(text, _ManualTokenizer(), examples=_sample_examples())
    expected_user = (
        "Translate the following English sentence into ASL gloss.\n\n"
        "Examples:\n"
        "English: The cat sleeps on the sofa\n"
        "ASL gloss: CAT SLEEP SOFA\n"
        "\n"
        "English: A dog runs in the park\n"
        "ASL gloss: DOG RUN PARK\n"
        "\n"
        "Now translate:\n"
        "English: The man walks into the house."
    )
    assert prompt == _legacy_manual_prompt(expected_user)


def test_few_shot_prompt_contains_all_parts():
    """All structural markers and examples appear, in order."""
    text = "The man walks into the house."
    prompt = build_t2g_prompt(text, _ManualTokenizer(), examples=_sample_examples())

    assert prompt.startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert "Translate the following English sentence into ASL gloss." in prompt
    assert "Examples:" in prompt
    assert "Now translate:" in prompt
    # Examples preserved in the given order.
    assert prompt.index("English: The cat sleeps on the sofa") < prompt.index(
        "English: A dog runs in the park"
    )
    assert "ASL gloss: CAT SLEEP SOFA" in prompt
    assert "ASL gloss: DOG RUN PARK" in prompt
    # Query appears exactly once, under the "Now translate" marker.
    assert f"Now translate:\nEnglish: {text}" in prompt
    assert prompt.count(f"English: {text}") == 1


def test_few_shot_prompt_with_chat_tokenizer():
    """The chat-template path also carries the few-shot user content."""
    tok = _ChatTokenizer()
    prompt = build_t2g_prompt(
        "The man walks into the house.",
        tok,
        examples=_sample_examples(),
    )
    assert "CHAT|" in prompt
    assert "Examples:" in prompt
    assert "Now translate:" in prompt
