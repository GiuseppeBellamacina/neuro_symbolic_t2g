#!/usr/bin/env python3
"""Verify Grammar & Constrained Decoding ? Test Script.

Validates:
  1. GlossVocabularyMask maps gloss tokens to tokenizer IDs
  2. GlossVocabularyLogitsProcessor correctly masks non-gloss tokens
  3. LogitsProcessor preserves shape and sets disallowed tokens to -inf
  4. decode_to_glosses works correctly

Usage:
    python tests/test_grammar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


def get_tokenizer():
    """Load a tokenizer ? try Qwen first as it's T2G's target, fallback to gpt2."""
    from transformers import AutoTokenizer

    for name in ("Qwen/Qwen2.5-0.5B-Instruct", "gpt2"):
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            print(f"  Using tokenizer: {name} (vocab_size={tok.vocab_size})")
            return tok
        except Exception:
            continue
    raise RuntimeError("No tokenizer available for testing")


def test_gloss_vocabulary_mask(tokenizer) -> None:
    print("\n-- 1. GlossVocabularyMask --")
    from src.grammar.gloss_grammar import GlossVocabularyMask

    test_vocab = [
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "IX",
        "MAN",
        "WALK",
        "HOUSE",
        "BOOK",
        "fs-JOHN",
        "fs-MARY",
        "NOT",
        "CAN",
    ]
    mask = GlossVocabularyMask(test_vocab, tokenizer)

    check(
        "Mask has token IDs",
        len(mask.token_ids) > 0,
        f"{len(mask.token_ids)} token IDs",
    )
    check("EOS is in token IDs", mask.eos_token_id in mask.token_ids)
    check("EOS is allowed", mask.is_allowed(mask.eos_token_id))
    check("Random high ID is NOT allowed", not mask.is_allowed(999999))

    # Tokenizer-specific: the subword tokenization might produce
    # different numbers of token IDs than number of glosses.
    # This is expected ? just verify at least EOS is included.
    check(
        "get_allowed_token_ids returns list",
        isinstance(mask.get_allowed_token_ids(), list),
    )
    check("Allowed IDs non-empty", len(mask.get_allowed_token_ids()) > 0)

    allowed = mask.get_allowed_token_ids()
    # Verify all returned IDs are actually in token_ids
    for tid in allowed[:20]:  # check first 20 only to avoid spam
        check(f"  ID {tid} in mask.token_ids", tid in mask.token_ids, "")


def test_logits_processor(tokenizer) -> None:
    print("\n-- 2. GlossVocabularyLogitsProcessor --")
    from src.grammar.gloss_grammar import GlossVocabularyMask
    from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor

    test_vocab = [
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "IX",
        "MAN",
        "WALK",
        "HOUSE",
        "BOOK",
        "fs-JOHN",
        "NOT",
        "CAN",
        "WANT",
    ]
    mask = GlossVocabularyMask(test_vocab, tokenizer)
    processor = GlossVocabularyLogitsProcessor(mask, device="cpu")

    vocab_size = tokenizer.vocab_size
    check("Vocab size detected", processor.vocab_size > 0, f"{processor.vocab_size}")

    # Create fake logits (uniform random) and input ids
    dummy_scores = torch.randn(1, vocab_size) * 0.1
    dummy_input_ids = torch.zeros(1, 5, dtype=torch.long)

    filtered = processor(dummy_input_ids, dummy_scores)
    check("Output shape preserved", filtered.shape == dummy_scores.shape)
    check("Step counter incremented", processor.step_count == 1)

    # Count allowed tokens
    num_allowed = sum(
        1 for i in range(vocab_size) if filtered[0, i].item() != float("-inf")
    )
    num_disallowed = vocab_size - num_allowed
    check("Disallowed tokens are -inf", num_disallowed > 0, f"{num_disallowed} masked")
    check("Allowed tokens are finite", num_allowed > 0, f"{num_allowed} allowed")

    # At least EOS should be allowed (guard against tokenizers
    # where eos_token_id > vocab_size, e.g. Qwen has extra added tokens)
    eos_id = tokenizer.eos_token_id
    if eos_id < vocab_size:
        eos_allowed = filtered[0, eos_id].item() != float("-inf")
        check("EOS token is allowed", eos_allowed)
    else:
        # EOS ID outside vocab range ? can't be in the logit tensor.
        # Verify it IS in the mask's allowed set (pre-generation check).
        check(
            f"EOS token {eos_id} >= vocab_size {vocab_size} (added token)",
            eos_id in mask.token_ids,
            "EOS is allowed by mask but outside logit range",
        )

    # Test reset
    processor.reset()
    check("Reset clears step counter", processor.step_count == 0)


def test_decode_to_glosses(tokenizer) -> None:
    print("\n-- 3. decode_to_glosses --")
    from src.grammar.gloss_grammar import GlossVocabularyMask

    test_vocab = [
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "IX",
        "MAN",
        "WALK",
    ]
    mask = GlossVocabularyMask(test_vocab, tokenizer)

    # Tokenize a gloss token that should decode back
    man_tokens = tokenizer.encode("MAN", add_special_tokens=False)
    house_tokens = tokenizer.encode("WALK", add_special_tokens=False)

    if man_tokens and house_tokens:
        decoded = mask.decode_to_glosses(man_tokens)
        check("decode_to_glosses returns list", isinstance(decoded, list))
        check("decode_to_glosses has content", len(decoded) > 0)

    # Test with EOS stops decoding
    fake_ids = (
        man_tokens + [tokenizer.eos_token_id] + house_tokens
        if man_tokens
        else [tokenizer.eos_token_id]
    )
    decoded = mask.decode_to_glosses(fake_ids)
    check(
        "decode_to_glosses stops at EOS",
        len(decoded) <= len(man_tokens) if man_tokens else True,
    )


def test_grammar_build(tokenizer) -> None:
    print("\n-- 4. Grammar Build --")
    from src.grammar.gloss_grammar import build_gloss_grammar

    test_vocab = [
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "IX",
        "MAN",
        "WALK",
    ]
    grammar = build_gloss_grammar(test_vocab, tokenizer)

    check("Grammar has S* start symbol", "S*" in grammar)
    check(
        "S* has productions",
        len(grammar["S*"]) > 0,
        f"{len(grammar['S*'])} productions",
    )
    check(
        "Productions contain gloss tokens",
        any("MAN" in p or "WALK" in p for p in grammar["S*"]),
    )
    check("Productions contain EOS", any("EOS" in p for p in grammar["S*"]))
    check(
        "Productions contain S* recursion",
        any("S*" in p for p in grammar["S*"] if "EOS" not in p),
    )

    # BOS and UNK should NOT appear in productions
    for p in grammar["S*"]:
        check(
            f"  Production '{p[:40]}' excludes BOS/UNK",
            "BOS" not in p and "UNK" not in p,
            "",
        )


def main() -> None:
    global PASS, FAIL
    print("=" * 60)
    print("TEST: Grammar & Constrained Decoding")
    print("=" * 60)

    try:
        tokenizer = get_tokenizer()
        test_gloss_vocabulary_mask(tokenizer)
        test_logits_processor(tokenizer)
        test_decode_to_glosses(tokenizer)
        test_grammar_build(tokenizer)
    except Exception as e:
        print(f"\n  !! CRASH: {e}")
        import traceback

        traceback.print_exc()
        FAIL += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
