#!/usr/bin/env python3
"""Integration Test ? End-to-End Coherence Check.

Validates the full chain:
    data ? grammar ? rewards ? metrics ? callbacks
all produce consistent values and types.

Usage:
    python tests/test_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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


def test_data_to_grammar_chain() -> None:
    print("\n-- 1. Data ? Grammar Coherence --")
    from src.datasets.aslg_dataset import (
        download_aslg_dataset,
        extract_gloss_vocabulary,
    )
    from src.grammar.gloss_grammar import GlossVocabularyMask

    # Download a small test subset
    dataset = download_aslg_dataset(cache_dir="data/test_integration_cache")
    vocab = extract_gloss_vocabulary(dataset, split="train")

    # Try loading a tokenizer (gpt2 as universal fallback)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  ??  Cannot load tokenizer for grammar; skipping tokenizer tests")
        tokenizer = None

    if tokenizer:
        mask = GlossVocabularyMask(vocab, tokenizer)
        check(
            "Mask built from real vocab",
            len(mask.token_ids) > 0,
            f"{len(mask.token_ids)} token IDs for {len(vocab)} glosses",
        )
        check("EOS allowed in mask", mask.is_allowed(mask.eos_token_id))

        # Verify vocab tokens decode correctly
        allowed = mask.get_allowed_token_ids()
        check("Allowed IDs non-empty from real vocab", len(allowed) > 0)

    return dataset, vocab


def test_grammar_to_rewards_chain() -> None:
    print("\n-- 2. Grammar ? Rewards Coherence --")
    from src.datasets.aslg_dataset import (
        download_aslg_dataset,
        extract_gloss_vocabulary,
    )
    from src.datasets.transition_matrix import compute_bigram_transitions
    from src.rewards.t2g_rewards import (
        build_t2g_reward_functions,
        initialize_rewards,
        structural_dense_reward,
    )

    dataset = download_aslg_dataset(cache_dir="data/test_integration_cache")
    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)

    initialize_rewards(bigram, vocab)

    # Verify reward functions are compatible with real data
    funcs, weights = build_t2g_reward_functions()
    check("4 reward functions built", len(funcs) == 4)

    # Generate completions compatible with vocabulary
    completions = [
        "IX MAN WALK HOUSE",
        "DOG CAT BIRD",
        "NOT CAN WANT GO",
    ]
    for fn in funcs:
        results = fn(completions)
        check(
            f"  {fn.__name__} returns {len(completions)} scores",
            len(results) == len(completions),
            f"got {len(results)}",
        )
        for r in results:
            check(f"  {fn.__name__} score is float", isinstance(r, float), f"{r:.4f}")

    # Test structural_dense directly
    sd = structural_dense_reward("IX MAN WALK", normalize=True)
    check("Structural dense from real data ? [0, 1]", 0.0 <= sd <= 1.0, f"{sd:.4f}")


def test_rewards_to_metrics_chain() -> None:
    print("\n-- 3. Rewards ? Metrics Coherence --")
    from src.datasets.aslg_dataset import (
        download_aslg_dataset,
        extract_gloss_vocabulary,
    )
    from src.datasets.transition_matrix import compute_bigram_transitions
    from src.rewards.t2g_rewards import initialize_rewards
    from src.utils.metrics import (
        compute_detailed_metrics,
        compute_pass_at_1,
        compute_reward_breakdown,
        rouge_l_score,
    )

    dataset = download_aslg_dataset(cache_dir="data/test_integration_cache")
    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    initialize_rewards(bigram, vocab)

    completions = ["IX MAN WALK", "DOG CAT BIRD", "NOT CAN WANT"]
    references = ["IX MAN WALK", "IX MAN GO", "NOT CAN COME"]

    # Metrics should be consistent with rewards
    pass1 = compute_pass_at_1(completions, references, threshold=0.3)
    check("Pass@1 ? [0, 1]", 0.0 <= pass1 <= 1.0, f"{pass1:.4f}")

    # Individual ROUGE-L should match reward expectations
    rl = rouge_l_score(completions[0], references[0])
    check("ROUGE-L perfect match = 1.0", abs(rl - 1.0) < 0.01, f"{rl:.4f}")

    # Reward breakdown should produce valid numbers
    breakdown = compute_reward_breakdown(completions)
    check("Breakdown has 4 keys", len(breakdown) >= 4)
    check(
        "All breakdown values finite", all(np.isfinite(v) for v in breakdown.values())
    )

    # Detailed metrics
    detailed = compute_detailed_metrics(completions, references)
    check("Detailed metrics has pass_rate", "overall_pass_rate" in detailed)


def test_callbacks_interface() -> None:
    print("\n-- 4. Callbacks Interface Coherence --")
    from src.datasets.aslg_dataset import (
        download_aslg_dataset,
        extract_gloss_vocabulary,
    )
    from src.datasets.transition_matrix import compute_bigram_transitions
    from src.rewards.t2g_rewards import (
        build_t2g_reward_functions,
        initialize_rewards,
    )
    from src.training.callbacks import (
        CompletionSampleCallback,
        CompletionSampleLogger,
        SFTSampleCallback,
    )

    dataset = download_aslg_dataset(cache_dir="data/test_integration_cache")
    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    initialize_rewards(bigram, vocab)

    reward_fns, reward_weights = build_t2g_reward_functions()
    check("reward_fns built", len(reward_fns) == 4)
    check("reward_weights built", len(reward_weights) == 4)

    logger = CompletionSampleLogger(reward_fns, reward_weights, n_samples=3)
    check("Logger created", logger is not None)
    check("Wrapped reward fns available", len(logger.wrapped_reward_fns) == 4)

    # Verify wrapped functions return lists of floats
    completions = ["IX MAN WALK", "DOG CAT"]
    for fn in logger.wrapped_reward_fns:
        try:
            result = fn(completions)
            check(f"  {fn.__name__} returns list", isinstance(result, list))
        except Exception as e:
            check(f"  {fn.__name__} callable", False, f"{e}")

    # Test _capture
    logger._capture(completions, None)
    check("Buffer has samples after capture", len(logger._buffer) > 0)

    # Test format_samples
    formatted = logger.format_samples()
    check("format_samples returns string", isinstance(formatted, str))
    check("format_samples is non-empty", len(formatted) > 0)
    check(
        "format_samples contains 'COMPLETION SAMPLES'",
        "COMPLETION SAMPLES" in formatted,
    )
    # When no gold is registered, GOLD section should not appear
    check(
        "format_samples has no GOLD when unregistered",
        "GOLD:" not in formatted,
    )

    # Register a gold gloss and verify GOLD appears in formatted output
    import hashlib

    from src.rewards.t2g_rewards import register_gold_glosses
    from src.utils.text_utils import extract_user_text

    # Build fake prompts matching the completions
    fake_prompt_1 = [{"role": "user", "content": "The man walks into the house."}]
    fake_prompt_2 = [{"role": "user", "content": "The dog chases the cat."}]
    fake_prompts = [fake_prompt_1, fake_prompt_2]
    sid1 = hashlib.sha256(
        extract_user_text(fake_prompt_1).encode("utf-8", errors="replace")
    ).hexdigest()
    sid2 = hashlib.sha256(
        extract_user_text(fake_prompt_2).encode("utf-8", errors="replace")
    ).hexdigest()
    register_gold_glosses([sid1, sid2], ["IX MAN WALK ENTER HOUSE", "DOG CHASE CAT"])

    logger._capture(completions, fake_prompts)
    formatted2 = logger.format_samples()
    check(
        "format_samples includes GOLD when registered",
        "GOLD:" in formatted2,
    )
    check(
        "format_samples includes correct gold text",
        "ENTER HOUSE" in formatted2,
    )
    # Verify match indicator appears (✗ since OUTPUT != GOLD)
    check(
        "format_samples includes match indicator",
        "✓" in formatted2 or "✗" in formatted2,
    )
    # Verify only active reward components (weight > 0) are in the breakdown.
    # build_t2g_reward_functions() with no config returns 4 active rewards:
    # translation, gold_structure, format, repetition.
    sample = logger._buffer[0]
    bd_keys = set(sample["breakdown"].keys())
    expected_active = {
        "translation_quality_reward",
        "gold_structure_reward",
        "gloss_format_reward",
        "gloss_repetition_reward",
    }
    check(
        "Only active reward components in breakdown (weight > 0)",
        bd_keys == expected_active,
        f"got: {bd_keys}, expected: {expected_active}",
    )
    check(
        "Inactive components not computed (viterbi not present)",
        "viterbi_distance_reward" not in bd_keys,
    )

    # Test callback creation
    cb = CompletionSampleCallback(logger, every_n_steps=5)
    check("Callback created", cb is not None)
    check("Callback has logger", cb._logger is logger)

    # Test SFTSampleCallback creation (no model/tokenizer — should not crash)
    sft_cb = SFTSampleCallback(
        tokenizer=None, model=None, dataset=None, every_n_steps=25
    )
    check("SFTSampleCallback created", sft_cb is not None)
    check("SFTSampleCallback has every_n_steps", sft_cb._every_n_steps == 25)


def test_module_imports() -> None:
    print("\n-- 5. Module Import Chain --")
    modules = [
        (
            "src.datasets.aslg_dataset",
            ["download_aslg_dataset", "extract_gloss_vocabulary", "build_t2g_dataset"],
        ),
        (
            "src.datasets.transition_matrix",
            [
                "compute_bigram_transitions",
                "load_transition_matrix",
                "soft_viterbi_score",
                "forward_log_probs",
                "backward_log_probs",
            ],
        ),
        ("src.grammar.gloss_grammar", ["GlossVocabularyMask"]),
        ("src.grammar.grammar_logits_processor", ["GlossVocabularyLogitsProcessor"]),
        (
            "src.rewards.t2g_rewards",
            [
                "build_t2g_reward_functions",
                "initialize_rewards",
                "soft_viterbi_distance_reward",
                "verifier_scaled_reward",
            ],
        ),
        (
            "src.training.callbacks",
            ["CompletionSampleLogger", "CompletionSampleCallback", "SFTSampleCallback"],
        ),
        (
            "src.utils.metrics",
            [
                "compute_pass_at_1",
                "compute_reward_breakdown",
                "compute_evaluation_report",
                "bootstrap_confidence_interval",
                "sentence_bleu",
                "corpus_bleu",
            ],
        ),
        ("src.utils.visualization", ["plot_training_curves", "plot_reward_breakdown"]),
        ("src.utils.chain_monitor", []),
        ("src.utils.live_training_table", []),
        ("src.utils.show_training_log", []),
    ]
    for mod_name, attrs in modules:
        try:
            mod = __import__(mod_name, fromlist=attrs)
            for attr in attrs:
                check(f"  {mod_name}.{attr} exists", hasattr(mod, attr))
        except Exception as e:
            check(f"  {mod_name} imports", False, f"{e}")


def main() -> None:
    global PASS, FAIL
    print("=" * 60)
    print("TEST: Integration & Coherence")
    print("=" * 60)

    try:
        test_data_to_grammar_chain()
        test_grammar_to_rewards_chain()
        test_rewards_to_metrics_chain()
        test_callbacks_interface()
        test_module_imports()
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
