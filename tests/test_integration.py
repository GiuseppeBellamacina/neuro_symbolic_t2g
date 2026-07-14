#!/usr/bin/env python3
"""Integration tests — end-to-end coherence check.

Validates the full chain:
    data → grammar → rewards → metrics → callbacks
all produce consistent values and types.

Tests requiring the dataset are skipped if offline.
"""

from __future__ import annotations

import hashlib

import numpy as np


def test_data_to_grammar_chain(dataset, tokenizer):
    """Data → Grammar: vocab from real data builds a valid mask."""
    from src.datasets.aslg_dataset import extract_gloss_vocabulary
    from src.grammar.gloss_grammar import GlossVocabularyMask

    vocab = extract_gloss_vocabulary(dataset, split="train")
    mask = GlossVocabularyMask(vocab, tokenizer)
    assert len(mask.token_ids) > 0, f"Token IDs for {len(vocab)} glosses"
    assert mask.is_allowed(mask.eos_token_id), "EOS allowed"
    allowed = mask.get_allowed_token_ids()
    assert len(allowed) > 0, "Allowed IDs non-empty"


def test_grammar_to_rewards_chain(dataset):
    """Grammar → Rewards: reward functions work with real data."""
    from src.datasets.aslg_dataset import extract_gloss_vocabulary
    from src.datasets.transition_matrix import compute_bigram_transitions
    from src.rewards.t2g_rewards import (
        build_t2g_reward_functions,
        initialize_rewards,
        structural_dense_reward,
    )

    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    initialize_rewards(bigram, vocab)

    funcs, weights = build_t2g_reward_functions()
    assert len(funcs) == 4, f"4 reward functions built, got {len(funcs)}"

    completions = ["IX MAN WALK HOUSE", "DOG CAT BIRD", "NOT CAN WANT GO"]
    for fn in funcs:
        results = fn(completions)
        assert len(results) == len(
            completions
        ), f"{fn.__name__} returns {len(completions)} scores"
        for r in results:
            assert isinstance(r, float), f"{fn.__name__} score is float"

    sd = structural_dense_reward("IX MAN WALK", normalize=True)
    assert -1.0 <= sd <= 1.0, f"Structural dense in [-1,1], got {sd:.4f}"


def test_rewards_to_metrics_chain(dataset):
    """Rewards → Metrics: metrics consistent with rewards."""
    from src.datasets.aslg_dataset import extract_gloss_vocabulary
    from src.datasets.transition_matrix import compute_bigram_transitions
    from src.rewards.t2g_rewards import initialize_rewards
    from src.utils.metrics import (
        compute_detailed_metrics,
        compute_pass_at_1,
        compute_reward_breakdown,
        rouge_l_score,
    )

    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    initialize_rewards(bigram, vocab)

    completions = ["IX MAN WALK", "DOG CAT BIRD", "NOT CAN WANT"]
    references = ["IX MAN WALK", "IX MAN GO", "NOT CAN COME"]

    pass1 = compute_pass_at_1(completions, references, threshold=0.3)
    assert 0.0 <= pass1 <= 1.0, f"Pass@1 in [0,1], got {pass1:.4f}"

    rl = rouge_l_score(completions[0], references[0])
    assert abs(rl - 1.0) < 0.01, f"ROUGE-L perfect match = 1.0, got {rl:.4f}"

    breakdown = compute_reward_breakdown(completions)
    assert len(breakdown) >= 4, "Breakdown has >=4 keys"
    assert all(np.isfinite(v) for v in breakdown.values()), "All values finite"

    detailed = compute_detailed_metrics(completions, references)
    assert "overall_pass_rate" in detailed, "Detailed metrics has pass_rate"


def test_callbacks_interface(dataset):
    """Callbacks: CompletionSampleLogger and SFTSampleCallback creation."""
    from src.datasets.aslg_dataset import extract_gloss_vocabulary
    from src.datasets.transition_matrix import compute_bigram_transitions
    from src.rewards.t2g_rewards import (
        build_t2g_reward_functions,
        initialize_rewards,
        register_gold_glosses,
    )
    from src.training.callbacks import (
        CompletionSampleCallback,
        CompletionSampleLogger,
        SFTSampleCallback,
    )
    from src.utils.text_utils import extract_user_text

    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    initialize_rewards(bigram, vocab)

    reward_fns, reward_weights = build_t2g_reward_functions()
    assert len(reward_fns) == 4
    assert len(reward_weights) == 4

    logger = CompletionSampleLogger(reward_fns, reward_weights, n_samples=3)
    assert logger is not None, "Logger created"
    assert len(logger.wrapped_reward_fns) == 4, "Wrapped reward fns available"

    completions = ["IX MAN WALK", "DOG CAT"]
    for fn in logger.wrapped_reward_fns:
        result = fn(completions)
        assert isinstance(result, list), f"{fn.__name__} returns list"

    logger._capture(completions, None)
    assert len(logger._buffer) > 0, "Buffer has samples after capture"

    formatted = logger.format_samples()
    assert isinstance(formatted, str), "format_samples returns string"
    assert len(formatted) > 0, "format_samples non-empty"
    assert "COMPLETION SAMPLES" in formatted, "Contains header"
    assert "GOLD:" not in formatted, "No GOLD when unregistered"

    # Register gold and verify GOLD appears
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
    assert "GOLD:" in formatted2, "GOLD appears when registered"
    assert "ENTER HOUSE" in formatted2, "Correct gold text"
    assert "✓" in formatted2 or "✗" in formatted2, "Match indicator present"

    # Only active reward components in breakdown
    sample = logger._buffer[0]
    bd_keys = set(sample["breakdown"].keys())
    expected_active = {
        "translation_quality_reward",
        "gold_structure_reward",
        "gloss_format_reward",
        "gloss_repetition_reward",
    }
    assert bd_keys == expected_active, f"Only active components: got {bd_keys}"
    assert "viterbi_distance_reward" not in bd_keys, "Inactive not computed"

    cb = CompletionSampleCallback(logger, every_n_steps=5)
    assert cb is not None, "Callback created"
    assert cb._logger is logger, "Callback has logger"

    sft_cb = SFTSampleCallback(
        tokenizer=None, model=None, dataset=None, every_n_steps=25
    )
    assert sft_cb is not None, "SFTSampleCallback created"
    assert sft_cb._every_n_steps == 25


def test_module_imports():
    """All key modules import without errors."""
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
        mod = __import__(mod_name, fromlist=attrs)
        for attr in attrs:
            assert hasattr(mod, attr), f"{mod_name}.{attr} exists"
