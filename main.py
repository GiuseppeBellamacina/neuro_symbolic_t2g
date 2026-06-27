"""
Neuro-Symbolic T2G Pipeline — Component Testing.

Tests each component of the pipeline independently before launching
full GRPO training on the cluster.

Usage:
    python main.py                          # Run all tests
    python main.py --task data              # Test data ingestion only
    python main.py --task grammar           # Test grammar/logits processor only
    python main.py --task rewards           # Test reward functions only
    python main.py --task single_generation # Test single constrained generation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("t2g-test")


def test_data_ingestion() -> None:
    """Test Task 1: Data ingestion and transition matrix computation."""
    logger.info("=" * 60)
    logger.info("TESTING: Data Ingestion & Transition Matrix")
    logger.info("=" * 60)

    from src.data.aslg_dataset import (
        build_t2g_dataset,
        download_aslg_dataset,
        extract_gloss_vocabulary,
        save_vocabulary,
        load_vocabulary,
    )
    from src.data.transition_matrix import (
        compute_bigram_transitions,
        save_transition_matrix,
        load_transition_matrix,
        transition_score,
        sequence_score_bigram,
    )

    # Download dataset (first ~500 samples for quick testing)
    logger.info("Downloading ASLG-PC12 dataset (test subset)...")
    dataset = download_aslg_dataset(cache_dir="data/aslg_pc12_test")

    # Extract vocabulary
    vocab = extract_gloss_vocabulary(dataset, split="train")
    assert len(vocab) > 10, f"Vocabulary too small: {len(vocab)} tokens"
    logger.info(f"✓ Vocabulary: {len(vocab)} unique gloss tokens")
    logger.info(f"  Sample: {vocab[:15]}")

    # Save and reload
    save_vocabulary(vocab, "data/test_vocab.txt")
    reloaded = load_vocabulary("data/test_vocab.txt")
    assert reloaded == vocab, "Vocabulary save/load mismatch"
    logger.info("✓ Vocabulary save/load: OK")

    # Compute bigram transitions
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    assert bigram.shape == (len(vocab), len(vocab)), f"Shape mismatch: {bigram.shape}"
    assert np.allclose(bigram.sum(axis=1), 1.0, atol=1e-5), "Rows must sum to 1"
    logger.info(f"✓ Bigram matrix: shape={bigram.shape}, normalized OK")

    # Save and reload
    save_transition_matrix(bigram, "data/test_bigram.npy")
    reloaded_bigram = load_transition_matrix("data/test_bigram.npy")
    assert np.allclose(bigram, reloaded_bigram), "Bigram save/load mismatch"
    logger.info("✓ Bigram matrix save/load: OK")

    # Test transition scoring
    score = transition_score(bigram, 0, 1)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    logger.info(f"✓ Transition score P(gloss[1]|gloss[0]): {score:.6f}")

    # Test sequence scoring
    indices = [0, 1, 2, 3, 4]  # BOS, three glosses, EOS
    seq_score = sequence_score_bigram(bigram, indices)
    logger.info(f"✓ Sequence log-prob: {seq_score:.4f}")

    # Build T2G dataset
    t2g_ds = build_t2g_dataset(dataset, split="train", max_samples=10)
    assert len(t2g_ds) == 10, f"Expected 10 samples, got {len(t2g_ds)}"
    logger.info(f"✓ T2G dataset: {len(t2g_ds)} prompt-completion pairs")
    logger.info(f"  Sample prompt: {t2g_ds[0]['prompt'][:80]}...")
    logger.info(f"  Sample completion: {t2g_ds[0]['completion'][:80]}...")

    logger.info("\n✅ Task 1: ALL TESTS PASSED\n")


def test_grammar_processor() -> None:
    """Test Task 2: Constrained decoding with GlossVocabularyLogitsProcessor."""
    logger.info("=" * 60)
    logger.info("TESTING: Grammar & Logits Processor")
    logger.info("=" * 60)

    import torch

    from src.grammar.gloss_grammar import GlossVocabularyMask, build_gloss_grammar
    from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor

    # We need a tokenizer. Try loading a small one.
    logger.info("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
    except Exception:
        logger.warning("Cannot load Qwen tokenizer; using gpt2 for testing")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Build a small test vocabulary
    test_vocab = ["<BOS>", "<EOS>", "<UNK>", "IX", "MAN", "WALK", "HOUSE", "BOOK",
                  "fs-JOHN", "fs-MARY", "NOT", "CAN", "WANT", "GO", "COME"]

    # Test GlossVocabularyMask
    mask = GlossVocabularyMask(test_vocab, tokenizer)
    assert len(mask.token_ids) > 0, "No token IDs found"
    logger.info(f"✓ Vocabulary mask: {len(test_vocab)} glosses → {len(mask.token_ids)} token IDs")

    allowed = mask.get_allowed_token_ids()
    assert mask.eos_token_id in mask.token_ids, "EOS must be allowed"
    logger.info(f"  EOS token ID: {mask.eos_token_id} (allowed={mask.is_allowed(mask.eos_token_id)})")

    # Test LogitsProcessor
    processor = GlossVocabularyLogitsProcessor(mask, device="cpu")
    logger.info(f"✓ Logits processor created: {processor}")

    # Simulate a generation step
    vocab_size = tokenizer.vocab_size
    dummy_scores = torch.randn(1, vocab_size) * 0.1
    dummy_input_ids = torch.zeros(1, 5, dtype=torch.long)

    filtered = processor(dummy_input_ids, dummy_scores)
    assert filtered.shape == dummy_scores.shape, "Shape must be preserved"
    # Check that disallowed tokens are -inf
    for tid in range(min(100, vocab_size)):
        if tid not in mask.token_ids:
            assert filtered[0, tid] == -float("inf"), f"Token {tid} should be masked"

    num_allowed = sum(1 for tid in range(vocab_size) if filtered[0, tid] != -float("inf"))
    logger.info(f"✓ After masking: {num_allowed} / {vocab_size} tokens allowed")

    # Test grammar building
    grammar = build_gloss_grammar(test_vocab, tokenizer)
    assert "S*" in grammar, "Grammar must have start symbol S*"
    assert len(grammar["S*"]) > 0, "S* must have productions"
    logger.info(f"✓ Grammar: S* → {len(grammar['S*'])} alternatives")

    logger.info("\n✅ Task 2: ALL TESTS PASSED\n")


def test_reward_functions() -> None:
    """Test Task 3: Reward functions."""
    logger.info("=" * 60)
    logger.info("TESTING: Reward Functions")
    logger.info("=" * 60)

    from src.data.aslg_dataset import (
        download_aslg_dataset,
        extract_gloss_vocabulary,
    )
    from src.data.transition_matrix import compute_bigram_transitions
    from src.rewards.t2g_rewards import (
        initialize_rewards,
        translation_quality_reward,
        structural_dense_reward,
        gloss_format_reward,
        gloss_repetition_reward,
        build_t2g_reward_functions,
    )

    # Load data
    dataset = download_aslg_dataset(cache_dir="data/aslg_pc12_test")
    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)

    # Initialize
    initialize_rewards(bigram, vocab)

    # Test translation quality reward
    gold = "IX MAN WALK HOUSE"
    generated = "IX MAN WALK HOUSE"
    score = translation_quality_reward(generated, gold)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    logger.info(f"✓ Translation quality (perfect match): {score:.4f}")

    generated_bad = "DOG CAT BIRD FISH"
    score_bad = translation_quality_reward(generated_bad, gold)
    assert score_bad < score, "Bad match should score lower"
    logger.info(f"✓ Translation quality (bad match): {score_bad:.4f}")

    # Test structural dense reward
    struct_score = structural_dense_reward("IX MAN WALK")
    assert 0.0 <= struct_score <= 1.0, f"Score out of range: {struct_score}"
    logger.info(f"✓ Structural dense reward: {struct_score:.4f}")

    # Test format reward
    format_score = gloss_format_reward("IX MAN WALK")
    assert format_score == 1.0, f"Clean gloss should score 1.0: {format_score}"
    logger.info(f"✓ Format reward (clean glosses): {format_score}")

    format_bad = gloss_format_reward("Here is the translation: IX MAN WALK")
    assert format_bad < 1.0, "Free text should score < 1.0"
    logger.info(f"✓ Format reward (free text): {format_bad}")

    # Test repetition reward
    rep_score = gloss_repetition_reward("IX MAN WALK HOUSE BOOK")
    assert rep_score == 1.0, "Non-repetitive should score 1.0"
    logger.info(f"✓ Repetition reward (normal): {rep_score}")

    rep_bad = gloss_repetition_reward("IX IX IX IX IX IX IX IX")
    assert rep_bad < 1.0, "Repetitive should score < 1.0"
    logger.info(f"✓ Repetition reward (repetitive): {rep_bad}")

    # Test build function
    funcs, weights = build_t2g_reward_functions()
    assert len(funcs) == len(weights), "Funcs and weights must match"
    assert len(funcs) == 4, f"Expected 4 reward funcs, got {len(funcs)}"
    logger.info(f"✓ Built {len(funcs)} reward functions with weights {[f'{w:.2f}' for w in weights]}")

    logger.info("\n✅ Task 3: ALL TESTS PASSED\n")


def test_single_generation() -> None:
    """Test a single constrained generation with a loaded model."""
    logger.info("=" * 60)
    logger.info("TESTING: Single Constrained Generation")
    logger.info("=" * 60)

    import torch

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning("Transformers not available; skipping generation test")
        return

    logger.info("Loading small model for test...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Cannot load model: {e}")
        logger.info("Skipping generation test.")
        return

    from src.data.aslg_dataset import (
        download_aslg_dataset,
        extract_gloss_vocabulary,
    )
    from src.grammar.gloss_grammar import GlossVocabularyMask
    from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor

    # Quick data load
    dataset = download_aslg_dataset(cache_dir="data/aslg_pc12_test")
    vocab = extract_gloss_vocabulary(dataset, split="train")

    # Build mask and processor
    mask = GlossVocabularyMask(vocab, tokenizer)
    processor = GlossVocabularyLogitsProcessor(mask, device=str(model.device))

    # Test prompt — use same format as training for consistency
    prompt = "The man walks into the house."
    system_prompt = (
        "You are an English-to-ASL-gloss translator. "
        "Translate the following English sentence into a sequence of "
        "ASL glosses. Output ONLY the gloss tokens separated by spaces. "
        "Do not include explanations or extra text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Use the tokenizer's built-in chat template (Qwen already has one set).
    # Do NOT manually override tokenizer.chat_template.
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        logger.info("  Using tokenizer's built-in chat template")
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted_prompt, return_tensors="pt"
        ).to(model.device)
        logger.info(f"  Formatted prompt starts with: {formatted_prompt[:80]}...")
    else:
        # Fallback: manual format
        formatted_prompt = (
            "<|im_start|>system\n" + system_prompt +
            "<|im_end|>\n<|im_start|>user\n" + prompt +
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(
            formatted_prompt, return_tensors="pt"
        ).to(model.device)

    logger.info(f"Source prompt: {prompt}")
    logger.info("Generating with gloss constraint...")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=0.7,
            logits_processor=[processor],
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input prompt — only decode the newly generated tokens.
    # The output includes both prompt + generation; we slice from the
    # prompt length onward.
    prompt_len = inputs["input_ids"].shape[1]
    new_token_ids = output[0][prompt_len:].tolist()

    generated = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    logger.info(f"Generated (new tokens only): {generated}")

    # Verify that all generated tokens are within the gloss vocabulary.
    # decode_to_glosses should be called with NEW tokens only.
    gloss_tokens = mask.decode_to_glosses(new_token_ids)
    logger.info(f"Gloss tokens: {gloss_tokens}")

    # Validate: each decoded gloss token should be in the vocab set
    invalid = [t for t in gloss_tokens if t not in mask.vocab_set]
    if invalid:
        logger.warning(f"  Unexpected tokens not in gloss vocab: {invalid}")
    else:
        logger.info("  All generated tokens are valid ASL glosses")

    logger.info("\n>>> Single generation test complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test T2G pipeline components")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "data", "grammar", "rewards", "single_generation"],
        help="Which task to test",
    )
    args = parser.parse_args()

    tasks = {
        "data": test_data_ingestion,
        "grammar": test_grammar_processor,
        "rewards": test_reward_functions,
        "single_generation": test_single_generation,
    }

    if args.task == "all":
        for name, fn in tasks.items():
            try:
                fn()
            except Exception as e:
                logger.error(f"❌ Task '{name}' FAILED: {e}")
                import traceback
                traceback.print_exc()
    else:
        tasks[args.task]()


if __name__ == "__main__":
    main()
