"""
T2G Evaluation Script.

Evaluates trained checkpoints on a held-out test set using multiple metrics:
    - ROUGE-L F1 (translation quality)
    - Bigram log-probability (structural plausibility)
    - Exact match accuracy

Usage:
    python -m src.training.eval_t2g --config config/grpo_t2g_qwen05.yaml --checkpoint path/to/ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data.aslg_dataset import (
    download_aslg_dataset,
    extract_gloss_vocabulary,
    load_vocabulary,
)
from src.data.transition_matrix import (
    load_transition_matrix,
    sequence_score_bigram,
)
from src.grammar.gloss_grammar import GlossVocabularyMask
from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor
from src.rewards.t2g_rewards import (
    initialize_rewards,
    translation_quality_reward,
)

logger = logging.getLogger("t2g-eval")


def load_model_for_eval(
    checkpoint_path: str,
    base_model_name: str,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load a trained model for evaluation.

    Handles both full model checkpoints and PEFT/LoRA adapter checkpoints.
    For adapter checkpoints, loads the base model first, then merges adapters.
    """
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_path = Path(checkpoint_path)
    is_peft = (ckpt_path / "adapter_config.json").exists()

    logger.info(f"Loading model from {checkpoint_path} (is_peft={is_peft})...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if is_peft:
        # Load base model + merge LoRA adapters
        logger.info(f"  Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ckpt_path))
        model = model.merge_and_unload()
        logger.info("  LoRA adapters merged and unloaded")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str,
    max_samples: int = 200,
) -> dict[str, float]:
    """Evaluate a single checkpoint on the test set.

    Returns a dict of metric names to scores.
    """
    ds_cfg = config["dataset"]
    grpo_cfg = config["grpo"]

    # Load test data
    dataset = download_aslg_dataset(cache_dir=ds_cfg.get("dataset_cache"))
    vocab = load_vocabulary(ds_cfg.get("vocab_path", "data/gloss_vocab.txt"))
    bigram = load_transition_matrix(
        ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy")
    )

    initialize_rewards(bigram, vocab)
    token_to_idx = {t: i for i, t in enumerate(vocab)}

    # Load model
    model, tokenizer = load_model_for_eval(
        checkpoint_path, config["model"]["name"]
    )

    # Build vocabulary mask
    gloss_mask = GlossVocabularyMask(vocab, tokenizer)
    logits_processor = GlossVocabularyLogitsProcessor(
        gloss_mask, device=str(model.device)
    )

    # Prepare test samples
    test_ds = dataset["test"]
    if max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))

    metrics: dict[str, list[float]] = {
        "rouge_l": [],
        "bigram_log_prob": [],
        "exact_match": [],
    }

    system_prompt = (
        "Translate English to ASL glosses. "
        "Output ONLY space-separated gloss tokens."
    )

    for sample in tqdm(test_ds, desc="Evaluating"):
        text = sample["text"]
        gold = sample["gloss"]

        # Build prompt
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"Translate to ASL glosses: {text}\nGlosses:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate with constraint
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=grpo_cfg.get("max_completion_length", 256),
                do_sample=False,
                logits_processor=[logits_processor],
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Compute metrics
        metrics["rouge_l"].append(translation_quality_reward(generated, gold))
        metrics["exact_match"].append(1.0 if generated == gold.strip() else 0.0)

        # Bigram score
        tokens = generated.split()
        indices = [
            token_to_idx.get(t, token_to_idx.get("<UNK>", 0))
            for t in tokens
        ]
        if len(indices) >= 2:
            bos = token_to_idx.get("<BOS>", 0)
            eos = token_to_idx.get("<EOS>", 1)
            metrics["bigram_log_prob"].append(
                sequence_score_bigram(bigram, [bos] + indices + [eos])
            )
        else:
            metrics["bigram_log_prob"].append(-10.0)

    # Aggregate
    results: dict[str, float] = {}
    for k, vals in metrics.items():
        results[k] = float(np.mean(vals)) if vals else 0.0
        results[f"{k}_std"] = float(np.std(vals)) if vals else 0.0

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="T2G checkpoint evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--max_samples", type=int, default=200, help="Max test samples")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results JSON"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = evaluate_checkpoint(config, args.checkpoint, max_samples=args.max_samples)

    # Print results
    print("\n" + "=" * 50)
    print("T2G Evaluation Results")
    print("=" * 50)
    for k, v in sorted(results.items()):
        if not k.endswith("_std"):
            print(f"  {k}: {v:.4f}")
    print("=" * 50)

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Results saved to {out_path}")
    else:
        ckpt_name = Path(args.checkpoint).name
        out_path = Path(args.checkpoint).parent / f"eval_{ckpt_name}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
