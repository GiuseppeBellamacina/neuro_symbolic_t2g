"""
T2G Evaluation Script — Multi-metric with plots.

Evaluates trained checkpoints on the ASLG-PC12 test set using:
    - ROUGE-L F1 (translation quality)
    - Pass@1 / Pass@k (multiple sampling)
    - Gloss validity (free-text / repetition detection)
    - Bigram log-probability (structural plausibility)
    - Exact match accuracy
    - Per-component reward breakdown
    - Detailed metrics (percentiles, error distribution)

Optionally generates plots via ``visualization.py`` (plotnine):
    - Completion length distribution (valid vs invalid)
    - Baseline vs Post-GRPO comparison
    - Reward component breakdown

Usage:
    python -m src.training.eval_t2g --config config/grpo_t2g_qwen05.yaml --checkpoint checkpoints/qwen05/final
    python -m src.training.eval_t2g --config config/grpo_t2g_qwen05.yaml --checkpoint path/to/ckpt --num-samples 5 --plot
    python -m src.training.eval_t2g --config config/grpo_t2g_qwen05.yaml --checkpoint path/to/ckpt --baseline 0.15 --plot
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data.aslg_dataset import (
    download_aslg_dataset,
    load_vocabulary,
)
from src.data.transition_matrix import (
    load_transition_matrix,
    sequence_score_bigram,
)
from src.grammar.gloss_grammar import GlossVocabularyMask
from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor
import hashlib

from src.rewards.t2g_rewards import (
    initialize_rewards,
    register_gold_glosses,
)
from src.utils.prompting import build_t2g_prompt

logger = logging.getLogger("t2g-eval")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_for_eval(
    checkpoint_path: str,
    base_model_name: str,
) -> tuple[Any, Any]:
    """Load a trained model for evaluation.

    Handles both full model checkpoints and PEFT/LoRA adapter checkpoints.
    For adapter checkpoints, loads the base model first, then merges adapters.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_path = Path(checkpoint_path)
    is_peft = (ckpt_path / "adapter_config.json").exists()

    logger.info(f"Loading model from {checkpoint_path} (is_peft={is_peft})...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if is_peft:
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


# ---------------------------------------------------------------------------
# Single generation helper
# ---------------------------------------------------------------------------


def _generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    logits_processor: Any,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    """Generate one completion with constrained decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            logits_processor=[logits_processor],
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output[0][prompt_len:], skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str,
    max_samples: int = 200,
    num_samples: int = 1,
) -> dict[str, Any]:
    """Evaluate a checkpoint on the test set with full metrics.

    Args:
        config: Parsed YAML config.
        checkpoint_path: Path to the checkpoint directory.
        max_samples: Max test samples to evaluate.
        num_samples: Number of completions per prompt (1 = greedy, >1 = sampled).

    Returns:
        Dict with all computed metrics.
    """
    ds_cfg = config["dataset"]
    grpo_cfg = config["grpo"]
    reward_cfg = config.get("reward", {})

    # ── Load test data ───────────────────────────────────────────────────
    dataset = download_aslg_dataset(cache_dir=ds_cfg.get("dataset_cache"))
    vocab = load_vocabulary(ds_cfg.get("vocab_path", "data/gloss_vocab.txt"))
    bigram = load_transition_matrix(
        ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy"),
    )

    initialize_rewards(bigram, vocab)
    token_to_idx = {t: i for i, t in enumerate(vocab)}

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer = load_model_for_eval(
        checkpoint_path, config["model"]["name"],
    )

    # ── Constrained decoding ─────────────────────────────────────────────
    gloss_mask = GlossVocabularyMask(vocab, tokenizer)
    logits_processor = GlossVocabularyLogitsProcessor(
        gloss_mask, device=str(model.device),
    )

    # ── Prepare test samples ─────────────────────────────────────────────
    test_ds = dataset["test"]
    if max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))

    do_sample = num_samples > 1

    # ── Collect completions ──────────────────────────────────────────────
    # Multi-sample: list[list[str]] per prompt (always nested for consistency)
    all_completions: list[list[str]] = []
    all_references: list[str] = []
    all_sample_ids: list[str] = []
    all_bigram_scores: list[float] = []
    all_exact_matches: list[float] = []

    system_prompt = (
        "You are an English-to-ASL-gloss translator. "
        "Translate the following English sentence into a sequence of "
        "ASL glosses. Output ONLY the gloss tokens separated by spaces. "
        "Do not include explanations or extra text."
    )

    for sample in tqdm(test_ds, desc="Evaluating"):
        text = sample["text"]
        gold = sample["gloss"]

        # Build prompt with centralized template (same as training)
        prompt = build_t2g_prompt(text, tokenizer)

        # Generate N completions
        completions: list[str] = []
        for _ in range(num_samples):
            temp = 0.7 if do_sample else 1.0  # greedy ignores temperature
            gen = _generate_one(
                model, tokenizer, prompt, logits_processor,
                max_new_tokens=grpo_cfg.get("max_completion_length", 256),
                do_sample=do_sample, temperature=temp,
            )
            logits_processor.reset()
            completions.append(gen)

        # Store
        all_completions.append(completions)

        all_references.append(gold)
        all_sample_ids.append(
            hashlib.sha256(
                str(text).encode("utf-8", errors="replace")
            ).hexdigest()
        )

        # Bigram score (on first completion)
        tokens = completions[0].split()
        indices = [token_to_idx.get(t, token_to_idx.get("<UNK>", 0)) for t in tokens]
        if len(indices) >= 2:
            bos = token_to_idx.get("<BOS>", 0)
            eos = token_to_idx.get("<EOS>", 1)
            all_bigram_scores.append(
                sequence_score_bigram(bigram, [bos] + indices + [eos]),
            )
        else:
            all_bigram_scores.append(-10.0)

        # Exact match (first completion)
        all_exact_matches.append(1.0 if completions[0] == gold.strip() else 0.0)

    # ── Compute metrics via metrics.py ───────────────────────────────────
    from src.utils.metrics import (
        check_gloss_validity,
        compute_detailed_metrics,
        compute_pass_at_1,
        compute_pass_at_k,
        compute_reward_breakdown,
        rouge_l_score,
    )

    # Register gold glosses using stable sample IDs (SHA256 of user text).
    # This matches the format-agnostic lookup in _lookup_gold_gloss.
    register_gold_glosses(all_sample_ids, all_references)

    # Flatten completions and sample_ids for per-completion metrics.
    # For num_samples=1, there is 1 completion per prompt.
    # For num_samples>1, all completions are scored individually.
    flat_completions: list[str] = [
        c for comps in all_completions for c in comps
    ]
    flat_sample_ids: list[str] = [
        sid for i, sid in enumerate(all_sample_ids)
        for _ in all_completions[i]
    ]

    # Validity stats
    validity: list[tuple[bool, str]] = [
        check_gloss_validity(c) for c in flat_completions
    ]
    valid_count = sum(1 for v, _ in validity if v)
    error_counts = Counter(err for _, err in validity if err)

    # Pass@1
    pass1 = compute_pass_at_1(flat_completions, all_references, threshold=0.3)

    # Pass@k (multi-sample only — uses nested list)
    passk: dict[str, float] = {}
    if num_samples > 1:
        passk = compute_pass_at_k(
            all_completions, all_references,
            k_values=tuple(range(1, min(num_samples + 1, 11))),
            threshold=0.3,
        )

    # Detailed metrics
    detailed = compute_detailed_metrics(flat_completions, all_references)

    # Per-component reward breakdown (all completions with sample_ids)
    reward_components = compute_reward_breakdown(
        flat_completions, sample_ids=flat_sample_ids,
    )

    # ROUGE-L mean/std
    rouge_scores = [rouge_l_score(c, r) for c, r in zip(flat_completions, all_references)]

    # ── Assemble results ─────────────────────────────────────────────────
    results: dict[str, Any] = {
        "num_samples_evaluated": len(all_references),
        "num_completions_per_prompt": num_samples,
        "rouge_l_mean": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "rouge_l_std": float(np.std(rouge_scores)) if rouge_scores else 0.0,
        "pass_at_1": pass1,
        "bigram_log_prob_mean": float(np.mean(all_bigram_scores)) if all_bigram_scores else 0.0,
        "bigram_log_prob_std": float(np.std(all_bigram_scores)) if all_bigram_scores else 0.0,
        "exact_match": float(np.mean(all_exact_matches)) if all_exact_matches else 0.0,
        "validity_rate": valid_count / max(len(flat_completions), 1),
        "valid_count": valid_count,
        "invalid_count": len(flat_completions) - valid_count,
        "error_distribution": dict(error_counts.most_common(20)),
        "reward_breakdown": reward_components,
        "detailed_metrics": detailed,
        "total_completions": len(flat_completions),
    }
    if passk:
        results["pass_at_k"] = passk

    return results, flat_completions, validity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="T2G checkpoint evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--max-samples", type=int, default=200, help="Max test samples")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Completions per prompt (1=greedy, >1=sampled for Pass@k)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--plot", action="store_true",
                        help="Generate evaluation plots via visualization.py")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Baseline Pass@1 for comparison plot")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    import yaml

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_tag = Path(args.checkpoint).parent.name if Path(args.checkpoint).name == "final" else Path(args.checkpoint).name

    results, completions, validity = evaluate_checkpoint(
        config, args.checkpoint,
        max_samples=args.max_samples,
        num_samples=args.num_samples,
    )

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  T2G Evaluation Results — {model_tag}")
    print("=" * 60)
    print(f"  Samples evaluated:       {results['num_samples_evaluated']}")
    print(f"  Completions per prompt:  {results['num_completions_per_prompt']}")
    print(f"  ROUGE-L (mean ± std):    {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}")
    print(f"  Pass@1:                  {results['pass_at_1']:.4f}")
    if "pass_at_k" in results:
        for k, v in results["pass_at_k"].items():
            print(f"  {k}:{' ' * (25 - len(k))}{v:.4f}")
    print(f"  Exact match:             {results['exact_match']:.4f}")
    print(f"  Bigram log-prob (mean):  {results['bigram_log_prob_mean']:.4f}")
    print(f"  Validity rate:           {results['validity_rate']:.4f}  "
          f"({results['valid_count']} valid / {results['invalid_count']} invalid)")
    print(f"\n  ── Reward Breakdown ──")
    for k, v in results["reward_breakdown"].items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  ── Error Distribution ──")
    for err, count in results["error_distribution"].items():
        print(f"    {err}: {count}")
    print("=" * 60)

    # ── Save JSON ────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        ckpt_name = Path(args.checkpoint).name
        out_path = Path(args.checkpoint).parent / f"eval_{ckpt_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")

    # ── Generate plots ───────────────────────────────────────────────────
    if args.plot:
        from src.utils.visualization import (
            plot_baseline_vs_grpo,
            plot_completion_length_distribution,
            plot_reward_breakdown,
        )

        figures_dir = Path(args.checkpoint).parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        valid_mask = [v for v, _ in validity]

        # 1. Completion length distribution
        plot_completion_length_distribution(
            completions, valid_mask=valid_mask,
            title=f"Gloss Length — {model_tag}",
            output_path=str(figures_dir / "completion_lengths.png"),
        )

        # 2. Baseline vs GRPO (if baseline provided)
        if args.baseline is not None:
            plot_baseline_vs_grpo(
                baseline_pass1=args.baseline,
                grpo_pass1=results["pass_at_1"],
                model_name=model_tag,
                output_path=str(figures_dir / "baseline_vs_grpo.png"),
            )

        # 3. Reward breakdown
        rewards_cfg = config.get("reward", {})
        weights = {
            "translation_quality_reward": rewards_cfg.get("weight_translation", 0.4),
            "structural_dense_reward": rewards_cfg.get("weight_structure", 0.4),
            "gloss_format_reward": rewards_cfg.get("weight_format", 0.1),
            "gloss_repetition_reward": rewards_cfg.get("weight_repetition", 0.1),
        }
        plot_reward_breakdown(
            [{"label": model_tag, "scores": results["reward_breakdown"]}],
            reward_weights=weights,
            model_name=model_tag,
            output_path=str(figures_dir / "reward_breakdown.png"),
        )

        print(f"\n  Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
