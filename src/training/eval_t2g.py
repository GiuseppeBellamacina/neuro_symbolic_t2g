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
    # Single checkpoint eval
    python -m src.training.eval_t2g --config experiments/configs/t2g/grpo_qwen05.yaml --checkpoint path/to/ckpt --plot

    # Compare baseline (zero-shot) vs GRPO — generates comparison plots + JSON
    python -m src.training.eval_t2g --config experiments/configs/t2g/grpo_qwen05.yaml --checkpoint path/to/ckpt --compare

    # Best-of-N selection (helps small models generalize — requires num_samples>1)
    python -m src.training.eval_t2g --config experiments/configs/t2g/grpo_qwen05.yaml --checkpoint path/to/ckpt --compare --best-of-n

    # Baseline-only eval (generates baseline JSON for later comparison)
    python -m src.training.eval_t2g --config experiments/configs/t2g/grpo_qwen05.yaml --eval-baseline-only --plot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Silence noisy transformers FutureWarnings (AttentionMaskConverter deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings(
    "ignore",
    message=".*AttentionMaskConverter.*",
    category=FutureWarning,
)

from src.datasets.aslg_dataset import (
    download_aslg_dataset,
    load_vocabulary,
)
from src.datasets.transition_matrix import (
    load_transition_matrix,
    sequence_score_bigram,
)
from src.grammar.gloss_grammar import GlossVocabularyMask
from src.grammar.grammar_logits_processor import (
    GlossVocabularyLogitsProcessor,
    GrammarPDALogitsProcessor,
)
from src.rewards.t2g_rewards import (
    initialize_rewards,
    register_gold_glosses,
)
from src.utils.config import load_config
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
        checkpoint_path,
        trust_remote_code=True,
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
        model = model.merge_and_unload()  # type: ignore[call-arg]
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
# Batched generation helper
# ---------------------------------------------------------------------------


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    logits_processor: Any,
    num_return_sequences: int = 1,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> list[str]:
    """Generate N completions in a single call via num_return_sequences.

    Uses ``num_return_sequences`` to generate all completions for a prompt
    in one ``model.generate()`` call, which is ~5x faster than calling
    ``model.generate()`` N times separately.

    When ``logits_processor`` is ``None``, generation is unconstrained
    (used for ablation studies — base model zero-shot or GRPO without grammar).

    Returns:
        List of N decoded completion strings.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_return_sequences,
    }
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = [logits_processor]
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    completions: list[str] = []
    for seq in outputs:
        text = tokenizer.decode(
            seq[prompt_len:],
            skip_special_tokens=True,
        ).strip()
        completions.append(text)
    return completions


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | None,
    max_samples: int = 200,
    num_samples: int = 1,
    best_of_n: bool = False,
) -> tuple[
    dict[str, Any],
    list[str],
    list[tuple[bool, str]],
    list[str],
    list[float],
    list[dict[str, Any]],
]:
    """Evaluate a checkpoint on the test set with full metrics.

    Args:
        config: Parsed YAML config.
        checkpoint_path: Path to the checkpoint directory, or ``None`` for
            zero-shot evaluation (loads the base model without LoRA).
        max_samples: Max test samples to evaluate.
        num_samples: Number of completions per prompt (1 = greedy, >1 = sampled).
        best_of_n: If True and num_samples > 1, select the best completion
            per prompt (highest ROUGE-L among valid ones) instead of using
            the first. This helps small models generalize by leveraging
            multiple samples at inference time (Pass@1 → Best-of-N).

    Returns:
        Tuple of ``(results, flat_completions, validity, all_references,
        rouge_scores, generations)`` where *results* is a dict with all
        computed metrics, *flat_completions* is a list of generated gloss
        strings, *validity* is a list of ``(is_valid, reason)`` tuples,
        *all_references* is the list of gold glosses, *rouge_scores* is the
        list of per-completion ROUGE-L scores, and *generations* is a list
        of per-completion dicts (text/gold/completion/valid/rouge_l) suitable
        for a standalone JSON dump (mirrors grpo-strict-generation's
        ``completions_*.json`` format).
    """
    ds_cfg = config["dataset"]
    # Support both generation (SFT) and grpo (GRPO) — generation preferred
    gen_cfg = config.get("generation", config.get("grpo", {}))

    # ── Load test data ───────────────────────────────────────────────────
    dataset = download_aslg_dataset(
        cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
    )
    vocab = load_vocabulary(ds_cfg.get("vocab_path", "data/gloss_vocab.txt"))
    bigram = load_transition_matrix(
        ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy"),
    )

    initialize_rewards(
        bigram,
        vocab,
        viterbi_diversity=config.get("grammar", {}).get("viterbi_diversity"),
    )
    token_to_idx = {t: i for i, t in enumerate(vocab)}

    # ── Load model ───────────────────────────────────────────────────────
    if checkpoint_path is None:
        logger.info(f"Zero-shot mode: loading base model {config['model']['name']}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name"],
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model, tokenizer = load_model_for_eval(
            checkpoint_path,
            config["model"]["name"],
        )

    # ── Constrained decoding ─────────────────────────────────────────────
    grammar_enabled = config.get("grammar", {}).get("enabled", True)
    if grammar_enabled:
        use_pda = config.get("grammar", {}).get("use_grammarllm_pda", False)
        if use_pda:
            # Lazy import to avoid hard dependency on grammarllm at module level
            from src.grammar.gloss_grammar import create_grammarllm_pipeline

            logger.info("Using GrammarLLM PDA for constrained decoding (eval)")
            _, _, pda = create_grammarllm_pipeline(
                vocab,
                tokenizer,
                temperature=gen_cfg.get("temperature", 0.7),
            )
            logits_processor = GrammarPDALogitsProcessor(
                tokenizer,
                pda,
                temperature=float(
                    config.get("grammar", {}).get("pda_temperature", 1.0)
                ),
            )
        else:
            gloss_mask = GlossVocabularyMask(vocab, tokenizer)
            logits_processor = GlossVocabularyLogitsProcessor(
                gloss_mask,
                device=str(model.device),
            )
    else:
        logger.info("⚠️  grammar.enabled=false — unconstrained generation (ablation)")
        logits_processor = None

    # ── Prepare test samples ─────────────────────────────────────────────
    test_ds = dataset["test"]
    if max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))

    do_sample = num_samples > 1
    if best_of_n and num_samples <= 1:
        logger.warning(
            "best_of_n=True but num_samples=%d (need >1). Disabling best_of_n.",
            num_samples,
        )
        best_of_n = False
    if best_of_n:
        logger.info(
            "Best-of-N selection enabled: generating %d samples/prompt and "
            "selecting the best (highest ROUGE-L among valid).",
            num_samples,
        )

    # ── Collect completions ──────────────────────────────────────────────
    # Multi-sample: list[list[str]] per prompt (always nested for consistency)
    all_completions: list[list[str]] = []
    all_references: list[str] = []
    all_sample_ids: list[str] = []
    all_texts: list[str] = []
    all_bigram_scores: list[float] = []
    all_exact_matches: list[float] = []

    for sample in tqdm(test_ds, desc="Evaluating"):
        text = sample["text"]  # type: ignore[index]
        gold = sample["gloss"]  # type: ignore[index]

        # Build prompt with centralized template (same as training)
        prompt = build_t2g_prompt(text, tokenizer)

        # Generate N completions in a single model.generate() call
        temp = 0.7 if do_sample else 1.0  # greedy ignores temperature
        completions = _generate_batch(
            model,
            tokenizer,
            prompt,
            logits_processor,
            num_return_sequences=num_samples,
            max_new_tokens=gen_cfg.get("max_completion_length", 256),
            do_sample=do_sample,
            temperature=temp,
        )
        if logits_processor is not None:
            logits_processor.reset()

        # Store
        all_completions.append(completions)

        all_references.append(gold)
        all_texts.append(str(text))
        all_sample_ids.append(
            hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()
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
        compute_evaluation_report,
        compute_pass_at_1,
        compute_pass_at_k,
        compute_reward_breakdown,
        rouge_l_score,
    )

    # Register gold glosses using stable sample IDs (SHA256 of user text).
    # This matches the format-agnostic lookup in _lookup_gold_gloss.
    register_gold_glosses(all_sample_ids, all_references)

    # ── Best-of-N selection ─────────────────────────────────────────────
    # If best_of_n is enabled, for each prompt we select the best completion
    # (highest ROUGE-L among valid ones; if none valid, highest ROUGE-L
    # overall). This effectively turns Pass@N into a stronger Pass@1 by
    # leveraging multiple samples at inference time — a well-known technique
    # to help small models generalize better without additional training.
    if best_of_n and num_samples > 1:
        selected_completions: list[str] = []
        for i, comps in enumerate(all_completions):
            gold = all_references[i]
            scored = [
                (rouge_l_score(c, gold), check_gloss_validity(c), c) for c in comps
            ]
            # Prefer valid completions; among those, pick highest ROUGE-L.
            valid_scored = [(rl, c) for rl, (v, _), c in scored if v]
            if valid_scored:
                best = max(valid_scored, key=lambda x: x[0])[1]
            else:
                # No valid completion — pick highest ROUGE-L overall.
                best = max(scored, key=lambda x: x[0])[2]
            selected_completions.append(best)
        # Replace flat lists with selected (1 per prompt)
        flat_completions = selected_completions
        flat_sample_ids = list(all_sample_ids)
        flat_texts = list(all_texts)
        flat_references = list(all_references)
        # Keep all_completions nested for Pass@k computation
        logger.info(
            "Best-of-N: selected 1 of %d completions per prompt (%d total).",
            num_samples,
            len(flat_completions),
        )
    else:
        # Flatten completions and sample_ids for per-completion metrics.
        # For num_samples=1, there is 1 completion per prompt.
        # For num_samples>1, all completions are scored individually.
        flat_completions = [c for comps in all_completions for c in comps]
        flat_sample_ids = [
            sid for i, sid in enumerate(all_sample_ids) for _ in all_completions[i]
        ]
        flat_texts = [
            txt for i, txt in enumerate(all_texts) for _ in all_completions[i]
        ]
        flat_references = [
            ref for i, ref in enumerate(all_references) for _ in all_completions[i]
        ]

    # Validity stats
    validity: list[tuple[bool, str]] = [
        check_gloss_validity(c) for c in flat_completions
    ]
    valid_count = sum(1 for v, _ in validity if v)
    error_counts = Counter(err for _, err in validity if err)

    # Pass@1
    pass1 = compute_pass_at_1(flat_completions, flat_references, threshold=0.3)

    # Pass@k (multi-sample only — uses nested list)
    passk: dict[str, float] = {}
    if num_samples > 1:
        passk = compute_pass_at_k(
            all_completions,
            all_references,
            k_values=tuple(range(1, min(num_samples + 1, 11))),
            threshold=0.3,
        )

    # Detailed metrics
    detailed = compute_detailed_metrics(flat_completions, flat_references)

    # Per-component reward breakdown (all completions with sample_ids)
    # Only compute components with weight > 0 to save computation
    rewards_cfg = config.get("reward", {})
    reward_weight_map = {
        "translation_quality_reward": rewards_cfg.get("weight_translation", 0.0),
        "structural_dense_reward": rewards_cfg.get("weight_structure", 0.0),
        "gold_structure_reward": rewards_cfg.get("weight_gold_structure", 0.0),
        "viterbi_distance_reward": rewards_cfg.get("weight_viterbi", 0.0),
        "soft_viterbi_distance_reward": rewards_cfg.get("weight_soft_viterbi", 0.0),
        "verifier_scaled_reward": rewards_cfg.get("weight_verifier_scaled", 0.0),
        "gloss_order_reward": rewards_cfg.get("weight_gloss_order", 0.0),
        "gloss_format_reward": rewards_cfg.get("weight_format", 0.0),
        "gloss_repetition_reward": rewards_cfg.get("weight_repetition", 0.0),
    }
    reward_components = compute_reward_breakdown(
        flat_completions,
        sample_ids=flat_sample_ids,
        reward_weights=reward_weight_map,
    )

    # ROUGE-L mean/std
    rouge_scores = [
        rouge_l_score(c, r) for c, r in zip(flat_completions, flat_references)
    ]

    # ── Per-completion generations log (stile grpo-strict-generation) ────
    # Salva ogni generazione con testo sorgente, gold, completion, validità
    # ed ROUGE-L, per ispezione manuale e debugging (es. quali frasi vanno
    # male, quali errori di validità sono più comuni, ecc).
    generations: list[dict[str, Any]] = []
    for i, (txt, ref, comp, (is_valid, err), rl, sid) in enumerate(
        zip(
            flat_texts,
            flat_references,
            flat_completions,
            validity,
            rouge_scores,
            flat_sample_ids,
        )
    ):
        entry: dict[str, Any] = {
            "index": i,
            "sample_id": sid,
            "text": txt,
            "gold_gloss": ref,
            "completion": comp,
            "valid": is_valid,
            "rouge_l": round(rl, 4),
        }
        if not is_valid:
            entry["error"] = err
        generations.append(entry)

    # ── Assemble results ─────────────────────────────────────────────────
    # Comprehensive evaluation report with BLEU + bootstrap CI (RECIPE-inspired)
    eval_report = compute_evaluation_report(
        flat_completions, flat_references, n_bootstrap=1000
    )

    results: dict[str, Any] = {
        "num_samples_evaluated": len(all_references),
        "num_completions_per_prompt": num_samples,
        "best_of_n": best_of_n,
        "rouge_l_mean": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "rouge_l_std": float(np.std(rouge_scores)) if rouge_scores else 0.0,
        "rouge_l_median": float(np.median(rouge_scores)) if rouge_scores else 0.0,
        "pass_at_1": pass1,
        "bigram_log_prob_mean": (
            float(np.mean(all_bigram_scores)) if all_bigram_scores else 0.0
        ),
        "bigram_log_prob_std": (
            float(np.std(all_bigram_scores)) if all_bigram_scores else 0.0
        ),
        "exact_match": float(np.mean(all_exact_matches)) if all_exact_matches else 0.0,
        "validity_rate": valid_count / max(len(flat_completions), 1),
        "valid_count": valid_count,
        "invalid_count": len(flat_completions) - valid_count,
        "error_distribution": dict(error_counts.most_common(20)),
        "reward_breakdown": reward_components,
        "detailed_metrics": detailed,
        "total_completions": len(flat_completions),
        # ── Comprehensive report (BLEU + bootstrap CI 95%) ──
        "evaluation_report": eval_report,
    }
    if passk:
        results["pass_at_k"] = passk

    return (
        results,
        flat_completions,
        validity,
        all_references,
        rouge_scores,
        generations,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="T2G checkpoint evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (omit for zero-shot base model evaluation)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max test samples (overrides config evaluation.max_samples)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Completions per prompt (1=greedy, >1=sampled for Pass@k). "
        "Overrides config evaluation.num_samples.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results JSON"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots via visualization.py",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Baseline Pass@1 for comparison plot",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Path to baseline eval JSON for full comparison plot",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also evaluate the base model (zero-shot) and generate "
        "baseline-vs-GRPO comparison plots + JSON. Implies --plot.",
    )
    parser.add_argument(
        "--best-of-n",
        action="store_true",
        help="Select the best completion per prompt (highest ROUGE-L among "
        "valid) instead of using the first. Helps small models generalize. "
        "Requires --num-samples > 1.",
    )
    parser.add_argument(
        "--eval-baseline-only",
        action="store_true",
        help="Evaluate only the base model (zero-shot, no checkpoint). "
        "Useful for generating the baseline JSON to compare against later.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config(args.config)

    # ── Resolve eval params from config (CLI args override) ──────────────
    eval_cfg = config.get("evaluation", {})
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = eval_cfg.get("max_samples", 200)
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = eval_cfg.get("num_samples", 1)
    # best_of_n: CLI flag overrides config; config default is False
    best_of_n = args.best_of_n or eval_cfg.get("best_of_n", False)

    # ── Log eval configuration ───────────────────────────────────────────
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint or 'zero-shot (base model)'}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Completions per prompt: {num_samples}")
    logger.info(f"Grammar enabled: {config.get('grammar', {}).get('enabled', True)}")
    logger.info(
        f"Use PDA: {config.get('grammar', {}).get('use_grammarllm_pda', False)}"
    )
    logger.info(f"Plot: {args.plot}")
    logger.info(f"Compare: {args.compare}")
    logger.info(f"Best-of-N: {args.best_of_n}")
    logger.info(f"Eval baseline only: {args.eval_baseline_only}")
    if args.baseline is not None:
        logger.info(f"Baseline Pass@1: {args.baseline}")
    if args.baseline_json is not None:
        logger.info(f"Baseline JSON: {args.baseline_json}")

    # --compare implies --plot
    if args.compare:
        args.plot = True

    # ── Set random seeds for reproducibility ─────────────────────────────
    seed = config["dataset"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Reproducibility: seed={seed} (random, numpy, torch, cuda)")

    # ── Resolve model_name, run_id, and directory paths ──────────────────
    from datetime import datetime

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).resolve()
        parts = checkpoint_path.parts
        if "checkpoints" in parts:
            idx = parts.index("checkpoints")
            if len(parts) > idx + 2:
                model_name = parts[idx + 1]
                run_id = parts[idx + 2]
            else:
                model_name = parts[idx + 1]
                run_id = "default_run"
        else:
            model_name = config.get("wandb", {}).get("run_name", "t2g-model")
            run_id = (
                checkpoint_path.parent.name
                if checkpoint_path.name in ["final", "checkpoint-*"]
                else checkpoint_path.name
            )

        model_tag = run_id
    else:
        raw_model_name = config["model"]["name"].split("/")[-1].lower()
        model_name = raw_model_name.replace(".", "")
        if "run_name" in config.get("wandb", {}):
            model_name = config["wandb"]["run_name"]
        run_id = f"zero_shot_{run_timestamp}"
        model_tag = "zero-shot"

    results_dir = Path("experiments/results") / model_name / run_id
    figures_dir = Path("experiments/figures") / model_name / run_id
    logs_dir = Path("experiments/logs") / model_name / run_id

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine eval mode ─────────────────────────────────────────────
    # Three modes:
    #   1. --eval-baseline-only: eval base model, save as baseline_results.json
    #   2. --compare: eval baseline (or load from --baseline-json) + eval GRPO,
    #      then generate comparison plots + JSON
    #   3. default: eval single checkpoint (or zero-shot)
    baseline_results: dict[str, Any] | None = None
    baseline_generations: list[dict[str, Any]] | None = None

    if args.eval_baseline_only:
        # Mode 1: baseline-only eval
        logger.info("=" * 60)
        logger.info("BASELINE EVALUATION (zero-shot base model)")
        logger.info("=" * 60)
        results, completions, validity, all_references, rouge_scores, generations = (
            evaluate_checkpoint(
                config,
                checkpoint_path=None,
                max_samples=max_samples,
                num_samples=num_samples,
                best_of_n=best_of_n,
            )
        )
        model_tag = "baseline"

    elif args.compare:
        # Mode 2: baseline + GRPO comparison
        # Step A: Load or evaluate baseline
        if args.baseline_json is not None and Path(args.baseline_json).exists():
            logger.info(f"Loading baseline results from {args.baseline_json}")
            baseline_results = json.loads(
                Path(args.baseline_json).read_text(encoding="utf-8")
            )
            # Try to load baseline generations too
            bl_gen_path = (
                Path(args.baseline_json).parent
                / f"generations_{Path(args.baseline_json).stem.removeprefix('eval_')}.json"
            )
            if bl_gen_path.exists():
                baseline_generations = json.loads(
                    bl_gen_path.read_text(encoding="utf-8")
                )
        else:
            logger.info("=" * 60)
            logger.info("BASELINE EVALUATION (zero-shot base model)")
            logger.info("=" * 60)
            # Baseline uses greedy decoding (num_samples=1) — generating
            # multiple samples from an untrained model is not informative
            # and wastes ~3h of compute time.
            bl_num_samples = 1
            logger.info(
                "  Baseline forced to num_samples=%d (greedy) for speed.",
                bl_num_samples,
            )
            (
                baseline_results,
                _bl_comps,
                _bl_val,
                _bl_refs,
                _bl_rouge,
                baseline_generations,
            ) = evaluate_checkpoint(
                config,
                checkpoint_path=None,
                max_samples=max_samples,
                num_samples=bl_num_samples,
                best_of_n=False,
            )
            # Save baseline results for future reuse
            bl_out_dir = results_dir
            bl_path = bl_out_dir / "eval_baseline.json"
            bl_path.write_text(
                json.dumps(baseline_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            bl_gen_path = bl_out_dir / "generations_baseline.json"
            bl_gen_path.write_text(
                json.dumps(baseline_generations, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Baseline results saved to {bl_path}")
            logger.info(f"Baseline generations saved to {bl_gen_path}")

        # Step B: Evaluate GRPO checkpoint
        logger.info("=" * 60)
        logger.info("GRPO EVALUATION")
        logger.info("=" * 60)
        results, completions, validity, all_references, rouge_scores, generations = (
            evaluate_checkpoint(
                config,
                args.checkpoint,
                max_samples=max_samples,
                num_samples=num_samples,
                best_of_n=best_of_n,
            )
        )

    else:
        # Mode 3: single eval (default behavior)
        results, completions, validity, all_references, rouge_scores, generations = (
            evaluate_checkpoint(
                config,
                args.checkpoint,
                max_samples=max_samples,
                num_samples=num_samples,
                best_of_n=best_of_n,
            )
        )

    # ── Log key metrics ─────────────────────────────────────────────────
    logger.info("Evaluation complete. Key metrics:")
    logger.info(
        f"  ROUGE-L mean: {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}"
    )
    logger.info(f"  ROUGE-L median: {results.get('rouge_l_median', 0.0):.4f}")
    logger.info(f"  Pass@1: {results['pass_at_1']:.4f}")
    if results.get("pass_at_k"):
        for k, v in results["pass_at_k"].items():
            logger.info(f"  Pass@{k}: {v:.4f}")
    logger.info(f"  Validity rate: {results['validity_rate']:.4f}")
    logger.info(
        f"  Valid: {results['valid_count']}, Invalid: {results['invalid_count']}"
    )

    # ── Log sample predictions ──────────────────────────────────────────
    logger.info("Sample predictions (first 5):")
    for i in range(min(5, len(completions))):
        comp = completions[i]
        is_valid, reason = validity[i]
        ref = all_references[i] if i < len(all_references) else "N/A"
        logger.info(f"  [{i+1}] valid={is_valid} ({reason})")
        logger.info(f"      gold: {ref[:120]}{'...' if len(ref) > 120 else ''}")
        logger.info(f"      pred: {comp[:120]}{'...' if len(comp) > 120 else ''}")

    # ── Log reward breakdown ────────────────────────────────────────────
    if results.get("reward_breakdown"):
        logger.info("Reward breakdown:")
        for name, val in results["reward_breakdown"].items():
            logger.info(f"  {name}: {val:.4f}")

    # ── Log error distribution ──────────────────────────────────────────
    if results.get("error_distribution"):
        logger.info("Error distribution:")
        for err, count in results["error_distribution"].items():
            logger.info(f"  {err}: {count}")

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  T2G Evaluation Results — {model_tag}")
    print("=" * 60)
    print(f"  Samples evaluated:       {results['num_samples_evaluated']}")
    print(f"  Completions per prompt:  {results['num_completions_per_prompt']}")
    print(
        f"  ROUGE-L (mean ± std):    {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}"
    )
    print(f"  Pass@1:                  {results['pass_at_1']:.4f}")
    if "pass_at_k" in results:
        for k, v in results["pass_at_k"].items():
            print(f"  {k}:{' ' * (25 - len(k))}{v:.4f}")
    print(f"  Exact match:             {results['exact_match']:.4f}")
    print(f"  Bigram log-prob (mean):  {results['bigram_log_prob_mean']:.4f}")
    print(
        f"  Validity rate:           {results['validity_rate']:.4f}  "
        f"({results['valid_count']} valid / {results['invalid_count']} invalid)"
    )

    # ── Comprehensive report (BLEU + bootstrap CI 95%) ──
    if "evaluation_report" in results:
        er = results["evaluation_report"]
        print("\n  ── BLEU & Confidence Intervals (95% CI) ──")
        if "rouge_l" in er:
            rl = er["rouge_l"]
            print(
                f"    ROUGE-L:  {rl['mean']:.4f}  "
                f"CI: [{rl['ci_95'][0]:.4f}, {rl['ci_95'][1]:.4f}]"
            )
        if "bleu" in er:
            bl = er["bleu"]
            print(
                f"    BLEU:     corpus={bl['corpus']:.4f}  "
                f"sentence={bl['sentence_mean']:.4f}  "
                f"CI: [{bl['ci_95'][0]:.4f}, {bl['ci_95'][1]:.4f}]"
            )
        if "pass_at_1" in er:
            pa = er["pass_at_1"]
            print(
                f"    Pass@1:   {pa['mean']:.4f}  "
                f"CI: [{pa['ci_95'][0]:.4f}, {pa['ci_95'][1]:.4f}]"
            )
        if "gloss_validity_rate" in er:
            print(f"    Gloss validity rate: {er['gloss_validity_rate']:.4f}")

    print("\n  ── Reward Breakdown ──")
    for k, v in results["reward_breakdown"].items():
        print(f"    {k}: {v:.4f}")
    print("\n  ── Error Distribution ──")
    for err, count in results["error_distribution"].items():
        print(f"    {err}: {count}")
    print("=" * 60)

    # ── Save JSON ────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        ckpt_name = Path(args.checkpoint).name if args.checkpoint else "zero_shot"
        out_path = results_dir / f"eval_{ckpt_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Results saved to {out_path}")

    # ── Save generations JSON (stile grpo-strict-generation) ────────────
    # File separato con ogni singola generazione (testo/gold/pred/valid/
    # rouge_l), utile per ispezione manuale e per costruire dataset di
    # analisi degli errori — analogo a completions_{name}.json in
    # grpo-strict-generation/src/evaluation/eval_grpo.py.
    gen_path = results_dir / f"generations_{out_path.stem.removeprefix('eval_')}.json"
    gen_path.write_text(
        json.dumps(generations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Generations saved to {gen_path}")
    print(f"  Generations saved to: {gen_path}")

    # ── Generate plots ───────────────────────────────────────────────────
    if args.plot:
        from src.utils.visualization import (
            plot_baseline_vs_grpo,
            plot_baseline_vs_grpo_comparison,
            plot_completion_examples,
            plot_completion_length_distribution,
            plot_error_breakdown,
            plot_pass_at_k_curve,
            plot_reward_breakdown,
            plot_reward_radar,
            plot_rouge_distribution,
            plot_validity_pie,
        )

        # figures_dir is pre-resolved at the start of main()
        valid_mask = [v for v, _ in validity]

        logger.info("Generating evaluation figures...")

        # 1. Completion length distribution
        plot_completion_length_distribution(
            completions,
            valid_mask=valid_mask,
            title=f"Gloss Length — {model_tag}",
            output_path=str(figures_dir / "completion_lengths.png"),
        )

        # 2. ROUGE-L score distribution
        plot_rouge_distribution(
            rouge_scores,
            model_name=model_tag,
            output_path=str(figures_dir / "rouge_distribution.png"),
        )

        # 3. Pass@k curve (if multi-sample)
        if results.get("pass_at_k"):
            plot_pass_at_k_curve(
                results["pass_at_k"],
                model_name=model_tag,
                output_path=str(figures_dir / "pass_at_k.png"),
            )

        # 4. Error breakdown pie chart
        plot_error_breakdown(
            results["error_distribution"],
            model_name=model_tag,
            output_path=str(figures_dir / "error_breakdown.png"),
        )

        # 5. Validity pie chart
        plot_validity_pie(
            valid_count=results["valid_count"],
            invalid_count=results["invalid_count"],
            model_name=model_tag,
            output_path=str(figures_dir / "validity_pie.png"),
        )

        # 6. Reward breakdown bar chart
        rewards_cfg = config.get("reward", {})
        structure_weight = rewards_cfg.get(
            "weight_gold_structure",
            rewards_cfg.get("weight_structure", 0.4),
        )
        weights = {
            "translation_quality_reward": rewards_cfg.get("weight_translation", 0.4),
            "structural_dense_reward": structure_weight,
            "gold_structure_reward": structure_weight,
            "viterbi_distance_reward": rewards_cfg.get("weight_viterbi", 0.0),
            "soft_viterbi_distance_reward": rewards_cfg.get("weight_soft_viterbi", 0.0),
            "verifier_scaled_reward": rewards_cfg.get("weight_verifier_scaled", 0.0),
            "gloss_order_reward": rewards_cfg.get("weight_gloss_order", 0.0),
            "gloss_format_reward": rewards_cfg.get("weight_format", 0.1),
            "gloss_repetition_reward": rewards_cfg.get("weight_repetition", 0.1),
        }
        plot_reward_breakdown(
            [{"label": model_tag, "scores": results["reward_breakdown"]}],
            reward_weights=weights,
            model_name=model_tag,
            output_path=str(figures_dir / "reward_breakdown.png"),
        )

        # 7. Reward radar chart
        plot_reward_radar(
            results["reward_breakdown"],
            reward_weights=weights,
            model_name=model_tag,
            output_path=str(figures_dir / "reward_radar.png"),
        )

        # 8. Completion examples (best & worst)
        plot_completion_examples(
            completions,
            all_references,
            rouge_scores,
            n_examples=10,
            model_name=model_tag,
            output_path=str(figures_dir / "completion_examples.png"),
        )

        # 9. Baseline vs GRPO (if baseline Pass@1 provided via CLI)
        if args.baseline is not None:
            plot_baseline_vs_grpo(
                baseline_pass1=args.baseline,
                grpo_pass1=results["pass_at_1"],
                model_name=model_tag,
                output_path=str(figures_dir / "baseline_vs_grpo.png"),
            )

        # 10. Baseline vs GRPO full comparison
        # Triggered by: --compare (baseline_results from in-run eval),
        # or --baseline-json (baseline_results from saved JSON).
        comparison_metrics: dict[str, Any] | None = None
        if baseline_results is not None:
            comparison_metrics = baseline_results
        elif args.baseline_json is not None and Path(args.baseline_json).exists():
            comparison_metrics = json.loads(
                Path(args.baseline_json).read_text(encoding="utf-8")
            )

        if comparison_metrics is not None:
            plot_baseline_vs_grpo_comparison(
                baseline_metrics=comparison_metrics,
                grpo_metrics=results,
                model_name=model_tag,
                output_path=str(figures_dir / "baseline_vs_grpo_comparison.png"),
            )

            # ── Print comparison summary ───────────────────────────────
            print("\n" + "=" * 60)
            print("  BASELINE vs GRPO COMPARISON")
            print("=" * 60)
            for metric_key, metric_label in [
                ("rouge_l_mean", "ROUGE-L mean"),
                ("pass_at_1", "Pass@1"),
                ("exact_match", "Exact Match"),
                ("validity_rate", "Validity Rate"),
            ]:
                bl_val = comparison_metrics.get(metric_key, 0.0)
                gr_val = results.get(metric_key, 0.0)
                delta = gr_val - bl_val
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                print(
                    f"  {metric_label:20s}  "
                    f"BL={bl_val:.4f}  GRPO={gr_val:.4f}  "
                    f"Δ={delta:+.4f} {arrow}"
                )
            print("=" * 60)

            # ── Save comparison JSON ────────────────────────────────────
            comparison_json = {
                "baseline": {
                    k: comparison_metrics.get(k, 0.0)
                    for k in [
                        "rouge_l_mean",
                        "pass_at_1",
                        "exact_match",
                        "validity_rate",
                        "bigram_log_prob_mean",
                    ]
                },
                "grpo": {
                    k: results.get(k, 0.0)
                    for k in [
                        "rouge_l_mean",
                        "pass_at_1",
                        "exact_match",
                        "validity_rate",
                        "bigram_log_prob_mean",
                    ]
                },
                "delta": {
                    k: results.get(k, 0.0) - comparison_metrics.get(k, 0.0)
                    for k in [
                        "rouge_l_mean",
                        "pass_at_1",
                        "exact_match",
                        "validity_rate",
                        "bigram_log_prob_mean",
                    ]
                },
            }
            comp_path = out_path.parent / "comparison.json"
            comp_path.write_text(
                json.dumps(comparison_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Comparison saved to {comp_path}")
            print(f"  Comparison saved to: {comp_path}")

        print(f"\n  Figures saved to: {figures_dir}")

    # ── wandb logging ───────────────────────────────────────────────────
    # Log all metrics to wandb (offline mode on cluster). Tags distinguish
    # baseline vs GRPO runs, mirroring grpo-strict-generation's approach.
    try:
        import os

        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DISABLE_WEAVE"] = "true"

        import wandb

        wandb_cfg = config.get("wandb", {})
        base_run_name = (
            wandb_cfg.get("run_name") or config["model"]["name"].split("/")[-1]
        )

        # Determine wandb tags based on eval mode
        wandb_tags = wandb_cfg.get("tags", ["t2g", "eval"])
        if "eval" not in wandb_tags:
            wandb_tags = list(wandb_tags) + ["eval"]
        if args.eval_baseline_only:
            wandb_tags = list(wandb_tags) + ["baseline"]
            wandb_run_name = f"eval-baseline-{base_run_name}"
        elif args.compare:
            wandb_tags = list(wandb_tags) + ["compare", "grpo"]
            wandb_run_name = f"eval-compare-{base_run_name}"
        else:
            wandb_tags = list(wandb_tags) + ["grpo"]
            wandb_run_name = f"eval-{base_run_name}-{model_tag}"

        wandb_dir = logs_dir

        wandb.init(
            project=wandb_cfg.get("project", "neuro-symbolic-t2g"),
            name=wandb_run_name,
            config=config,
            tags=wandb_tags,
            dir=str(wandb_dir),
            mode="offline",
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_bytes=1_000_000,
                console_chunk_max_seconds=60,
            ),
        )

        # Log all scalar metrics
        wandb.log(results)

        # Log comparison metrics if available
        if baseline_results is not None:
            wandb.log(
                {
                    "baseline/rouge_l_mean": baseline_results.get("rouge_l_mean", 0.0),
                    "baseline/pass_at_1": baseline_results.get("pass_at_1", 0.0),
                    "baseline/exact_match": baseline_results.get("exact_match", 0.0),
                    "baseline/validity_rate": baseline_results.get(
                        "validity_rate", 0.0
                    ),
                    "delta/rouge_l_mean": results.get("rouge_l_mean", 0.0)
                    - baseline_results.get("rouge_l_mean", 0.0),
                    "delta/pass_at_1": results.get("pass_at_1", 0.0)
                    - baseline_results.get("pass_at_1", 0.0),
                    "delta/exact_match": results.get("exact_match", 0.0)
                    - baseline_results.get("exact_match", 0.0),
                    "delta/validity_rate": results.get("validity_rate", 0.0)
                    - baseline_results.get("validity_rate", 0.0),
                }
            )

        # Log figures as images
        if args.plot:
            # figures_dir is pre-resolved
            for fig_file in figures_dir.glob("*.png"):
                wandb.log({f"figures/{fig_file.stem}": wandb.Image(str(fig_file))})

        wandb.finish()
        logger.info(f"wandb run logged: {wandb_run_name} (tags={wandb_tags})")
    except Exception as e:
        logger.warning(f"wandb logging failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
