"""GRPO training for constrained text-to-gloss generation."""

from __future__ import annotations

import argparse
import gc

# Normalize tuple-valued optional-dependency flags before importing TRL.
import importlib
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings(
    "ignore",
    message=".*AttentionMaskConverter.*",
    category=FutureWarning,
)

_trl_iu = importlib.import_module("trl.import_utils")  # noqa: E402
if isinstance(_trl_iu._mergekit_available, tuple):
    _trl_iu._mergekit_available = False
if isinstance(_trl_iu._llm_blender_available, tuple):
    _trl_iu._llm_blender_available = False

import wandb
from dotenv import load_dotenv
from transformers.trainer_callback import (
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments
from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]

from datasets import Dataset
from src.datasets.aslg_dataset import (
    build_t2g_dataset,
    download_aslg_dataset,
    extract_gloss_vocabulary,
    save_vocabulary,
)
from src.grammar.gloss_grammar import GlossVocabularyMask, create_grammarllm_pipeline
from src.grammar.grammar_logits_processor import (
    GlossVocabularyLogitsProcessor,
    GrammarPDALogitsProcessor,
)
from src.models.model_loader import load_model_and_tokenizer
from src.retrieval import ExampleRetriever
from src.rewards.t2g_rewards import (
    build_t2g_reward_functions,
    initialize_rewards,
)
from src.training.retrieval_setup import (
    build_train_retriever,
    retrieve_few_shot_batch,
)
from src.utils.cache_meta import cache_is_current as _cache_is_current
from src.utils.cache_meta import write_cache_meta as _write_cache_meta
from src.utils.config import load_config
from src.utils.live_status import live_status_set
from src.utils.paths import (
    RunPath,
    training_run_paths,
    wandb_name,
    wandb_tags,
)
from src.utils.prompting import build_t2g_prompt

# ───────────────────────────────────────────────────────────────────────────


load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _build_grpo_config(
    training_cfg: dict[str, Any],
    grpo_cfg: dict[str, Any],
    full_config: dict[str, Any] | None = None,
    reward_weights: list[float] | None = None,
) -> GRPOConfig:
    """Build a ``GRPOConfig`` from config sections."""
    loss_type = grpo_cfg.get("loss_type", "dr_grpo")
    scale_rewards = grpo_cfg.get("scale_rewards", "none")
    importance_level = grpo_cfg.get("importance_sampling_level", "token")
    mask_truncated = grpo_cfg.get("mask_truncated_completions", True)
    if loss_type not in {"grpo", "bnpo", "dr_grpo"}:
        raise ValueError(f"Invalid grpo.loss_type: {loss_type!r}")
    if scale_rewards not in {"group", "batch", "none", True, False}:
        raise ValueError(f"Invalid grpo.scale_rewards: {scale_rewards!r}")
    if importance_level not in {"token", "sequence"}:
        raise ValueError(
            f"Invalid grpo.importance_sampling_level: {importance_level!r}"
        )
    if not isinstance(mask_truncated, bool):
        raise ValueError("grpo.mask_truncated_completions must be boolean")
    epsilon = float(grpo_cfg.get("epsilon", 0.2))
    if epsilon < 0:
        raise ValueError("grpo.epsilon must be non-negative")
    epsilon_high = grpo_cfg.get("epsilon_high")
    if epsilon_high is not None and float(epsilon_high) < epsilon:
        raise ValueError("grpo.epsilon_high must be at least grpo.epsilon")
    beta = float(grpo_cfg.get("beta", 0.04))
    if beta < 0:
        raise ValueError("grpo.beta must be non-negative")
    output_dir = training_cfg["output_dir"]
    log_dir = training_cfg["log_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    wandb_cfg = (full_config or {}).get("wandb", {})
    from datetime import datetime

    base_name = wandb_cfg.get("run_name", "grpo-t2g")
    run_timestamp = training_cfg.get("run_timestamp") or datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_name = f"{base_name}-{run_timestamp}"

    grpo_options: dict[str, Any] = {
        "loss_type": loss_type,
        "scale_rewards": scale_rewards,
        "mask_truncated_completions": mask_truncated,
        "importance_sampling_level": importance_level,
        "epsilon": epsilon,
    }
    if epsilon_high is not None:
        grpo_options["epsilon_high"] = float(epsilon_high)

    return GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        seed=training_cfg.get(
            "seed", (full_config or {}).get("dataset", {}).get("seed", 42)
        ),
        max_steps=training_cfg.get("max_steps", 1500),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 50),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.1),
        bf16=training_cfg.get("bf16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        logging_steps=training_cfg.get("logging_steps", 5),
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 256),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 256),
        beta=beta,
        temperature=grpo_cfg.get("temperature", 0.7),
        reward_weights=reward_weights,
        # HighPrecisionLogCallback owns W&B scalar logging, so Trainer's
        # automatic W&B integration stays disabled.
        report_to="none",
        **grpo_options,
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _prepare_t2g_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    vocab: list[str],
    dataset: Any = None,
    *,
    retriever: ExampleRetriever | None = None,
    retrieval_cfg: dict[str, Any] | None = None,
) -> Dataset:
    """Build GRPO prompts while preserving columns consumed by rewards."""
    del vocab
    ds_cfg = config["dataset"]
    if dataset is None:
        dataset = download_aslg_dataset(
            cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
        )

    t2g_ds = build_t2g_dataset(
        dataset,
        split=ds_cfg.get("split", "train"),
        max_samples=ds_cfg.get("max_samples"),
    )

    top_k = int((retrieval_cfg or {}).get("top_k", 3))
    max_self_similarity = float((retrieval_cfg or {}).get("max_self_similarity", 0.98))
    texts = [t2g_ds[i]["prompt"] for i in range(len(t2g_ds))]
    examples_batch = (
        retrieve_few_shot_batch(retriever, texts, top_k, max_self_similarity)
        if retriever is not None
        else None
    )

    # Format prompts with the centralized T2G prompt builder.
    # This guarantees train/eval/test use identical formatting.
    formatted: list[dict[str, str]] = []
    for i in range(len(t2g_ds)):
        sample = t2g_ds[i]
        text = sample["prompt"]

        prompt = build_t2g_prompt(
            text,
            tokenizer,
            examples=examples_batch[i] if examples_batch is not None else None,
        )

        formatted.append(
            {
                "prompt": prompt,
                "text": sample.get("text", text),
                "completion": sample["completion"],
                "gold_gloss": sample.get("gold_gloss", sample["completion"]),
                "difficulty": sample.get("difficulty", "medium"),
                "sample_id": sample.get("sample_id", ""),
            }
        )

    result = Dataset.from_list(formatted)
    logger.info(f"[dataset] T2G training set: {len(result)} prompts")
    return result


# ---------------------------------------------------------------------------
# Vocabulary-constrained generation config for GRPO
# ---------------------------------------------------------------------------


def _build_generation_kwargs(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build GenerationConfig-safe rollout kwargs."""
    grpo_cfg = config.get("generation", config.get("grpo", {}))
    return {
        "max_new_tokens": grpo_cfg.get("max_completion_length", 128),
    }


# ---------------------------------------------------------------------------
# Curriculum learning (difficulty-scheduled sampling)
# ---------------------------------------------------------------------------


class CurriculumSchedule:
    """Three-stage progressive difficulty schedule."""

    _STAGES: list[dict[str, float]] = [
        {"simple": 0.10, "medium": 0.65, "hard": 0.25},
        {"simple": 0.05, "medium": 0.40, "hard": 0.55},
        {"simple": 0.03, "medium": 0.30, "hard": 0.67},
    ]

    def __init__(self, max_steps: int) -> None:
        self._max_steps = max(max_steps, 1)
        self._stage_size = max(self._max_steps // 3, 1)

    def get_stage(self, step: int) -> int:
        """Return current curriculum stage (0, 1, or 2)."""
        return min(step // self._stage_size, len(self._STAGES) - 1)

    def get_distribution(self, step: int) -> dict[str, float]:
        """Return difficulty distribution for the current step."""
        return self._STAGES[self.get_stage(step)]

    @property
    def stage_size(self) -> int:
        return self._stage_size


class CurriculumFilteredDataset:
    """Dataset view resampled to the current difficulty distribution."""

    def __init__(
        self,
        dataset: Dataset,
        schedule: CurriculumSchedule,
        stage: int,
    ) -> None:
        self._full_dataset = dataset
        self._schedule = schedule
        self._stage = stage
        self._indices: list[int] = []
        self.column_names = dataset.column_names
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild index list to match the current stage's difficulty distribution."""
        distribution = self._schedule._STAGES[self._stage]

        # Group indices by difficulty label
        by_diff: dict[str, list[int]] = {"simple": [], "medium": [], "hard": []}
        for i, row in enumerate(self._full_dataset):
            diff = row.get("difficulty", "medium")
            if diff not in by_diff:
                diff = "medium"
            by_diff[diff].append(i)

        total = len(self._full_dataset)
        indices: list[int] = []
        for diff, target_pct in distribution.items():
            count = min(int(total * target_pct), len(by_diff[diff]))
            if count > 0 and by_diff[diff]:
                indices.extend(random.sample(by_diff[diff], count))

        if not indices:
            indices = list(range(total))

        # Shuffle so items are mixed, not grouped by difficulty
        random.shuffle(indices)

        # Pad/truncate to maintain constant length
        # (prevents DataLoader sampler from generating out-of-bounds indices)
        target_len = len(self._full_dataset)
        if len(indices) < target_len:
            indices.extend(random.choices(indices, k=target_len - len(indices)))
        elif len(indices) > target_len:
            indices = indices[:target_len]

        self._indices = indices

    def update_stage(self, stage: int) -> None:
        """Transition to a new curriculum stage (rebuilds index list)."""
        if stage != self._stage:
            self._stage = stage
            self._rebuild()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._full_dataset[self._indices[idx]]

    def __getattr__(self, name: str) -> Any:
        # Forward attribute access to the underlying Dataset when
        # the attribute isn't defined on the wrapper itself.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._full_dataset, name)


class CurriculumCallback(TrainerCallback):
    """Apply curriculum transitions at stage boundaries."""

    def __init__(
        self,
        schedule: CurriculumSchedule,
        curriculum_dataset: CurriculumFilteredDataset,
    ) -> None:
        self._schedule = schedule
        self._dataset = curriculum_dataset
        self._last_stage = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        current_stage = self._schedule.get_stage(state.global_step)

        if current_stage != self._last_stage:
            self._last_stage = current_stage

            # Rebuild the dataset index list for the new difficulty distribution
            self._dataset.update_stage(current_stage)

            distribution = self._schedule._STAGES[current_stage]

            # Stage transition banner for stdout (parsed by chain_monitor)
            print(f"\n{'=' * 60}")
            print(f"  CURRICULUM STAGE {current_stage + 1}/3")
            print(
                f"  Distribution: simple={distribution['simple']:.0%} "
                f"medium={distribution['medium']:.0%} "
                f"hard={distribution['hard']:.0%}"
            )
            print(f"{'=' * 60}\n")

            # Keep custom diagnostics aligned with the trainer's explicit step.
            try:
                import wandb

                if wandb.run:
                    wandb.log(
                        {
                            "curriculum/stage": float(current_stage + 1),
                            "curriculum/difficulty_distribution": distribution,
                        },
                        step=state.global_step,
                        commit=False,
                    )
            except Exception:
                logger.debug("Failed to log curriculum metrics to wandb", exc_info=True)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="GRPO training for Text-to-Gloss (T2G) with constrained decoding"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Only prepare data (download dataset, compute transitions, save vocab)",
    )
    parser.add_argument(
        "--force-sft",
        action="store_true",
        help="Ignore SFT adapter reuse and always retrain SFT from scratch "
        "(bypasses fingerprint matching for saved SFT adapters)",
    )
    return parser


def resolve_reusable_sft_adapter(
    sft_config: dict[str, Any],
    current_run_dir: str | Path,
    model_checkpoint_root: str | Path,
) -> tuple[Path, str] | None:
    """Resolve a canonical reusable SFT adapter for a GRPO run.

    Discovery is deliberately delegated to the single canonical cross-method
    search, which covers both standalone SFT finals and SFT-GRPO subphases.
    """
    from src.training.sft_train import (
        compute_sft_fingerprint,
        find_reusable_sft_adapter_cross_method,
    )

    current_run = Path(current_run_dir)
    found = find_reusable_sft_adapter_cross_method(
        model_checkpoint_root,
        current_run,
        compute_sft_fingerprint(sft_config),
    )
    return found


def main() -> None:
    """Main entry point for T2G GRPO training."""
    args = build_arg_parser().parse_args()

    config = load_config(args.config)
    # ── Resolve timestamped output/log directories and resume logic ──────
    output_dir, log_dir, run_id, cell = training_run_paths(config, resume=args.resume)
    run_timestamp = run_id.removeprefix("run_")
    print(f"[grpo] Resolved training run. Output dir: {output_dir}")

    config["training"]["output_dir"] = str(output_dir)
    config["training"]["log_dir"] = str(log_dir)
    config["training"]["run_timestamp"] = run_timestamp
    if config.get("experiment"):
        identity = RunPath(cell, run_id)
        config.setdefault("wandb", {})["run_name"] = wandb_name(identity)
        config["wandb"]["tags"] = list(wandb_tags(identity))

    # Safe config access: support both 'grpo' (GRPO) and 'generation' (SFT) keys
    grpo_cfg = config.get("generation", config.get("grpo", {}))

    # ── Setup logging ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    # Quiet down external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Remove library handlers that duplicate records propagated to root.
    from src.utils.log_dedup import dedupe_library_loggers

    dedupe_library_loggers()

    # ── Set random seeds for reproducibility ─────────────────────────────
    seed = config["dataset"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[grpo] Reproducibility: seed={seed} (random, numpy, torch, cuda)")

    # ── Step 1: Data preparation ─────────────────────────────────────────
    ds_cfg = config["dataset"]
    vocab_path = ds_cfg.get("vocab_path", "data/gloss_vocab.txt")

    print(f"\n{'=' * 60}")
    print("STEP 1: Data Preparation")

    # Download dataset
    dataset = download_aslg_dataset(
        cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
    )

    # Caches are keyed by (seed, train_size): if either changes (e.g. a new
    # seed, or dataset dedup changing the split composition), the vocab and
    # bigram artifacts must be regenerated. Caches without a sidecar
    # are never trusted (see _cache_is_current).
    train_size = len(dataset["train"])

    # Extract vocabulary (or load from cache if still current)
    if _cache_is_current(vocab_path, seed, train_size):
        from src.datasets.aslg_dataset import load_vocabulary

        vocab = load_vocabulary(vocab_path)
    else:
        vocab = extract_gloss_vocabulary(dataset, split="train")
        save_vocabulary(vocab, vocab_path)
        _write_cache_meta(vocab_path, seed, train_size)

    print(f"  Data prepared: |V|={len(vocab)}")

    if args.prepare_data:
        print("Data preparation complete. Exiting.")
        return

    # ── Step 1.5: Optional SFT Pre-training ─────────────────────────────
    sft_adapter_path: str | None = None
    sft_pretrain_cfg = config.get("sft_pretrain", {})
    if sft_pretrain_cfg.get("enabled", False):
        print(f"\n{'=' * 60}")
        print("STEP 1.5: SFT Pre-training")
        # Live status: the SFT phase begins (adapter reuse skips this block).
        live_status_set(phase="sft", note="SFT pre-training")

        # Build a synthetic config for run_sft using sft_pretrain section
        sft_config = {
            **config,
            "training": {
                **config["training"],
                **sft_pretrain_cfg.get("training", {}),
                "output_dir": str(
                    sft_pretrain_cfg.get(
                        "output_dir",
                        Path(config["training"]["output_dir"]) / "sft_pretrain",
                    )
                ),
                "log_dir": str(
                    sft_pretrain_cfg.get(
                        "log_dir",
                        Path(config["training"]["log_dir"]) / "sft_pretrain",
                    )
                ),
                "trainer": "sft",
            },
        }

        # ── SFT adapter reuse ──────────────────────────────────────────
        # If a previous run already produced an SFT adapter trained with the
        # SAME SFT configuration (same fingerprint of model/lora/dataset/
        # system prompt/SFT hyperparams — see compute_sft_fingerprint),
        # retraining SFT is wasteful: reuse it.  Reuse only skips Step 1.5;
        # GRPO still trains (or resumes) from its OWN checkpoints, so
        # `--resume` behaviour is unaffected.
        #   - sft_pretrain.reuse_adapter: false → always retrain
        #   - sft_pretrain.adapter_path: <dir>  → explicit adapter (no search)
        #   - --force-sft                        → always retrain (CLI override)
        from src.training.sft_train import (
            is_complete_adapter_dir,
            run_sft,
        )

        explicit_adapter = sft_pretrain_cfg.get("adapter_path")
        reuse_adapter = sft_pretrain_cfg.get("reuse_adapter", True)
        reused_adapter: str | None = None

        if explicit_adapter is not None:
            if is_complete_adapter_dir(explicit_adapter):
                reused_adapter = str(explicit_adapter)
                print(f"  Reusing SFT adapter (explicit path): {reused_adapter}")
            else:
                print(
                    "  ⚠️  Explicit adapter_path missing or incomplete: "
                    f"{explicit_adapter} — ignoring it"
                )
        elif reuse_adapter and not args.force_sft:
            resolved = resolve_reusable_sft_adapter(
                sft_config,
                config["training"]["output_dir"],
                Path("experiments/checkpoints") / cell.model_tag,
            )
            if resolved is not None:
                source, source_cell = resolved
                reused_adapter = str(source)
                print(
                    f"  Reusing SFT adapter from cell '{source_cell}' "
                    "(identical SFT config — fingerprint match), "
                    f"at {source} — skipping SFT training"
                )

        if reused_adapter is not None:
            sft_adapter_path = reused_adapter
        else:
            if args.force_sft:
                print(
                    "  --force-sft: retraining SFT from scratch "
                    "(adapter reuse disabled)"
                )
            sft_adapter_path = run_sft(sft_config, resume=args.resume)
            print(f"  SFT adapter saved to: {sft_adapter_path}")

        # Aggressive cleanup between SFT and GRPO
        gc.collect()
        torch.cuda.empty_cache()

    # ── Step 2: Model loading ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 2: Model Loading")

    model, tokenizer = load_model_and_tokenizer(config, adapter_path=sft_adapter_path)

    # ── Step 3: Constrained decoding setup ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: Constrained Decoding Setup")

    # Grammar toggle: set ``grammar.enabled: false`` to disable constrained
    # decoding (for ablation study — GRPO without grammar).
    grammar_enabled = config.get("grammar", {}).get("enabled", True)
    if not grammar_enabled:
        print(
            "  ⚠️  grammar.enabled=false — GRPO rollouts will use UNCONSTRAINED "
            "generation (no vocabulary mask).  This is intended for ablation "
            "studies only."
        )
        logits_processor_for_gen = None
    else:
        # Determine which constrained decoding strategy to use.
        # Set ``use_grammarllm_pda: true`` in the config to enable the full
        # grammarllm PDA pipeline (LL(1) parsing).  Default is lightweight
        # vocabulary mask (faster, sufficient for most gloss constraints).
        use_pda = config.get("grammar", {}).get("use_grammarllm_pda", False)

        if use_pda:
            print("  Using FULL grammarllm PDA pipeline for constrained decoding")
            # grammarllm v0.5.0: create_grammarllm_pipeline returns
            # (pdas: list[PushdownAutomaton], streamer, pda) — the first
            # element is now a list of base PDA templates, not a logit_processor.
            # token_lookahead=True (default) enables native BPE token emission
            # across grammar boundaries — a key v0.5.0 improvement.
            grammar_cfg = config.get("grammar", {})
            pdas, streamer, pda = create_grammarllm_pipeline(
                vocab,
                tokenizer,
                temperature=grpo_cfg.get("temperature", 0.7),
                num_return_sequences=1,  # GRPO: 1 sequence per prompt during rollouts
                token_lookahead=grammar_cfg.get("token_lookahead", True),
            )
            # Pass the full pdas list (not just pda=pdas[0]) and
            # track_score_history from config so the StatelessLogitsProcessor
            # can optionally accumulate logit history for debugging.
            grammar_lp = GrammarPDALogitsProcessor(
                tokenizer,
                pdas,
                temperature=grpo_cfg.get("temperature", 0.7),
                track_score_history=grammar_cfg.get("track_score_history", False),
            )
            logits_processor_for_gen = grammar_lp
            print("  GrammarLLM PDA pipeline ready")
        else:
            print("  Using lightweight GlossVocabularyMask for constrained decoding")
            gloss_mask = GlossVocabularyMask(vocab, tokenizer)
            logits_processor_for_gen = GlossVocabularyLogitsProcessor(
                gloss_mask, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("  Vocabulary mask ready")

    # ── Step 4: Dataset preparation ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: Dataset Preparation")

    # ── Optional few-shot retrieval (train-split demonstrations) ─────────
    # Build (or load from cache) the ExampleRetriever over the deduplicated
    # TRAIN split.  When enabled, every GRPO prompt is augmented with top_k
    # similar (text→gloss) examples; the query itself and near-duplicates
    # are always excluded (anti-leakage).  SFT pre-training stays zero-shot.
    retrieval_cfg = config.get("retrieval", {})
    retriever = build_train_retriever(
        dataset,
        retrieval_cfg,
        seed=ds_cfg.get("seed", 42),
    )
    if retriever is not None:
        print(
            f"  Few-shot retrieval ENABLED: backend={retriever.backend}, "
            f"top_k={retrieval_cfg.get('top_k', 3)}, "
            f"max_self_similarity="
            f"{retrieval_cfg.get('max_self_similarity', 0.98)}"
        )
        # Few-shot examples (~40-60 tokens each) inflate prompt length; the
        # default 256 can silently truncate them.  Warn — never force.
        max_prompt_length = grpo_cfg.get("max_prompt_length", 256)
        if max_prompt_length < 768:
            logger.warning(
                "[retrieval] Few-shot retrieval is enabled but "
                "grpo.max_prompt_length=%d < 768: few-shot prompts may be "
                "truncated during rollout. Consider raising it in the config "
                "(e.g. grpo.max_prompt_length: 768). Not forced automatically.",
                max_prompt_length,
            )
    else:
        print("  Few-shot retrieval disabled — zero-shot prompts")

    t2g_dataset = _prepare_t2g_dataset(
        config,
        tokenizer,
        vocab,
        dataset=dataset,
        retriever=retriever,
        retrieval_cfg=retrieval_cfg,
    )

    # NOTE: no gold-gloss registry anymore.  The ``gold_gloss`` column is
    # preserved on the dataset and TRL 0.24 forwards it to the reward
    # functions as a kwarg (see t2g_rewards._make_gloss_reward_fn), which
    # eliminates SHA256-of-text-only collisions when duplicate English
    # sentences map to different gold glosses.

    # ── Step 5: Reward functions ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 5: Reward Functions")

    initialize_rewards(vocab)
    reward_fns, reward_weights = build_t2g_reward_functions(config.get("reward"))

    # ── Curriculum Learning setup ────────────────────────────────────────
    curriculum_cfg = config.get("curriculum", {})
    curriculum_callback: CurriculumCallback | None = None

    if curriculum_cfg.get("enabled", False):
        print(f"\n{'─' * 60}")
        print("CURRICULUM LEARNING: ENABLED")
        print("  3-stage progressive difficulty curriculum (project-original)")

        max_steps = config["training"].get("max_steps", 1500)
        curriculum_schedule = CurriculumSchedule(max_steps)

        # Wrap the training dataset with curriculum filtering (Stage 1)
        t2g_dataset = CurriculumFilteredDataset(
            t2g_dataset, curriculum_schedule, stage=0
        )

        dist = curriculum_schedule.get_distribution(0)
        print(
            f"  Stage 1/3 — Distribution: simple={dist['simple']:.0%} "
            f"medium={dist['medium']:.0%} hard={dist['hard']:.0%}"
        )
        print(
            f"  Stage size: {curriculum_schedule.stage_size} steps × 3 "
            f"= {curriculum_schedule.stage_size * 3}"
        )
        print(f"  Effective samples: {len(t2g_dataset)}")
        print(f"{'─' * 60}")

        curriculum_callback = CurriculumCallback(curriculum_schedule, t2g_dataset)

    # ── Step 6: GRPO configuration ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 6: GRPO Configuration")

    grpo_config = _build_grpo_config(
        config["training"],
        grpo_cfg,
        config,
        reward_weights=reward_weights,
    )

    # The instantiated TRL config is the sole source of truth for group size.
    from src.training.callbacks import (
        CompletionSampleCallback,
        CompletionSampleLogger,
        HighPrecisionLogCallback,
        TqdmOnlyProgressCallback,
    )

    sample_logger = CompletionSampleLogger(
        reward_fns,
        reward_weights,
        n_samples=3,
        group_size=int(grpo_config.num_generations),
    )
    assert sample_logger._group_size == grpo_config.num_generations
    sample_logger.set_difficulty_map(t2g_dataset)
    wrapped_reward_fns = sample_logger.wrapped_reward_fns
    sample_callback = CompletionSampleCallback(
        sample_logger,
        every_n_steps=5,
        logits_processor=logits_processor_for_gen,
    )

    print(
        f"[grpo] max_steps={grpo_config.max_steps}, "
        f"batch={grpo_config.per_device_train_batch_size}, "
        f"grad_accum={grpo_config.gradient_accumulation_steps}, "
        f"lr={grpo_config.learning_rate}, "
        f"num_gen={grpo_config.num_generations}, "
        f"beta={grpo_config.beta}, "
        f"max_completion={grpo_config.max_completion_length}"
    )

    # ── Workaround: unsloth-zoo autocast dtype defaults to float16 ───────
    # `unsloth_zoo.rl_replacements.grpo_accumulated_loss` (materialized as
    # unsloth_compiled_cache/UnslothGRPOTrainer.py on the cluster) lazily
    # initializes `trainer._autocast_dtype` on the FIRST training step via:
    #
    #   trainer._autocast_dtype = (
    #       torch.float16
    #       if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16'
    #       else torch.bfloat16
    #   )
    #
    # This reads the RAW `ACCELERATE_MIXED_PRECISION` env var directly,
    # bypassing HF Accelerate's own `AcceleratorState().mixed_precision`
    # bookkeeping entirely. `GRPOConfig(bf16=True)` sets
    # `TrainingArguments.mixed_precision = "bf16"` as a *Python attribute*
    # and forwards it straight into `Accelerator(mixed_precision="bf16")`
    # — this never touches `os.environ`. The env var is only ever set by
    # the `accelerate launch` CLI or DeepSpeed, neither of which this
    # project uses (script is run directly via `python -m src.training`).
    # Result: the env var is unset → unsloth-zoo defaults to 'fp16' →
    # `trainer._autocast_dtype = torch.float16`, wrapping GRPO's forward
    # pass in a FLOAT16 autocast context that conflicts with the model's
    # actual bfloat16 weights/LoRA adapters, causing:
    #   RuntimeError: self and mat2 must have the same dtype, but got
    #   Half and Float  (in unsloth/kernels/utils.py:matmul_lora)
    #
    # Fix: explicitly set the env var to match `grpo_config.bf16` BEFORE
    # `GRPOTrainer` is constructed / trained, so unsloth-zoo's lazy check
    # picks up the correct dtype on its first (and only) evaluation.
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16" if grpo_config.bf16 else "fp16"

    # ── Resume logic ─────────────────────────────────────────────────────
    resume_from: str | None = None
    if args.resume:
        ckpts = sorted(Path(grpo_config.output_dir).glob("checkpoint-*"))
        if ckpts:
            resume_from = str(ckpts[-1])
            print(f"[grpo] Resuming from {resume_from}")

    # ── Wandb setup ──────────────────────────────────────────────────────
    # Keep W&B offline on network-isolated compute nodes.
    wandb_cfg = config.get("wandb", {})
    log_dir = config["training"]["log_dir"]
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    # Disable weave (wandb 0.25.0 tenta il login anche offline).
    os.environ["WANDB_DISABLE_WEAVE"] = "true"
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "neuro-symbolic-t2g")
    os.environ["WANDB_DIR"] = log_dir
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get("tags", ["grpo", "t2g", "constrained-decoding"])
    )

    if not wandb.run:
        wandb.init(
            project=wandb_cfg.get("project", "neuro-symbolic-t2g"),
            name=grpo_config.run_name,
            config=config,
            tags=wandb_cfg.get("tags", ["grpo", "t2g"]),
            dir=log_dir,
            mode="offline",
            # ── Fix: output.log missing on Files tab ──────────────────
            # Without console_multipart, W&B buffers the ENTIRE stdout/
            # stderr in memory and only writes/uploads output.log when
            # wandb.finish() completes successfully.  On a SLURM cluster,
            # jobs are frequently killed by OOM/timeout/SIGKILL before
            # reaching finish() — losing the whole log.  With
            # console_multipart=True, W&B writes timestamped chunks under
            # wandb/run-*/files/logs/ incrementally, so partial logs
            # survive a crash. See:
            # https://docs.wandb.ai/models/app/console-logs (Multipart
            # console logging).
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_bytes=1_000_000,
                console_chunk_max_seconds=60,
            ),
        )

    # ── Tee stdout → output.log (sync_cluster download) ─────────────────
    # console_multipart salva i log in chunk sotto wandb/run-*/files/logs/
    # ma sync_cluster.ps1 si aspetta un singolo output.log.  Teeiamo stdout
    # così abbiamo entrambi: crash safety (multipart) + comodità (file singolo).
    _output_log_path = os.path.join(log_dir, "output.log")
    _sys_stdout = sys.stdout
    _output_log_fh = open(_output_log_path, "a", buffering=1)

    class _Tee:
        def write(self, data):
            _sys_stdout.write(data)
            _output_log_fh.write(data)

        def flush(self):
            _sys_stdout.flush()
            _output_log_fh.flush()

    sys.stdout = _Tee()

    # ── Step 7: Training ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 7: GRPO Training")
    # Live status: the GRPO phase begins (SFT phase, if any, is over).
    live_status_set(
        phase="grpo",
        total_steps=int(config["training"].get("max_steps", 1500)),
        note="GRPO training",
    )

    # ── Workaround: transformers 5.3.0 + peft non espongono  ──────────
    # model.warnings_issued, ma trl 0.24.0 lo usa in GRPOTrainer.__init__.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # ── Generation kwargs for vocabulary-constrained rollout generation ──
    # In trl 0.24.0, generation_kwargs goes into GRPOConfig (args), NOT
    # directly into GRPOTrainer.__init__().
    # NOTE: logits_processor CANNOT be in generation_kwargs because trl
    # 0.24.0 does GenerationConfig(**generation_kwargs) and transformers
    # 5.3.0 rejects logits_processor in GenerationConfig.
    # Workaround: monkey-patch model.generate() to inject the processor.
    gen_kwargs = _build_generation_kwargs(config)
    grpo_config.generation_kwargs = gen_kwargs

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=t2g_dataset,
        reward_funcs=wrapped_reward_fns,
        processing_class=tokenizer,
        callbacks=(
            [sample_callback]
            if curriculum_callback is None
            else [sample_callback, curriculum_callback]
        ),
    )

    # ── Defensive: force unsloth-zoo's internal autocast dtype directly ──
    # Belt-and-suspenders alongside the `ACCELERATE_MIXED_PRECISION` env
    # var fix above: pre-set `trainer._autocast_dtype` on the trainer
    # instance itself so `grpo_accumulated_loss`'s
    # `if not hasattr(trainer, '_autocast_dtype')` lazy-init check is a
    # no-op regardless of env var propagation timing/caching quirks.
    trainer._autocast_dtype = torch.bfloat16 if grpo_config.bf16 else torch.float16

    # ── Monkey-patch model.generate() AFTER trainer init ────────────────
    # IMPORTANT: The patch must be applied AFTER GRPOTrainer.__init__()
    # because the trainer may wrap/store the model differently than the
    # object we passed in.  We patch `trainer.model` directly to ensure
    # TRL's internal rollout generation calls our patched method.
    #
    # Two things this patch does:
    #   1. AUTOCAST: model.generate() during GRPO rollouts runs OUTSIDE the
    #      trainer's autocast context.  With 4-bit quantization + LoRA,
    #      prepare_model_for_kbit_training() upcasts LoRA adapters to float32,
    #      but lm_head stays in bfloat16 (from dtype=bfloat16 at load time).
    #      Without autocast, lm_head receives float32 hidden states → crash:
    #        RuntimeError: expected scalar type BFloat16 but found Float
    #      Wrapping generate() in autocast harmonizes all dtypes.
    #   2. LOGITS PROCESSOR: transformers 5.3.0 GenerationConfig rejects
    #      logits_processor as a kwarg, but model.generate() accepts it.
    #      Inject the vocabulary mask here when grammar is enabled.
    _generation_model = trainer.model
    _orig_generate = _generation_model.generate
    _autocast_dtype = torch.bfloat16 if grpo_config.bf16 else torch.float16
    _lp_called = False

    def _patched_generate(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal _lp_called
        if logits_processor_for_gen is not None:
            # Prompt offsets are generation-specific; never reuse cached state.
            logits_processor_for_gen.reset()
            _kwargs["logits_processor"] = [logits_processor_for_gen] + _kwargs.get(
                "logits_processor", []
            )
            if not _lp_called:
                _lp_called = True
                print("  [constrained-decoding] logits_processor ACTIVE in generate()")
                allowed_count = 0
                if hasattr(logits_processor_for_gen, "allowed_ids"):
                    allowed_count = len(logits_processor_for_gen.allowed_ids)
                elif hasattr(logits_processor_for_gen, "get_valid_tokens"):
                    allowed_count = len(logits_processor_for_gen.get_valid_tokens())
                elif hasattr(logits_processor_for_gen, "mask") and hasattr(
                    logits_processor_for_gen.mask, "token_ids"
                ):
                    allowed_count = len(logits_processor_for_gen.mask.token_ids)
                print(f"  [constrained-decoding] allowed tokens: {allowed_count}")
        with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
            return _orig_generate(*_args, **_kwargs)

    _generation_model.generate = _patched_generate  # type: ignore[method-assign]
    print(
        "  model.generate monkey-patched on trainer.model (autocast + logits_processor)"
    )

    # Keep tqdm while suppressing duplicate progress log lines.
    try:
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(TqdmOnlyProgressCallback)
        trainer.add_callback(HighPrecisionLogCallback())
    except Exception:
        pass

    # ── Fix: guarantee wandb.finish() even on crash/exception ───────────
    # Previously, if trainer.train() raised (OOM, CUDA error, SLURM kill
    # signal caught as exception, etc.), wandb.finish() was never reached,
    # so the run stayed "crashed"/unfinished and output.log never made it
    # to the Files tab.  Wrapping in try/finally ensures the run is always
    # finalized and whatever log chunks were written get flushed.
    try:
        print("\n[grpo] Starting GRPO training...")
        trainer.train(resume_from_checkpoint=resume_from)

        # ── Save final model ─────────────────────────────────────────────
        final_path = Path(grpo_config.output_dir) / "final"
        print(f"\n[grpo] Saving final model to {final_path}...")
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        # ── Clean up duplicate final step checkpoint ──────────────────────
        global_step = trainer.state.global_step
        last_ckpt = Path(grpo_config.output_dir) / f"checkpoint-{global_step}"
        if last_ckpt.exists():
            import shutil

            print(
                f"[grpo] Cleaning up duplicate final step checkpoint folder: {last_ckpt}"
            )
            shutil.rmtree(last_ckpt, ignore_errors=True)
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        if wandb.run:
            wandb.finish()

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("GRPO T2G training complete!")
    print(f"  Model: {final_path}")
    print(f"  Logs:  {config['training']['log_dir']}")


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run this script directly. " "Use 'python -m src.training --config ...'"
    )
