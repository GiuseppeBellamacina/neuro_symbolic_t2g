"""
GRPO T2G Training Loop — Phase 1.

Integrates Constrained Decoding with Group Relative Policy Optimization (GRPO)
for Text-to-Gloss (T2G) translation using HuggingFace + PEFT and TRL.

Architecture:
    1. Load model + tokenizer via HuggingFace (LoRA + 4-bit quantization).
    2. Load ASLG-PC12 dataset and build prompt-completion pairs.
    3. Compute/load bigram transition matrix (Viterbi proxy).
    4. Build gloss vocabulary mask for constrained decoding.
    5. Define reward functions (translation quality + structural proxy).
    6. Train with ``trl.GRPOTrainer``, constraining generation rollouts
       to ASL gloss tokens only.

Usage:
    python -m src.training --config experiments/configs/t2g/grpo_qwen05.yaml
    CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc
import hashlib

# ── Workaround: _is_package_available in transformers 5.3.0 restituisce  ─
# una TUPLA (bool, str) invece di un bool.  In Python, una tupla non vuota
# è sempre truthy → (False, None) è True → trl prova a importare mergekit e
# llm_blender anche quando non sono installati.
# Fix: usiamo importlib per caricare trl.import_utils bypassando il pigro
# __getattr__, correggiamo le variabili a False, poi importiamo GRPOTrainer.
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

# ── Silence noisy transformers FutureWarnings ──────────────────────────
# transformers 5.3.0 prints 5 FutureWarning lines per generate() call about
# the deprecated AttentionMaskConverter API. These are internal to transformers
# and will be fixed in v5.10. Suppress them to keep the training log clean.
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
from transformers.integrations.integration_utils import WandbCallback
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
from src.datasets.transition_matrix import (
    compute_bigram_transitions,
    load_transition_matrix,
    save_transition_matrix,
)
from src.grammar.gloss_grammar import GlossVocabularyMask, create_grammarllm_pipeline
from src.grammar.grammar_logits_processor import (
    GlossVocabularyLogitsProcessor,
    GrammarPDALogitsProcessor,
)
from src.models.model_loader import load_model_and_tokenizer
from src.rewards.t2g_rewards import (
    build_t2g_reward_functions,
    initialize_rewards,
    register_gold_glosses,
)
from src.utils.config import load_config
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
    output_dir = training_cfg["output_dir"]
    log_dir = training_cfg["log_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    warmup_kwargs: dict[str, Any] = {}
    if "warmup_steps" in training_cfg:
        warmup_kwargs["warmup_steps"] = training_cfg["warmup_steps"]
    else:
        warmup_kwargs["warmup_steps"] = 50

    wandb_cfg = (full_config or {}).get("wandb", {})
    from datetime import datetime

    base_name = wandb_cfg.get("run_name", "grpo-t2g")
    run_timestamp = training_cfg.get("run_timestamp") or datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_name = f"{base_name}-{run_timestamp}"

    # Set tensorboard logging dir via env var (logging_dir kwarg is deprecated
    # since transformers 5.2).
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", log_dir)

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
        **warmup_kwargs,
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.1),
        bf16=training_cfg.get("bf16", True),
        # Gradient checkpointing: recompute forward activations during backward
        # to trade ~20% compute for substantial VRAM savings. Essential for
        # num_generations=8 on 22GB GPUs (cluster). Reads from
        # training.gradient_checkpointing in YAML; defaults to False to preserve
        # behavior of other configs that don't set it.
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        logging_steps=training_cfg.get("logging_steps", 5),
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 256),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 256),
        beta=grpo_cfg.get("beta", 0.04),
        temperature=grpo_cfg.get("temperature", 0.7),
        reward_weights=reward_weights,
        report_to="wandb",
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _prepare_t2g_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    vocab: list[str],
    dataset: Any = None,
) -> Dataset:
    """Load ASLG-PC12 and build prompt-completion pairs for GRPO.

    The dataset has columns: ``prompt``, ``completion`` (gold gloss), ``difficulty``.

    Args:
        config: Full config dict.
        tokenizer: Hugging Face tokenizer.
        vocab: Gloss vocabulary (unused here, kept for API compatibility).
        dataset: Optional pre-loaded ``DatasetDict``. If ``None``, downloads it.
    """
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

    # Format prompts with the centralized T2G prompt builder.
    # This guarantees train/eval/test use identical formatting.
    formatted: list[dict[str, str]] = []
    for i in range(len(t2g_ds)):
        sample = t2g_ds[i]
        text = sample["prompt"]

        prompt = build_t2g_prompt(text, tokenizer)

        # Stable sample ID: SHA256 of the user instruction (English sentence).
        # This survives any prompt format changes by TRL, enabling reliable
        # gold gloss lookup in reward functions.
        sample_id = hashlib.sha256(
            str(text).encode("utf-8", errors="replace")
        ).hexdigest()

        formatted.append(
            {
                "prompt": prompt,
                "completion": sample["completion"],
                "difficulty": sample.get("difficulty", "medium"),
                "sample_id": sample_id,
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
    """Build generation kwargs for GRPO rollouts.

    Set via ``grpo_config.generation_kwargs`` before creating GRPOTrainer.
    In trl 0.24.0, generation_kwargs lives on GRPOConfig (args), not on
    GRPOTrainer.__init__() directly.

    **IMPORTANT**: ``logits_processor`` is NOT included here.  trl 0.24.0
    passes generation_kwargs to ``GenerationConfig(**kwargs)``, and
    transformers 5.3.0 rejects ``logits_processor`` as a GenerationConfig
    argument.  Instead, the logits processor is injected via a monkey-patch
    of ``model.generate()`` in ``main()``.

    Args:
        config: Full config dict.

    Returns:
        Dict of generation kwargs compatible with ``GenerationConfig()``.
    """
    grpo_cfg = config.get("generation", config.get("grpo", {}))
    kwargs: dict[str, Any] = {
        "max_new_tokens": grpo_cfg.get("max_completion_length", 128),
    }
    return kwargs


# ---------------------------------------------------------------------------
# Curriculum Learning (G²RPO-A, ACL 2026)
# ---------------------------------------------------------------------------


class CurriculumSchedule:
    """Progressive difficulty curriculum for GRPO training.

    Implements 3-stage curriculum based on G²RPO-A findings that sorting
    training data by difficulty prevents small models from getting stuck
    on hard examples early in training.

    Stage 1 (0-33% steps): 10% simple, 65% medium, 25% hard
    Stage 2 (33-66% steps): 5% simple, 40% medium, 55% hard
    Stage 3 (66-100% steps): 3% simple, 30% medium, 67% hard
    """

    _STAGES: list[dict[str, float]] = [
        # Stage 1: 10% simple (usa quasi tutti i disponibili, 9.3%), 65% medium, 25% hard
        {"simple": 0.10, "medium": 0.65, "hard": 0.25},
        # Stage 2: 5% simple, 40% medium, 55% hard
        {"simple": 0.05, "medium": 0.40, "hard": 0.55},
        # Stage 3: 3% simple, 30% medium, 67% hard
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
    """Mutable dataset wrapper that filters by curriculum difficulty distribution.

    Wraps a Hugging Face ``Dataset`` and maintains a shuffled index list
    matching the current stage's target difficulty proportions.  The
    dataset length is kept constant (padded/truncated) so DataLoader
    samplers never go out of bounds on stage transitions.

    The underlying data is NOT copied — only the index list is rebuilt.
    """

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
    """TrainerCallback that drives curriculum stage transitions during GRPO.

    On each step, checks whether the global step has crossed a stage
    boundary.  When a new stage begins, triggers a dataset rebuild and
    logs the transition to stdout and wandb.
    """

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

            # Log curriculum metrics to wandb
            try:
                import wandb

                if wandb.run:
                    wandb.log(
                        {
                            "curriculum/stage": float(current_stage + 1),
                            "curriculum/difficulty_distribution": distribution,
                        },
                        step=state.global_step,
                    )
            except Exception:
                logger.debug("Failed to log curriculum metrics to wandb", exc_info=True)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for T2G GRPO training."""
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
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Resolve timestamped output/log directories and resume logic ──────
    from datetime import datetime

    training_cfg = config["training"]
    base_output_dir = Path(training_cfg["output_dir"])
    base_log_dir = Path(training_cfg["log_dir"])

    run_timestamp = None
    if args.resume:
        run_folders = sorted(base_output_dir.glob("run_*"))
        if run_folders:
            output_dir = run_folders[-1]
            run_timestamp = output_dir.name.removeprefix("run_")
            log_dir = base_log_dir / f"run_{run_timestamp}"
            print(f"[grpo] Resuming training in existing directory: {output_dir}")
        else:
            print(
                f"[grpo] Warning: No existing run directory found in {base_output_dir} to resume. Creating a new run."
            )

    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / f"run_{run_timestamp}"
        log_dir = base_log_dir / f"run_{run_timestamp}"
        print(f"[grpo] Starting new training run. Output dir: {output_dir}")

    config["training"]["output_dir"] = str(output_dir)
    config["training"]["log_dir"] = str(log_dir)
    config["training"]["run_timestamp"] = run_timestamp

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
    bigram_path = ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy")

    print(f"\n{'=' * 60}")
    print("STEP 1: Data Preparation")

    # Download dataset
    dataset = download_aslg_dataset(
        cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
    )

    # Extract vocabulary (or load from cache)
    if Path(vocab_path).exists():
        from src.datasets.aslg_dataset import load_vocabulary

        vocab = load_vocabulary(vocab_path)
    else:
        vocab = extract_gloss_vocabulary(dataset, split="train")
        save_vocabulary(vocab, vocab_path)

    # Compute transition matrix (or load from cache)
    if Path(bigram_path).exists():
        bigram_matrix = load_transition_matrix(bigram_path)
    else:
        bigram_matrix = compute_bigram_transitions(
            dataset, vocab, split="train", smoothing=1.0
        )
        save_transition_matrix(bigram_matrix, bigram_path)

    print(f"  Data prepared: |V|={len(vocab)}, bigram shape={bigram_matrix.shape}")

    if args.prepare_data:
        print("Data preparation complete. Exiting.")
        return

    # ── Step 1.5: Optional SFT Pre-training ─────────────────────────────
    sft_adapter_path: str | None = None
    sft_pretrain_cfg = config.get("sft_pretrain", {})
    if sft_pretrain_cfg.get("enabled", False):
        print(f"\n{'=' * 60}")
        print("STEP 1.5: SFT Pre-training")

        from src.training.sft_train import run_sft

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

    t2g_dataset = _prepare_t2g_dataset(config, tokenizer, vocab, dataset=dataset)

    # Register gold glosses for the translation quality reward function.
    # Uses stable sample IDs (SHA256 of user instruction) for format-agnostic
    # matching, so TRL prompt reformatting doesn'’t break the lookup.
    register_gold_glosses(
        sample_ids=list(t2g_dataset["sample_id"]),
        gold_glosses=list(t2g_dataset["completion"]),
    )

    # ── Step 5: Reward functions ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 5: Reward Functions")

    initialize_rewards(
        bigram_matrix,
        vocab,
        viterbi_diversity=config.get("grammar", {}).get("viterbi_diversity"),
    )
    reward_fns, reward_weights = build_t2g_reward_functions(config.get("reward"))

    # ── Wire completion sample logging (for live chain_monitor display) ─
    from src.training.callbacks import (
        CompletionSampleCallback,
        CompletionSampleLogger,
        HighPrecisionLogCallback,
        TqdmOnlyProgressCallback,
    )

    sample_logger = CompletionSampleLogger(reward_fns, reward_weights, n_samples=3)
    sample_logger.set_difficulty_map(t2g_dataset)
    wrapped_reward_fns = sample_logger.wrapped_reward_fns
    sample_callback = CompletionSampleCallback(
        sample_logger,
        every_n_steps=5,
        logits_processor=logits_processor_for_gen,
    )

    # ── Curriculum Learning setup ────────────────────────────────────────
    curriculum_cfg = config.get("curriculum", {})
    curriculum_callback: CurriculumCallback | None = None

    if curriculum_cfg.get("enabled", False):
        print(f"\n{'─' * 60}")
        print("CURRICULUM LEARNING: ENABLED")
        print("  G²RPO-A 3-stage progressive difficulty curriculum")

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
    # Modalità offline (cluster senza internet) — come grpo-strict-generation.
    # WANDB_MODE=offline è già esportato da train.sh; lo rinforziamo qui.
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
            # CRITICAL: reset the processor's prompt_len cache before each
            # generate() call. TRL generates completions for different prompts
            # in sequence — if prompt_len is stale from a previous rollout,
            # the Trie traces from the wrong offset in input_ids, producing
            # garbage allowed-token sets. This was the root cause of the
            # DEBUTRECHT/HOWEVERY garbage tokens in the 2026-07-08 run.
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

    # Replace default ProgressCallback with TqdmOnlyProgressCallback
    # (keeps tqdm bar, suppresses duplicate log lines — same as grpo-strict-generation)
    try:
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(TqdmOnlyProgressCallback)
        trainer.add_callback(HighPrecisionLogCallback())
        trainer.remove_callback(WandbCallback)
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
