"""
SFT T2G Training Script — Text-to-Gloss Supervised Fine-Tuning.

Trains Qwen2.5-0.5B-Instruct via teacher forcing on gold ASL gloss sequences
using ``trl.SFTTrainer``.  No reward shaping, no constrained decoding —
the model simply learns to replicate the gold gloss given the English input.

The dataset uses trl's native prompt-completion conversational format
(``prompt`` = ``[system, user]`` message list, ``completion`` = the gold
gloss) with ``completion_only_loss=True``, so the loss masks the prompt
tokens and only the gold gloss is counted.  A small seeded holdout is carved
from the train split and used with early stopping to guard against
overfitting.  Prompt formatting is identical to the GRPO rollout prompts
(see ``src/utils/prompting.py``).

Usage:
    python -m src.training --config experiments/configs/qwen25-05b/sft/zero-shot.yaml
    CONFIG=experiments/configs/qwen25-05b/sft/zero-shot.yaml sbatch cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb

# Silence noisy transformers FutureWarnings (AttentionMaskConverter deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings(
    "ignore",
    message=".*AttentionMaskConverter.*",
    category=FutureWarning,
)
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer  # type: ignore[import]

from datasets import Dataset
from src.datasets.aslg_dataset import (
    build_t2g_dataset,
    download_aslg_dataset,
    extract_gloss_vocabulary,
    save_vocabulary,
)
from src.datasets.structured_transitions import (
    StructuredTransitionGraph,
    build_structured_transition_graph,
)
from src.models.model_loader import load_model_and_tokenizer
from src.models.structured_gloss_head import StructuredGraphLoss
from src.utils.cache_meta import (
    cache_is_current,
    validate_dataset_name,
    write_cache_meta,
)
from src.utils.config import load_config
from src.utils.live_status import live_status_reset, live_status_set
from src.utils.paths import (
    Cell,
    RunPath,
    cell_from_config,
    training_run_paths,
    wandb_name,
    wandb_tags,
)
from src.utils.prompting import SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFT dataset preparation
# ---------------------------------------------------------------------------


def _build_prompt_completion_example(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw T2G row into a prompt-completion SFT example.

    ``prompt`` is the conversational message list ``[system, user]`` and
    ``completion`` the single ``[assistant]`` gold-gloss message.  trl 0.24
    tokenizes the prompt with ``apply_chat_template(prompt,
    add_generation_prompt=True)`` (``trl/trainer/sft_trainer.py:956-962``),
    producing byte-identical prompts to the GRPO rollout path
    (``build_t2g_prompt`` in ``src/utils/prompting.py``).  The full sequence
    is tokenized from ``prompt + completion`` and a ``completion_mask`` marks
    everything after the prompt (``trl/trainer/sft_trainer.py:997-1000``), so
    with ``completion_only_loss=True`` the loss only counts the gold gloss.

    Args:
        sample: Row from ``build_t2g_dataset`` (``prompt``, ``completion``,
            ``difficulty``; optionally ``gold_gloss`` and ``sample_id``).

    Returns:
        Dict with ``prompt`` and ``completion`` message lists plus the
        metadata columns ``gold_gloss``, ``difficulty``, ``sample_id``.
    """
    text = str(sample["prompt"]).strip()
    gold = str(sample.get("gold_gloss") or sample["completion"]).strip()
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "completion": [{"role": "assistant", "content": gold}],
        "gold_gloss": gold,
        "difficulty": str(sample.get("difficulty", "medium")),
        "sample_id": str(
            sample.get("sample_id") or hashlib.sha256(text.encode("utf-8")).hexdigest()
        ),
    }


def split_eval_holdout(
    dataset: Dataset,
    eval_fraction: float = 0.02,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Split a Hugging Face ``Dataset`` into ``(train, eval)`` holdout subsets.

    Seeded and deterministic (``Dataset.train_test_split`` with a fixed
    ``seed``), so the same inputs always produce the same partition and the
    two subsets are disjoint by construction.  Used to carve a small held-out
    set from the ASLG train split for evaluation and early stopping.

    Args:
        dataset: Source ``Dataset`` (the built SFT prompt-completion set).
        eval_fraction: Fraction of rows held out for evaluation.  Values
            ``<= 0`` return an empty eval set.
        seed: RNG seed for the shuffle (use the dataset seed).

    Returns:
        ``(train_ds, eval_ds)``.
    """
    if eval_fraction <= 0.0:
        return dataset, dataset.select([])
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed, shuffle=True)
    return split["train"], split["test"]


def _prepare_sft_dataset(
    config: dict[str, Any],
    dataset: Any = None,
) -> tuple[Dataset, Dataset]:
    """Build prompt-completion train/eval datasets for SFT.

    Uses trl 0.24's native prompt-completion conversational format: the
    ``prompt`` column is the message list ``[system, user]`` and the
    ``completion`` column the single ``[assistant]`` gold-gloss message
    (see ``_build_prompt_completion_example``).  ``SFTTrainer`` tokenizes
    these with the tokenizer chat template and builds a ``completion_mask``
    (``trl/trainer/sft_trainer.py:949-1000``), so the loss — with
    ``completion_only_loss=True`` — only covers the gold gloss.

    A small seeded holdout is carved from the train split
    (``config["training"]["eval_fraction"]``, default 0.02) for evaluation
    and early stopping.

    Args:
        config: Full config dict.
        dataset: Optional pre-loaded ``DatasetDict``. If ``None``, downloads it.

    Returns:
        ``(train_ds, eval_ds)`` pair with columns ``prompt``, ``completion``,
        ``gold_gloss``, ``difficulty``, ``sample_id``.
    """
    ds_cfg = config["dataset"]
    validate_dataset_name(ds_cfg.get("dataset_name"))
    if dataset is None:
        dataset = download_aslg_dataset(
            cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
        )

    t2g_ds = build_t2g_dataset(
        dataset,
        split=ds_cfg.get("split", "train"),
        max_samples=ds_cfg.get("max_samples"),
    )

    rows = [_build_prompt_completion_example(sample) for sample in t2g_ds]
    sft_ds = Dataset.from_list(rows)
    logger.info(
        "[sft] SFT dataset: %d prompt-completion pairs (columns=%s)",
        len(sft_ds),
        list(sft_ds.column_names),
    )

    eval_fraction = config.get("training", {}).get("eval_fraction", 0.02)
    seed = ds_cfg.get("seed", 42)
    train_ds, eval_ds = split_eval_holdout(
        sft_ds, eval_fraction=eval_fraction, seed=seed
    )
    logger.info(
        "[sft] Eval holdout: eval_fraction=%.3f → train=%d, eval=%d",
        eval_fraction,
        len(train_ds),
        len(eval_ds),
    )
    return train_ds, eval_ds


def prepare_structured_sft_datasets(
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer: Any,
    structured_cfg: dict[str, Any],
    *,
    max_sequence_length: int,
) -> tuple[Dataset, Dataset, StructuredTransitionGraph, dict[str, Any]]:
    """Build a train-only graph and attach precomputed row metadata.

    The LM row sets are returned intact and in their original order. Eligibility
    affects only graph construction and the structured objective.
    """
    max_gloss_length = int(structured_cfg["max_gloss_length"])

    def eligible(row: Any) -> bool:
        tokens = str(row["gold_gloss"]).split()
        if not (0 < len(tokens) <= max_gloss_length):
            return False
        encoded = tokenizer.apply_chat_template(
            [*row["prompt"], *row["completion"]],
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.reshape(-1).tolist()
        return (
            len(encoded) <= max_sequence_length
            and bool(encoded)
            and int(encoded[-1]) == int(tokenizer.eos_token_id)
        )

    train_rows = [dict(row) for row in train_ds]
    eval_rows = [dict(row) for row in eval_ds]
    train_ids = [str(row["sample_id"]) for row in train_rows]
    eval_ids = [str(row["sample_id"]) for row in eval_rows]
    if len(set(train_ids)) != len(train_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("structured SFT requires unique sample IDs within each split")
    if not set(train_ids).isdisjoint(eval_ids):
        raise ValueError("structured train/eval sample IDs overlap")
    eligible_rows = [
        {"sample_id": row["sample_id"], "gloss": row["gold_gloss"]}
        for row in train_rows
        if eligible(row)
    ]
    excluded_ids = [str(row["sample_id"]) for row in train_rows if not eligible(row)]
    if not eligible_rows:
        raise ValueError("structured SFT has no eligible finalized train rows")
    graph = build_structured_transition_graph(
        eligible_rows,
        top_k=int(structured_cfg["top_k"]),
        alpha=float(structured_cfg["alpha"]),
    )
    manifest = {
        **graph.manifest(),
        "train_split_sample_ids": train_ids,
        "eval_split_sample_ids": eval_ids,
        "eligibility_policy": "complete-nonempty-whitespace-gloss-within-max-length-and-full-chat-fits-through-eos-v1",
        "max_gloss_length": max_gloss_length,
        "excluded_train_sample_ids": excluded_ids,
        "excluded_train_rate": len(excluded_ids) / max(len(train_ids), 1),
        "transition_scale": float(structured_cfg["transition_scale"]),
    }

    def add_metadata(row: dict[str, Any]) -> dict[str, Any]:
        tokens = str(row["gold_gloss"]).split()
        row_eligible = eligible(row)
        states = graph.map_glosses(tokens) if row_eligible else []
        return {
            "structured_gold_states": states + [0] * (max_gloss_length - len(states)),
            "structured_length": len(states),
            "structured_eligible": row_eligible,
        }

    return train_ds.map(add_metadata), eval_ds.map(add_metadata), graph, manifest


# ---------------------------------------------------------------------------
# SFT adapter fingerprint & reuse
# ---------------------------------------------------------------------------

#: Version of the fingerprint schema.  Bump when the fingerprinted fields
#: change (e.g. a new field starts affecting the adapter) — the version is
#: part of the hash, so a bump invalidates every previously stored
#: fingerprint automatically.
_SFT_FINGERPRINT_VERSION = 4

#: Training keys that never affect the SFT adapter weights (paths/timestamps).
_NON_DETERMINISTIC_TRAINING_KEYS = ("output_dir", "log_dir", "run_timestamp", "trainer")


def _sft_training_fingerprint_source(config: dict[str, Any]) -> dict[str, Any]:
    """SFT-relevant training hyperparameters (path/timestamp keys excluded).

    In the GRPO flow ``sft_config["sft_pretrain"]["training"]`` carries the
    SFT hyperparameters, while the merged ``training`` section additionally
    holds GRPO-only keys such as ``max_steps`` that must NOT invalidate the
    SFT adapter. The canonical ``sft/zero-shot.yaml`` flow has no ``sft_pretrain``
    section, so the effective ``training`` section is used instead. This makes
    the same SFT fingerprint reusable from the ``sft/zero-shot`` cell by the
    ``sft-grpo`` zero-shot and few-shot cells.
    """
    pretrain_training = config.get("sft_pretrain", {}).get("training", {})
    if isinstance(pretrain_training, dict) and pretrain_training:
        training = dict(pretrain_training)
    else:
        training = dict(config.get("training", {}))
    for key in _NON_DETERMINISTIC_TRAINING_KEYS:
        training.pop(key, None)
    return training


def _sft_fingerprint_payload(
    config: dict[str, Any], auxiliary_manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The exact dict hashed to produce the SFT fingerprint.

    Contains every field that determines the SFT adapter (model + loading,
    LoRA shape, dataset selection, SFT hyperparameters, system prompt).
    Output/log paths and run timestamps are deliberately excluded — they
    never affect the adapter weights.
    """
    model = config.get("model", {})
    lora = config.get("lora", {})
    dataset = config.get("dataset", {})
    mass_enabled = bool(
        config.get("auxiliary_loss", {}).get("mass", {}).get("enabled", False)
    )
    structured_cfg = config.get("auxiliary_loss", {}).get("structured", {})
    structured_enabled = bool(structured_cfg.get("enabled", False))
    if mass_enabled and auxiliary_manifest is None:
        raise ValueError(
            "enabled auxiliary mass fingerprint requires a runtime Trie manifest"
        )
    if structured_enabled and auxiliary_manifest is None:
        raise ValueError(
            "enabled auxiliary structured fingerprint requires a graph manifest"
        )
    trie_manifest = None
    graph_manifest = None
    if auxiliary_manifest:
        # Accept the original mass-only bare manifest while using a combined
        # envelope for mass+structured runs.
        trie_manifest = (
            auxiliary_manifest.get("trie", auxiliary_manifest) if mass_enabled else None
        )
        graph_manifest = (
            auxiliary_manifest.get("structured") if structured_enabled else None
        )
    if structured_enabled and graph_manifest is None:
        raise ValueError(
            "enabled auxiliary structured fingerprint requires graph manifest"
        )
    return {
        "version": _SFT_FINGERPRINT_VERSION,
        "model": {
            key: model.get(key)
            for key in ("name", "quantization", "dtype", "use_unsloth")
            if key in model
        },
        "lora": {
            key: lora.get(key)
            for key in (
                "r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "random_state",
            )
            if key in lora
        },
        "dataset": {
            key: dataset.get(key)
            for key in ("dataset_name", "seed", "split", "max_samples", "thinking")
            if key in dataset
        },
        "sft_training": _sft_training_fingerprint_source(config),
        "auxiliary_objective": {
            "protocol": "allowed-mass-v1",
            "mass": {
                "enabled": mass_enabled,
                "lambda": float(
                    config.get("auxiliary_loss", {}).get("mass", {}).get("lambda", 0.0)
                ),
                "warmup_steps": int(
                    config.get("auxiliary_loss", {})
                    .get("mass", {})
                    .get("warmup_steps", 0)
                ),
            },
            "structured": {
                "enabled": structured_enabled,
                **{
                    key: structured_cfg.get(key)
                    for key in (
                        "lambda",
                        "warmup_steps",
                        "top_k",
                        "max_gloss_length",
                        "alpha",
                        "transition_scale",
                    )
                    if key in structured_cfg
                },
                "head_architecture": (
                    "boundary-position-layernorm-linear-v1"
                    if structured_enabled
                    else None
                ),
                "eligibility_policy": (
                    "lm-all_structured-complete-whitespace-max64-eos-retained-v1"
                    if structured_enabled
                    else None
                ),
            },
            "trie_manifest": trie_manifest,
            "graph_manifest": graph_manifest,
        },
        "system_prompt": SYSTEM_PROMPT,
    }


def compute_sft_fingerprint(
    sft_config: dict[str, Any], auxiliary_manifest: dict[str, Any] | None = None
) -> str:
    """SHA-256 fingerprint of everything that determines an SFT adapter.

    Two runs with the same fingerprint are expected to produce equivalent
    adapters, so the SFT phase can be skipped and the previously saved
    adapter reused.  Changing any SFT hyperparameter, the model, the LoRA
    config, the dataset or the system prompt changes the fingerprint and
    forces a retrain.

    Args:
        sft_config: Full config dict as passed to :func:`run_sft`.

    Returns:
        64-char hex SHA-256 of the canonical JSON of the fingerprinted
        fields.
    """
    canonical = json.dumps(
        _sft_fingerprint_payload(sft_config, auxiliary_manifest),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_sft_fingerprint(
    final_path: str | Path,
    sft_config: dict[str, Any],
    auxiliary_manifest: dict[str, Any] | None = None,
) -> Path:
    """Write ``sft_fingerprint.json`` next to a freshly-trained SFT adapter.

    The file records the fingerprint plus the fingerprinted config so a
    later GRPO run can decide whether this adapter is reusable.  It is
    written ONLY after training completed (called at the end of
    :func:`run_sft`); a ``final/`` without it is never reused.

    Args:
        final_path: Directory of the saved SFT adapter (``.../final``).
        sft_config: Full config dict used to train the adapter.

    Returns:
        Path of the written ``sft_fingerprint.json`` file.
    """
    document = {
        "fingerprint": compute_sft_fingerprint(sft_config, auxiliary_manifest),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": _sft_fingerprint_payload(sft_config, auxiliary_manifest),
    }
    final_dir = Path(final_path)
    final_dir.mkdir(parents=True, exist_ok=True)
    out = final_dir / "sft_fingerprint.json"
    out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def is_complete_adapter_dir(path: str | Path) -> bool:
    """Whether *path* looks like a loadable (PEFT/merged) adapter directory.

    Accepts a PEFT LoRA adapter (``adapter_config.json`` + weights) or a
    merged model directory (``config.json`` + weights).  Used to guard
    adapter reuse: a directory containing only ``sft_fingerprint.json`` is
    NOT a usable adapter.

    Args:
        path: Candidate adapter directory.

    Returns:
        ``True`` if the directory has a config file AND weight files.
    """
    d = Path(path)
    if not d.is_dir():
        return False
    has_config = (d / "adapter_config.json").is_file() or (d / "config.json").is_file()
    has_weights = any(
        (d / name).is_file()
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        )
    )
    return has_config and has_weights


def _adapter_if_matching(candidate: Path, fingerprint: str) -> Path | None:
    """Return the adapter dir when *candidate*'s fingerprint matches.

    Shared by the same-cell and cross-method searches: skips unreadable
    fingerprint files and fingerprint matches with missing weight files.
    """
    try:
        meta = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning(
            "[sft-reuse] Unreadable sft_fingerprint.json, skipping: %s", candidate
        )
        return None
    stored_mass = (
        meta.get("config", {})
        .get("auxiliary_objective", {})
        .get("mass", {})
        .get("enabled", False)
    )
    stored_structured = (
        meta.get("config", {})
        .get("auxiliary_objective", {})
        .get("structured", {})
        .get("enabled", False)
    )
    if stored_mass or stored_structured:
        logger.info(
            "[sft-reuse] Skipping non-reusable auxiliary adapter: %s", candidate
        )
        return None
    if meta.get("fingerprint") != fingerprint:
        return None
    adapter_dir = candidate.parent
    if not is_complete_adapter_dir(adapter_dir):
        logger.warning(
            "[sft-reuse] Fingerprint match but adapter files missing, " "skipping: %s",
            adapter_dir,
        )
        return None
    return adapter_dir


def find_reusable_sft_adapter_cross_method(
    checkpoints_root: str | Path,
    exclude_parent: str | Path,
    fingerprint: str,
    *,
    sft_config: dict[str, Any] | None = None,
) -> tuple[Path, str] | None:
    """Find a matching adapter in the canonical SFT layouts for one model.

    Only these layouts are searched beneath *checkpoints_root*:

    * ``sft/zero-shot/run_*/final/sft_fingerprint.json``
    * ``sft-grpo/*/run_*/sft_pretrain/final/sft_fingerprint.json``

    The current run (or cell base, when that is passed) is excluded. Matches
    are ordered by fingerprint-file mtime newest-first, then by path for a
    stable result when mtimes tie.

    Args:
        checkpoints_root: Canonical model root, e.g.
            ``experiments/checkpoints/qwen25-05b``.
        exclude_parent: Current canonical run directory or cell base to skip.
        fingerprint: Expected SFT fingerprint.

    Returns:
        ``(adapter_dir, cell_name)`` of the newest match, or ``None``.
    """
    auxiliary = (sft_config or {}).get("auxiliary_loss", {})
    if any(
        auxiliary.get(name, {}).get("enabled", False) for name in ("mass", "structured")
    ):
        logger.info("[sft-reuse] Auxiliary pilot always trains; reuse disabled")
        return None
    root = Path(checkpoints_root)
    excluded = Path(exclude_parent).resolve()
    if not root.is_dir():
        return None
    candidates = [
        *root.glob("sft/zero-shot/run_*/final/sft_fingerprint.json"),
        *root.glob("sft-grpo/*/run_*/sft_pretrain/final/sft_fingerprint.json"),
    ]
    candidates.sort(
        key=lambda p: (p.stat().st_mtime, p.as_posix()),
        reverse=True,
    )
    for candidate in candidates:
        is_subphase = candidate.parent.parent.name == "sft_pretrain"
        run_dir = (
            candidate.parent.parent.parent if is_subphase else candidate.parent.parent
        )
        candidate_base = run_dir.parent
        if excluded in {run_dir.resolve(), candidate_base.resolve()}:
            continue
        adapter_dir = _adapter_if_matching(candidate, fingerprint)
        if adapter_dir is not None:
            source_cell = candidate_base.relative_to(root).as_posix()
            logger.info(
                "[sft-reuse] Cross-method match: reusing SFT adapter from "
                "cell '%s': %s",
                source_cell,
                adapter_dir,
            )
            return adapter_dir, source_cell
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def resolve_sft_run_paths(
    config: dict[str, Any], resume: bool = False
) -> tuple[Path, Path, str, Cell]:
    """Resolve standalone canonical paths or preserve explicit GRPO subphase paths.

    Standalone canonical SFT configs intentionally omit ``training.output_dir``
    and ``training.log_dir``. Explicit directories are only treated as a
    subphase when both are present and the output path is inside a ``run_*``.
    """
    training_cfg = config["training"]
    auxiliary = config.get("auxiliary_loss", {})
    auxiliary_enabled = any(
        auxiliary.get(name, {}).get("enabled", False) for name in ("mass", "structured")
    )
    if auxiliary_enabled and (
        training_cfg.get("packing", False) or training_cfg.get("padding_free", False)
    ):
        raise ValueError("auxiliary loss does not support packing or padding_free")
    explicit_output = training_cfg.get("output_dir")
    explicit_log = training_cfg.get("log_dir")
    if explicit_output is not None and explicit_log is not None:
        output_dir = Path(explicit_output)
        if any(part.startswith("run_") for part in output_dir.parts):
            cell = cell_from_config(config)
            run_timestamp = next(
                part.removeprefix("run_")
                for part in reversed(output_dir.parts)
                if part.startswith("run_")
            )
            return output_dir, Path(explicit_log), run_timestamp, cell

    output_dir, log_dir, run_id, cell = training_run_paths(config, resume=resume)
    return output_dir, log_dir, run_id.removeprefix("run_"), cell


def run_sft(config: dict[str, Any], resume: bool = False) -> str:
    """Run SFT training and return the path to the saved adapter.

    This function is designed to be called from grpo_t2g_train.py for
    SFT pre-training before GRPO.  It aggressively cleans up GPU memory
    when done so GRPO can use the full VRAM.

    Args:
        config: Full config dict (same format as YAML).

    Returns:
        Path to the saved SFT LoRA adapter directory.
    """

    from src.training.auxiliary_sft_trainer import (
        require_single_process_for_auxiliary,
        validate_mass_config,
        validate_structured_config,
    )

    mass_cfg = validate_mass_config(config)
    structured_cfg = validate_structured_config(config)
    require_single_process_for_auxiliary(config)

    # ── Setup logging ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    # Quiet down external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # HF libraries attach their own StreamHandler AND propagate to root —
    # every library warning printed twice (slurm-train-7073). Strip the
    # library-owned handlers so each record prints exactly once.
    from src.utils.log_dedup import dedupe_library_loggers

    dedupe_library_loggers()

    # ── Set random seeds for reproducibility ─────────────────────────────
    seed = config["dataset"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Reproducibility: seed=%d (random, numpy, torch, cuda)", seed)

    # ── Step 1: Data preparation ─────────────────────────────────────────
    ds_cfg = config["dataset"]
    vocab_path = ds_cfg.get("vocab_path", "data/gloss_vocab.txt")
    validate_dataset_name(ds_cfg.get("dataset_name"))

    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)

    dataset = download_aslg_dataset(
        cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
    )

    # Vocabulary cache is valid only with the same dataset seed/train size.
    train_size = len(dataset["train"])
    if cache_is_current(vocab_path, seed, train_size):
        from src.datasets.aslg_dataset import load_vocabulary

        vocab = load_vocabulary(vocab_path)
    else:
        vocab = extract_gloss_vocabulary(dataset, split="train")
        save_vocabulary(vocab, vocab_path)
        write_cache_meta(vocab_path, seed, train_size)

    logger.info("Data prepared: |V|=%d (bigram not used by SFT)", len(vocab))

    # (prepare-data is handled in main(), not here)

    # ── Step 2: Model loading ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Model Loading")
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(config)

    # ── Step 3: SFT dataset preparation ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: SFT Dataset Preparation")
    logger.info("=" * 60)

    sft_train_ds, sft_eval_ds = _prepare_sft_dataset(config, dataset=dataset)

    # Log a few sample pairs for verification
    logger.info("[sft] Sample prompt-completion pairs (first 2):")
    for i in range(min(2, len(sft_train_ds))):
        sample = sft_train_ds[i]
        user_text = sample["prompt"][-1]["content"]
        gold_text = sample["completion"][0]["content"]
        logger.info("[sft]   #%d  EN: %s", i, user_text[:80])
        logger.info("[sft]        GOLD: %s", gold_text[:80])

    # ── Step 4: SFT configuration ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: SFT Configuration")
    logger.info("=" * 60)

    training_cfg = config["training"]
    mass_enabled = bool(mass_cfg.get("enabled", False))
    structured_enabled = bool(structured_cfg.get("enabled", False))
    output_dir, log_dir, run_timestamp, cell = resolve_sft_run_paths(
        config, resume=resume
    )
    is_subphase = "sft_pretrain" in output_dir.parts
    if is_subphase:
        logger.info("SFT running as GRPO sub-phase. Using path: %s", output_dir)
    else:
        logger.info("Resolved SFT run directory: %s", output_dir)

    output_dir = str(output_dir)
    log_dir = str(log_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    wandb_cfg = config.get("wandb", {})
    identity = RunPath(cell, f"run_{run_timestamp}")
    run_name = wandb_name(identity)
    wandb_cfg = {**wandb_cfg, "tags": list(wandb_tags(identity))}

    # Set tensorboard logging dir via env var (logging_dir kwarg is deprecated
    # since transformers 5.2).
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", log_dir)

    eval_enabled = len(sft_eval_ds) > 0
    if not eval_enabled:
        logger.warning(
            "[sft] eval_fraction<=0 → eval_strategy='no', early stopping disabled"
        )

    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        seed=training_cfg.get("seed", config["dataset"].get("seed", 42)),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 100),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        max_length=training_cfg.get(
            "max_seq_length", 768
        ),  # renamed from max_seq_length in TRL 0.20+
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        # ── Loss masking: only the gold gloss (completion) counts ───────
        # The dataset uses trl's prompt-completion conversational format, so
        # SFTTrainer builds a completion_mask and, when completion_only_loss
        # is None (default), auto-enables completion-only loss for
        # prompt-completion datasets (trl/trainer/sft_trainer.py:733-739).
        # Set explicitly for clarity and forward-compatibility.
        completion_only_loss=True,
        # ── Held-out eval + early stopping (overfitting guard) ──────────
        # ``sft_pretrain.training`` is merged into ``training`` by
        # grpo_t2g_train.py, so eval_fraction/eval_steps/... are read here
        # from the same key regardless of the entry point.
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=training_cfg.get("eval_steps", 200),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=eval_enabled,
        report_to="wandb",
    )

    logger.info(
        "[sft] epochs=%d, batch=%d, grad_accum=%d, lr=%.1e, max_len=%d",
        sft_config.num_train_epochs,
        sft_config.per_device_train_batch_size,
        sft_config.gradient_accumulation_steps,
        sft_config.learning_rate,
        sft_config.max_length,
    )
    logger.info(
        "[sft] warmup=%d, weight_decay=%.3f, scheduler=%s, optim=%s, bf16=%s",
        sft_config.warmup_steps,
        sft_config.weight_decay,
        sft_config.lr_scheduler_type,
        sft_config.optim,
        sft_config.bf16,
    )
    logger.info(
        "[sft] dataset_size=%d, effective_batch=%d, total_optim_steps≈%d",
        len(sft_train_ds),
        sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps,
        max(
            1,
            len(sft_train_ds)
            // (
                sft_config.per_device_train_batch_size
                * sft_config.gradient_accumulation_steps
            ),
        )
        * sft_config.num_train_epochs,
    )
    logger.info(
        "[sft] eval: strategy=%s, eval_steps=%d, eval_size=%d, "
        "completion_only_loss=%s",
        sft_config.eval_strategy,
        sft_config.eval_steps,
        len(sft_eval_ds),
        sft_config.completion_only_loss,
    )

    # ── Resume logic ─────────────────────────────────────────────────────
    resume_from: str | None = None
    if resume:
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"))
        if ckpts:
            resume_from = str(ckpts[-1])
            logger.info("Resuming from %s", resume_from)

    # ── Wandb setup ──────────────────────────────────────────────────────
    # Modalità offline — come grpo-strict-generation.
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    # Disable weave (wandb 0.25.0 tenta il login anche offline).
    os.environ["WANDB_DISABLE_WEAVE"] = "true"
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "neuro-symbolic-t2g")
    os.environ["WANDB_DIR"] = log_dir
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get("tags", ["sft", "t2g", "supervised"])
    )

    if not wandb.run:
        wandb.init(
            project=wandb_cfg.get("project", "neuro-symbolic-t2g"),
            name=run_name,
            config=config,
            tags=wandb_cfg.get("tags", ["sft", "t2g"]),
            dir=log_dir,
            mode="offline",
            # ── Fix: output.log missing on Files tab ──────────────────
            # See grpo_t2g_train.py for full explanation: without
            # console_multipart, W&B only flushes output.log on a clean
            # wandb.finish(). SLURM OOM/timeout kills lose the log entirely.
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_bytes=1_000_000,
                console_chunk_max_seconds=60,
            ),
        )

    # ── Tee stdout → output.log (sync_cluster download) ─────────────────
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

    # ── Step 5: Training ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: SFT Training")
    logger.info("=" * 60)

    # ── Workaround: transformers 5.3.0 + peft non espongono  ──────────
    # model.warnings_issued.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    from transformers.integrations.integration_utils import WandbCallback
    from transformers.trainer_callback import EarlyStoppingCallback, ProgressCallback

    from src.training.callbacks import (
        HighPrecisionLogCallback,
        SFTSampleCallback,
        TqdmOnlyProgressCallback,
    )

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": sft_train_ds,
        "eval_dataset": sft_eval_ds if eval_enabled else None,
        "processing_class": tokenizer,
    }
    trie_manifest = None
    graph_manifest = None
    graph = None
    if structured_enabled:
        max_gloss_length = int(structured_cfg["max_gloss_length"])
        max_sequence_length = int(training_cfg.get("max_seq_length", 768))
        sft_train_ds, sft_eval_ds, graph, graph_manifest = (
            prepare_structured_sft_datasets(
                sft_train_ds,
                sft_eval_ds,
                tokenizer,
                structured_cfg,
                max_sequence_length=max_sequence_length,
            )
        )

        from src.models.auxiliary_sft_model import AuxiliarySFTModel
        from src.models.structured_gloss_head import StructuredGlossHead

        hidden_size = int(getattr(model.config, "hidden_size"))
        head = StructuredGlossHead(hidden_size, graph.num_states, max_gloss_length)
        head.to(model.device)
        model = AuxiliarySFTModel(model, head)
        trainer_kwargs["model"] = model

    if mass_enabled or structured_enabled:
        from src.grammar.grammar_logits_processor import DualRootGlossTrie
        from src.training.auxiliary_sft_trainer import (
            AuxiliaryMassSFTTrainer,
            CompletionMetadataCollator,
            build_auxiliary_trie_manifest,
            canonical_vocabulary,
        )

        trie = None
        if mass_enabled:
            canonical_vocab = list(canonical_vocabulary(vocab))
            trie_manifest = build_auxiliary_trie_manifest(vocab, vocab_path, tokenizer)
            trie = DualRootGlossTrie.from_vocabulary(canonical_vocab, tokenizer)
        collator = CompletionMetadataCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            completion_only_loss=True,
        )
        trainer = AuxiliaryMassSFTTrainer(
            **trainer_kwargs,
            data_collator=collator,
            mass_state_machine=trie,
            mass_lambda=float(mass_cfg.get("lambda", 0.1)),
            mass_warmup_steps=int(mass_cfg.get("warmup_steps", 200)),
            structured_loss=(
                StructuredGraphLoss(graph, float(structured_cfg["transition_scale"]))
                if structured_enabled and graph is not None
                else None
            ),
            structured_lambda=float(structured_cfg.get("lambda", 0.0)),
            structured_warmup_steps=int(structured_cfg.get("warmup_steps", 0)),
            graph_manifest=graph_manifest,
        )
        logger.info(
            "[sft] auxiliary objectives: mass=%s structured=%s (training only; single-GPU)",
            mass_enabled,
            structured_enabled,
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    auxiliary_manifest = (
        {"trie": trie_manifest, "structured": graph_manifest}
        if mass_enabled or structured_enabled
        else None
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

    # Early stopping on eval_loss: stop if it does not improve for
    # `early_stopping_patience` evaluations (guards overfitting on the
    # prompt-redundant 78K train samples).
    if eval_enabled:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg.get("early_stopping_patience", 3)
            )
        )

    # SFT sample + loss tracking callback for visibility into pre-training
    sft_sample_cb = SFTSampleCallback(
        tokenizer=tokenizer,
        model=model,
        dataset=sft_train_ds,
        every_n_steps=training_cfg.get("logging_steps", 10) * 5,
        sample_every_n_steps=training_cfg.get("sft_sample_every_n_steps", 100),
        n_samples=2,
    )
    trainer.add_callback(sft_sample_cb)

    # ── Fix: guarantee wandb.finish() even on crash/exception ───────────
    # See grpo_t2g_train.py for full explanation.
    final_path_str: str
    try:
        logger.info("Starting SFT training...")
        # Live status: SFT training loop starts (estimated total steps).
        live_status_set(
            phase="sft",
            total_steps=(
                int(training_cfg["max_steps"])
                if training_cfg.get("max_steps")
                else None
            ),
            note="SFT training",
        )
        trainer.train(resume_from_checkpoint=resume_from)

        # ── Best metric (tracked by load_best_model_at_end) ─────────────
        best_metric = getattr(trainer.state, "best_metric", None)
        if best_metric is not None:
            logger.info(
                "[sft] Best eval_loss=%.6f (best checkpoint=%s)",
                best_metric,
                trainer.state.best_model_checkpoint,
            )
            live_status_set(eval_loss_best=float(best_metric))
        else:
            logger.info("[sft] No eval metric tracked (evaluation disabled).")

        # ── Save final model ─────────────────────────────────────────────
        # With load_best_model_at_end=True the trainer already re-loaded the
        # best checkpoint weights, so `final` holds the best adapter.
        final_path = Path(output_dir) / "final"
        logger.info("Saving final model to %s...", final_path)
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        final_path_str = str(final_path)

        # ── Record SFT fingerprint (adapter reuse in the GRPO flow) ──────
        # Written ONLY after a completed training: a ``final/`` without
        # sft_fingerprint.json is never reused by grpo_t2g_train.
        try:
            fingerprint_path = write_sft_fingerprint(
                final_path, config, auxiliary_manifest
            )
            logger.info("SFT fingerprint written to %s", fingerprint_path)
        except Exception as exc:  # metadata only — never fail a completed training
            logger.warning("[sft] Failed to write sft_fingerprint.json: %s", exc)

        # ── Clean up duplicate final step checkpoint ──────────────────────
        global_step = trainer.state.global_step
        last_ckpt = Path(output_dir) / f"checkpoint-{global_step}"
        if last_ckpt.exists() and not structured_enabled:
            import shutil

            logger.info(
                "Cleaning up duplicate final step checkpoint folder: %s", last_ckpt
            )
            shutil.rmtree(last_ckpt, ignore_errors=True)
    finally:
        # ── Cleanup (aggressive: free VRAM for GRPO phase) ───────────────
        if wandb.run:
            wandb.finish()

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("SFT T2G training complete!")
    logger.info("  Model: %s", final_path_str)
    logger.info("  Logs:  %s", log_dir)
    # Live status: SFT done — back to idle (the GRPO phase, if any, will
    # set its own phase right after).
    live_status_reset(note=f"SFT completato: {final_path_str}")
    logger.info("=" * 60)

    return final_path_str


def main() -> None:
    """Standalone entry point for SFT training (used by __main__.py)."""
    parser = argparse.ArgumentParser(description="SFT training for Text-to-Gloss (T2G)")
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

    if args.prepare_data:
        # Handle prepare-data separately
        ds_cfg = config["dataset"]
        from src.datasets.aslg_dataset import download_aslg_dataset

        validate_dataset_name(ds_cfg.get("dataset_name"))
        download_aslg_dataset(
            cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
        )
        print("Data preparation complete.")
        return

    run_sft(config, resume=args.resume)


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run this script directly. "
        "Use 'python -m src.training --config ...' to ensure "
        "Unsloth is imported before trl/transformers/peft for optimizations."
    )
