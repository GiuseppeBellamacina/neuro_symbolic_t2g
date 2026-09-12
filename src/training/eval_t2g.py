"""
T2G Evaluation Script — Multi-metric with plots.

Evaluates trained checkpoints on the ASLG-PC12 test set using:
    - ROUGE-L F1 (translation quality)
    - BLEU (sacreBLEU sentence + corpus)
    - chrF2 (sacreBLEU, sentence + corpus)
    - Token-level gloss F1 (micro + sentence mean)
    - Non-copy token accuracy (metrica primaria copy-insensitive)
    - Pass@1 / Pass@k (multiple sampling)
    - Gloss validity (free-text / repetition detection)
    - Bigram log-probability (structural plausibility)
    - Exact match accuracy
    - Per-component reward breakdown (direct calls — no registry)
    - Detailed metrics (percentiles, error distribution)

**Honest primary metrics.**  The primary metric block is computed over
*all* ``num_samples`` completions per prompt (never just the first one),
so ROUGE-L/BLEU/chrF/F1/exact-match/bigram/validity are honest numbers
for the actual decoding strategy.  ``pass_at_1`` is the fraction of
prompts whose *first* completion reaches ROUGE-L >= 0.3 (a single
honest draw), computed with the shared ``compute_pass_at_k`` helper
(k=1) for consistency with the Pass@k curve.  NOTE: the CI block
(``evaluation_report["pass_at_1"]``) reports a DIFFERENT pass@1
estimator — the empirical pass rate over ALL completions with a
bootstrap CI.  The two are labeled distinctly in the output and agree
within sampling noise (docs/EVALUATION.md §2a).

**Best-of-N is oracle-only.**  When ``evaluation.best_of_n`` is enabled, the
gold reference is used to pick the best completion per prompt; those metrics
are reported in a *separate* ``oracle_best_of_n`` block and never overwrite
the primary (deployable) metrics.

**Dual prompting (attivo di default sulle celle addestrabili).** Con
``evaluation.dual_prompting: true`` la cella è valutata in ENTRAMBE le
modalità di prompting (quella della config + la complementare): la seconda
passata riesegue l'intera pipeline nello stesso processo, scrive file con
suffisso ``__<mode>`` (mai sovrascritti: il suffisso copre eval/generations/
baseline E comparison.json, le figure vanno in una sottodirectory per
modalità, e un eventuale ``evaluation.output`` esplicito viene suffissato) e
usa una cache baseline separata (fingerprint diverso). Le celle
``baseline/*`` lo disattivano: sono già una
griglia esplicita di prompting e il dual le duplicherebbe. COSTO: l'eval
raddoppia (~25 min per passata a 5000 prompt); al primo giro la passata
complementare valuta anche la SUA baseline del base model.

**Partial-state resume (walltime-safe).** Su cluster condiviso il ritmo varia
fino a 6x senza preavviso e il JSON dei risultati esiste solo a fine passata:
un TIMEOUT azzera ore di generazione. Per questo ogni passata salva uno stato
parziale degli accumulatori (``resume_state_<soggetto>[__<mode>].json``) ogni
``evaluation.resume_every`` prompt (default 100) e, al rilancio, riprende dai
prompt già fatti dopo una validazione rigorosa del contesto (fingerprint,
metrics_version, campione deterministico, checkpoint). A passata completata
lo stato è rimosso; il JSON finale dichiara le riprese in ``resumed_from``.

Optionally generates plots via ``visualization.py`` (plotnine):
    - Completion length distribution (valid vs invalid)
    - Baseline vs Post-GRPO comparison
    - Reward component breakdown

Usage — TUTTI i knob comportamentali vivono nella sezione ``evaluation:`` del
config; l'unica superficie CLI è ``--config`` + ``--checkpoint``, che
identificano COSA valutare (non COME):

    # Single checkpoint eval (plot/compare/best_of_n/prompting/… dal config)
    python -m src.training.eval_t2g --config experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml --checkpoint path/to/ckpt

    # Compare baseline vs checkpoint — SAME decoding AND same prompting for
    # both (the baseline reuses the cell's config, incl. retrieval.enabled).
    # Sulle celle di training è il default: evaluation.compare: null =
    # deduzione automatica da training.output_dir, niente flag da passare.

    # Best-of-N selection (DIAGNOSTIC ONLY — oracle; reported separately):
    # evaluation.best_of_n: true nel config (richiede num_samples > 1).

    # Baseline-only eval (generates baseline JSON for later comparison):
    # dedotto automaticamente sulle celle eval-only (baseline/*), oppure
    # evaluation.eval_baseline_only: true.
    python -m src.training.eval_t2g --config experiments/configs/qwen25-05b/baseline/few-shot.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

# tqdm may not be available in all Apptainer containers on the cluster.
# Provide a no-op fallback so eval doesn't crash at import time —
# progress bars are a convenience, not a requirement.
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        """No-op fallback when tqdm is not installed."""
        return iterable if iterable is not None else iter(())


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
from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor
from src.models.model_loader import resolve_model_source
from src.rewards.t2g_rewards import initialize_rewards
from src.training.retrieval_setup import (
    build_train_retriever,
    retrieve_few_shot_batch,
)
from src.utils.config import load_config
from src.utils.live_status import live_status_reset, live_status_set
from src.utils.log_dedup import dedupe_library_loggers
from src.utils.metrics import (
    METRICS_VERSION,
    bleu_corpus,
    bleu_sentence,
    check_gloss_validity,
    chrf_score,
    compute_detailed_metrics,
    compute_evaluation_report,
    compute_pass_at_k,
    compute_reward_breakdown,
    corpus_chrf,
    corpus_gloss_f1,
    gloss_f1,
    non_copy_token_accuracy,
    rouge_l_score,
    seeded_sample_indices,
)
from src.utils.phase_timing import phase
from src.utils.prompting import SYSTEM_PROMPT, build_t2g_prompt
from src.utils.run_paths import split_checkpoint_path

logger = logging.getLogger("t2g-eval")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _sft_pretrain_adapter_sibling(checkpoint_path: Path) -> Path | None:
    """Locate the SFT-phase adapter for an ``sft_pretrain``-enabled GRPO run.

    ``grpo_t2g_train.py`` merges the SFT adapter into the base model
    IN-MEMORY ONLY before attaching a fresh LoRA for GRPO — the merge is
    never written to disk (see ``model_loader.py::_load_with_transformers`` /
    ``_load_with_unsloth``). ``trainer.save_model`` on the resulting PEFT
    model therefore writes only the GRPO adapter's delta, computed relative
    to a base+SFT it never records. Loading that adapter over the RAW base
    (as this function used to do unconditionally) silently drops the entire
    SFT contribution — the checkpoint that consumed most of the training
    wall-clock ends up not represented in the evaluated model at all.

    The SFT adapter DOES survive on disk: ``grpo_t2g_train.py`` always
    trains-or-copies it to ``<run_dir>/sft_pretrain/final/``, a sibling of
    the GRPO ``final/`` directory, specifically so the run stays
    self-contained. This looks for that sibling and returns it if present.
    """
    candidate = checkpoint_path.parent / "sft_pretrain" / "final"
    if (candidate / "adapter_config.json").exists():
        return candidate
    return None


def load_model_for_eval(
    checkpoint_path: str,
    base_model_name: str,
) -> tuple[Any, Any]:
    """Load a trained model for evaluation.

    Handles both full model checkpoints and PEFT/LoRA adapter checkpoints.
    For adapter checkpoints, loads the base model first, then merges adapters
    — including the SFT-phase adapter, when the checkpoint is an
    ``sft_pretrain``-enabled GRPO run (see :func:`_sft_pretrain_adapter_sibling`).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_path = Path(checkpoint_path)
    is_peft = (ckpt_path / "adapter_config.json").exists()
    sft_adapter_path = _sft_pretrain_adapter_sibling(ckpt_path) if is_peft else None

    logger.info(
        f"Loading model from {checkpoint_path} (is_peft={is_peft}, "
        f"sft_phase={'yes: ' + str(sft_adapter_path) if sft_adapter_path else 'no'})..."
    )

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
            # Local snapshot when cached — no hub HEAD retries on DNS-less
            # nodes (see src.models.model_loader.resolve_model_source).
            resolve_model_source(base_model_name),
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import PeftModel

        if sft_adapter_path is not None:
            logger.info(f"  Merging SFT-phase adapter first: {sft_adapter_path}")
            model = PeftModel.from_pretrained(model, str(sft_adapter_path))
            model = model.merge_and_unload()  # type: ignore[call-arg]

        model = PeftModel.from_pretrained(model, str(ckpt_path))
        model = model.merge_and_unload()  # type: ignore[call-arg]
        logger.info("  LoRA adapters merged and unloaded")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16,
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
# Cached-baseline reuse (compare mode)
# ---------------------------------------------------------------------------


def _config_prompting_mode(config: dict[str, Any]) -> str:
    """Prompting mode implied by the config: ``few-shot`` when retrieval is
    enabled, ``zero-shot`` otherwise.

    Unica fonte di verità per derivare la modalità dalla config — usata dalla
    risoluzione dei knob (``_resolve_prompting``) sia per il default
    (``evaluation.prompting: config``) sia per capire se un override
    dichiarato nel config cambia davvero qualcosa.
    """
    return (
        "few-shot" if config.get("retrieval", {}).get("enabled", False) else "zero-shot"
    )


def _prompt_context_fingerprint(
    config: dict[str, Any],
    num_samples: int,
    prompting: str | None = None,
) -> str:
    """Stable identity of the baseline eval context.

    The baseline depends only on: base model, dataset + seed, system
    prompt, few-shot retrieval settings, constrained-decoding setup and the
    decoding ``num_samples``.  Hashing these lets a cached
    ``eval_baseline.json`` be reused across SIBLING runs of the same model
    tag (the baseline is re-evaluated identically in every one of them) —
    while config changes that alter baseline generations (retrieval or
    grammar toggles, different num_samples, new system prompt) invalidate
    the cache and force a re-evaluation.

    ``prompting`` è la modalità effettiva della passata: entra nel payload
    SOLO quando differisce da quella implicita nella config (override
    dichiarato in ``evaluation.prompting`` o passata dual). I fingerprint
    delle run di default (e degli override ridondanti, che generano le stesse
    completions) restano byte-identici a quelli già calcolati, quindi la
    cache esistente sul cluster NON viene invalidata gratuitamente; una
    modalità diversa (es. il complemento zero-shot di una cella few-shot)
    produce un fingerprint diverso e forza la ricomputo — è la valida
    invalidazione della cache baseline al cambio di modalità richiesta dal
    dual eval.
    """
    grammar_cfg = config.get("grammar", {})
    config_mode = _config_prompting_mode(config)
    payload = {
        "version": 1,
        "model": config.get("model", {}).get("name"),
        "dataset_name": config.get("dataset", {}).get("dataset_name"),
        "seed": config.get("dataset", {}).get("seed", 42),
        "system_prompt": SYSTEM_PROMPT,
        "retrieval": config.get("retrieval", {}),
        "grammar": {
            "enabled": grammar_cfg.get("enabled", True),
        },
        "num_samples": num_samples,
    }
    if prompting in ("zero-shot", "few-shot") and prompting != config_mode:
        payload["prompting_override"] = prompting
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _try_cached_baseline(
    results_dir: Path,
    num_samples: int,
    max_samples: int | None,
    fingerprint: str,
    filename: str = "eval_baseline.json",
) -> tuple[dict[str, Any], list[dict[str, Any]] | None] | None:
    """Check ONE results dir for a compatible cached baseline JSON.

    Compatible = same metric definitions (``metrics_version``), same prompt
    context (``prompt_context_fingerprint``), same decoding
    (``num_completions_per_prompt``) and same prompt count (``max_samples``
    or full test set).  Returns the baseline (+ generations) or None.

    ``filename`` distingue le baseline per modalità di prompting: una passata
    con modalità diversa dal default (override dichiarato o passata dual)
    salva/rilegge ``eval_baseline__<mode>.json`` così i due cache non si
    sovrascrivono; la compatibilità resta comunque garantita dal fingerprint.
    """
    bl_path = results_dir / filename
    if not bl_path.exists():
        return None
    try:
        baseline = json.loads(bl_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if baseline.get("metrics_version") != METRICS_VERSION:
        return None  # computed with older metric definitions — recompute
    if baseline.get("prompt_context_fingerprint") != fingerprint:
        return None  # different prompting context (retrieval/grammar/…) — recompute
    if baseline.get("num_completions_per_prompt") != num_samples:
        return None  # different decoding — comparison would be unfair
    n_eval = baseline.get("num_samples_evaluated")
    if max_samples is not None:
        if n_eval != max_samples:
            return None
    elif n_eval != baseline.get("test_set_size"):
        return None  # cached was a subset, current wants the full test set

    generations: list[dict[str, Any]] | None = None
    gen_path = results_dir / filename.replace(
        "eval_baseline", "generations_baseline", 1
    )
    if gen_path.exists():
        try:
            generations = json.loads(gen_path.read_text(encoding="utf-8"))
        except Exception:
            generations = None
    return baseline, generations


def _load_cached_baseline(
    results_dir: Path,
    num_samples: int,
    max_samples: int | None,
    fingerprint: str,
    filename: str = "eval_baseline.json",
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, Path] | None:
    """Load a compatible cached baseline JSON — this run first,
    then SIBLING runs of the same model tag (newest first), then runs under
    OTHER model tags (cross-tag, fingerprint-guarded — ablation cells share
    the same base-model baseline).

    The base-model baseline is identical for every run of the same prompt
    context (same base model, dataset, prompting, decoding): re-evaluating
    it (~28 min GPU on 500 prompts) for each new training run is pure waste.
    Sibling reuse is guarded by the compatibility checks in
    :func:`_try_cached_baseline`.

    Args:
        results_dir: Run directory to search first.
        num_samples: Completions per prompt of the current eval.
        max_samples: Prompt count of the current eval (``None`` = full set).
        fingerprint: Prompt-context fingerprint of the current eval.
        filename: Baseline filename (mode-suffixed quando l'override CLI
            cambia la modalità di prompting).

    Returns:
        ``(baseline_results, generations_or_None, source_dir)`` or
        ``None`` when absent/stale/incompatible (caller re-evaluates).
    """
    # Current run dir first
    cached = _try_cached_baseline(
        Path(results_dir), num_samples, max_samples, fingerprint, filename
    )
    if cached is not None:
        return *cached, Path(results_dir)

    # Sibling run dirs of the same model tag, newest first
    parent = Path(results_dir).parent
    if parent.is_dir():
        siblings = sorted(
            (
                d
                for d in parent.iterdir()
                if d.is_dir() and d.name.startswith("run_") and d != Path(results_dir)
            ),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for d in siblings:
            cached = _try_cached_baseline(
                d, num_samples, max_samples, fingerprint, filename
            )
            if cached is not None:
                return *cached, d

    # Cross-tag: altre tag modello sotto experiments/results/. La baseline e'
    # lo STESSO base model (nessun peso addestrato) valutato con lo stesso
    # prompt context — ri-valutarla per ogni tag e' spreco (~28 GPU-min
    # ciascuna); il fingerprint del contesto in _try_cached_baseline
    # garantisce il match esatto.
    results_root = parent.parent
    if results_root.is_dir():
        for tag_dir in sorted(
            (t for t in results_root.iterdir() if t.is_dir() and t != parent),
            key=lambda t: t.stat().st_mtime,
            reverse=True,
        ):
            for d in sorted(
                (r for r in tag_dir.glob("run_*") if r.is_dir()),
                key=lambda r: r.stat().st_mtime,
                reverse=True,
            ):
                cached = _try_cached_baseline(
                    d, num_samples, max_samples, fingerprint, filename
                )
                if cached is not None:
                    return *cached, d
    return None


# ---------------------------------------------------------------------------
# Partial-state resume (walltime-safe eval)
# ---------------------------------------------------------------------------
# Su cluster condiviso il ritmo di generazione varia fino a 6x senza preavviso
# (1,86 s/prompt con GPU libera, 11,67 s/prompt con GPU contesa): un walltime
# calibrato sul caso migliore è una scommessa, e il JSON dei risultati viene
# scritto SOLO a fine passata — un TIMEOUT azzera ore di lavoro. Il rimedio è
# un checkpoint parziale degli accumulatori scritto durante la generazione e
# ricaricato al rilancio (il campione è deterministico: seeded_sample_indices
# con seed fisso restituisce sempre le stesse posizioni nello stesso ordine,
# quindi il resume per indice è sicuro).


# Marcatore di tipo dentro il file di stato: protegge dal caricare un JSON
# estraneo che per caso finisse sul path dello stato.
_RESUME_STATE_KIND = "eval_partial_state"
# Versione dello schema: un cambio di formato invalida gli stati vecchi.
_RESUME_STATE_SCHEMA_VERSION = 1
# Default di evaluation.resume_every: con 2000 prompt sono 20 scritture
# (pochi MB l'una, secondi in totale) su una passata di ore — il costo di I/O
# è trascurabile e la perdita massima in caso di kill è resume_every prompt.
_DEFAULT_RESUME_EVERY = 100
# Marcatore del soggetto valutato quando non c'è un checkpoint (base model).
_NO_CHECKPOINT_KEY = "__no_checkpoint__"


def _checkpoint_identity(checkpoint_path: str | None) -> str:
    """Identità del checkpoint valutato per la validazione dello stato.

    Il fingerprint del prompt context NON copre l'adattatore LoRA (descrive
    modello base + dataset + prompting + decoding): senza questo marcatore lo
    stato di un eval su ``final`` potrebbe essere ripreso da un eval su un
    checkpoint intermedio dello stesso contesto.
    """
    return str(checkpoint_path) if checkpoint_path else _NO_CHECKPOINT_KEY


def _resume_every(config: dict[str, Any]) -> int:
    """Cadenza di salvataggio dello stato parziale (``evaluation.resume_every``).

    Valori <= 0 disabilitano del tutto il meccanismo (né salvataggio né
    ripresa): è l'escape hatch per riprodurre il comportamento storico.
    """
    raw = config.get("evaluation", {}).get("resume_every", _DEFAULT_RESUME_EVERY)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "evaluation.resume_every=%r non è un intero: uso il default %d",
            raw,
            _DEFAULT_RESUME_EVERY,
        )
        return _DEFAULT_RESUME_EVERY


def _save_resume_state(
    path: Path,
    *,
    fingerprint: str,
    metrics_version: int,
    max_samples: int | None,
    num_samples: int,
    prompting_mode: str,
    checkpoint_key: str,
    test_set_size: int,
    sampled_count: int,
    resume_count: int,
    all_completions: list[list[str]],
    all_references: list[str],
    all_sample_ids: list[str],
    all_texts: list[str],
    all_difficulties: list[str],
) -> None:
    """Scrivi lo stato parziale della passata in modo ATOMICO.

    Il job muore per TIMEOUT in un istante arbitrario: la scrittura va su un
    file .tmp e poi os.replace() — un kill DURANTE la scrittura lascia il
    vecchio stato valido intatto, mai uno stato corrotto al suo posto. JSON
    con indent=2: pochi MB (2000 prompt x 5 completions), leggibile a mano
    in debug.
    """
    state = {
        "kind": _RESUME_STATE_KIND,
        "state_schema_version": _RESUME_STATE_SCHEMA_VERSION,
        "prompt_context_fingerprint": fingerprint,
        "metrics_version": metrics_version,
        "max_samples": max_samples,
        "num_samples": num_samples,
        "prompting_mode": prompting_mode,
        "checkpoint_key": checkpoint_key,
        "test_set_size": test_set_size,
        "sampled_count": sampled_count,
        "completed_prompts": len(all_references),
        "resume_count": resume_count,
        "accumulators": {
            "completions": all_completions,
            "references": all_references,
            "sample_ids": all_sample_ids,
            "texts": all_texts,
            "difficulties": all_difficulties,
        },
    }
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as e:
        # Un I/O NFS transienti non deve uccidere una passata di ore: si
        # perde al peggio la ripresa dal punto corrente, mai il lavoro.
        logger.warning("Could not save partial eval state %s: %s", path, e)
        return
    logger.info(
        "  eval partial state saved: %d prompts -> %s",
        len(all_references),
        path.name,
    )


def _load_resume_state(
    path: Path,
    *,
    fingerprint: str,
    metrics_version: int,
    max_samples: int | None,
    num_samples: int,
    prompting_mode: str,
    checkpoint_key: str,
    test_set_size: int,
    sampled_count: int,
    test_ds: Any,
) -> dict[str, Any] | None:
    """Carica lo stato parziale SE (e solo se) appartiene a questa valutazione.

    La validazione è volontariamente paranoidale: riprendere da uno stato di
    un contesto diverso produrrebbe metriche su un insieme misto di prompt,
    indistinguibile da un risultato valido. Qualunque controllo fallito ⇒
    lo stato è scartato (con il motivo in log) e la passata riparte da zero —
    sempre l'opzione sicura. Il file scartato viene rimosso: la presenza
    dello stato deve significare solo "questa passata è incompleta".

    Returns:
        Lo stato parsato, oppure ``None`` (assente o scartato).
    """

    def _discard(reason: str) -> None:
        logger.warning(
            "Discarding partial eval state %s: %s — restarting this pass "
            "from scratch.",
            path.name,
            reason,
        )
        try:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Could not remove discarded state %s: %s", path, e)

    if not path.is_file():
        return None  # assente: caso normale (primo avvio), nessun log
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        _discard(f"unreadable or corrupt file ({e})")
        return None
    if not isinstance(state, dict) or state.get("kind") != _RESUME_STATE_KIND:
        _discard(f"kind is not {_RESUME_STATE_KIND!r}")
        return None
    if state.get("state_schema_version") != _RESUME_STATE_SCHEMA_VERSION:
        _discard(
            f"state_schema_version {state.get('state_schema_version')!r} != "
            f"{_RESUME_STATE_SCHEMA_VERSION}"
        )
        return None
    if state.get("metrics_version") != metrics_version:
        _discard(
            f"metrics_version {state.get('metrics_version')!r} != {metrics_version}"
        )
        return None
    if state.get("max_samples") != max_samples:
        _discard(f"max_samples {state.get('max_samples')!r} != {max_samples!r}")
        return None
    if state.get("num_samples") != num_samples:
        _discard(f"num_samples {state.get('num_samples')!r} != {num_samples}")
        return None
    if state.get("prompting_mode") != prompting_mode:
        _discard(
            f"prompting_mode {state.get('prompting_mode')!r} != {prompting_mode!r}"
        )
        return None
    if state.get("checkpoint_key") != checkpoint_key:
        _discard(
            f"checkpoint {state.get('checkpoint_key')!r} != {checkpoint_key!r} "
            "(different checkpoint evaluated)"
        )
        return None
    if state.get("prompt_context_fingerprint") != fingerprint:
        _discard(
            "prompt_context_fingerprint mismatch (model/dataset/seed/system "
            "prompt/retrieval/grammar/decoding changed)"
        )
        return None
    if state.get("test_set_size") != test_set_size:
        _discard(
            f"test_set_size {state.get('test_set_size')!r} != {test_set_size} "
            "(dataset changed?)"
        )
        return None
    if state.get("sampled_count") != sampled_count:
        _discard(
            f"sampled_count {state.get('sampled_count')!r} != {sampled_count} "
            "(different deterministic sample)"
        )
        return None

    completed = state.get("completed_prompts")
    accs = state.get("accumulators")
    if not isinstance(accs, dict) or not isinstance(completed, int) or completed <= 0:
        _discard("missing or invalid accumulators/completed_prompts")
        return None
    acc_keys = ("completions", "references", "sample_ids", "texts", "difficulties")
    lengths = {k: len(accs[k]) if k in accs else None for k in acc_keys}
    if any(ln != completed for ln in lengths.values()):
        _discard(f"accumulator lengths incoherent: {lengths} (completed={completed})")
        return None
    if completed > sampled_count:
        _discard(f"completed_prompts {completed} > sampled_count {sampled_count}")
        return None
    # Ogni prompt ha esattamente num_samples completions (invariante del loop).
    if any(len(c) != num_samples for c in accs["completions"]):
        _discard("a saved prompt does not carry exactly num_samples completions")
        return None

    # ── Allineamento col campione deterministico ─────────────────────────
    # La protezione contro il guasto peggiore: se gli sample_id salvati non
    # coincidono con quelli che il campione produce alle stesse posizioni, lo
    # stato appartiene a un campione diverso e le metriche sarebbero calcolate
    # su prompt eterogenei — senza questo check, indistinguibili da un
    # risultato valido.
    for i, sid in enumerate(accs["sample_ids"]):
        expected = hashlib.sha256(
            str(test_ds[i]["text"]).encode("utf-8", errors="replace")
        ).hexdigest()
        if sid != expected:
            _discard(
                f"sample_id mismatch at position {i}: the state belongs to a "
                "different sample"
            )
            return None
    return state


def _remove_resume_state(path: Path | None) -> None:
    """Rimuovi lo stato parziale (e l'eventuale .tmp) a passata completata."""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
        Path(str(path) + ".tmp").unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Could not remove partial eval state %s: %s", path, e)


def _collect_completions_with_resume(
    *,
    test_ds: Any,
    tokenizer: Any,
    examples_batch: list[Any] | None,
    generate_fn: Any,
    num_samples: int,
    resume_state_path: Path | None,
    resume_every: int,
    fingerprint: str,
    metrics_version: int,
    max_samples: int | None,
    prompting_mode: str,
    checkpoint_key: str,
    test_set_size: int,
) -> tuple[
    list[list[str]],
    list[str],
    list[str],
    list[str],
    list[str],
    dict[str, Any],
]:
    """Loop di generazione con stato parziale: accumula le completions,
    salva ogni ``resume_every`` prompt e riprende da uno stato valido.

    ``generate_fn(prompt) -> list[str]`` è la unità di generazione (nel
    chiamante incapsula ``_generate_batch`` + reset del logits processor): è
    il punto di stub per i test. Gli accumulatori restano allineati per
    indice (la riga i-esima di ognuno appartiene allo stesso prompt).

    Returns:
        ``(all_completions, all_references, all_sample_ids, all_texts,
        all_difficulties, resume_info)`` con
        ``resume_info = {"resumed", "resume_count", "recovered_prompts"}``.
    """
    n_total = len(test_ds)
    all_completions: list[list[str]] = []
    all_references: list[str] = []
    all_sample_ids: list[str] = []
    all_texts: list[str] = []
    all_difficulties: list[str] = []

    # ── Ripresa ──────────────────────────────────────────────────────────
    start_idx = 0
    resume_count = 0
    if resume_state_path is not None and resume_every > 0:
        state = _load_resume_state(
            Path(resume_state_path),
            fingerprint=fingerprint,
            metrics_version=metrics_version,
            max_samples=max_samples,
            num_samples=num_samples,
            prompting_mode=prompting_mode,
            checkpoint_key=checkpoint_key,
            test_set_size=test_set_size,
            sampled_count=n_total,
            test_ds=test_ds,
        )
        if state is not None:
            accs = state["accumulators"]
            all_completions = [list(c) for c in accs["completions"]]
            all_references = list(accs["references"])
            all_sample_ids = list(accs["sample_ids"])
            all_texts = list(accs["texts"])
            all_difficulties = list(accs["difficulties"])
            start_idx = int(state["completed_prompts"])
            # Quante RIPRESE ha subito questa passata: entra nel JSON finale
            # (results["resumed_from"]) così chi legge sa che il risultato è
            # stato prodotto in più rilanci — stesso valore, provenienza nota.
            resume_count = int(state.get("resume_count", 0)) + 1
            logger.info("=" * 60)
            logger.info(
                "RESUMED partial eval state: %d/%d prompts recovered, "
                "%d remaining (resume #%d of this pass)",
                start_idx,
                n_total,
                n_total - start_idx,
                resume_count,
            )
            logger.info("=" * 60)
            # Il monitor esterno non deve mostrare 0/N finché non arriva il
            # primo tick periodico: lo stato live parte dalla posizione
            # recuperata.
            live_status_set(
                step=start_idx,
                total_steps=n_total,
                eval_progress=f"{start_idx}/{n_total}",
            )

    for idx, sample in enumerate(
        tqdm(test_ds, desc="Evaluating", initial=start_idx, total=n_total)
    ):
        # Ripresa: le posizioni già completate sono nei buffer caricati dallo
        # stato — si salta senza rigenerare né ricalcolare la difficoltà.
        if idx < start_idx:
            continue
        text = sample["text"]  # type: ignore[index]
        gold = sample["gloss"]  # type: ignore[index]

        # Difficulty from the GOLD gloss (same heuristic as the training
        # dataset builder): ≤5 tokens simple, ≤15 medium, else hard.
        n_gold_tokens = len(gold.strip().split())
        if n_gold_tokens <= 5:
            all_difficulties.append("simple")
        elif n_gold_tokens <= 15:
            all_difficulties.append("medium")
        else:
            all_difficulties.append("hard")

        # Periodic progress line for long runs (e.g. the full 8771-sample
        # test set) — visible in output.log/slurm logs even if the tqdm
        # bar is buffered/mangled. Parsable by chain_monitor.
        if (idx + 1) % 50 == 0:
            logger.info(
                "  eval progress: %d/%d (%.1f%%)",
                idx + 1,
                n_total,
                (idx + 1) / max(n_total, 1) * 100,
            )
            # Live status: eval progress (same cadence as the log line).
            live_status_set(
                step=idx + 1,
                total_steps=n_total,
                eval_progress=f"{idx + 1}/{n_total}",
            )

        # Build prompt with centralized template (same as training).
        # With retrieval enabled, examples mirror the GRPO few-shot prompts.
        prompt = build_t2g_prompt(
            text,
            tokenizer,
            examples=examples_batch[idx] if examples_batch is not None else None,
        )

        completions = generate_fn(prompt)

        # Store
        all_completions.append(completions)
        all_references.append(gold)
        all_texts.append(str(text))
        all_sample_ids.append(
            hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()
        )

        # Stato parziale periodico: il lavoro fatto fin qui sopravvive a un
        # kill per walltime. Quando la cadenza cade sull'ultimo prompt lo
        # stato copre anche la fase di metriche + JSON che segue (minuti):
        # riprenderla da zero costerebbe di nuovo tutta la generazione.
        if (
            resume_state_path is not None
            and resume_every > 0
            and (idx + 1) % resume_every == 0
        ):
            _save_resume_state(
                Path(resume_state_path),
                fingerprint=fingerprint,
                metrics_version=metrics_version,
                max_samples=max_samples,
                num_samples=num_samples,
                prompting_mode=prompting_mode,
                checkpoint_key=checkpoint_key,
                test_set_size=test_set_size,
                sampled_count=n_total,
                resume_count=resume_count,
                all_completions=all_completions,
                all_references=all_references,
                all_sample_ids=all_sample_ids,
                all_texts=all_texts,
                all_difficulties=all_difficulties,
            )

    return (
        all_completions,
        all_references,
        all_sample_ids,
        all_texts,
        all_difficulties,
        {
            "resumed": start_idx > 0,
            "resume_count": resume_count,
            "recovered_prompts": start_idx,
        },
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def _compute_primary_metrics(
    flat_completions: list[str],
    flat_references: list[str],
    all_completions: list[list[str]],
    all_references: list[str],
    token_to_idx: dict[str, int],
    bigram: Any,
    reward_weights: dict[str, float],
    flat_sources: list[str],
    n_bootstrap: int = 1000,
) -> tuple[
    dict[str, Any], list[float], list[tuple[bool, str]], list[float], list[float]
]:
    """Compute the honest primary metric block over all completions.

    Every metric here is averaged over *all* completions (not just the
    first one per prompt), so the numbers reflect the real decoding
    strategy.  ``pass_at_1`` is the fraction of prompts whose *first*
    completion reaches ROUGE-L >= 0.3 (a single honest draw) — computed
    with the shared ``compute_pass_at_k`` helper (k=1) for consistency
    with the Pass@k curve.  It never uses the gold to select among
    samples; the higher pass@N (N = num_samples) is reported under
    ``pass_at_k``.

    DUE STIMATORI pass@1 DIVERSI con lo stesso scopo (probabilità che una
    singola completion campionata superi la soglia) — ora con etichette
    distinte nell'output (docs/EVALUATION.md §2a):

    - ``pass_at_1`` (blocco primario): frazione di prompt la cui PRIMA
      completion raggiunge ROUGE-L >= 0.3 — un draw onesto per prompt, ed è
      l'ancora k=1 della curva Pass@k (che usa le prime k completions).
    - ``evaluation_report["pass_at_1"]["mean"]`` (blocco CI): media empirica
      dell'indicatore ROUGE-L >= 0.3 su TUTTE le completions (num_samples
      draw per prompt) — più efficiente (5x i campioni), con CI bootstrap.

    Nessuno dei due è lo stimatore combinatorio non distorto
    ``1 - C(n-c, k) / C(n, k)``: sono entrambe medie empiriche su
    sottoinsiemi diversi dello stesso campione e coincidono quando
    ``num_samples = 1``. I valori NON sono stati modificati: solo le
    etichette distinguono i due numeri.

    Args:
        flat_completions: All completions flattened (one entry per
            completion across all prompts).
        flat_references: Gold reference per completion (same order and
            length as ``flat_completions``).
        all_completions: Nested completions (one list per prompt).
        all_references: Gold reference per prompt (one per prompt).
        token_to_idx: Gloss token → index mapping for bigram scoring.
        bigram: Bigram transition matrix.
        reward_weights: Reward weight map (weight > 0 ⇒ computed).
        flat_sources: Testo inglese sorgente per completion (stesso ordine
            e lunghezza di ``flat_completions``). Serve alla metrica primaria
            ``non_copy_token_accuracy``: senza il source non si possono
            distinguere i token di reference copiabili (uppercase del source)
            da quelli non banali. È SEMPRE disponibile in ``evaluate_checkpoint``
            (colonna ``text`` del dataset, propagata come ``flat_texts``): la
            metrica viene calcolata senza fallback silenziosi.
        n_bootstrap: Bootstrap resamples for the evaluation report CIs.

    Returns:
        Tuple of ``(metrics, rouge_scores, validity, bleu_scores,
        chrf_scores)`` where *metrics* is the primary metric dict,
        *rouge_scores*/*bleu_scores*/*chrf_scores* are the per-completion
        score lists, and *validity* is the per-completion ``(is_valid,
        reason)`` list.
    """
    # ── Per-completion quality metrics ──────────────────────────────────
    rouge_scores = [
        rouge_l_score(c, r) for c, r in zip(flat_completions, flat_references)
    ]
    bleu_scores = [
        bleu_sentence(c, r) for c, r in zip(flat_completions, flat_references)
    ]
    chrf_scores = [chrf_score(c, r) for c, r in zip(flat_completions, flat_references)]
    gloss_f1_scores = [
        gloss_f1(c, r) for c, r in zip(flat_completions, flat_references)
    ]

    # Validity (over ALL completions)
    validity: list[tuple[bool, str]] = [
        check_gloss_validity(c) for c in flat_completions
    ]
    valid_count = sum(1 for v, _ in validity if v)
    validity_rate = valid_count / max(len(flat_completions), 1)
    error_counts = Counter(err for _, err in validity if err)

    # Bigram log-prob & exact match over ALL completions (not just the first)
    bigram_scores: list[float] = []
    exact_matches: list[int] = []
    for c, r in zip(flat_completions, flat_references):
        tokens = c.split()
        indices = [token_to_idx.get(t, token_to_idx.get("<UNK>", 0)) for t in tokens]
        if len(indices) >= 2:
            bos = token_to_idx.get("<BOS>", 0)
            eos = token_to_idx.get("<EOS>", 1)
            bigram_scores.append(
                sequence_score_bigram(bigram, [bos] + indices + [eos]),
            )
        else:
            bigram_scores.append(-10.0)
        exact_matches.append(1 if c == r.strip() else 0)

    # Metrica primaria copy-insensitive: SOLO i token di reference non
    # ottenibili uppercaseando il source (case-sensitive, a multinsieme).
    # Separa "copia l'inglese" da "produce gloss": il ~62% del gloss e' il
    # source maiuscolizzato (docs/EVALUATION.md §2). Denominatore sempre
    # riportato: sui 2000 prompt storici erano 9350 posizioni non banali.
    non_copy_acc, non_copy_hits, non_copy_total = non_copy_token_accuracy(
        flat_completions, flat_sources, flat_references
    )

    # Pass@1 / Pass@k via the shared helper (k=1 for pass@1)
    pass_at_1 = compute_pass_at_k(
        all_completions, all_references, k_values=(1,), threshold=0.3
    )["pass@1"]
    n_per_prompt = len(all_completions[0]) if all_completions else 0
    passk_full: dict[str, float] = {}
    if n_per_prompt > 1:
        passk_full = compute_pass_at_k(
            all_completions,
            all_references,
            k_values=tuple(range(1, min(n_per_prompt + 1, 11))),
            threshold=0.3,
        )

    # Detailed metrics / reward breakdown / comprehensive report
    detailed = compute_detailed_metrics(flat_completions, flat_references)
    reward_components = compute_reward_breakdown(
        flat_completions,
        references=flat_references,
        reward_weights=reward_weights,
    )
    # Valori corpus-level calcolati UNA volta e riusati dal report: stessa
    # funzione sullo stesso input, ricalcolare raddoppiava le passate corpus
    # (e l'avviso sacrebleu sui dati tokenizzati).
    bleu_corpus_value = bleu_corpus(flat_completions, flat_references)
    chrf_corpus_value = corpus_chrf(flat_completions, flat_references)
    gloss_f1_micro_value = corpus_gloss_f1(flat_completions, flat_references)["micro"]
    eval_report = compute_evaluation_report(
        flat_completions,
        flat_references,
        n_bootstrap=n_bootstrap,
        corpus_bleu_score=bleu_corpus_value,
        corpus_chrf_score=chrf_corpus_value,
        gloss_f1_micro_score=gloss_f1_micro_value,
    )

    rouge_mean = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    results: dict[str, Any] = {
        "rouge_l_mean": rouge_mean,
        "rouge_l_std": float(np.std(rouge_scores)) if rouge_scores else 0.0,
        "rouge_l_median": float(np.median(rouge_scores)) if rouge_scores else 0.0,
        # Valid ROUGE-L: rouge_l_mean × validity_rate (penalizza output
        # invalidi). PRESERVATO com'e': ogni eval_final.json storico
        # (metrics_version 2) usa questa forma prodotto; la definizione
        # corretta per riga e' emessa sotto come valid_rouge_l_mean_v3.
        "valid_rouge_l_mean": rouge_mean * validity_rate,
        # Definizione corretta: media per riga di (rouge if valid else 0);
        # la forma prodotto scala anche le righe valide, sottovalutando la
        # qualita'. Misurato sulle generazioni salvate: few-shot base
        # 0.4581 -> 0.4645, zero-shot+grammar 0.1243 -> 0.1335. Chiave
        # separata per non toccare la comparabilita' v2.
        "valid_rouge_l_mean_v3": (
            float(
                np.mean(
                    [
                        score if is_valid else 0.0
                        for score, (is_valid, _) in zip(rouge_scores, validity)
                    ]
                )
            )
            if rouge_scores
            else 0.0
        ),
        "bleu_sentence_mean": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        "bleu_corpus": bleu_corpus_value,
        "chrf_sentence_mean": float(np.mean(chrf_scores)) if chrf_scores else 0.0,
        "chrf_corpus": chrf_corpus_value,
        "gloss_f1_sentence_mean": (
            float(np.mean(gloss_f1_scores)) if gloss_f1_scores else 0.0
        ),
        "gloss_f1_micro": gloss_f1_micro_value,
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        # ── Metrica primaria copy-insensitive + denominatore ──────────────
        # Accanto a exact_match: le due metriche primarie dichiarate in
        # PRIMARY_METRICS (src/utils/metrics.py). hits/total consentono di
        # giudicare la stabilità del valore e di ri-aggregarlo su sottogruppi.
        "non_copy_token_accuracy": non_copy_acc,
        "non_copy_token_hits": non_copy_hits,
        "non_copy_token_total": non_copy_total,
        "bigram_log_prob_mean": (
            float(np.mean(bigram_scores)) if bigram_scores else 0.0
        ),
        "bigram_log_prob_std": (float(np.std(bigram_scores)) if bigram_scores else 0.0),
        "validity_rate": validity_rate,
        "valid_count": valid_count,
        "invalid_count": len(flat_completions) - valid_count,
        "pass_at_1": pass_at_1,
        "reward_breakdown": reward_components,
        "detailed_metrics": detailed,
        "error_distribution": dict(error_counts.most_common(20)),
        "total_completions": len(flat_completions),
        # ── Comprehensive report (BLEU + chrF + gloss F1 + bootstrap CI 95%) ──
        "evaluation_report": eval_report,
    }
    if passk_full:
        results["pass_at_k"] = passk_full

    # bleu/chrf per-completion lists are returned for the distribution
    # plots (they are NOT stored in results — they would bloat the JSON).
    return results, rouge_scores, validity, bleu_scores, chrf_scores


def _select_best_of_n(
    all_completions: list[list[str]],
    all_references: list[str],
) -> list[str]:
    """Select the best completion per prompt (ORACLE — uses the gold).

    Picks the completion with the highest ROUGE-L among valid ones; if
    none is valid, the highest ROUGE-L overall.

    WARNING: this uses the gold reference, so it is NOT deployable.  It
    exists purely to quantify the headroom that best-of-N sampling could
    unlock with a perfect reranker (see ``oracle_best_of_n`` in the
    results JSON).
    """
    selected: list[str] = []
    for comps, gold in zip(all_completions, all_references):
        scored = [(rouge_l_score(c, gold), check_gloss_validity(c), c) for c in comps]
        # Prefer valid completions; among those, pick highest ROUGE-L.
        valid_scored = [(rl, c) for rl, (v, _), c in scored if v]
        if valid_scored:
            selected.append(max(valid_scored, key=lambda x: x[0])[1])
        else:
            # No valid completion — pick highest ROUGE-L overall.
            selected.append(max(scored, key=lambda x: x[0])[2])
    return selected


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | None,
    max_samples: int | None = None,
    num_samples: int = 1,
    best_of_n: bool = False,
    prompting_mode: str | None = None,
    prompting_source: str | None = None,
    resume_state_path: str | Path | None = None,
) -> tuple[
    dict[str, Any],
    list[str],
    list[tuple[bool, str]],
    list[str],
    list[float],
    list[dict[str, Any]],
    list[float],
    list[float],
]:
    """Evaluate a checkpoint on the test set with full metrics.

    Args:
        config: Parsed YAML config.
        checkpoint_path: Path to the checkpoint directory, or ``None`` to
            evaluate the base model without trained weights (no LoRA).
        max_samples: Max test samples to evaluate, or ``None`` for the full
            test set.  When set, a *seeded random sample* of the test set
            is used (seed from ``dataset.seed``), never the first N.
        num_samples: Number of completions per prompt (1 = greedy, >1 = sampled).
        best_of_n: If True and num_samples > 1, select the best completion
            per prompt (highest ROUGE-L among valid ones). **Oracle**: uses
            the gold reference, so it is NOT deployable.  Results are
            reported in a separate ``oracle_best_of_n`` block and never
            overwrite the primary metrics.
        prompting_mode: Modalità di prompting effettiva (``zero-shot`` /
            ``few-shot``). ``None`` (default, comportamento legacy) la deriva
            da ``retrieval.enabled`` nella config. Un valore esplicito SOVRASCRIVE
            la config: ``zero-shot`` forza il retriever a None, ``few-shot``
            forza il retrieval attivo.
        prompting_source: Provenienza della modalità (``config`` = derivata
            da ``retrieval.enabled``, ``config-override`` = forzata da
            ``evaluation.prompting``, ``config-dual`` = passata complementare
            del dual eval), stampata nei risultati accanto alla modalità.
        resume_state_path: Path dello stato parziale per il resume da
            walltime (``None`` = comportamento storico, nessun salvataggio
            né ripresa). Con ``evaluation.resume_every <= 0`` il meccanismo
            è disattivato anche quando il path è fornito.

    Returns:
        Tuple of ``(results, flat_completions, validity, all_references,
        rouge_scores, generations, bleu_scores, chrf_scores)`` where
        *results* is a dict with all computed metrics (primary block over
        ALL completions, plus an optional ``oracle_best_of_n`` sub-block),
        *flat_completions* is a list of generated gloss strings, *validity*
        is a list of ``(is_valid, reason)`` tuples, *all_references* is the
        list of gold glosses (one per prompt), *rouge_scores*,
        *bleu_scores* and *chrf_scores* are the per-completion score lists
        (fed to the distribution plots), and *generations* is a list of
        per-completion dicts (text/gold/completion/valid/rouge_l) suitable
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

    # ── Optional few-shot retrieval (same strategy as GRPO training) ─────
    # Stessi esempi top_k (text→gloss) dal TRAIN split: baseline e checkpoint
    # condividono lo stesso retriever, cosi' --compare resta equo (stesso
    # decoding E stesso prompting). Copia locale della sezione retrieval per
    # NON mutare la config condivisa; zero-shot forza enabled=False, few-shot
    # True (None/config = comportamento legacy).
    retrieval_cfg = dict(config.get("retrieval", {}))
    if prompting_mode == "zero-shot":
        retrieval_cfg["enabled"] = False
    elif prompting_mode == "few-shot":
        retrieval_cfg["enabled"] = True
    if prompting_mode is None:
        prompting_mode = _config_prompting_mode(config)
    if prompting_source is None:
        prompting_source = "config"
    retriever = build_train_retriever(
        dataset,
        retrieval_cfg,
        seed=ds_cfg.get("seed", 42),
    )
    top_k = int(retrieval_cfg.get("top_k", 3))
    max_self_similarity = float(retrieval_cfg.get("max_self_similarity", 0.98))
    if retriever is not None:
        logger.info(
            "Few-shot retrieval enabled: backend=%s, top_k=%d, "
            "max_self_similarity=%.2f (examples from TRAIN split, "
            "query excluded)",
            retriever.backend,
            top_k,
            max_self_similarity,
        )

    initialize_rewards(
        bigram,
        vocab,
    )
    token_to_idx = {t: i for i, t in enumerate(vocab)}

    # ── Load model ───────────────────────────────────────────────────────
    if checkpoint_path is None:
        # Nessun checkpoint: si valuta il base model senza pesi addestrati.
        # "zero-shot" indicherebbe l'assenza di checkpoint MA e' anche il nome
        # della modalità prompting: nel log si usa "no-checkpoint".
        logger.info(
            "No-checkpoint mode: loading base model %s (prompting: %s)",
            config["model"]["name"],
            prompting_mode,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Offline-first: risolve l'hub id allo snapshot LOCALE se in cache.
        # transformers 5.3 chiama model_info() (rete) per id non locali — su
        # nodi senza DNS aveva CRASHATO l'eval (slurm-eval-7077); un path
        # locale cortocircuita il check e salta i retry HEAD.
        base_src = resolve_model_source(config["model"]["name"])
        if base_src != config["model"]["name"]:
            logger.info(f"  Using cached snapshot: {base_src}")

        tokenizer = AutoTokenizer.from_pretrained(
            base_src,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            base_src,
            dtype=torch.float16,
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
        gloss_mask = GlossVocabularyMask(vocab, tokenizer)
        logits_processor = GlossVocabularyLogitsProcessor(
            gloss_mask,
            device=str(model.device),
        )
    else:
        logger.info("⚠️  grammar.enabled=false — unconstrained generation (ablation)")
        logits_processor = None

    # ── Prepare test samples (seeded sampling, never the first N) ────────
    test_ds = dataset["test"]
    test_set_size = len(test_ds)
    sample_indices = seeded_sample_indices(
        test_set_size, max_samples, seed=ds_cfg.get("seed", 42)
    )
    logger.info(
        "Evaluating %d/%d samples (seeded sample)", len(sample_indices), test_set_size
    )
    test_ds = test_ds.select(sample_indices)

    # Live status: the eval phase begins (progress updated in the loop).
    live_status_set(
        phase="eval",
        step=0,
        total_steps=len(test_ds),
        note=f"eval {len(test_ds)} sample",
    )

    # Pre-compute few-shot examples for the selected test samples using the
    # SAME per-query anti-leakage as GRPO training (exclude the query's own
    # normalized text; drop near-duplicates above max_self_similarity).
    examples_batch = (
        retrieve_few_shot_batch(
            retriever,
            [str(s["text"]) for s in test_ds],
            top_k,
            max_self_similarity,
        )
        if retriever is not None
        else None
    )

    do_sample = num_samples > 1
    if best_of_n and num_samples <= 1:
        logger.warning(
            "best_of_n=True but num_samples=%d (need >1). Disabling best_of_n.",
            num_samples,
        )
        best_of_n = False
    if best_of_n:
        logger.info(
            "Best-of-N enabled: generating %d samples/prompt and selecting the "
            "best per prompt (highest ROUGE-L among valid). NOTE: this is an "
            "ORACLE selection (uses the gold reference) — results are reported "
            "in a separate 'oracle_best_of_n' block and never override the "
            "primary metrics.",
            num_samples,
        )

    # ── Collect completions (con stato parziale per il resume) ───────────
    # Multi-sample: list[list[str]] per prompt (always nested for consistency).
    # Gli accumulatori sono allineati per indice e vivono nel checkpoint
    # parziale: una passata interrotta e ripresa li ricostituisce identici.
    fingerprint = _prompt_context_fingerprint(config, num_samples, prompting_mode)

    # Unità di generazione stubbabile nei test: incapsula _generate_batch e
    # il reset del logits processor (la temperature non dipende dal prompt).
    temp = 0.7 if do_sample else 1.0  # greedy ignores temperature

    def _generate_one(prompt: str) -> list[str]:
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
        return completions

    (
        all_completions,
        all_references,
        all_sample_ids,
        all_texts,
        all_difficulties,
        resume_info,
    ) = _collect_completions_with_resume(
        test_ds=test_ds,
        tokenizer=tokenizer,
        examples_batch=examples_batch,
        generate_fn=_generate_one,
        num_samples=num_samples,
        resume_state_path=Path(resume_state_path) if resume_state_path else None,
        resume_every=_resume_every(config),
        fingerprint=fingerprint,
        metrics_version=METRICS_VERSION,
        max_samples=max_samples,
        prompting_mode=prompting_mode,
        checkpoint_key=_checkpoint_identity(checkpoint_path),
        test_set_size=test_set_size,
    )

    # Flatten completions for per-completion metrics. For num_samples=1 there
    # is 1 completion per prompt; for num_samples>1 ALL completions are scored
    # individually — primary metrics are honest averages over every completion.
    flat_completions = [c for comps in all_completions for c in comps]
    flat_sample_ids = [
        sid for i, sid in enumerate(all_sample_ids) for _ in all_completions[i]
    ]
    flat_texts = [txt for i, txt in enumerate(all_texts) for _ in all_completions[i]]
    flat_references = [
        ref for i, ref in enumerate(all_references) for _ in all_completions[i]
    ]

    # Reward weight map — only components with weight > 0 are computed.
    rewards_cfg = config.get("reward", {})
    reward_weight_map = {
        "translation_quality_reward": rewards_cfg.get("weight_translation", 0.0),
        "bleu_reward": rewards_cfg.get("weight_bleu", 0.0),
        "gold_structure_reward": rewards_cfg.get("weight_gold_structure", 0.0),
        "verifier_scaled_reward": rewards_cfg.get("weight_verifier_scaled", 0.0),
        "gloss_order_reward": rewards_cfg.get("weight_gloss_order", 0.0),
        "gloss_format_reward": rewards_cfg.get("weight_format", 0.0),
        "gloss_repetition_reward": rewards_cfg.get("weight_repetition", 0.0),
        "edit_validity_reward": rewards_cfg.get("weight_edit_validity", 0.0),
    }

    # Ripetizioni bootstrap del blocco CI: è il default storico di
    # _compute_primary_metrics, reso esplicito qui solo per dichiararlo nel
    # log di fase (il valore non cambia).
    n_bootstrap_resamples = 1000

    # ── Primary metrics (honest, averaged over ALL completions) ─────────
    # Fase piu' lunga dell'eval: il messaggio esce PRIMA del lavoro e la
    # barra tqdm del bootstrap copre le ripetizioni.
    with phase(
        "Computing per-completion metrics + bootstrap CIs",
        detail=(
            f"{len(flat_completions)} completions, "
            f"5 bootstrap CIs x {n_bootstrap_resamples} resamples"
        ),
    ):
        results, rouge_scores, validity, bleu_scores, chrf_scores = (
            _compute_primary_metrics(
                flat_completions,
                flat_references,
                all_completions,
                all_references,
                token_to_idx=token_to_idx,
                bigram=bigram,
                reward_weights=reward_weight_map,
                # flat_texts = colonna "text" del dataset espansa per
                # completion (stessa espansione di flat_references): il source
                # richiesto da non_copy_token_accuracy, senza fallback.
                flat_sources=flat_texts,
                n_bootstrap=n_bootstrap_resamples,
            )
        )
    results["num_samples_evaluated"] = len(all_references)
    results["test_set_size"] = test_set_size
    results["num_completions_per_prompt"] = num_samples
    results["best_of_n"] = best_of_n
    # Modalità di prompting effettiva + provenienza (dalla config o forzata
    # da CLI): senza questo stamp due run con prompting diverso producono
    # JSON indistinguibili — il difetto peggiore in un contesto sperimentale.
    results["prompting"] = {
        "mode": prompting_mode,
        "source": prompting_source,
    }
    # Stamp the metric definitions used — cached-baseline reuse (see
    # _load_cached_baseline) refuses to reuse results computed with a
    # different version (e.g. pre corpus-BLEU-fix numbers).
    results["metrics_version"] = METRICS_VERSION
    # Stamp the prompting context (model/dataset/system prompt/retrieval/
    # grammar/decoding, più l'eventuale override CLI) — guards sibling-run
    # baseline reuse against config changes that would alter baseline
    # generations.
    results["prompt_context_fingerprint"] = fingerprint
    results["decoding"] = {
        "do_sample": do_sample,
        "temperature": (0.7 if do_sample else None),
        "num_samples": num_samples,
    }
    # Provenienza del risultato: una passata ripresa da uno stato parziale
    # produce le STESSE metriche di una completata in un colpo solo (gli
    # accumulatori sono identici), ma chi legge il JSON deve saperlo —
    # "resumed_from" dichiara se c'è stata una ripresa, quante volte e quanti
    # prompt sono stati recuperati all'avvio dell'ultima.
    results["resumed_from"] = {
        "resumed": resume_info["resumed"],
        "resume_count": resume_info["resume_count"],
        "recovered_prompts": resume_info["recovered_prompts"],
    }

    # ── Per-difficulty breakdown ────────────────────────────────────────
    # Aggregates the headline metrics per gold-difficulty level (same
    # heuristic as the training dataset: gold token count). Small dict —
    # safe to serialize into eval_*.json and feeds the difficulty plot.
    if all_difficulties and len(all_difficulties) == len(all_references):
        flat_difficulties = [d for d in all_difficulties for _ in range(num_samples)]
        breakdown: dict[str, dict[str, float]] = {}
        for level in ("simple", "medium", "hard"):
            idxs = [i for i, d in enumerate(flat_difficulties) if d == level]
            if not idxs:
                continue
            level_rouge = [rouge_scores[i] for i in idxs]
            level_bleu = [bleu_scores[i] for i in idxs]
            level_chrf = [chrf_scores[i] for i in idxs]
            level_valid = [validity[i][0] for i in idxs]
            breakdown[level] = {
                "n_prompts": sum(1 for d in all_difficulties if d == level),
                "rouge_l_mean": float(np.mean(level_rouge)),
                "bleu_sentence_mean": float(np.mean(level_bleu)),
                "chrf_sentence_mean": float(np.mean(level_chrf)),
                "validity_rate": float(np.mean(level_valid)),
            }
        # Pass@1 per difficulty: needs the nested completions grouped by
        # prompt — walk prompts and test the FIRST completion only.
        if all_completions and len(all_completions) == len(all_references):
            for level in breakdown:
                first_rouge = [
                    rouge_l_score(comps[0], ref)
                    for comps, ref, d in zip(
                        all_completions, all_references, all_difficulties
                    )
                    if d == level and comps
                ]
                if first_rouge:
                    breakdown[level]["pass_at_1"] = float(
                        np.mean([r >= 0.3 for r in first_rouge])
                    )
        if breakdown:
            results["difficulty_breakdown"] = breakdown
            logger.info(
                "Per-difficulty breakdown: %s",
                {k: round(v["rouge_l_mean"], 3) for k, v in breakdown.items()},
            )

    # ── Oracle best-of-N (separate block, never overrides the primary) ──
    if best_of_n and num_samples > 1:
        selected = _select_best_of_n(all_completions, all_references)
        oracle_metrics, _, _, _, _ = _compute_primary_metrics(
            selected,
            list(all_references),
            [[c] for c in selected],
            all_references,
            token_to_idx=token_to_idx,
            bigram=bigram,
            reward_weights=reward_weight_map,
            # Una completion selezionata per prompt: il source allineato è
            # all_texts (uno per prompt), stessa lunghezza di selected.
            flat_sources=list(all_texts),
        )
        oracle_metrics["num_samples_evaluated"] = len(all_references)
        oracle_metrics["num_completions_per_prompt"] = num_samples
        oracle_metrics["note"] = (
            "Oracle best-of-N: the gold reference was used to select the best "
            "completion per prompt (highest ROUGE-L among valid). NOT deployable "
            "— reported separately and never part of the primary metrics."
        )
        results["oracle_best_of_n"] = oracle_metrics
        logger.info(
            "Oracle best-of-N: selected 1 of %d completions per prompt (%d total). "
            "Reported under 'oracle_best_of_n' only.",
            num_samples,
            len(selected),
        )

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

    # Live status: generation loop done (metrics computation follows, which
    # is fast) — clear the phase so the monitor doesn't show a stale eval.
    live_status_reset(note="eval completato")

    return (
        results,
        flat_completions,
        validity,
        all_references,
        rouge_scores,
        generations,
        bleu_scores,
        chrf_scores,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Risoluzione knob eval — SOLO dal config (nessun flag CLI comportamentale)
# ---------------------------------------------------------------------------


def _complement_prompting(mode: str) -> str:
    """Modalità di prompting complementare (per la passata dual).

    zero-shot ↔ few-shot: il dual eval valuta la cella in ENTRAMBE le
    modalità per separare "il modello ha interiorizzato la mappatura" da
    "dipende dal prompt come stampella".
    """
    return "few-shot" if mode == "zero-shot" else "zero-shot"


def _resolve_prompting(
    eval_cfg: dict[str, Any], config: dict[str, Any]
) -> tuple[str, str, bool]:
    """Risolvi la modalità di prompting dell'eval dalla sezione ``evaluation``.

    Fonte unica della verità per la coppia (modalità, provenienza):

    - ``evaluation.prompting: config`` (default storico) → modalità derivata
      da ``retrieval.enabled`` (``_config_prompting_mode``), provenienza
      ``config``.
    - ``evaluation.prompting: zero-shot|few-shot`` → modalità forzata. Se
      coincide con quella derivata è un override ridondante (stesse
      completions del default: nessun suffisso file, nessuna invalidazione di
      cache); se differisce è un override vero (provenienza
      ``config-override``, suffisso ``__<mode>`` sui file e fingerprint che
      invalida la cache della baseline).

    Returns:
        ``(prompting_mode, prompting_source, prompting_changed)``.
    """
    declared = eval_cfg.get("prompting", "config")
    config_mode = _config_prompting_mode(config)
    if declared in ("zero-shot", "few-shot"):
        source = "config" if declared == config_mode else "config-override"
        return declared, source, declared != config_mode
    return config_mode, "config", False


def _deduce_eval_modes(
    eval_cfg: dict[str, Any],
    has_checkpoint: bool,
    has_output_dir: bool,
) -> tuple[bool, bool]:
    """Deduci ``eval_baseline_only`` / ``compare`` quando non dichiarati.

    Replica la deduzione che storicamente viveva in cluster/eval.sh (dalla
    presenza di ``training.output_dir``), così le celle baseline/* — eval-only
    — continuano a funzionare senza dichiarare nulla:

    - cella di training (``training.output_dir``) o checkpoint esplicito →
      ``compare`` (baseline base-model cachata + checkpoint);
    - eval-only (niente output_dir E niente checkpoint) → solo base model.

    Un valore esplicito nel config (``compare`` / ``eval_baseline_only``) è
    sovrascrivibile e vince sulla deduzione automatica.

    Returns:
        ``(eval_baseline_only, do_compare)``.
    """
    declared_only = eval_cfg.get("eval_baseline_only")
    if declared_only is not None:
        baseline_only = bool(declared_only)
    else:
        baseline_only = not has_checkpoint and not has_output_dir
    declared_compare = eval_cfg.get("compare")
    if declared_compare is not None:
        do_compare = bool(declared_compare)
    else:
        do_compare = not baseline_only
    return baseline_only, do_compare


def _checkpoint_completeness_stamp(checkpoint_path: str | None) -> dict[str, Any]:
    """Marchio di incompletezza del checkpoint valutato (dict da fondere nei
    risultati; vuoto = nessuno stamp).

    ``final`` viene scritto SOLO a training completato (grpo_t2g_train.py:
    ``save_model`` su ``output_dir/final``); i checkpoint intermedi si
    chiamano ``checkpoint-<step>``. Un eval su un ``checkpoint-<N>`` misura
    quindi un modello PARZIALE (training interrotto, TIMEOUT, o ancora in
    corso): senza marchio il suo eval_final.json sarebbe indistinguibile da
    quello di un modello completo — l'errore peggiore in una campagna
    comparativa. La presenza dei campi nel JSON (e quindi in wandb e in
    comparison.json) rende il risultato auto-descrittivo e greppabile;
    l'ASSENZA dello stamp equivale a "checkpoint completo o base model".
    """
    if not checkpoint_path:
        return {}
    match = re.fullmatch(r"checkpoint-(\d+)", Path(checkpoint_path).name)
    if match is None:
        return {}
    return {"checkpoint_incomplete": True, "checkpoint_step": int(match.group(1))}


def _run_eval_pass(
    config: dict[str, Any],
    checkpoint_arg: str | None,
    *,
    config_path: str | None = None,
    eval_baseline_only: bool,
    do_compare: bool,
    plot: bool,
    best_of_n: bool,
    max_samples: int | None,
    num_samples: int,
    force_baseline_eval: bool,
    output_override: str | None,
    baseline_pass_at1: float | None,
    baseline_json: str | None,
    prompting_mode: str,
    prompting_source: str,
    prompting_changed: bool,
    model_tag_default: str,
    results_dir: Path,
    figures_dir: Path,
    logs_dir: Path,
) -> None:
    """Esegui UNA passata di eval completa (eval + JSON + figure + wandb).

    Estratta dal vecchio ``main()`` per il dual prompting: la seconda passata
    riesegue la STESSA pipeline con la modalità complementare — l'equivalente
    in-processo delle due invocazioni che un tempo cluster/eval.sh lanciava
    con ``DUAL_EVAL=1``. I knob comportamentali arrivano dal chiamante (che
    li ha risolti dal config); le directory di output sono condivise e la
    modalità diversa produce file con suffisso ``__<mode>``.
    """
    # Il tag identifica la passata nei print/figure/wandb: la modalità
    # baseline-only ridefinisce il tag (storico: eval del SOLO base model).
    model_tag = "baseline" if eval_baseline_only else model_tag_default

    logger.info("=" * 60)
    if prompting_changed:
        logger.info(
            "EVAL PASS — prompting: %s (source: %s) — differs from the "
            "config-implied mode: output files carry the __%s suffix",
            prompting_mode,
            prompting_source,
            prompting_mode,
        )
    else:
        logger.info(
            "EVAL PASS — prompting: %s (source: %s)",
            prompting_mode,
            prompting_source,
        )
    logger.info("=" * 60)

    # compare implica plot (storico: il confronto senza figure era mezzo output)
    if do_compare:
        plot = True

    # ── Set random seeds for reproducibility ─────────────────────────────
    # Riseminati PER PASSATA (non una sola volta in main): le due passate del
    # dual partono così dalle stesse condizioni random e sono riproducibili
    # indipendentemente l'una dall'altra.
    seed = config["dataset"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Reproducibility: seed={seed} (random, numpy, torch, cuda)")

    # ── Determine eval mode ─────────────────────────────────────────────
    # Tre modalita' (dedotte in main() da training.output_dir / checkpoint /
    # evaluation.compare / evaluation.eval_baseline_only): 1. eval-baseline-only,
    # 2. compare (baseline + checkpoint + confronto), 3. single.
    # Suffisso __<mode> sui file di output SOLO quando la modalità effettiva
    # differisce da quella implicita nella config: due eval della stessa cella
    # in modalità diverse non si sovrascrivono; default byte-compatibile.
    prompting_suffix = f"__{prompting_mode}" if prompting_changed else ""
    baseline_filename = f"eval_baseline{prompting_suffix}.json"
    baseline_fingerprint = _prompt_context_fingerprint(
        config, num_samples, prompting_mode
    )

    # ── Stati parziali per il resume da walltime ─────────────────────────
    # Uno per SOGGETTO valutato e per modalità di prompting: baseline e
    # checkpoint della stessa passata non devono mai condividere lo stato,
    # né le due passate del dual (ognuna ha il proprio suffisso __<mode>).
    # Il nome NON matcha i pattern eval_*.json di ablation_summary e
    # campaign_report: un file di stato incompleto non è mai raccolto come
    # risultato. A passata completata vengono rimossi (vedi sotto).
    baseline_state_path = results_dir / f"resume_state_baseline{prompting_suffix}.json"
    ckpt_name_for_state = Path(checkpoint_arg).name if checkpoint_arg else "zero_shot"
    checkpoint_state_path = (
        results_dir / f"resume_state_{ckpt_name_for_state}{prompting_suffix}.json"
    )

    # ── Isolamento degli artefatti della passata (dual prompting) ────────
    # Il suffisso __<mode> copre TUTTI gli artefatti della passata: figure in
    # sottodirectory per modalità, comparison.json e evaluation.output
    # esplicito con suffisso. Altrimenti la seconda passata riscriverebbe
    # comparison.json (path NON suffissato) e ablation_summary avrebbe
    # pubblicato i delta CROSS-prompting. Contratto: la passata primaria
    # (prompting_changed=False) NON cambia alcun nome file — i JSON esistenti
    # e la lettura fissa di comparison.json in src/utils/ablation_summary.py
    # restano validi.
    pass_figures_dir = (
        figures_dir / prompting_mode if prompting_changed else figures_dir
    )

    baseline_results: dict[str, Any] | None = None
    baseline_generations: list[dict[str, Any]] | None = None

    if eval_baseline_only:
        # Mode 1: baseline-only eval
        logger.info("=" * 60)
        logger.info("BASELINE EVALUATION (base model, no trained weights)")
        logger.info("=" * 60)
        (
            results,
            completions,
            validity,
            all_references,
            rouge_scores,
            generations,
            bleu_scores,
            chrf_scores,
        ) = evaluate_checkpoint(
            config,
            checkpoint_path=None,
            max_samples=max_samples,
            num_samples=num_samples,
            best_of_n=best_of_n,
            prompting_mode=prompting_mode,
            prompting_source=prompting_source,
            resume_state_path=baseline_state_path,
        )

    elif do_compare:
        # Mode 2: baseline + checkpoint comparison (SAME decoding for both)
        # Step A: Load or evaluate baseline
        if baseline_json is not None and Path(baseline_json).exists():
            logger.info(f"Loading baseline results from {baseline_json}")
            baseline_results = json.loads(
                Path(baseline_json).read_text(encoding="utf-8")
            )
            # Try to load baseline generations too
            bl_gen_path = (
                Path(baseline_json).parent
                / f"generations_{Path(baseline_json).stem.removeprefix('eval_')}.json"
            )
            if bl_gen_path.exists():
                baseline_generations = json.loads(
                    bl_gen_path.read_text(encoding="utf-8")
                )
        elif not force_baseline_eval:
            cached = _load_cached_baseline(
                results_dir,
                num_samples=num_samples,
                max_samples=max_samples,
                fingerprint=baseline_fingerprint,
                filename=baseline_filename,
            )
            if cached is not None:
                baseline_results, baseline_generations, cached_from = cached
                logger.info(
                    "=" * 60
                    + f"\n  REUSING CACHED BASELINE: {cached_from}/{baseline_filename}"
                    + "\n  (compatible metrics/prompt-context/decoding/sample count —"
                    "\n   set evaluation.force_baseline_eval: true to re-evaluate)"
                    + "\n"
                    + "=" * 60
                )
        if baseline_results is None:
            logger.info("=" * 60)
            logger.info("BASELINE EVALUATION (base model, no trained weights)")
            logger.info("=" * 60)
            # Baseline e checkpoint usano lo STESSO decoding (stesso
            # num_samples, stessa temperatura) E lo stesso prompting: su una
            # cella few-shot la baseline di confronto e' few-shot. Il log qui
            # sotto lo dichiara esplicitamente per chi legge l'output.
            logger.info(
                "  Baseline uses the same decoding (num_samples=%d) AND the "
                "same prompting (%s, source: %s) as the checkpoint.",
                num_samples,
                prompting_mode,
                prompting_source,
            )
            (
                baseline_results,
                _bl_comps,
                _bl_val,
                _bl_refs,
                _bl_rouge,
                baseline_generations,
                _bl_bleu,
                _bl_chrf,
            ) = evaluate_checkpoint(
                config,
                checkpoint_path=None,
                max_samples=max_samples,
                num_samples=num_samples,
                best_of_n=False,
                prompting_mode=prompting_mode,
                prompting_source=prompting_source,
                resume_state_path=baseline_state_path,
            )
            # Salva la baseline per riuso futuro: filename con suffisso di
            # modalità quando la modalità della passata differisce da quella
            # implicita nella config (il fingerprint protegge il riuso).
            bl_out_dir = results_dir
            bl_path = bl_out_dir / baseline_filename
            bl_path.write_text(
                json.dumps(baseline_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            bl_gen_path = bl_out_dir / baseline_filename.replace(
                "eval_baseline", "generations_baseline", 1
            )
            bl_gen_path.write_text(
                json.dumps(baseline_generations, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Baseline results saved to {bl_path}")
            logger.info(f"Baseline generations saved to {bl_gen_path}")

        # Step B: Evaluate checkpoint
        logger.info("=" * 60)
        logger.info("CHECKPOINT EVALUATION")
        logger.info("=" * 60)
        (
            results,
            completions,
            validity,
            all_references,
            rouge_scores,
            generations,
            bleu_scores,
            chrf_scores,
        ) = evaluate_checkpoint(
            config,
            checkpoint_arg,
            max_samples=max_samples,
            num_samples=num_samples,
            best_of_n=best_of_n,
            prompting_mode=prompting_mode,
            prompting_source=prompting_source,
            resume_state_path=checkpoint_state_path,
        )

    else:
        # Mode 3: single eval (default behavior)
        (
            results,
            completions,
            validity,
            all_references,
            rouge_scores,
            generations,
            bleu_scores,
            chrf_scores,
        ) = evaluate_checkpoint(
            config,
            checkpoint_arg,
            max_samples=max_samples,
            num_samples=num_samples,
            best_of_n=best_of_n,
            prompting_mode=prompting_mode,
            prompting_source=prompting_source,
            resume_state_path=checkpoint_state_path,
        )

    # ── Marchio di incompletezza del checkpoint ──────────────────────────
    # Un path checkpoint-<step> (invece di final) è un checkpoint intermedio:
    # senza questo stamp l'eval di una cella interrotta sarebbe
    # indistinguibile da quella di un modello completo (dettagli in
    # _checkpoint_completeness_stamp). L'eval baseline-only non ha
    # checkpoint: nessuno stamp.
    results.update(
        _checkpoint_completeness_stamp(None if eval_baseline_only else checkpoint_arg)
    )

    # ── Log key metrics ─────────────────────────────────────────────────
    logger.info("Evaluation complete. Key metrics (averaged over ALL completions):")
    # Order = literature-based metric hierarchy (docs/EVALUATION.md):
    # BLEU-4 first (T2G standard), then chrF, ROUGE-L, Gloss F1, Pass@1
    # (project-specific threshold metric), validity.
    logger.info(f"  BLEU-4 (corpus):         {results['bleu_corpus']:.4f}")
    logger.info(f"  chrF2 (corpus):         {results['chrf_corpus']:.2f}")
    logger.info(
        f"  ROUGE-L (sent mean):    {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}"
    )
    logger.info(f"  Gloss F1 (micro):       {results['gloss_f1_micro']:.4f}")
    # Metrica primaria copy-insensitive: il denominatore (posizioni non banali)
    # è stampato accanto perché il valore non è giudicabile senza di esso.
    logger.info(
        f"  Non-copy token acc:     {results['non_copy_token_accuracy']:.4f} "
        f"({results['non_copy_token_hits']}/{results['non_copy_token_total']} "
        f"non-copy ref tokens)"
    )
    # pass@1 = frazione di prompt la cui PRIMA completion raggiunge la
    # soglia (k=1 della curva Pass@k). L'etichetta esplicita evita di
    # confonderlo con la media su tutte le completions del blocco CI.
    logger.info(f"  Pass@1 (first completion): {results['pass_at_1']:.4f}")
    if results.get("pass_at_k"):
        for k, v in results["pass_at_k"].items():
            if k == "pass@1":
                continue  # già stampato come "Pass@1 (first completion)" sopra
            logger.info(f"  Pass@{k}: {v:.4f}")
    logger.info(f"  Validity rate: {results['validity_rate']:.4f}")
    logger.info(
        f"  Valid: {results['valid_count']}, Invalid: {results['invalid_count']}"
    )
    if results.get("difficulty_breakdown"):
        for lvl, agg in results["difficulty_breakdown"].items():
            logger.info(
                f"  [{lvl}] n={agg['n_prompts']}  ROUGE-L={agg['rouge_l_mean']:.3f}  "
                f"Pass@1={agg.get('pass_at_1', float('nan')):.3f}"
            )

    # ── Log sample predictions ──────────────────────────────────────────
    # ``completions`` is FLAT (one per completion) while ``all_references``
    # is one per prompt: they misalign when num_samples > 1. Use the aligned
    # per-completion gold carried by ``generations`` instead.
    flat_refs = (
        [g["gold_gloss"] for g in generations] if generations else list(all_references)
    )
    logger.info("Sample predictions (first 5):")
    for i in range(min(5, len(completions))):
        comp = completions[i]
        is_valid, reason = validity[i]
        ref = flat_refs[i] if i < len(flat_refs) else "N/A"
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
    print(
        f"  Samples evaluated:       {results['num_samples_evaluated']}"
        f" / {results.get('test_set_size', '?')} test set"
    )
    print(f"  Completions per prompt:  {results['num_completions_per_prompt']}")
    print(
        f"  ROUGE-L (mean ± std):    {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}"
    )
    print(
        f"  Valid ROUGE-L:           {results['valid_rouge_l_mean']:.4f}  "
        f"(rouge_l × validity = {results['rouge_l_mean']:.4f} × {results['validity_rate']:.4f})"
    )
    print(
        f"  BLEU (sent/corpus):      {results['bleu_sentence_mean']:.4f} / "
        f"{results['bleu_corpus']:.4f}"
    )
    print(
        f"  chrF2 (sent/corpus):     {results['chrf_sentence_mean']:.2f} / "
        f"{results['chrf_corpus']:.2f}"
    )
    print(
        f"  Gloss F1 (sent/micro):   {results['gloss_f1_sentence_mean']:.4f} / "
        f"{results['gloss_f1_micro']:.4f}"
    )
    # pass@1 (first completion) = frazione di prompt la cui PRIMA completion
    # raggiunge ROUGE-L >= 0.3: un solo draw onesto per prompt, ancora k=1
    # della curva Pass@k (che usa le prime k completions). Il loop pass@k
    # sotto salta k=1: stamparlo due volte con lo stesso valore era rumore.
    print(f"  Pass@1 (first completion): {results['pass_at_1']:.4f}")
    if "pass_at_k" in results:
        for k, v in results["pass_at_k"].items():
            if k == "pass@1":
                continue  # già stampato come "Pass@1 (first completion)" sopra
            print(f"  {k}:{' ' * (25 - len(k))}{v:.4f}")
    print(f"  Exact match:             {results['exact_match']:.4f}")
    # Metrica primaria (PRIMARY_METRICS): separa "copia l'inglese" da
    # "produce gloss" — su questo corpus ROUGE-L/BLEU premiano la copia.
    # Denominatore sempre visibile: il valore si giudica solo sapendo su
    # quante posizioni non banali si basa.
    print(
        f"  Non-copy token acc:      {results['non_copy_token_accuracy']:.4f}  "
        f"({results['non_copy_token_hits']} hits / "
        f"{results['non_copy_token_total']} non-copy ref tokens)"
    )
    print(f"  Bigram log-prob (mean):  {results['bigram_log_prob_mean']:.4f}")
    print(
        f"  Validity rate:           {results['validity_rate']:.4f}  "
        f"({results['valid_count']} valid / {results['invalid_count']} invalid)"
    )

    # ── Comprehensive report (BLEU + chrF + gloss F1 + bootstrap CI 95%) ──
    if "evaluation_report" in results:
        er = results["evaluation_report"]
        print("\n  ── Metrics & Confidence Intervals (95% CI) ──")
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
        if "chrf" in er:
            cf = er["chrf"]
            print(
                f"    chrF2:    corpus={cf['corpus']:.2f}  "
                f"sentence={cf['sentence_mean']:.2f}  "
                f"CI: [{cf['ci_95'][0]:.2f}, {cf['ci_95'][1]:.2f}]"
            )
        if "gloss_f1" in er:
            gf = er["gloss_f1"]
            print(
                f"    Gloss F1: micro={gf['micro']:.4f}  "
                f"sentence={gf['sentence_mean']:.4f}  "
                f"CI: [{gf['ci_95'][0]:.4f}, {gf['ci_95'][1]:.4f}]"
            )
        if "pass_at_1" in er:
            pa = er["pass_at_1"]
            # STIMATORE DIVERSO dal "Pass@1 (first completion)" del blocco
            # principale: media empirica dell'indicatore ROUGE-L >= 0.3 su
            # TUTTE le completions (num_samples draw per prompt), con CI
            # bootstrap. NON e' lo stesso numero (docs/EVALUATION.md §2a).
            print(
                f"    Pass@1 (all completions):  {pa['mean']:.4f}  "
                f"CI: [{pa['ci_95'][0]:.4f}, {pa['ci_95'][1]:.4f}]"
            )
        if "gloss_validity_rate" in er:
            print(f"    Gloss validity rate: {er['gloss_validity_rate']:.4f}")

    # ── Oracle best-of-N (separate, diagnostic only) ────────────────────
    if results.get("oracle_best_of_n"):
        obn = results["oracle_best_of_n"]
        print("\n  ── Oracle Best-of-N (NOT deployable — diagnostic only) ──")
        print(f"    ROUGE-L:      {obn.get('rouge_l_mean', 0.0):.4f}")
        print(f"    BLEU:         {obn.get('bleu_sentence_mean', 0.0):.4f}")
        print(f"    chrF2:        {obn.get('chrf_sentence_mean', 0.0):.2f}")
        print(f"    Gloss F1:     {obn.get('gloss_f1_sentence_mean', 0.0):.4f}")
        print(f"    Exact match:  {obn.get('exact_match', 0.0):.4f}")

    print("\n  ── Reward Breakdown ──")
    for k, v in results["reward_breakdown"].items():
        print(f"    {k}: {v:.4f}")
    print("\n  ── Error Distribution ──")
    for err, count in results["error_distribution"].items():
        print(f"    {err}: {count}")
    print("=" * 60)

    # Provenienza: QUALE config e QUALE checkpoint hanno prodotto questo JSON.
    # Senza questi campi un file di risultati è anonimo — se due celle finiscono
    # nella stessa directory (è successo: vedi src/utils/run_paths.py) l'unico
    # modo di attribuirlo è la forensics sui log SLURM.
    results["provenance"] = {
        "config": config_path,
        "checkpoint": str(checkpoint_arg) if checkpoint_arg else None,
        "results_dir": str(results_dir),
    }

    # ── Save JSON ────────────────────────────────────────────────────────
    if output_override:
        out_path = Path(output_override)
        # evaluation.output esplicito è UN SOLO path condiviso fra le passate:
        # con modalità effettiva diversa da quella della config la passata
        # dual sovrascriverebbe l'output della primaria. Il suffisso __<mode>
        # va applicato anche qui (stesso contratto dei file standard); il
        # file generations_ derivato dal stem eredita il suffisso.
        if prompting_suffix:
            out_path = out_path.with_stem(f"{out_path.stem}{prompting_suffix}")
    else:
        ckpt_name = Path(checkpoint_arg).name if checkpoint_arg else "zero_shot"
        out_path = results_dir / f"eval_{ckpt_name}{prompting_suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fase istrumentata: la scrittura delle decine di MB di generations JSON
    # può durare secondi-minuti e il log del path usciva solo DOPO la scrittura.
    with phase(
        "Saving eval + generations JSON",
        detail=f"{len(results)} metric keys, {len(generations)} generations",
    ):
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Results saved to {out_path}")

        # ── Save generations JSON (stile grpo-strict-generation) ────────────
        # Ogni generazione (testo/gold/pred/valid/rouge_l) per ispezione e
        # analisi errori. Naming: sempre dallo stem dell'eval file, scritto
        # ACCANTO — un evaluation.output custom resta coerente.
        gen_stem = out_path.stem
        if gen_stem.startswith("eval_"):
            gen_stem = gen_stem[len("eval_") :]
        gen_path = out_path.parent / f"generations_{gen_stem}.json"
        gen_path.write_text(
            json.dumps(generations, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Generations saved to {gen_path}")
    print(f"  Generations saved to: {gen_path}")

    # ── Pulizia dello stato parziale ─────────────────────────────────────
    # La passata è completata con successo (JSON + generations scritti): lo
    # stato va RIMOSSO — la sua presenza deve significare solo "questa
    # passata è incompleta", mai un orfano accanto a un eval_*.json finito.
    # Solo uno dei due è mai esistito (baseline o checkpoint): unlink con
    # missing_ok copre entrambi.
    _remove_resume_state(baseline_state_path)
    _remove_resume_state(checkpoint_state_path)

    # ── Generate plots ───────────────────────────────────────────────────
    if plot:
        # La passata con modalità override/dual renderizza nella SUA
        # sottodirectory (pass_figures_dir): mkdir esplicito qui — main()
        # crea solo la root delle figure.
        pass_figures_dir.mkdir(parents=True, exist_ok=True)
        with phase(
            "Rendering evaluation figures",
            detail="11-12 ggplot figures via plotnine (per-figure Saved: lines follow)",
        ):
            from src.utils.visualization import (
                dump_completion_examples,
                plot_baseline_vs_grpo,
                plot_baseline_vs_grpo_comparison,
                plot_completion_length_distribution,
                plot_difficulty_breakdown,
                plot_error_breakdown,
                plot_metrics_dashboard,
                plot_pass_at_k_curve,
                plot_reward_breakdown,
                plot_reward_radar,
                plot_rouge_distribution,
                plot_score_distribution,
                plot_validity_pie,
            )

            # pass_figures_dir è derivato da figures_dir (pre-risolto in
            # main()): root per la passata primaria, sottodirectory per
            # modalità quando la modalità effettiva è un override/dual.
            valid_mask = [v for v, _ in validity]

            # 1. Metrics dashboard — THE comparison figure (headline metrics,
            #    baseline vs checkpoint, ordered by relevance). Rendered FIRST
            #    so it's the "one figure to look at" in the figures dir.
            dashboard_baseline = None
            if baseline_results is not None:
                dashboard_baseline = baseline_results
            elif baseline_json is not None and Path(baseline_json).exists():
                dashboard_baseline = json.loads(
                    Path(baseline_json).read_text(encoding="utf-8")
                )
            plot_metrics_dashboard(
                dashboard_baseline,
                results,
                model_name=model_tag,
                label=model_tag,
                output_path=str(pass_figures_dir / "metrics_dashboard.png"),
            )

            # 2. Difficulty breakdown (per gold-difficulty metrics)
            if results.get("difficulty_breakdown"):
                plot_difficulty_breakdown(
                    results["difficulty_breakdown"],
                    model_name=model_tag,
                    output_path=str(pass_figures_dir / "difficulty_breakdown.png"),
                )

            # 3. Completion length distribution
            plot_completion_length_distribution(
                completions,
                valid_mask=valid_mask,
                title=f"Gloss Length - {model_tag}",
                output_path=str(pass_figures_dir / "completion_lengths.png"),
            )

            # 4. Metric score distributions (BLEU-4, chrF2, ROUGE-L — same
            #    histogram format for the three headline content metrics)
            plot_score_distribution(
                bleu_scores,
                metric_name="BLEU-4 (sentence)",
                xlabel="BLEU-4 Score",
                model_name=model_tag,
                output_path=str(pass_figures_dir / "bleu_distribution.png"),
                valid_mask=valid_mask,
            )
            plot_score_distribution(
                chrf_scores,
                metric_name="chrF2 (sentence)",
                xlabel="chrF2 Score",
                model_name=model_tag,
                output_path=str(pass_figures_dir / "chrf_distribution.png"),
                valid_mask=valid_mask,
            )
            plot_rouge_distribution(
                rouge_scores,
                model_name=model_tag,
                output_path=str(pass_figures_dir / "rouge_distribution.png"),
            )

            # 5. Pass@k curve (if multi-sample)
            if results.get("pass_at_k"):
                plot_pass_at_k_curve(
                    results["pass_at_k"],
                    model_name=model_tag,
                    output_path=str(pass_figures_dir / "pass_at_k.png"),
                )

            # 6. Error breakdown pie chart
            plot_error_breakdown(
                results["error_distribution"],
                model_name=model_tag,
                output_path=str(pass_figures_dir / "error_breakdown.png"),
            )

            # 7. Validity pie chart
            plot_validity_pie(
                valid_count=results["valid_count"],
                invalid_count=results["invalid_count"],
                model_name=model_tag,
                output_path=str(pass_figures_dir / "validity_pie.png"),
            )

            # 8. Reward breakdown bar chart
            rewards_cfg = config.get("reward", {})
            structure_weight = rewards_cfg.get("weight_gold_structure", 0.4)
            weights = {
                "translation_quality_reward": rewards_cfg.get(
                    "weight_translation", 0.4
                ),
                "bleu_reward": rewards_cfg.get("weight_bleu", 0.0),
                "gold_structure_reward": structure_weight,
                "verifier_scaled_reward": rewards_cfg.get(
                    "weight_verifier_scaled", 0.0
                ),
                "gloss_order_reward": rewards_cfg.get("weight_gloss_order", 0.0),
                "gloss_format_reward": rewards_cfg.get("weight_format", 0.1),
                "gloss_repetition_reward": rewards_cfg.get("weight_repetition", 0.1),
                "edit_validity_reward": rewards_cfg.get("weight_edit_validity", 0.0),
            }
            plot_reward_breakdown(
                [{"label": model_tag, "scores": results["reward_breakdown"]}],
                reward_weights=weights,
                model_name=model_tag,
                output_path=str(pass_figures_dir / "reward_breakdown.png"),
            )

            # 9. Reward radar chart
            plot_reward_radar(
                results["reward_breakdown"],
                reward_weights=weights,
                model_name=model_tag,
                output_path=str(pass_figures_dir / "reward_radar.png"),
            )

            # 10. Completion examples (best & worst) — JSON + HTML
            # ``completions``/``rouge_scores`` are FLAT (one per completion):
            # pass the aligned per-completion gold from ``generations``.
            # ``all_references`` is ONE PER PROMPT and would misalign gold vs
            # prompt whenever num_samples > 1 (gold shown from another sample).
            prompts = [g["text"] for g in generations]
            flat_refs = [g["gold_gloss"] for g in generations]
            dump_completion_examples(
                completions,
                flat_refs,
                rouge_scores,
                prompts=prompts,
                n_examples=10,
                model_name=model_tag,
                output_dir=str(pass_figures_dir),
            )

            # 11. Baseline vs GRPO (if a baseline Pass@1 is set in the config)
            if baseline_pass_at1 is not None:
                plot_baseline_vs_grpo(
                    baseline_pass1=baseline_pass_at1,
                    grpo_pass1=results["pass_at_1"],
                    model_name=model_tag,
                    output_path=str(pass_figures_dir / "baseline_vs_grpo.png"),
                )

            # 12. Baseline vs GRPO full comparison
            # Triggered by: compare mode (baseline_results from in-run eval),
            # or evaluation.baseline_json (baseline_results from saved JSON).
            comparison_metrics: dict[str, Any] | None = None
            if baseline_results is not None:
                comparison_metrics = baseline_results
            elif baseline_json is not None and Path(baseline_json).exists():
                comparison_metrics = json.loads(
                    Path(baseline_json).read_text(encoding="utf-8")
                )

            if comparison_metrics is not None:
                plot_baseline_vs_grpo_comparison(
                    baseline_metrics=comparison_metrics,
                    grpo_metrics=results,
                    model_name=model_tag,
                    label=model_tag,
                    output_path=str(
                        pass_figures_dir / "baseline_vs_grpo_comparison.png"
                    ),
                )

                # ── Print comparison summary ───────────────────────────────
                # Both models use the SAME decoding (same num_samples and
                # temperature), so the comparison is fair.
                compare_keys = [
                    "rouge_l_mean",
                    "valid_rouge_l_mean",
                    "pass_at_1",
                    "exact_match",
                    "validity_rate",
                    "bleu_sentence_mean",
                    "bleu_corpus",
                    "chrf_sentence_mean",
                    "chrf_corpus",
                    "gloss_f1_sentence_mean",
                    "gloss_f1_micro",
                    "bigram_log_prob_mean",
                ]
                compare_labels = {
                    "rouge_l_mean": "ROUGE-L mean",
                    "valid_rouge_l_mean": "Valid ROUGE-L",
                    "pass_at_1": "Pass@1",
                    "exact_match": "Exact Match",
                    "validity_rate": "Validity Rate",
                    "bleu_sentence_mean": "BLEU (sent)",
                    "bleu_corpus": "BLEU (corpus)",
                    "chrf_sentence_mean": "chrF2 (sent)",
                    "chrf_corpus": "chrF2 (corpus)",
                    "gloss_f1_sentence_mean": "Gloss F1 (sent)",
                    "gloss_f1_micro": "Gloss F1 (micro)",
                    "bigram_log_prob_mean": "Bigram LP",
                }
                print("\n" + "=" * 60)
                # Modalità di prompting della baseline: in compare mode e'
                # identica per costruzione (stessa config + override); per
                # evaluation.baseline_json si legge lo stamp nel JSON: se
                # dichiara una modalità DIVERSA il confronto non e' a parità
                # di prompting e va dichiarato.
                bl_stamp = (
                    comparison_metrics.get("prompting")
                    if isinstance(comparison_metrics, dict)
                    else None
                )
                bl_mode_label = (
                    f"{prompting_mode} (source: {prompting_source})"
                    if not isinstance(bl_stamp, dict)
                    or bl_stamp.get("mode") in (None, prompting_mode)
                    else f"{bl_stamp.get('mode')} — DIFFERS from checkpoint prompting!"
                )
                print(
                    "  BASELINE vs CHECKPOINT COMPARISON "
                    f"(same decoding; baseline prompting = {bl_mode_label})"
                )
                print("=" * 60)
                for metric_key in compare_keys:
                    bl_val = comparison_metrics.get(metric_key, 0.0)
                    gr_val = results.get(metric_key, 0.0)
                    delta = gr_val - bl_val
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                    print(
                        f"  {compare_labels[metric_key]:20s}  "
                        f"BL={bl_val:.4f}  CKPT={gr_val:.4f}  "
                        f"Δ={delta:+.4f} {arrow}"
                    )
                print("=" * 60)

                # ── Save comparison JSON ────────────────────────────────────
                comparison_json = {
                    "decoding": results.get("decoding", {}),
                    "baseline": {
                        k: comparison_metrics.get(k, 0.0) for k in compare_keys
                    },
                    "checkpoint": {k: results.get(k, 0.0) for k in compare_keys},
                    "delta": {
                        k: results.get(k, 0.0) - comparison_metrics.get(k, 0.0)
                        for k in compare_keys
                    },
                }
                # Suffisso __<mode> anche qui (vedi pass_figures_dir): senza,
                # la passata dual riscriverebbe comparison.json della primaria
                # e ablation_summary leggerebbe i delta della modalità sbagliata.
                comp_path = out_path.parent / f"comparison{prompting_suffix}.json"
                comp_path.write_text(
                    json.dumps(comparison_json, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(f"Comparison saved to {comp_path}")
                print(f"  Comparison saved to: {comp_path}")

            print(f"\n  Figures saved to: {pass_figures_dir}")

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
        wandb_tags = wandb_cfg.get("tags", ["T2G", "eval"])
        if "eval" not in wandb_tags:
            wandb_tags = list(wandb_tags) + ["eval"]
        if eval_baseline_only:
            wandb_tags = list(wandb_tags) + ["baseline"]
            wandb_run_name = f"eval-baseline-{base_run_name}"
        elif do_compare:
            wandb_tags = list(wandb_tags) + ["compare", "grpo"]
            wandb_run_name = f"eval-compare-{base_run_name}"
        else:
            wandb_tags = list(wandb_tags) + ["grpo"]
            wandb_run_name = f"eval-{base_run_name}-{model_tag}"

        wandb_dir = logs_dir

        with phase(
            "Logging results to wandb",
            detail="offline run: metrics + comparison + figures, then finish()",
        ):
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
                wandb_delta: dict[str, Any] = {
                    "baseline/rouge_l_mean": baseline_results.get("rouge_l_mean", 0.0),
                    "baseline/valid_rouge_l_mean": baseline_results.get(
                        "valid_rouge_l_mean", 0.0
                    ),
                    "baseline/pass_at_1": baseline_results.get("pass_at_1", 0.0),
                    "baseline/exact_match": baseline_results.get("exact_match", 0.0),
                    "baseline/validity_rate": baseline_results.get(
                        "validity_rate", 0.0
                    ),
                    "baseline/bleu_sentence_mean": baseline_results.get(
                        "bleu_sentence_mean", 0.0
                    ),
                    "baseline/bleu_corpus": baseline_results.get("bleu_corpus", 0.0),
                    "baseline/chrf_sentence_mean": baseline_results.get(
                        "chrf_sentence_mean", 0.0
                    ),
                    "baseline/gloss_f1_sentence_mean": baseline_results.get(
                        "gloss_f1_sentence_mean", 0.0
                    ),
                    "baseline/gloss_f1_micro": baseline_results.get(
                        "gloss_f1_micro", 0.0
                    ),
                    "delta/rouge_l_mean": results.get("rouge_l_mean", 0.0)
                    - baseline_results.get("rouge_l_mean", 0.0),
                    "delta/valid_rouge_l_mean": results.get("valid_rouge_l_mean", 0.0)
                    - baseline_results.get("valid_rouge_l_mean", 0.0),
                    "delta/pass_at_1": results.get("pass_at_1", 0.0)
                    - baseline_results.get("pass_at_1", 0.0),
                    "delta/exact_match": results.get("exact_match", 0.0)
                    - baseline_results.get("exact_match", 0.0),
                    "delta/validity_rate": results.get("validity_rate", 0.0)
                    - baseline_results.get("validity_rate", 0.0),
                    "delta/bleu_sentence_mean": results.get("bleu_sentence_mean", 0.0)
                    - baseline_results.get("bleu_sentence_mean", 0.0),
                    "delta/bleu_corpus": results.get("bleu_corpus", 0.0)
                    - baseline_results.get("bleu_corpus", 0.0),
                    "delta/chrf_sentence_mean": results.get("chrf_sentence_mean", 0.0)
                    - baseline_results.get("chrf_sentence_mean", 0.0),
                    "delta/gloss_f1_sentence_mean": results.get(
                        "gloss_f1_sentence_mean", 0.0
                    )
                    - baseline_results.get("gloss_f1_sentence_mean", 0.0),
                    "delta/gloss_f1_micro": results.get("gloss_f1_micro", 0.0)
                    - baseline_results.get("gloss_f1_micro", 0.0),
                }
                wandb.log(wandb_delta)

            # Log figures as images
            if plot:
                # pass_figures_dir: la sottodirectory della passata (override/
                # dual) o la root — mai le figure di un'altra modalità.
                for fig_file in pass_figures_dir.glob("*.png"):
                    wandb.log({f"figures/{fig_file.stem}": wandb.Image(str(fig_file))})

            wandb.finish()
        logger.info(f"wandb run logged: {wandb_run_name} (tags={wandb_tags})")
    except Exception as e:
        logger.warning(f"wandb logging failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# CLI — SOLO identificatori di run, nessun knob comportamentale
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point dell'eval: superficie CLI ridotta a --config + --checkpoint.

    Tutti i knob comportamentali (plot, compare, dual_prompting, prompting,
    best_of_n, max_samples, num_samples, eval_baseline_only,
    force_baseline_eval, output, baseline_pass_at1, baseline_json) vivono
    nella sezione ``evaluation:`` del config — richiesta esplicita: "non voglio
    passare altro che il file di config giusto al programma giusto". I due
    flag restanti sono gli UNICI irriducibili: identificano l'esecuzione
    (quale config, quale checkpoint prodotto a runtime dalla catena di job),
    non il comportamento.
    """
    parser = argparse.ArgumentParser(description="T2G checkpoint evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (omit to evaluate the base model WITHOUT "
        "trained weights). IDENTIFICATORE di run, non un knob: il path è "
        "prodotto a runtime dalla catena di job (timestamped run dirs) e non "
        "può stare nel config. Ogni knob comportamentale vive nella sezione "
        "evaluation: del config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # HF libraries attach their own StreamHandler AND propagate to root —
    # every library warning printed twice (slurm-eval-7077). Strip the
    # library-owned handlers so each record prints exactly once.
    dedupe_library_loggers()

    config = load_config(args.config)

    # ── Resolve eval knobs ONLY from the config ──────────────────────────
    # La sezione evaluation: è l'unica fonte dei knob (prima metà da CLI
    # flag + metà da env var dello sbatch: MAX_SAMPLES/PROMPTING/DUAL_EVAL/
    # BEST_OF_N — tutti rimossi).
    eval_cfg = dict(config.get("evaluation", {}))
    max_samples = eval_cfg.get("max_samples")  # None = full test set
    num_samples = int(eval_cfg.get("num_samples", 1))
    best_of_n = bool(eval_cfg.get("best_of_n", False))
    plot = bool(eval_cfg.get("plot", False))
    force_baseline_eval = bool(eval_cfg.get("force_baseline_eval", False))
    output_override = eval_cfg.get("output")
    baseline_pass_at1 = eval_cfg.get("baseline_pass_at1")
    baseline_json = eval_cfg.get("baseline_json")
    dual_prompting = bool(eval_cfg.get("dual_prompting", False))

    # ── Resolve the effective prompting mode (passata primaria) ──────────
    # La coppia (modalità, provenienza) viene stampata nel log, stampata nei
    # JSON e usata nel fingerprint del contesto prompt: due run con prompting
    # diverso devono restare distinguibili ovunque.
    prompting_mode, prompting_source, prompting_changed = _resolve_prompting(
        eval_cfg, config
    )

    # Validazione runtime (replica il vincolo di tests/validate_configs.py
    # per la sezione retrieval): forzare few-shot senza budget di prompt
    # adeguato troncherebbe gli esempi, rendendo la cella indistinguibile
    # dallo zero-shot. FALLIMENTO LOUD, non warning. Vale solo per la
    # modalità PRIMARIA: la passata dual e' la misura DELIBERATA del
    # cross-prompting (non troncata da max_prompt_length; suffisso __<mode>
    # + stamp source: config-dual la distinguono).
    if prompting_mode == "few-shot":
        max_prompt = config.get("grpo", {}).get("max_prompt_length") or config.get(
            "generation", {}
        ).get("max_prompt_length")
        if max_prompt is None or int(max_prompt) < 512:
            parser.error(
                f"evaluation.prompting: few-shot richiede max_prompt_length "
                f">= 512, trovato {max_prompt!r} (config: {args.config}). Con "
                f"un budget di prompt più corto gli esempi few-shot verrebbero "
                f"troncati e la cella misurerebbe altro. Correggi la config "
                f"(grpo.max_prompt_length o generation.max_prompt_length) "
                f"oppure imposta evaluation.prompting: config."
            )

    # ── Deduzione della modalità eval (storico di cluster/eval.sh) ───────
    # compare/eval-baseline-only NON sono più flag: si deducono dalla
    # presenza di training.output_dir (celle di training → compare; celle
    # eval-only → solo base model), con override esplicito nel config.
    has_output_dir = bool(config.get("training", {}).get("output_dir"))
    eval_baseline_only, do_compare = _deduce_eval_modes(
        eval_cfg,
        has_checkpoint=args.checkpoint is not None,
        has_output_dir=has_output_dir,
    )

    # Guardia anti-eval-silenzioso (stessa politica loud-fail che
    # cluster/eval.sh applica sull'auto-detect): un "compare" senza
    # checkpoint valuterebbe il base model DUE volte producendo numeri senza
    # senso senza alcun errore.
    if do_compare and not eval_baseline_only and args.checkpoint is None:
        parser.error(
            "Modalità compare richiesta (dedotta da training.output_dir o da "
            "evaluation.compare: true) ma nessun --checkpoint: il 'checkpoint' "
            "sarebbe il base model e il confronto misurerebbe il modello "
            "contro se stesso. Passa --checkpoint <path>, oppure imposta "
            "evaluation.eval_baseline_only: true per valutare SOLO il base "
            "model."
        )

    # ── Log eval configuration ───────────────────────────────────────────
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint or 'base model, no trained weights'}")
    logger.info(
        f"Prompting: {prompting_mode} (source: {prompting_source})"
        + (
            f" — config implies {_config_prompting_mode(config)}"
            if prompting_changed
            else ""
        )
    )
    logger.info(
        f"Max samples: {max_samples if max_samples is not None else 'all (full test set)'}"
    )
    logger.info(f"Completions per prompt: {num_samples}")
    logger.info(f"Grammar enabled: {config.get('grammar', {}).get('enabled', True)}")
    logger.info(f"Plot: {plot}")
    logger.info(f"Compare: {do_compare}")
    logger.info(f"Dual prompting: {dual_prompting}")
    logger.info(f"Best-of-N: {best_of_n}")
    logger.info(f"Eval baseline only: {eval_baseline_only}")
    if baseline_pass_at1 is not None:
        logger.info(f"Baseline Pass@1: {baseline_pass_at1}")
    if baseline_json is not None:
        logger.info(f"Baseline JSON: {baseline_json}")

    # Dichiarazioni contraddittorie nel config: vince eval_baseline_only,
    # ma il chiamante deve saperlo (silenzio qui = confusione nei risultati).
    if eval_baseline_only and do_compare:
        logger.warning(
            "evaluation: eval_baseline_only e compare sono entrambi true — "
            "vince eval_baseline_only (compare ignorato)."
        )
    if eval_baseline_only and args.checkpoint is not None:
        logger.warning(
            "--checkpoint fornito ma eval_baseline_only=true: il checkpoint "
            "viene IGNORATO (si valuta il base model)."
        )

    # ── Resolve model_name, run_id, and directory paths ──────────────────
    # Dipendono SOLO dal checkpoint/config, non dalla modalità di prompting:
    # le due passate del dual condividono le stesse directory (i file si
    # distinguono per il suffisso __<mode>).
    from datetime import datetime

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).resolve()
        split = split_checkpoint_path(checkpoint_path)
        if split is not None:
            model_name, run_id = split
        else:
            model_name = config.get("wandb", {}).get("run_name", "t2g-model")
            run_id = (
                checkpoint_path.parent.name
                if checkpoint_path.name in ["final", "checkpoint-*"]
                else checkpoint_path.name
            )
        model_tag_default = run_id
    else:
        raw_model_name = config["model"]["name"].split("/")[-1].lower()
        model_name = raw_model_name.replace(".", "")
        if "run_name" in config.get("wandb", {}):
            model_name = config["wandb"]["run_name"]
        run_id = f"zero_shot_{run_timestamp}"
        model_tag_default = "zero-shot"

    results_dir = Path("experiments/results") / model_name / run_id
    figures_dir = Path("experiments/figures") / model_name / run_id
    logs_dir = Path("experiments/logs") / model_name / run_id

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Argomenti comuni alle passate (primaria + eventuale dual): cambiano
    # SOLO la tripletta prompting (mode/source/changed).
    pass_common: dict[str, Any] = {
        "config": config,
        "checkpoint_arg": args.checkpoint,
        "config_path": args.config,
        "plot": plot,
        "best_of_n": best_of_n,
        "max_samples": max_samples,
        "num_samples": num_samples,
        "force_baseline_eval": force_baseline_eval,
        "output_override": output_override,
        "baseline_pass_at1": baseline_pass_at1,
        "baseline_json": baseline_json,
        "model_tag_default": model_tag_default,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
    }

    # ── Passata primaria ─────────────────────────────────────────────────
    _run_eval_pass(
        eval_baseline_only=eval_baseline_only,
        do_compare=do_compare,
        prompting_mode=prompting_mode,
        prompting_source=prompting_source,
        prompting_changed=prompting_changed,
        **pass_common,
    )

    # ── Dual prompting (evaluation.dual_prompting) ───────────────────────
    # Seconda passata con la modalità complementare: misura se il modello ha
    # interiorizzato la mappatura o dipende dal prompt come stampella. File
    # con suffisso __<mode> e cache baseline separata (fingerprint diverso).
    # COSTO: eval raddoppia (~25 min per passata a 5000 prompt); al primo
    # giro valuta anche la SUA baseline del base model (poi cachata).
    if dual_prompting:
        complement = _complement_prompting(prompting_mode)
        # La passata dual NON è soggetta al vincolo max_prompt_length >= 512
        # (vedi il commento alla validazione sopra): è la misura deliberata
        # del cross-prompting, sempre distinguibile nei risultati.
        dual_changed = complement != _config_prompting_mode(config)
        logger.info("=" * 60)
        logger.info(
            "DUAL PROMPTING: seconda passata con prompting=%s (complemento "
            "di %s; source: config-dual)",
            complement,
            prompting_mode,
        )
        logger.info("=" * 60)
        _run_eval_pass(
            eval_baseline_only=eval_baseline_only,
            do_compare=do_compare,
            prompting_mode=complement,
            prompting_source="config-dual",
            prompting_changed=dual_changed,
            **pass_common,
        )


if __name__ == "__main__":
    main()
