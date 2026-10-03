#!/usr/bin/env python3
"""
Campaign Report — structured cross-factor comparison of eval runs for the
ablation matrix.

Usage:
    python -m src.analysis.campaign_report
    python -m src.analysis.campaign_report --results-dir experiments/results
    python -m src.analysis.campaign_report --output-dir experiments/figures

Complementare a ``src.utils.ablation_summary`` (che produce la tabella
piatta per config): qui i run vengono appaiati per FATTORI SPERIMENTALI.
Per ogni fattore si isolano le coppie di run che differiscono SOLO per
quello fattore e si riportano i delta di ogni metrica — esattamente le
righe che servono per compilare la matrice di ablazione. Il modulo LEGGE
i JSON di eval: non ricalcola nessuna metrica.

Deduzione dei fattori (dichiarata, nessun default silenzioso):
  - ``method``: dal token del nome cella nel percorso
    (``sft-grpo`` / ``sft-only`` / ``grpo-only`` / ``t2g-``); i file
    ``eval_baseline.json`` sono sempre ``method=baseline`` (nessun
    checkpoint). Non deducibile → ``unknown``.
  - ``variant``: token rimanenti dopo modello/metodo/marcatori noti
    (es. ``structure``, ``viterbi``, ``all-rewards``).
  - ``prompting``: (1) stamp ``prompting.mode`` nel JSON; (2) suffisso
    ``__zero-shot`` / ``__few-shot`` nel nome file (dual/override);
    (3) convenzione di naming ``t2g-zero-shot*`` (solo celle baseline);
    altrimenti ``unknown`` — non si assume mai la modalità della config,
    perché con il dual prompting può differire da quella effettiva.
  - ``grammar``: ``no-grammar`` nel nome cella → off; ``grammar`` nella
    famiglia ``t2g-*`` → on; altrimenti ``unknown``.
  - ``reward_stack`` e ``rl_objective``: NON deducibili da percorso/stamp/
    campi del JSON (richiederebbero il config risolto, che l'eval non
    salva nel payload) → sempre ``unknown`` e DICHIARATI tali. Ne consegue
    che una coppia appaiata è "differiscono solo per F nei fattori
    deducibili": i fattori non deducibili sono riportati come caveat
    esplicito su ogni coppia, mai assunti uguali in silenzio.

Avvertenze obbligatorie incluse nell'output (§ del report):
  - numero di prompt valutati per run + avviso se discordante;
  - ``metrics_version`` per run + avviso se diversa;
  - data di ogni run (dal nome della run dir, fallback mtime del file
    dichiarato come tale): run di campagne diverse non sono confrontabili;
  - soglia di interpretabilità: a seme fissato l'addestramento è
    deterministico bit per bit, quindi non esiste una stima di variabilità
    fra semi → delta < NOISE_THRESHOLD marcati esplicitamente come non
    generalizzabili, mai presentati come risultati.

Output (default sotto ``experiments/figures/``):
    - ``campaign_report.json`` — struttura leggibile da altri strumenti;
    - ``campaign_report.md``   — matrice di ablazione tabellare;
    - ``campaign_pairwise_deltas.png`` — delta appaiati per fattore;
    - ``campaign_matrix.png``  — vista d'insieme della matrice.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import chart_style
from src.utils.metrics import PRIMARY_METRICS, SATURATED_OVERLAP_METRICS
from src.utils.run_paths import DATASET_KEYS, DEFAULT_DATASET_KEY, cell_sort_key

logger = logging.getLogger(__name__)

# Soglia di interpretabilità dei delta (scala [0,1]). Due esecuzioni della
# stessa cella con la stessa configurazione e lo stesso seme producono
# generazioni identiche byte per byte: la dispersione fra run è nulla, quindi
# NON è quella che la soglia protegge. Nessuna cella è stata replicata con
# semi diversi, per cui non esiste alcuna stima di variabilità fra semi: un
# delta sotto ~0.02 si riprodurrà a ogni riesecuzione ma non è per questo
# generalizzabile, e potrebbe invertirsi cambiando seme.
NOISE_THRESHOLD = 0.02

# Metriche lette dai JSON: (chiave, etichetta, scala). Nessuna metrica è
# ricalcolata: se la chiave manca nel JSON, il delta è semplicemente assente.
# fmt: off
METRIC_REGISTRY = [
    ("rouge_l_mean",            "ROUGE-L",        "0-1"),
    ("valid_rouge_l_mean",      "Valid ROUGE-L",  "0-1"),
    ("exact_match",             "Exact Match",    "0-1"),
    ("non_copy_token_accuracy", "Non-copy Tok Acc","0-1"),
    ("pass_at_1",               "Pass@1",         "0-1"),
    ("validity_rate",           "Validity",       "0-1"),
    ("bleu_corpus",             "BLEU (corpus)",  "0-1"),
    ("bleu_sentence_mean",      "BLEU (sent)",    "0-1"),
    ("chrf_corpus",             "chrF2 (corpus)", "0-100"),
    ("chrf_sentence_mean",      "chrF2 (sent)",   "0-100"),
    ("gloss_f1_micro",          "Gloss F1 (mic)", "0-1"),
    ("gloss_f1_sentence_mean",  "Gloss F1 (sent)","0-1"),
]
# fmt: on
METRIC_KEYS = [k for k, _, _ in METRIC_REGISTRY]
METRIC_LABELS = {k: lbl for k, lbl, _ in METRIC_REGISTRY}
METRIC_SCALES = {k: sc for k, _, sc in METRIC_REGISTRY}

# Fattori su cui si appaiano i confronti (ordine di presentazione). Gli altri
# fattori (reward_stack, rl_objective) non sono appaiabili: non deducibili.
PAIRABLE_FACTORS = ["method", "prompting", "grammar"]

UNKNOWN = "unknown"

# Suffisso dual/override sui nomi file: l'eval usa `__zero-shot` (dash,
# f"__{prompting_mode}"); si tollera anche la forma con underscore.
_PROMPTING_SUFFIX_RE = re.compile(r"__(zero[-_]shot|few[-_]shot)$")
# Data/ora nel nome della run dir: run_20260904_062558
_RUN_TS_RE = re.compile(r"(\d{8})_(\d{6})")

# Testo dichiarativo riprodotto in JSON e Markdown: la deduzione dei fattori
# DEVE essere trasparente, altrimenti il lettore non sa cosa c'è in una riga
# della matrice (e non può fidarsi della coppia appaiata).
FACTOR_DEDUCTION_DOC = [
    {
        "factor": "method",
        "rule": "token del nome cella nel percorso (sft-grpo / sft-only / "
        "grpo-only / t2g-*); eval_baseline.json => baseline (nessun checkpoint)",
        "if_not_deducible": "unknown (never guessed)",
    },
    {
        "factor": "variant",
        "rule": "remaining cell-name tokens after model tag, method and known "
        "markers (e.g. structure, viterbi, all-rewards); no-grammar is a "
        "grammar marker, not a variant",
        "if_not_deducible": "empty string (plain cell)",
    },
    {
        "factor": "prompting",
        "rule": "1) 'prompting.mode' stamp in the eval JSON; 2) filename "
        "suffix __zero-shot / __few-shot (dual/override pass); 3) t2g- "
        "zero-shot naming convention (baseline cells only)",
        "if_not_deducible": "unknown — the config-implied mode is NEVER "
        "assumed, it can differ from the effective one under dual prompting",
    },
    {
        "factor": "grammar",
        "rule": "'no-grammar' in cell name => off; 'grammar' in t2g-* cell "
        "name => on (that family names the toggle explicitly)",
        "if_not_deducible": "unknown (never guessed)",
    },
    {
        "factor": "reward_stack",
        "rule": "NOT deducible from path / JSON stamp / payload fields (it "
        "lives in the resolved config, which eval JSONs do not embed)",
        "if_not_deducible": "always unknown, declared in every pairing caveat",
    },
    {
        "factor": "rl_objective",
        "rule": "NOT deducible from path / JSON stamp / payload fields",
        "if_not_deducible": "always unknown, declared in every pairing caveat",
    },
]


# ---------------------------------------------------------------------------
# Deduzione dei fattori
# ---------------------------------------------------------------------------


def _tokens_to_string(tokens: list[str]) -> str:
    """Ricomposizione dei token rimanenti in un'etichetta variante."""
    return "-".join(t for t in tokens if t)


def _split_cell_and_run(parts: tuple[str, ...]) -> tuple[str | None, str]:
    """Separa il nome cella dall'identificativo di run in un percorso di eval.

    Ogni eval vive in ``<cella>/run_<ts>/`` (o in una sua sotto-directory di
    sola valutazione), e la cella è annidata quanto il suo config:
    ``qwen25-05b/sft/zero-shot/run_<ts>/``, ``qwen25-05b/baseline/zero-shot/
    run_<ts>/``. Prendere ``parts[0]`` come cella, com'era in origine,
    collassava tutte le celle annidate su ``qwen25-05b``: una sola tipologia,
    tutti gli altri run marcati superseded, e una matrice che accostava celle
    diverse come se fossero la stessa.

    L'ancora è l'ULTIMO segmento ``run_*``: ciò che lo precede è la cella, ciò
    che lo segue è una variante di sola valutazione dello stesso checkpoint
    (per esempio ``decoding-greedy``), tenuta distinta perché ha metriche
    proprie. Un file senza segmento di run è un orfano e viene scartato.
    """
    runs = [i for i, part in enumerate(parts[:-1]) if part.startswith("run_")]
    if not runs or runs[-1] == 0:
        return (None, "")
    i = runs[-1]
    return ("-".join(parts[:i] + parts[i + 1 : -1]), parts[i])


def deduce_cell_factors(cell: str) -> tuple[dict, dict]:
    """Deduci method/variant/grammar/prompting/model_tag dal nome cella.

    Returns:
        (factors, sources): due dict con le stesse chiavi; ``sources`` dice
        DA DOVE viene ogni valore (per la dichiarazione nel report).
    """
    factors: dict[str, str] = {}
    sources: dict[str, str] = {}
    tokens = cell.split("-")
    model_tag_tokens: list[str] = []

    # Marcatori consumati dai token; ciò che resta dopo il marcatore del
    # metodo è la variante. L'ordine conta: sft-grpo prima di sft/grpo.
    method = UNKNOWN
    consumed_until: int | None = None
    for i in range(len(tokens)):
        pair = tokens[i : i + 2]
        if pair == ["sft", "grpo"]:
            method = "sft-grpo"
            consumed_until = i + 2
            break
        if pair == ["sft", "only"]:
            method = "sft"
            consumed_until = i + 2
            break
        if pair == ["grpo", "only"]:
            method = "grpo"
            consumed_until = i + 2
            break
        if tokens[i] == "t2g":
            # Famiglia baseline: il nome della cella dichiara il setup di
            # eval (zero-shot / grammar), non c'è training.
            method = "baseline"
            consumed_until = i + 1
            break
    if consumed_until is None:
        # Niente marcatore noto: method unknown, ma la tipologia resta
        # univoca perché l'intero nome cella entra nella variante.
        method = UNKNOWN
        factors["variant"] = _tokens_to_string(tokens) or UNKNOWN
        sources["variant"] = "directory name (no method marker: full cell "
        "name kept as variant to avoid typology collisions)"
    else:
        model_tag_tokens = tokens[:consumed_until]
        rest = tokens[consumed_until:]
        # Marcatori di fattore noti nel resto (dichiarati, non default):
        # no-grammar e zero-shot/grammar nella famiglia baseline.
        if ["no", "grammar"] in [rest[j : j + 2] for j in range(len(rest) - 1)]:
            factors["grammar"] = "off"
            sources["grammar"] = "directory name ('no-grammar' marker)"
            # rimozione della coppia esatta (non dei singoli token, che
            # potrebbero appartenere al nome della variante):
            idx = next(
                j for j in range(len(rest) - 1) if rest[j : j + 2] == ["no", "grammar"]
            )
            rest = rest[:idx] + rest[idx + 2 :]
        if tokens[consumed_until - 1] == "t2g":
            if ["zero", "shot"] in [rest[j : j + 2] for j in range(len(rest) - 1)]:
                factors["prompting"] = "zero-shot"
                sources["prompting"] = "directory name (t2g-* convention)"
                idx = next(
                    j
                    for j in range(len(rest) - 1)
                    if rest[j : j + 2] == ["zero", "shot"]
                )
                rest = rest[:idx] + rest[idx + 2 :]
            if "grammar" in rest:
                factors["grammar"] = "on"
                sources["grammar"] = "directory name (t2g-* convention)"
                rest = [t for t in rest if t != "grammar"]
            elif "grammar" not in factors:
                # Convenzione DICHIARATA della famiglia t2g-*: solo le celle
                # col suffisso "grammar" attivano il vincolo; il nome nudo è
                # quello senza. Non è un default silenzioso: è la regola di
                # naming della famiglia, riprodotta nella tabella del report.
                factors["grammar"] = "off"
                sources["grammar"] = (
                    "directory name (t2g-* convention: no 'grammar' marker "
                    "means unconstrained)"
                )
        factors["variant"] = _tokens_to_string(rest)
        sources["variant"] = "directory name (remaining tokens after markers)"

    factors["method"] = method
    sources["method"] = "directory name"
    if "model_tag" not in factors:
        factors["model_tag"] = (
            _tokens_to_string(model_tag_tokens) if model_tag_tokens else UNKNOWN
        )
        sources["model_tag"] = "directory name (tokens before the method marker)"
    if "grammar" not in factors:
        factors["grammar"] = UNKNOWN
        sources["grammar"] = "not deducible from path"
    if "prompting" not in factors:
        factors["prompting"] = UNKNOWN
        sources["prompting"] = "not deducible from path (no stamp / no suffix)"
    # Non deducibili per costruzione: dichiarati, MAI riempiti con default.
    factors["reward_stack"] = UNKNOWN
    sources["reward_stack"] = "not deducible (config not embedded in eval JSON)"
    factors["rl_objective"] = UNKNOWN
    sources["rl_objective"] = "not deducible (config not embedded in eval JSON)"
    return factors, sources


def _parse_run_timestamp(run_dir_name: str) -> str | None:
    """ISO timestamp dal nome della run dir (run_YYYYMMDD_HHMMSS)."""
    m = _RUN_TS_RE.search(run_dir_name)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return dt.isoformat(sep=" ")


# ---------------------------------------------------------------------------
# Scoperta dei run
# ---------------------------------------------------------------------------


def discover_runs(results_dir: Path) -> dict:
    """Scandi results_dir e costruisci i descrittori dei run.

    Returns:
        dict con ``runs`` (descrivibili, selezionabili), ``excluded``
        (checkpoint incompleti), ``malformed`` (JSON illeggibili/parziali).
        Un run interrotto è un caso NORMALE: mai sollevare, sempre segnalare.
    """
    runs: list[dict] = []
    excluded: list[dict] = []
    malformed: list[dict] = []

    if not results_dir.exists():
        return {"runs": runs, "excluded": excluded, "malformed": malformed}

    for eval_path in sorted(results_dir.rglob("eval_*.json")):
        rel = eval_path.relative_to(results_dir)
        # Layout atteso: <cella>/<run_id>/[<sub-eval>/]eval_<ckpt>[__mode].json
        if len(rel.parts) < 3:
            continue
        cell, run_id = _split_cell_and_run(rel.parts)
        if cell is None:
            continue
        rel_str = str(rel).replace("\\", "/")

        try:
            with open(eval_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            # JSON incompleto/parziale: caso normale di run interrotto.
            malformed.append({"path": rel_str, "error": str(e)})
            continue
        if not isinstance(data, dict):
            malformed.append(
                {"path": rel_str, "error": "top-level JSON is not an object"}
            )
            continue

        if data.get("checkpoint_incomplete"):
            # Valutazione su checkpoint parziale: ESCLUSA dall'appaiamento —
            # confrontarla con un modello completo produrrebbe conclusioni false.
            excluded.append(
                {
                    "path": rel_str,
                    "reason": "checkpoint_incomplete: true",
                    "checkpoint_step": data.get("checkpoint_step"),
                }
            )
            continue

        kind = (
            "final"
            if "final" in eval_path.stem
            else ("baseline" if "baseline" in eval_path.stem else "checkpoint")
        )
        factors, sources = deduce_cell_factors(cell)
        # Il suffisso __<mode> è la modalità EFFETTIVA della passata: ha
        # precedenza sulla deduzione dal nome cella (che per le celle
        # addestrate resta comunque "unknown").
        suffix_m = _PROMPTING_SUFFIX_RE.search(eval_path.stem)
        if suffix_m:
            mode = suffix_m.group(1).replace("_", "-")
            factors["prompting"] = mode
            sources["prompting"] = "filename suffix (forced/dual pass)"
        # Lo stamp `prompting` nel JSON è la verità sul campo quando esiste.
        stamp = data.get("prompting")
        if isinstance(stamp, dict) and stamp.get("mode") in ("zero-shot", "few-shot"):
            factors["prompting"] = stamp["mode"]
            sources["prompting"] = f"JSON stamp (source: {stamp.get('source')})"

        if kind == "baseline":
            # eval_baseline = modello BASE (nessun checkpoint) nel contesto
            # di eval della cella: è un run method=baseline, non della cella.
            factors["method"] = "baseline"
            sources["method"] = "eval file kind (no-checkpoint baseline)"

        # Il modello viene dal percorso: ogni cella vive sotto
        # results/<dataset>/<modello>/ (legacy: results/<modello>/). Dedurlo
        # dal nome riusciva solo con un marcatore di metodo nel nome, e senza
        # il modello nella tipologia due modelli con la stessa cella
        # collassavano in una sola (l'altra superseded). Il dataset entra nel
        # model_tag (``<dataset>/<modello>``): tipologie e coppie restano
        # confinate a un solo dataset, mai confrontate attraverso corpora.
        if rel.parts[0] in DATASET_KEYS and len(rel.parts) > 3:
            factors["dataset"] = rel.parts[0]
            factors["model_tag"] = f"{rel.parts[0]}/{rel.parts[1]}"
            sources["dataset"] = "first path segment (results/<dataset>/...)"
            sources["model_tag"] = (
                "first two path segments (results/<dataset>/<model>/...)"
            )
        else:
            factors["dataset"] = DEFAULT_DATASET_KEY
            factors["model_tag"] = rel.parts[0]
            sources["dataset"] = "legacy layout without dataset segment (default)"
            sources["model_tag"] = "first path segment (results/<model>/...)"
        cell_path = "/".join(part for part in rel.parts[:-1] if part != run_id)

        timestamp = _parse_run_timestamp(run_id)
        ts_source = "run directory name" if timestamp else None
        if timestamp is None:
            timestamp = datetime.fromtimestamp(
                eval_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
            ts_source = "file mtime (directory name had no timestamp)"

        metrics = {}
        for key in METRIC_KEYS:
            val = data.get(key)
            # Solo numeri veri: un JSON parziale può avere null/NaN.
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                metrics[key] = float(val)

        runs.append(
            {
                "path": rel_str,
                "cell": cell,
                "cell_path": cell_path,
                "run_id": run_id,
                "kind": kind,
                "factors": factors,
                "factor_sources": sources,
                "timestamp": timestamp,
                "timestamp_source": ts_source,
                "num_samples_evaluated": data.get("num_samples_evaluated"),
                "metrics_version": data.get("metrics_version"),
                "prompt_context_fingerprint": data.get("prompt_context_fingerprint"),
                "metrics": metrics,
                "_mtime": eval_path.stat().st_mtime,
            }
        )

    return {"runs": runs, "excluded": excluded, "malformed": malformed}


def select_latest(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per ogni tipologia (combinazione dei fattori) tieni il run più recente.

    Priorità: eval_final > checkpoint eval (i checkpoint intermedi non devono
    oscurare il finale della stessa tipologia); a parità, timestamp poi mtime.

    Returns:
        (selected, superseded)
    """
    kind_rank = {"final": 2, "checkpoint": 1, "baseline": 0}
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        f = run["factors"]
        key = (
            f["model_tag"],
            f["method"],
            f["variant"],
            f["prompting"],
            f["grammar"],
        )
        groups.setdefault(key, []).append(run)

    selected: list[dict] = []
    superseded: list[dict] = []
    for key in sorted(groups, key=str):
        candidates = sorted(
            groups[key],
            key=lambda r: (
                kind_rank.get(r["kind"], 0),
                r["timestamp"] or "",
                r["_mtime"],
            ),
        )
        # eval_baseline resta in gara solo se è l'UNICO rappresentante:
        # un eval_final della stessa tipologia ha sempre precedenza.
        chosen = candidates[-1]
        selected.append(chosen)
        for r in candidates[:-1]:
            superseded.append(
                {
                    "path": r["path"],
                    "reason": "older run of the same typology",
                    "typology": "|".join(str(k) for k in key),
                }
            )
    return selected, superseded


# ---------------------------------------------------------------------------
# Confronti appaiati
# ---------------------------------------------------------------------------


def _known(value: str | None) -> bool:
    return value is not None and value != UNKNOWN and value != ""


def build_comparisons(
    selected: list[dict],
) -> tuple[list[dict], list[dict], list[str]]:
    """Appaia i run che differiscono SOLO per un fattore (deducibile).

    Returns:
        (comparisons, missing_pairs, warnings)
    """
    comparisons: list[dict] = []
    missing_pairs: list[dict] = []
    warnings: list[str] = []

    for factor in PAIRABLE_FACTORS:
        other = [f for f in PAIRABLE_FACTORS if f != factor]
        groups: dict[tuple, list[dict]] = {}
        for run in selected:
            f = run["factors"]
            # Il modello fa parte del gruppo ma non è un fattore appaiabile:
            # un confronto fra due modelli diversi confonderebbe l'effetto del
            # fattore con quello del modello.
            key = (f["model_tag"], *(f[o] for o in other))
            groups.setdefault(key, []).append(run)

        factor_pairs: list[dict] = []
        blocked_groups: list[list[dict]] = []
        for key in sorted(groups, key=str):
            members = list(groups[key])
            # Appaiamento greedy: ad ogni iterazione la miglior coppia
            # disponibile (stesso checkpoint > più recente), poi i due run
            # escono dal pool. Così un gruppo con più celle (es. sft e
            # sft-grpo entrambi "plain") produce ANCHE la coppia sft vs
            # sft-grpo, non solo la prima trovata.
            while True:
                pairable = [r for r in members if _known(r["factors"][factor])]
                if len(pairable) < 2 or (
                    len({r["factors"][factor] for r in pairable}) < 2
                ):
                    if len(members) >= 2:
                        blocked_groups.append(members)
                    break
                best: tuple[dict, dict] | None = None
                best_score: tuple = ()
                for i in range(len(pairable)):
                    for j in range(i + 1, len(pairable)):
                        a, b = pairable[i], pairable[j]
                        if a["factors"][factor] == b["factors"][factor]:
                            continue
                        same_run = a["run_id"] == b["run_id"] and a["cell"] == b["cell"]
                        recency = min(a["timestamp"] or "", b["timestamp"] or "")
                        score = (1 if same_run else 0, recency)
                        if best is None or score > best_score:
                            best, best_score = (a, b), score
                if best is None:
                    if len(members) >= 2:
                        blocked_groups.append(members)
                    break
                factor_pairs.append(
                    _make_comparison(factor, best[0], best[1], warnings)
                )
                # Consuma i due run appaiati: niente run usati due volte
                # (ogni confronto del report è fra run distinti).
                members = [r for r in members if r not in best]

        if factor_pairs:
            comparisons.extend(factor_pairs)
        else:
            detail = (
                f"{len(blocked_groups)} group(s) with 2+ runs but no pair with a "
                "known, differing value of this factor"
                if blocked_groups
                else "fewer than 2 runs with known factor values"
            )
            missing_pairs.append(
                {
                    "factor": factor,
                    "reason": detail,
                    "available_typologies": sorted(
                        {
                            "|".join(
                                f"{r['factors'][factor]}"
                                if _known(r["factors"][factor])
                                else f"{factor}={UNKNOWN}"
                            )
                            for r in selected
                        }
                    ),
                }
            )
    return comparisons, missing_pairs, warnings


def _make_comparison(factor: str, a: dict, b: dict, warnings: list[str]) -> dict:
    """Costruisci il confronto fra due run appaiati, con i delta e i caveat."""
    fa, fb = a["factors"], b["factors"]
    caveats: list[str] = []

    # Avvertenza obbligatoria 1: numero di prompt diverso => NON confrontabili.
    na, nb = a["num_samples_evaluated"], b["num_samples_evaluated"]
    if na is not None and nb is not None and na != nb:
        caveats.append(
            f"num_samples_evaluated DIFFERS (a={na}, b={nb}): metrics computed "
            "on different prompt counts are NOT comparable"
        )
        warnings.append(f"[{factor}] {a['path']} vs {b['path']}: {caveats[-1]}")
    if na is None or nb is None:
        caveats.append(
            "num_samples_evaluated missing on at least one side — comparability "
            "cannot be verified"
        )

    # Avvertenza obbligatoria 2: metrics_version diversa => scala non coerente.
    va, vb = a["metrics_version"], b["metrics_version"]
    if va is not None and vb is not None and va != vb:
        caveats.append(f"metrics_version DIFFERS (a={va}, b={vb})")
        warnings.append(f"[{factor}] {a['path']} vs {b['path']}: {caveats[-1]}")

    # Fattori non deducibili: NON si assume mai che siano uguali. Se noti e
    # diversi oltre al fattore target, la coppia è debole; se unknown, lo si
    # dichiara (la differenza osservata può includere quel fattore).
    unverified = [
        o
        for o in ("reward_stack", "rl_objective")
        if not _known(fa[o]) or not _known(fb[o])
    ]
    if unverified:
        caveats.append(
            "factors not deducible on at least one side, equality NOT verified: "
            + ", ".join(sorted(set(unverified)))
        )
    confounded = [
        o
        for o in ("reward_stack", "rl_objective")
        if _known(fa[o]) and _known(fb[o]) and fa[o] != fb[o]
    ]
    if confounded:
        caveats.append(f"CONFOUNDED by additional differing factors: {confounded}")

    # Il fingerprint del contesto prompt copre model/dataset/retrieval/
    # grammar/num_samples/prompting_override: per le coppie METHOD dovrebbe
    # coincidere (il fattore method non entra nel payload). Se non coincide,
    # la coppia non è "solo method".
    fpa, fpb = a["prompt_context_fingerprint"], b["prompt_context_fingerprint"]
    if factor == "method" and fpa and fpb and fpa != fpb:
        caveats.append(
            "prompt_context_fingerprint DIFFERS: the eval contexts (prompting/"
            "grammar/num_samples/model) are not identical — this pair may "
            "differ in more than the method factor"
        )
        warnings.append(f"[{factor}] {a['path']} vs {b['path']}: {caveats[-1]}")

    # Delta: SOLO metriche presenti (come float) su ENTRAMBI i lati. Se una
    # chiave manca, il delta è assente — non si ricostruisce con euristiche.
    deltas: dict[str, dict] = {}
    for key in METRIC_KEYS:
        va_, vb_ = a["metrics"].get(key), b["metrics"].get(key)
        if va_ is None or vb_ is None:
            continue
        delta = round(vb_ - va_, 6)
        scale = METRIC_SCALES[key]
        if scale == "0-1" and abs(delta) < NOISE_THRESHOLD:
            interpretation = (
                f"NOT GENERALIZABLE (|delta| < {NOISE_THRESHOLD}): "
                "reproducible (training is bit-deterministic at fixed seed) "
                "but single-seed, so it may invert under another seed — "
                "do NOT interpret"
            )
        else:
            interpretation = (
                "above noise threshold"
                if scale == "0-1"
                else (f"scale {scale}: 0.02 noise threshold not applicable")
            )
        deltas[key] = {
            "a": va_,
            "b": vb_,
            "delta": delta,
            "interpretation": interpretation,
        }

    # Il confronto più interessante: stessa cella valutata in modalità diverse
    # (dual) — misura se il modello ha interiorizzato o dipende dal prompt.
    same_checkpoint = a["run_id"] == b["run_id"] and a["cell"] == b["cell"]
    if factor == "prompting" and same_checkpoint:
        kind = "prompt-dependence (dual eval, same checkpoint)"
        caveats.insert(
            0,
            "HIGHLIGHT: same model evaluated in both prompting "
            "modes — how much does it depend on the prompt as a crutch?",
        )
    elif factor == "prompting":
        kind = "prompting (cross-run)"
        caveats.insert(0, "cross-run comparison: checkpoint identity NOT guaranteed")
    elif factor == "method" and same_checkpoint:
        kind = "training effect (same config, baseline vs checkpoint)"
    else:
        kind = f"{factor} effect (cross-run)"

    return {
        "factor": factor,
        "kind": kind,
        "a": {
            "path": a["path"],
            "timestamp": a["timestamp"],
            "factors": {
                k: fa[k]
                for k in ("model_tag", "method", "variant", "prompting", "grammar")
            },
            "num_samples_evaluated": na,
            "metrics_version": va,
        },
        "b": {
            "path": b["path"],
            "timestamp": b["timestamp"],
            "factors": {
                k: fb[k]
                for k in ("model_tag", "method", "variant", "prompting", "grammar")
            },
            "num_samples_evaluated": nb,
            "metrics_version": vb,
        },
        "deltas": deltas,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Uscite: JSON / Markdown / Grafici
# ---------------------------------------------------------------------------


def build_json_report(
    selected: list[dict],
    superseded: list[dict],
    excluded: list[dict],
    malformed: list[dict],
    comparisons: list[dict],
    missing_pairs: list[dict],
    warnings: list[str],
) -> dict:
    """Struttura JSON completa, leggibile da altri strumenti."""
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "factor_deduction": FACTOR_DEDUCTION_DOC,
        "noise_threshold": {
            "value": NOISE_THRESHOLD,
            "scale": "0-1 metrics",
            "rationale": "training is bit-deterministic at fixed seed, so "
            "same-cell dispersion is zero; but no cell was replicated across "
            "seeds, so deltas below this threshold are reproducible yet not "
            "generalizable",
        },
        "runs": [{k: v for k, v in r.items() if k != "_mtime"} for r in selected],
        "superseded_runs": superseded,
        "excluded_runs": excluded,
        "malformed_files": malformed,
        "warnings": warnings,
        "comparisons": comparisons,
        "missing_pairs": missing_pairs,
    }


def _fmt_metric(v: float | None, delta: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+.4f}" if delta else f"{v:.4f}"


def build_markdown(
    selected: list[dict],
    excluded: list[dict],
    malformed: list[dict],
    comparisons: list[dict],
    missing_pairs: list[dict],
    warnings: list[str],
    superseded: list[dict],
    matrix_metric: str = "rouge_l_mean",
) -> str:
    """Matrice di ablazione in forma tabellare, pronta da leggere."""
    lines: list[str] = []
    lines.append("# Campaign Report — Cross-Factor Ablation Comparison")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append("")
    lines.append("> Complementare ad `ablation-summary`: qui i run sono appaiati")
    lines.append("> per fattore sperimentale. **Legge i JSON di eval: non")
    lines.append("> ricalcola nessuna metrica.**")
    lines.append("")

    # ── Avvertenze obbligatorie ────────────────────────────────────────────
    lines.append("## ⚠️ Mandatory caveats — read before any conclusion")
    lines.append("")
    lines.append(
        f"1. **Interpretability threshold {NOISE_THRESHOLD}** (metrics on the "
        "[0,1] scale): training is bit-deterministic at a fixed seed — two "
        "executions of the same cell produce byte-identical generations — so "
        "there is no run-to-run noise. But no cell was replicated across "
        f"seeds, so **deltas below {NOISE_THRESHOLD} are reproducible yet not "
        "generalizable**: they may invert under another seed. A delta of "
        "0.003 is not a result."
    )
    lines.append(
        "2. **Prompt counts** (`num_samples_evaluated`) are shown for every "
        "run; a pair with differing counts is explicitly flagged (metrics on "
        "different samples are not comparable). Historically this number moved "
        "from 2000 to 3000 prompts."
    )
    lines.append(
        "3. **`metrics_version`** is shown for every run and flagged when it "
        "differs across a pair."
    )
    lines.append(
        "4. **Run dates** are shown for every run: older runs come from "
        "different configurations (e.g. `max_steps` moved 2000→5000)."
    )
    lines.append(
        "5. **Saturated overlap metrics — do not rank on these**: "
        f"{', '.join(f'`{m}`' for m in SATURATED_OVERLAP_METRICS)} are all "
        "within noise of a context-free rule baseline on ASLG-PC12 "
        "(ROUGE-L ~0.97, EM 0.59, see `src/analysis/rule_baseline.py`). "
        f"Use the reward-independent primary metrics instead — "
        f"{', '.join(f'`{m}`' for m in PRIMARY_METRICS)} — for any claim "
        "that a cell translates better, not merely that it scores higher "
        "on the same metric family the reward optimizes."
    )
    next_idx = 6
    if malformed:
        lines.append(
            f"{next_idx}. **{len(malformed)} malformed/partial JSON file(s)** "
            "were skipped (interrupted runs are normal):"
        )
        next_idx += 1
        for m in malformed:
            lines.append(f"   - `{m['path']}` — {m['error'][:120]}")
    if excluded:
        lines.append(
            f"{next_idx}. **{len(excluded)} run(s) EXCLUDED for "
            "`checkpoint_incomplete: true`** (partial checkpoints):"
        )
        next_idx += 1
        for e in excluded:
            lines.append(f"   - `{e['path']}` — {e['reason']}")
    if warnings:
        lines.append(f"{next_idx}. **Pairing warnings:**")
        for w in warnings:
            lines.append(f"   - {w}")
    lines.append("")

    # ── Deduzione dei fattori ──────────────────────────────────────────────
    lines.append("## How factors are deduced")
    lines.append("")
    lines.append("| Factor | Rule | If not deducible |")
    lines.append("|---|---|---|")
    for d in FACTOR_DEDUCTION_DOC:
        lines.append(f"| `{d['factor']}` | {d['rule']} | {d['if_not_deducible']} |")
    lines.append("")

    # ── Run selezionati ────────────────────────────────────────────────────
    if not selected:
        lines.append("## ❌ No eval runs found")
        lines.append("")
        lines.append("Nothing to compare. Run evaluations first.")
        return "\n".join(lines) + "\n"

    lines.append(f"## Selected runs ({len(selected)} typologies, latest per typology)")
    lines.append("")
    lines.append(
        "| Run | method | variant | prompting | grammar | Date | Prompts | "
        "metrics_version |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(selected, key=lambda x: x["path"]):
        f = r["factors"]
        lines.append(
            f"| `{r['path']}` | {f['method']} | {f['variant'] or '—'} | "
            f"{f['prompting']} | {f['grammar']} | {(r['timestamp'] or '?')[:10]} | "
            f"{r['num_samples_evaluated'] if r['num_samples_evaluated'] is not None else '?'} | "
            f"{r['metrics_version'] if r['metrics_version'] is not None else '?'} |"
        )
    lines.append("")
    if superseded:
        lines.append(
            f"Superseded (older runs of the same typology): {len(superseded)} "
            "(see JSON for the full list)."
        )
        lines.append("")

    # ── Matrice di ablazione (vista d'insieme) ─────────────────────────────
    lbl = METRIC_LABELS.get(matrix_metric, matrix_metric)
    lines.append(f"## Ablation matrix overview — {lbl}")
    lines.append("")
    lines.append(
        "One row per cell (its path under `results/`), one column per "
        "prompting mode that has at least one value. `—` = the cell was "
        "not evaluated in that mode. `*` = multiple runs collapsed, latest "
        "shown. Each cell's own `eval_baseline` (the base model in that "
        "cell's context) is not a row here: it repeats the `baseline/` "
        "cells and stays in the paired comparisons below."
    )
    lines.append("")
    lines.extend(_matrix_table(selected, matrix_metric))
    lines.append("")

    # ── Tutte le metriche, per ogni dataset diverso da ASLG-PC12 ───────────
    other_datasets = sorted(
        {r["factors"].get("dataset") for r in selected if r["kind"] != "baseline"}
        - {DEFAULT_DATASET_KEY, None}
    )
    for dataset in other_datasets:
        lines.append(f"## {dataset} — all metrics")
        lines.append("")
        lines.append(
            "Same rows as the matrix above, restricted to this dataset, one "
            "column per metric. On CoNLL-2003 and WOS-46985 read Exact Match "
            "and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU "
            "split `PER:Mary` into `per` + `mary` and give partial credit to "
            "a wrong label of the same type."
        )
        lines.append("")
        lines.extend(_all_metrics_table(selected, dataset))
        lines.append("")

    # ── Confronti appaiati: una tabella per fattore ────────────────────────
    lines.append("## Paired comparisons — one table per factor")
    lines.append("")
    for factor in PAIRABLE_FACTORS:
        pairs = [c for c in comparisons if c["factor"] == factor]
        lines.append(f"### Factor: `{factor}`")
        lines.append("")
        if not pairs:
            mp = next((m for m in missing_pairs if m["factor"] == factor), None)
            lines.append(
                "**No paired comparison available.** " + (mp["reason"] if mp else "")
            )
            lines.append("")
            continue
        for c in pairs:
            highlight = "🔴 " if "HIGHLIGHT" in " ".join(c["caveats"]) else ""
            lines.append(
                f"{highlight}**{c['kind']}** — `{c['a']['path']}` "
                f"({c['a']['timestamp'][:10] if c['a']['timestamp'] else '?'}, "
                f"{c['a']['factors']['prompting']}, {c['a']['factors']['grammar']})"
                " → "
                f"`{c['b']['path']}` "
                f"({c['b']['timestamp'][:10] if c['b']['timestamp'] else '?'}, "
                f"{c['b']['factors']['prompting']}, {c['b']['factors']['grammar']})"
            )
            lines.append("")
            lines.append("| Metric | A | B | Δ (B−A) | Interpretation |")
            lines.append("|---|---|---|---|---|")
            for key in METRIC_KEYS:
                d = c["deltas"].get(key)
                if d is None:
                    continue
                flag = " ⚠️" if d["interpretation"].startswith("NOISE") else ""
                lines.append(
                    f"| {METRIC_LABELS[key]} | {_fmt_metric(d['a'])} | "
                    f"{_fmt_metric(d['b'])} | {_fmt_metric(d['delta'], True)}{flag} | "
                    f"{d['interpretation']} |"
                )
            for cv in c["caveats"]:
                lines.append(f"- ⚠️ {cv}")
            lines.append("")
    lines.append("")

    # ── Coppie mancanti ────────────────────────────────────────────────────
    if missing_pairs:
        lines.append("## Missing paired comparisons (declared, not fabricated)")
        lines.append("")
        for m in missing_pairs:
            lines.append(f"- **`{m['factor']}`**: {m['reason']}.")
            lines.append(
                "  Available typologies: "
                + "; ".join(f"`{t}`" for t in m["available_typologies"])
            )
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append(
        "- `campaign_pairwise_deltas.png` — one panel per factor: "
        "paired deltas as diverging bars with the ±"
        f"{NOISE_THRESHOLD} noise band shaded. Diverging bars make the "
        "SIGN of each effect immediately visible, and the shaded band "
        "makes it impossible to mistake sub-noise deltas for results."
    )
    lines.append(
        "- `campaign_matrix.png` — the ablation matrix as a heatmap "
        "(rows: method/variant, columns: prompting, annotated with "
        f"{lbl}); empty typologies stay visibly grey so gaps in the "
        "grid are impossible to miss."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


_PROMPTING_ORDER = ["zero-shot", "few-shot", UNKNOWN]


def _overview(
    selected: list[dict], metric: str
) -> tuple[list[str], list[str], list[list[float | None]], list[list[str]]]:
    """Righe = celle (percorso), colonne = modalità di prompting con almeno un dato.

    Stessa vista per la tabella Markdown e per ``campaign_matrix.png``, così le
    due non possono divergere.

    Le ``eval_baseline`` delle singole celle restano fuori da questa vista: sono
    il modello base valutato nel contesto di ogni cella, identico fra celle con
    lo stesso contesto, e comparivano come una riga "baseline / <cella>" per
    ognuna. Il modello base è già nelle tre celle ``baseline/``; le eval_baseline
    restano nel JSON e nei confronti appaiati ("training effect").

    Una colonna senza alcun valore non viene mostrata: ``prompting=unknown`` era
    una colonna sempre vuota.

    Returns:
        (celle, colonne, valori, annotazioni). ``None`` = tipologia assente.
    """
    by_cell: dict[str, dict[str, list[dict]]] = {}
    for r in selected:
        if r["kind"] == "baseline":
            continue
        by_cell.setdefault(r["cell_path"], {}).setdefault(
            r["factors"]["prompting"], []
        ).append(r)

    cells = sorted(by_cell, key=cell_sort_key)
    cols = [p for p in _PROMPTING_ORDER if any(p in by_cell[c] for c in cells)]
    values: list[list[float | None]] = []
    annots: list[list[str]] = []
    for c in cells:
        row_v: list[float | None] = []
        row_a: list[str] = []
        for p in cols:
            members = by_cell[c].get(p)
            if not members:
                row_v.append(None)
                row_a.append("—")
                continue
            # Niente medie (ricomputare valori è vietato): si mostra il run
            # più recente e si marca il collasso di più run con '*'.
            latest = max(members, key=lambda r: r["timestamp"] or "")
            mark = "*" if len(members) > 1 else ""
            v = latest["metrics"].get(metric)
            row_v.append(v)
            row_a.append(("n/a" if v is None else _fmt_metric(v)) + mark)
        values.append(row_v)
        annots.append(row_a)
    return cells, cols, values, annots


def _all_metrics_table(selected: list[dict], dataset: str) -> list[str]:
    """Tabella Markdown di un dataset: righe = cella × prompting, colonne = TUTTE
    le metriche di :data:`METRIC_REGISTRY`.

    Per i dataset diversi da ASLG-PC12 la matrice a metrica singola non basta:
    su CoNLL-2003 e WOS-46985 ROUGE-L e BLEU premiano un'etichetta sbagliata
    dello stesso tipo (``PER:John`` contro ``PER:Mary``) e si leggono exact
    match e gloss F1. Stesse regole di :func:`_overview`: niente eval_baseline
    per cella, run più recente, ``*`` se più run sono collassati.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in selected:
        if r["kind"] == "baseline" or r["factors"].get("dataset") != dataset:
            continue
        groups.setdefault((r["cell_path"], r["factors"]["prompting"]), []).append(r)
    keys = sorted(
        groups,
        key=lambda k: (
            cell_sort_key(k[0]),
            _PROMPTING_ORDER.index(k[1]) if k[1] in _PROMPTING_ORDER else 99,
        ),
    )
    lines = [
        "| cell | prompting | "
        + " | ".join(METRIC_LABELS[k] for k in METRIC_KEYS)
        + " |",
        "|---|---|" + "---|" * len(METRIC_KEYS),
    ]
    for cell, prompting in keys:
        members = groups[(cell, prompting)]
        latest = max(members, key=lambda r: r["timestamp"] or "")
        mark = "*" if len(members) > 1 else ""
        vals = [
            (
                "n/a"
                if latest["metrics"].get(k) is None
                else _fmt_metric(latest["metrics"][k])
            )
            for k in METRIC_KEYS
        ]
        lines.append(f"| {cell}{mark} | {prompting} | " + " | ".join(vals) + " |")
    return lines


def _matrix_table(selected: list[dict], metric: str) -> list[str]:
    """Tabella Markdown della matrice (righe = celle, colonne = prompting)."""
    cells, cols, _values, annots = _overview(selected, metric)
    lines = [
        "| cell | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for c, row in zip(cells, annots):
        lines.append(f"| {c} | " + " | ".join(row) + " |")
    return lines


def plot_pairwise_deltas(comparisons: list[dict], output_path: Path) -> bool:
    """Un pannello per fattore: delta appaiati come barre divergenti.

    Returns:
        True se almeno un pannello è stato disegnato.
    """
    factors = [
        f for f in PAIRABLE_FACTORS if any(c["factor"] == f for c in comparisons)
    ]
    if not factors:
        logger.warning("No paired comparisons to plot")
        return False

    fig, axes = plt.subplots(len(factors), 1, figsize=(11, 3.6 * len(factors)))
    if len(factors) == 1:
        axes = [axes]

    for ax, factor in zip(axes, factors):
        comp = next(c for c in comparisons if c["factor"] == factor)
        # Metrics with a delta on both sides, ordered by |delta| desc.
        entries = sorted(
            ((METRIC_LABELS[k], d["delta"]) for k, d in comp["deltas"].items()),
            key=lambda t: abs(t[1]),
            reverse=True,
        )
        if not entries:
            ax.text(0.5, 0.5, "no shared metrics", ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = [e[0] for e in entries]
        values = [e[1] for e in entries]
        y = range(len(entries))
        colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in values]
        ax.barh(y, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        # Banda di rumore: rende visivamente impossibile leggere un delta
        # sub-rumore come un effetto. Le metriche in scala 0-100 esulano.
        ax.axvspan(
            -NOISE_THRESHOLD, NOISE_THRESHOLD, color="grey", alpha=0.25, zorder=0
        )
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        for yi, v in zip(y, values):
            ax.text(
                v,
                yi,
                f" {v:+.4f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )
        ax.set_title(
            f"Effect of '{factor}' — {comp['kind']} (grey band = ±"
            f"{NOISE_THRESHOLD} noise, 0-1 scale metrics)"
        )
        ax.grid(axis="x", alpha=0.3)
        ax.margins(x=0.18)

    fig.suptitle("Campaign Report — paired cross-factor deltas", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pairwise deltas plot saved to %s", output_path)
    return True


def plot_matrix(
    selected: list[dict], output_path: Path, metric: str = "rouge_l_mean"
) -> bool:
    """Vista d'insieme: heatmap celle × modalità di prompting.

    Rampa a un solo tono (più scuro = più alto) su scala fissa 0-1, così il
    colore di una cella si confronta fra un report e l'altro; spazi di 2 px nel
    colore della superficie fra le caselle; colonne in alto e orizzontali.
    """
    cells, cols, values, annots = _overview(selected, metric)
    if not cells:
        return False
    import numpy as np

    lbl = METRIC_LABELS.get(metric, metric)
    cmap = chart_style.sequential_cmap()
    arr = np.ma.masked_invalid(
        np.array(
            [[np.nan if v is None else v for v in row] for row in values], dtype=float
        )
    )

    row_in = 0.32
    fig, ax = plt.subplots(
        figsize=(4.2 + 1.35 * len(cols), 1.3 + row_in * len(cells)),
        facecolor=chart_style.SURFACE,
    )
    ax.set_facecolor(chart_style.SURFACE)
    mesh = ax.pcolormesh(
        arr,
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolors=chart_style.SURFACE,
        linewidth=2,
    )
    for i in range(len(cells)):
        for j in range(len(cols)):
            v = values[i][j]
            if v is None:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=chart_style.INK_MUTED,
                )
                continue
            fill = cmap(float(v))
            hex_fill = "#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in fill[:3]))
            ax.text(
                j + 0.5,
                i + 0.5,
                annots[i][j],
                ha="center",
                va="center",
                fontsize=9,
                color=chart_style.ink_on(hex_fill),
            )

    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(cells), 0)
    ax.xaxis.tick_top()
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels(cols, fontsize=9.5, color=chart_style.INK_SECONDARY)
    ax.set_yticks([i + 0.5 for i in range(len(cells))])
    ax.set_yticklabels(cells, fontsize=8.5, color=chart_style.INK_SECONDARY)
    ax.tick_params(length=0, pad=6)
    for side in ax.spines.values():
        side.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=chart_style.INK_MUTED, length=0)
    starred = any("*" in a for row in annots for a in row)
    fig.suptitle(
        f"{lbl} by cell and prompting mode",
        x=0.01,
        ha="left",
        fontsize=12,
        color=chart_style.INK,
    )
    note = (
        "Latest run per cell. Per-cell base-model evals are in the paired comparisons."
    )
    if starred:
        note += " '*' = several runs collapsed, latest shown."
    fig.text(0.01, 0.005, note, fontsize=8, color=chart_style.INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_path, dpi=150, facecolor=chart_style.SURFACE)
    plt.close(fig)
    logger.info("Matrix plot saved to %s", output_path)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    # Console output carries non-ASCII markers; a cp1252 console (Windows
    # default) would raise on print and lose the run after the files are
    # written but before the summary is shown.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Campaign report: paired cross-factor comparisons of eval runs "
            "for the ablation matrix (JSON + Markdown + plots)"
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing eval results (default: experiments/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/figures",
        help="Output directory for the report files (default: experiments/figures)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rouge_l_mean",
        help="Metric for the matrix overview plot/table (default: rouge_l_mean)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_runs(results_dir)
    selected, superseded = select_latest(discovered["runs"])
    comparisons, missing_pairs, warnings = build_comparisons(selected)

    report = build_json_report(
        selected,
        superseded,
        discovered["excluded"],
        discovered["malformed"],
        comparisons,
        missing_pairs,
        warnings,
    )
    json_path = output_dir / "campaign_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = build_markdown(
        selected,
        discovered["excluded"],
        discovered["malformed"],
        comparisons,
        missing_pairs,
        warnings,
        superseded,
        matrix_metric=args.metric,
    )
    md_path = output_dir / "campaign_report.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\n{'=' * 64}")
    print("  Campaign Report — cross-factor paired comparisons")
    print(f"{'=' * 64}")
    if not selected:
        print(f"\n❌ No eval runs found in {results_dir}/")
        print(f"   JSON: {json_path}\n   MD:   {md_path}")
        return

    print(
        f"\n  Runs: {len(discovered['runs'])} discovered, "
        f"{len(selected)} typologies selected, "
        f"{len(superseded)} superseded"
    )
    if discovered["malformed"]:
        print(
            f"  ⚠️  {len(discovered['malformed'])} malformed/partial JSON "
            "file(s) skipped:"
        )
        for m in discovered["malformed"]:
            print(f"      - {m['path']} ({m['error'][:80]})")
    if discovered["excluded"]:
        print(
            f"  ⛔ {len(discovered['excluded'])} run(s) excluded "
            "(checkpoint_incomplete: true):"
        )
        for e in discovered["excluded"]:
            print(f"      - {e['path']}")

    print(
        f"\n  ⚠️  Interpretability threshold: deltas < {NOISE_THRESHOLD} on "
        "[0,1] metrics"
    )
    print(
        "      are reproducible (bit-deterministic at fixed seed) but "
        "single-seed, so NOT generalizable."
    )
    if warnings:
        print(f"\n  ⚠️  {len(warnings)} pairing warning(s):")
        for w in warnings:
            print(f"      - {w}")

    for factor in PAIRABLE_FACTORS:
        pairs = [c for c in comparisons if c["factor"] == factor]
        print(f"\n── Factor: {factor} ──")
        if not pairs:
            mp = next((m for m in missing_pairs if m["factor"] == factor), None)
            print(
                "  ❌ No paired comparison available"
                + (f" ({mp['reason']})" if mp else "")
            )
            continue
        for c in pairs:
            star = " 🔴" if "HIGHLIGHT" in " ".join(c["caveats"]) else ""
            print(f"  {c['kind']}{star}")
            print(f"    a: {c['a']['path']}")
            print(f"    b: {c['b']['path']}")
            keys = [k for k in METRIC_KEYS if k in c["deltas"]]
            for key in keys[:6]:
                d = c["deltas"][key]
                flag = " [NOISE]" if d["interpretation"].startswith("NOISE") else ""
                print(
                    f"      {METRIC_LABELS[key]:<16} "
                    f"{d['a']:.4f} → {d['b']:.4f}  Δ={d['delta']:+.4f}{flag}"
                )
            if len(keys) > 6:
                print(f"      … +{len(keys) - 6} more metrics (see JSON/MD)")
            for cv in c["caveats"]:
                print(f"      ⚠️  {cv}")

    if missing_pairs:
        print("\n  Missing paired comparisons (declared, not fabricated):")
        for m in missing_pairs:
            print(f"      - {m['factor']}: {m['reason']}")

    png1 = output_dir / "campaign_pairwise_deltas.png"
    png2 = output_dir / "campaign_matrix.png"
    plotted1 = plot_pairwise_deltas(comparisons, png1)
    plot_matrix(selected, png2, metric=args.metric)

    print(f"\n{'=' * 64}")
    print(f"  JSON:   {json_path}")
    print(f"  MD:     {md_path}")
    print(f"  Chart:  {png1} ({'written' if plotted1 else 'skipped: no pairs'})")
    print(f"  Matrix: {png2}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
