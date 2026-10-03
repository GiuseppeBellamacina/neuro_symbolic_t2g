"""Where a run's outputs live: ``experiments/{results,figures,logs}/<cella>/run_<ts>/``.

La cella rispecchia il percorso del config sotto ``experiments/configs/``, a qualunque
profondità, e comincia SEMPRE con ``<dataset>/<modello>/``:
``aslg-pc12/qwen25-05b/grpo/zero-shot``,
``aslg-pc12/qwen25-05b/ablations/loss/dr-grpo``,
``phoenix-2014t/qwen25-05b/baseline/zero-shot``. Ogni esecuzione è una directory
``run_<timestamp>`` dentro la cella, sempre: è l'unica invariante su cui si appoggiano
i lettori (ablation summary, campaign report, cache della baseline, helper remoto).

Perché dataset PRIMA del modello: i numeri sono confrontabili solo dentro lo stesso
dataset (test set, vocabolario del Trie, coppia di lingue e saturazione delle metriche
cambiano tutti con il corpus), mentre modelli diversi sullo stesso dataset sono
esattamente ciò che si confronta. Con ``<dataset>/`` in testa tutto ciò che è
confrontabile sta sotto una sola radice, e lo stesso confine vale per gli artefatti che
si possono riusare (adapter SFT, cache della baseline, vocabolario in ``data/``): mai
attraverso dataset diversi.

Il layout legacy senza dataset (``qwen25-05b/grpo/zero-shot``, ASLG-PC12 implicito) è
ancora letto da :func:`split_cell` / :func:`cell_tag` / :func:`cell_sort_key`, così i
risultati già prodotti restano leggibili; ``cluster/migrate_dataset_layout.sh`` li
sposta nel layout nuovo.

Due modi in cui questa invariante è stata violata, e che le funzioni qui sotto
escludono:

* prendere un numero FISSO di segmenti dopo ``checkpoints/`` faceva confluire nella
  stessa directory ogni cella che condivideva i primi due segmenti (``grpo/zero-shot`` e
  ``grpo/few-shot`` in ``results/qwen25-05b/grpo/``), ciascuna eval sovrascrivendo la
  precedente;
* senza un segmento ``run_*`` i file finivano direttamente nella directory di gruppo
  (``results/qwen25-05b/sft/eval_final.json``), orfani: senza identificativo di run né
  riferimento al config, quindi non attribuibili a nessuna cella.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "DATASET_KEYS",
    "DEFAULT_DATASET_KEY",
    "cell_from_config",
    "cell_sort_key",
    "cell_tag",
    "eval_output_location",
    "split_cell",
    "split_checkpoint_path",
]

#: Segmenti di dataset riconosciuti in testa a una cella. Rispecchia le chiavi di
#: ``src.datasets.registry.DATASETS`` (verificato da un test): duplicato qui perché
#: questo modulo resta importabile senza ``datasets``/``numpy``.
DATASET_KEYS: tuple[str, ...] = (
    "aslg-pc12",
    "phoenix-2014t",
    "wos-46985",
    "conll-2003",
)

#: Dataset implicito del layout legacy (cartelle senza segmento di dataset).
DEFAULT_DATASET_KEY = "aslg-pc12"


def split_cell(cell: str) -> tuple[str, str, str]:
    """Split a cell into ``(dataset, modello, resto)``.

    Il layout legacy (senza segmento di dataset) è ASLG-PC12 implicito.

    Examples:
        >>> split_cell("phoenix-2014t/qwen25-05b/ablations/loss/dr-grpo")
        ('phoenix-2014t', 'qwen25-05b', 'ablations/loss/dr-grpo')
        >>> split_cell("qwen25-05b/grpo/few-shot")
        ('aslg-pc12', 'qwen25-05b', 'grpo/few-shot')
    """
    parts = [p for p in cell.strip("/").split("/") if p]
    if parts and parts[0] in DATASET_KEYS:
        dataset, parts = parts[0], parts[1:]
    else:
        dataset = DEFAULT_DATASET_KEY
    model = parts[0] if parts else ""
    return dataset, model, "/".join(parts[1:])


def cell_tag(cell: str) -> str:
    """Tag di job/monitor di una cella: il resto sotto il modello, ``/`` → ``-``.

    Ogni tag comincia con la chiave del dataset, ASLG-PC12 compreso
    (``aslg-pc12-grpo-few-shot``, ``phoenix-2014t-grpo-few-shot``): due dataset
    non si contendono mai lo stesso tag e nella TUI ogni job dice a quale
    dataset appartiene. Il modello NON entra nel tag. Le celle legacy senza
    segmento di dataset sono ASLG-PC12 e ricevono lo stesso prefisso.

    Examples:
        >>> cell_tag("aslg-pc12/qwen25-05b/ablations/loss/dr-grpo")
        'aslg-pc12-ablations-loss-dr-grpo'
        >>> cell_tag("phoenix-2014t/qwen25-05b/grpo/few-shot")
        'phoenix-2014t-grpo-few-shot'
    """
    dataset, _model, rest = split_cell(cell)
    return f"{dataset}-" + rest.replace("/", "-").replace("_", "-")


def _after_last(parts: tuple[str, ...], anchor: str) -> tuple[str, ...] | None:
    """Segmenti dopo l'ULTIMA occorrenza di ``anchor``, o None se assente.

    L'ultima e non la prima: il percorso del repository stesso può contenere un
    segmento con lo stesso nome.
    """
    if anchor not in parts:
        return None
    return parts[len(parts) - parts[::-1].index(anchor) :]


def cell_from_config(config_path: str | Path) -> str | None:
    """Cella di un config: il suo percorso sotto ``configs/``, senza estensione.

    Examples:
        >>> cell_from_config(
        ...     "experiments/configs/aslg-pc12/qwen25-05b/baseline/zero-shot.yaml"
        ... )
        'aslg-pc12/qwen25-05b/baseline/zero-shot'
        >>> cell_from_config("/tmp/prova.yaml") is None
        True
    """
    after = _after_last(Path(config_path).resolve().with_suffix("").parts, "configs")
    return "/".join(after) if after else None


def split_checkpoint_path(path: str | Path) -> tuple[str, str] | None:
    """Split a checkpoint (or in-checkpoint file) path into ``(cella, run_id)``.

    ``cella`` is every segment between ``checkpoints/`` and the ``run_*`` directory,
    joined with ``/`` so callers can use it directly as a nested output path.
    ``run_id`` is the ``run_*`` directory itself.

    Args:
        path: Any path inside ``experiments/checkpoints/`` — an adapter dir, a ``final/``
            dir, or a file such as ``trainer_state.json``.

    Returns:
        ``(cella, run_id)``, or ``None`` when *path* has no ``checkpoints`` segment or no
        ``run_*`` segment below it. The caller then derives the cell from the config and
        opens a fresh ``run_<ts>``: guessing a pair from a non-standard path is what
        used to drop orphan files straight into a group directory.

    Examples:
        >>> split_checkpoint_path(
        ...     "experiments/checkpoints/aslg-pc12/qwen25-05b/grpo/zero-shot/run_1/final"
        ... )
        ('aslg-pc12/qwen25-05b/grpo/zero-shot', 'run_1')
        >>> split_checkpoint_path(
        ...     "experiments/checkpoints/aslg-pc12/qwen25-05b/grpo/final"
        ... ) is None
        True
    """
    after = _after_last(Path(path).resolve().parts, "checkpoints")
    if not after:
        return None

    # The LAST run_* wins: the SFT sub-phase nests under the GRPO run dir
    # (run_<ts>/sft_pretrain/final), never the other way around.
    run_positions = [i for i, seg in enumerate(after) if seg.startswith("run_")]
    if not run_positions or run_positions[-1] == 0:
        return None
    pos = run_positions[-1]
    return "/".join(after[:pos]), after[pos]


def eval_output_location(
    config_path: str | Path,
    config: Mapping[str, Any],
    checkpoint: str | Path | None,
    timestamp: str,
) -> tuple[str, str, str]:
    """Dove scrive un'eval: ``(cella, run_id, etichetta dei grafici)``.

    * Checkpoint sotto ``experiments/checkpoints/<cella>/run_*/``: stessa cella e
      stessa run del training, così l'eval sta accanto al modello che valuta.
    * Nessun checkpoint (le tre baseline) o checkpoint fuori da quell'albero: la
      cella è il percorso del config
      (``configs/aslg-pc12/qwen25-05b/baseline/zero-shot.yaml`` ->
      ``aslg-pc12/qwen25-05b/baseline/zero-shot``) e si apre una run nuova
      ``run_<timestamp>``. Solo un config fuori da ``configs/`` ricade sul nome
      del run o del modello.

    Raises:
        ValueError: se il risultato non è ``<cella>/run_*``. È l'invariante su cui
            si appoggiano tutti i lettori (ablation summary, campaign report,
            cache della baseline, helper remoto): un file scritto fuori da una run
            è un orfano, e meglio fermarsi che produrlo.
    """
    cell = cell_from_config(config_path) or (
        config.get("wandb", {}).get("run_name")
        or str(config["model"]["name"]).split("/")[-1].lower().replace(".", "")
    )
    fresh_run = f"run_{timestamp}"

    if checkpoint is None:
        location = (cell, fresh_run, "zero-shot")
    else:
        ckpt = Path(checkpoint).resolve()
        split = split_checkpoint_path(ckpt)
        if split is not None:
            location = (split[0], split[1], split[1])
        else:
            label = ckpt.parent.name if ckpt.name == "final" else ckpt.name
            location = (cell, fresh_run, label)

    if not location[0] or not location[1].startswith("run_"):
        raise ValueError(
            "eval output fuori dal layout <cella>/run_<ts>/: "
            f"cella={location[0]!r}, run={location[1]!r}"
        )
    return location


#: Ordine di lettura delle famiglie di celle dentro un modello: prima i
#: riferimenti, poi i metodi dal più semplice, infine le ablazioni.
_FAMILY_ORDER = ("baseline", "sft", "sft-grpo", "grpo", "ablations")


def cell_sort_key(cell: str) -> tuple[str, str, int, str]:
    """Chiave d'ordinamento di una cella (``aslg-pc12/qwen25-05b/grpo/few-shot``).

    Per dataset, poi per modello, poi per famiglia (baseline -> sft -> sft-grpo ->
    grpo -> ablazioni), poi alfabetico. Tabelle e grafici la condividono, così una
    cella occupa la stessa posizione ovunque e dataset e modelli restano raggruppati.
    Le celle legacy senza dataset si ordinano come ASLG-PC12.

    Examples:
        >>> sorted(["m/grpo/a", "m/ablations/x", "m/baseline/z"], key=cell_sort_key)
        ['m/baseline/z', 'm/grpo/a', 'm/ablations/x']
        >>> sorted(
        ...     ["phoenix-2014t/m/baseline/z", "aslg-pc12/m/grpo/a"], key=cell_sort_key
        ... )
        ['aslg-pc12/m/grpo/a', 'phoenix-2014t/m/baseline/z']
    """
    dataset, model, rest = split_cell(cell)
    family = rest.split("/", 1)[0] if rest else ""
    rank = (
        _FAMILY_ORDER.index(family) if family in _FAMILY_ORDER else len(_FAMILY_ORDER)
    )
    return (dataset, model, rank, cell)
