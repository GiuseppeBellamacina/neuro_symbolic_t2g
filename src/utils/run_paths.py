"""Where a run's outputs live: ``experiments/{results,figures,logs}/<cella>/run_<ts>/``.

La cella rispecchia il percorso del config sotto ``experiments/configs/``, a qualunque
profondità: ``qwen25-05b/grpo/zero-shot`` è a tre livelli,
``qwen25-05b/ablations/loss/dr-grpo`` a quattro, ``qwen25-05b/baseline/zero-shot`` a
tre. Ogni esecuzione è una directory ``run_<timestamp>`` dentro la cella, sempre: è
l'unica invariante su cui si appoggiano i lettori (ablation summary, campaign report,
cache della baseline, helper remoto).

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

__all__ = ["cell_from_config", "eval_output_location", "split_checkpoint_path"]


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
        >>> cell_from_config("experiments/configs/qwen25-05b/baseline/zero-shot.yaml")
        'qwen25-05b/baseline/zero-shot'
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
        >>> split_checkpoint_path("experiments/checkpoints/qwen25-05b/grpo/zero-shot/run_1/final")
        ('qwen25-05b/grpo/zero-shot', 'run_1')
        >>> split_checkpoint_path("experiments/checkpoints/qwen25-05b/grpo/final") is None
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
      cella è il percorso del config (``configs/qwen25-05b/baseline/zero-shot.yaml``
      -> ``qwen25-05b/baseline/zero-shot``) e si apre una run nuova
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
