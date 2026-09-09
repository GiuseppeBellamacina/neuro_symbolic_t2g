"""Guardia sull'entrypoint di training: le celle eval-only non sono addestrabili.

Contesto del bug reale (job SLURM 7294, 8 set 2026): lanciare
`baseline/zero-shot-no-grammar.yaml` con `cluster/train.sh` produceva
``KeyError: 'output_dir'`` DOPO aver caricato Unsloth, il modello e il dataset —
cioe' un messaggio incomprensibile a minuti dall'avvio, su un nodo GPU
allocato. Le celle `baseline/*` sono eval-only per costruzione: ereditano una
sezione ``training`` parziale da ``base.yaml`` e non dichiarano ne'
``output_dir`` ne' ``log_dir`` perche' non addestrano nulla.

Questi test verificano che l'errore sia ora esplicito e immediato, e che indichi
il comando corretto.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "experiments/configs/qwen25-05b"

# Celle dichiaratamente eval-only: nessun training, quindi nessuna output_dir.
EVAL_ONLY = [
    "baseline/zero-shot.yaml",
    "baseline/zero-shot-no-grammar.yaml",
    "baseline/few-shot.yaml",
]

# Celle addestrabili: devono avere output_dir/log_dir E step di training.
TRAINABLE = [
    "sft/zero-shot.yaml",
    "grpo/zero-shot.yaml",
    "grpo/few-shot.yaml",
    "sft-grpo/zero-shot.yaml",
    "sft-grpo/few-shot.yaml",
]


def _resolve(rel: str) -> dict:
    from src.utils.config import resolve_config

    return resolve_config(CONFIG_ROOT / rel)


@pytest.mark.parametrize("rel", EVAL_ONLY)
def test_eval_only_cells_declare_no_output_dir(rel):
    """Invariante di progetto: una cella eval-only non ha directory di output.

    Se questo test inizia a fallire, la cella e' diventata addestrabile e va
    spostata fuori da `baseline/`.

    WHY solo output_dir/log_dir e non le chiavi di step: `max_steps` e
    `num_train_epochs` vivono in `base.yaml` perche' sono comuni a tutte le
    celle addestrabili, quindi le celle eval-only le EREDITANO e la loro
    presenza non distingue piu' nulla. Il segnale e' `output_dir`, che e'
    per-cella per costruzione (ogni cella scrive in una directory propria) e
    che il resto del sistema usa gia': `eval_t2g.py` ne deduce
    `eval_baseline_only` e `compare`, la guardia in `src/training/__main__.py`
    rifiuta con exit 2 una cella senza `output_dir` lanciata come training, e
    `tests/validate_configs.py::_detect_kind` classifica allo stesso modo.
    """
    training = _resolve(rel).get("training", {})
    assert "output_dir" not in training
    assert "log_dir" not in training


@pytest.mark.parametrize("rel", TRAINABLE)
def test_trainable_cells_declare_output_dirs(rel):
    """Ogni cella addestrabile deve avere output_dir, log_dir e uno step budget.

    Questa e' la meta' che impedisce il bug opposto: un config che dichiara di
    addestrare ma non dice dove salvare.
    """
    training = _resolve(rel).get("training", {})
    assert "output_dir" in training, f"{rel}: manca training.output_dir"
    assert "log_dir" in training, f"{rel}: manca training.log_dir"
    assert {"max_steps", "num_train_epochs"} & set(
        training
    ), f"{rel}: manca max_steps/num_train_epochs"


def test_grpo_main_refuses_eval_only_config_with_actionable_message(tmp_path):
    """L'entrypoint GRPO deve rifiutare una cella eval-only con SystemExit.

    Si verifica il messaggio, non solo il tipo di eccezione: il valore di questo
    fix e' proprio che l'operatore capisca cosa fare senza leggere il codice.
    """
    from src.training import grpo_t2g_train

    cfg = {
        "model": {"name": "stub", "num_gpus": 1},
        "dataset": {"seed": 42},
        # Sezione training parziale, come quella ereditata da base.yaml.
        "training": {"per_device_train_batch_size": 1},
        "wandb": {"project": "p", "run_name": "r"},
    }
    config_path = tmp_path / "eval_only.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    import sys

    original_argv = sys.argv
    sys.argv = ["prog", "--config", str(config_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            grpo_t2g_train.main()
    finally:
        sys.argv = original_argv

    message = str(excinfo.value)
    assert "non addestrabile" in message
    assert "output_dir" in message
    assert "EVAL-ONLY" in message
    # Deve indicare il comando corretto, non solo lamentarsi.
    assert "cluster/eval.sh" in message


def test_grpo_main_reports_missing_dirs_when_steps_are_declared(tmp_path):
    """Se la cella dichiara step ma non le directory, il messaggio e' diverso.

    Qui non e' un errore d'uso: e' un config incompleto, e va detto cosi'.
    """
    from src.training import grpo_t2g_train

    cfg = {
        "model": {"name": "stub", "num_gpus": 1},
        "dataset": {"seed": 42},
        "training": {"max_steps": 100},
        "wandb": {"project": "p", "run_name": "r"},
    }
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    import sys

    original_argv = sys.argv
    sys.argv = ["prog", "--config", str(config_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            grpo_t2g_train.main()
    finally:
        sys.argv = original_argv

    message = str(excinfo.value)
    assert "non addestrabile" in message
    assert "aggiungi training.output_dir" in message
    # NON deve suggerire eval.sh: la cella vuole addestrare.
    assert "cluster/eval.sh" not in message
