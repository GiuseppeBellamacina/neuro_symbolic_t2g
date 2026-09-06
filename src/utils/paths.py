"""Canonical experiment identities and artifact paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

Method = Literal["base", "sft", "grpo", "sft-grpo"]
TrainPromptMode = Literal["none", "zero-shot", "few-shot"]
PromptMode = Literal["zero-shot", "retrieval", "few-shot"]
Variant = Literal["none", "pda", "hot"]
ExperimentKind = Literal["baseline", "train", "ablation", "probe"]
ArtifactKind = Literal["logs", "checkpoints", "results", "figures"]

METHODS = frozenset({"base", "sft", "grpo", "sft-grpo"})
TRAIN_PROMPT_MODES = frozenset({"none", "zero-shot", "few-shot"})
VARIANTS = frozenset({"none", "pda", "hot"})
EXPERIMENT_KINDS = frozenset({"baseline", "train", "ablation", "probe"})
PROMPT_LABELS: Mapping[str, str] = {
    "zero-shot": "zero-shot",
    "retrieval": "few-shot",
    "few-shot": "few-shot",
}
_TOKEN_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RUN_RE = re.compile(r"^run_(\d{8}_\d{6})$")


@dataclass(frozen=True, slots=True)
class Cell:
    model_tag: str
    method: Method
    train_prompt_mode: TrainPromptMode
    variant: Variant
    kind: ExperimentKind = "train"

    def __post_init__(self) -> None:
        if not _TOKEN_RE.fullmatch(self.model_tag):
            raise ValueError(f"invalid model tag: {self.model_tag!r}")
        if self.method not in METHODS:
            raise ValueError(f"invalid method: {self.method!r}")
        if self.train_prompt_mode not in TRAIN_PROMPT_MODES:
            raise ValueError(f"invalid train prompt mode: {self.train_prompt_mode!r}")
        if self.variant not in VARIANTS:
            raise ValueError(f"invalid variant: {self.variant!r}")
        if self.kind not in EXPERIMENT_KINDS:
            raise ValueError(f"invalid experiment kind: {self.kind!r}")
        if self.kind in {"train", "ablation"} and self.train_prompt_mode == "none":
            raise ValueError("training cells require a train prompt mode")
        if self.kind == "ablation" and self.variant == "none":
            raise ValueError("ablation cells require a variant")
        if self.kind != "ablation" and self.variant != "none":
            raise ValueError("variants belong only to ablation cells")


@dataclass(frozen=True, slots=True)
class RunPath:
    cell: Cell
    run_id: str
    prompt_mode: PromptMode | None = None

    def __post_init__(self) -> None:
        if not _RUN_RE.fullmatch(self.run_id):
            raise ValueError(f"invalid run id: {self.run_id!r}")
        if self.prompt_mode is not None and self.prompt_mode not in PROMPT_LABELS:
            raise ValueError(f"invalid prompt mode: {self.prompt_mode!r}")

    def path(self, root: str | Path, kind: ArtifactKind) -> Path:
        return cell_base_dir(root, kind, self.cell, self.prompt_mode) / self.run_id


def model_tag(model_id: str | None) -> str:
    value = (model_id or "").lower()
    if re.search(r"qwen(?:2[._-]?5|25).*0[._-]?5b", value):
        return "qwen25-05b"
    leaf = value.rstrip("/").rsplit("/", 1)[-1]
    tag = re.sub(r"[^a-z0-9]+", "-", leaf).strip("-")
    if not tag:
        raise ValueError("model ID is required when experiment.model_tag is absent")
    return tag


def prompt_label(prompt_mode: str) -> str:
    try:
        return PROMPT_LABELS[prompt_mode]
    except KeyError as exc:
        raise ValueError(f"invalid prompt mode: {prompt_mode!r}") from exc


def is_run_id(name: str) -> bool:
    return _RUN_RE.fullmatch(name) is not None


def new_run_id(now: datetime | None = None) -> str:
    return f"run_{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}"


def cell_from_config(config: Mapping[str, Any]) -> Cell:
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("config.experiment identity is required")
    missing = [
        key
        for key in ("method", "train_prompt_mode", "variant", "kind")
        if not experiment.get(key)
    ]
    if missing:
        raise ValueError("config.experiment requires: " + ", ".join(missing))
    model = experiment.get("model_tag")
    if not model:
        model_cfg = config.get("model")
        model = model_tag(
            model_cfg.get("name") if isinstance(model_cfg, Mapping) else None
        )
    return Cell(
        str(model),
        str(experiment["method"]),  # type: ignore[arg-type]
        str(experiment["train_prompt_mode"]),  # type: ignore[arg-type]
        str(experiment["variant"]),  # type: ignore[arg-type]
        str(experiment["kind"]),  # type: ignore[arg-type]
    )


def model_root(root: str | Path, kind: ArtifactKind, model: str) -> Path:
    return Path(root) / kind / model


def cell_base_dir(
    root: str | Path,
    kind: ArtifactKind,
    cell: Cell,
    prompt_mode: PromptMode | None = None,
) -> Path:
    path = model_root(root, kind, cell.model_tag)
    identity = "baseline" if cell.kind == "baseline" else cell.method
    path /= Path(identity) / cell.train_prompt_mode
    if cell.kind == "ablation":
        path /= Path("ablations") / cell.variant
    if kind in {"results", "figures"}:
        if cell.kind == "baseline":
            if (
                prompt_mode is not None
                and prompt_label(prompt_mode) != cell.train_prompt_mode
            ):
                raise ValueError("baseline evaluation prompt must match its identity")
            return path
        if prompt_mode is None:
            raise ValueError(f"prompt_mode is required for {kind}")
        path /= f"eval-{prompt_label(prompt_mode)}"
    elif prompt_mode is not None:
        raise ValueError(f"prompt_mode does not belong under {kind}")
    return path


def evaluation_log_dir(
    root: str | Path,
    run: RunPath,
) -> Path:
    """Return the prompt-specific log/W&B directory for one eval leg."""
    if run.prompt_mode is None:
        raise ValueError("prompt_mode is required for evaluation logs")
    return (
        cell_base_dir(root, "logs", run.cell)
        / run.run_id
        / f"eval-{prompt_label(run.prompt_mode)}"
    )


def evaluation_identifier(run: RunPath) -> str:
    """Stable method/train-prompt/ablation/eval-mode display identity."""
    if run.prompt_mode is None:
        raise ValueError("prompt_mode is required for evaluation identity")
    parts = [run.cell.method, run.cell.train_prompt_mode]
    if run.cell.kind == "ablation":
        parts.extend(("ablations", run.cell.variant))
    parts.append(f"eval-{prompt_label(run.prompt_mode)}")
    return "/".join(parts)


def latest_run_dir(base: str | Path) -> Path | None:
    runs = sorted(
        path
        for path in Path(base).glob("run_*")
        if path.is_dir() and is_run_id(path.name)
    )
    return runs[-1] if runs else None


def training_run_paths(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    root: str | Path = "experiments",
    now: datetime | None = None,
) -> tuple[Path, Path, str, Cell]:
    cell = cell_from_config(config)
    output_base = cell_base_dir(root, "checkpoints", cell)
    log_base = cell_base_dir(root, "logs", cell)
    existing = latest_run_dir(output_base) if resume else None
    run_id = existing.name if existing is not None else new_run_id(now)
    output = existing if existing is not None else output_base / run_id
    return output, log_base / run_id, run_id, cell


def cell_run_from_checkpoint(checkpoint: str | Path) -> RunPath:
    path = Path(checkpoint)
    run_dir = next(
        (parent for parent in (path, *path.parents) if is_run_id(parent.name)), None
    )
    if run_dir is None:
        raise ValueError(f"checkpoint is not inside a canonical run directory: {path}")
    leaf = run_dir.parent
    if leaf.parent.name == "ablations" and leaf.parent.parent.parent.name in METHODS:
        variant = leaf.name
        prompt_dir = leaf.parent.parent
        method_dir = prompt_dir.parent
        kind = "ablation"
    elif leaf.parent.name in METHODS:
        variant = "none"
        prompt_dir = leaf
        method_dir = leaf.parent
        kind = "train"
    else:
        raise ValueError(
            f"checkpoint is not in the canonical model/method/variant tree: {path}"
        )
    cell = Cell(
        method_dir.parent.name,
        method_dir.name,  # type: ignore[arg-type]
        prompt_dir.name,  # type: ignore[arg-type]
        variant,  # type: ignore[arg-type]
        kind,  # type: ignore[arg-type]
    )
    return RunPath(cell, run_dir.name)


def experiment_root(path: str | Path, kind: ArtifactKind) -> Path:
    value = Path(path)
    for parent in (value, *value.parents):
        if parent.name == kind and parent.parent.name == "experiments":
            return parent.parent
    raise ValueError(f"path is not under experiments/{kind}: {path}")


def iter_runs(
    root: str | Path,
    kind: ArtifactKind,
    cell: Cell,
    prompt_mode: PromptMode | None = None,
) -> Iterator[Path]:
    parent = cell_base_dir(root, kind, cell, prompt_mode)
    if parent.is_dir():
        yield from sorted(
            (
                child
                for child in parent.iterdir()
                if child.is_dir() and is_run_id(child.name)
            ),
            reverse=True,
        )


def newest_checkpoint(root: str | Path, cell: Cell) -> Path | None:
    for run in iter_runs(root, "checkpoints", cell):
        final = run / "final"
        if final.is_dir():
            return final
        checkpoints = sorted(
            (path for path in run.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: (
                int(path.name.partition("-")[2])
                if path.name.partition("-")[2].isdigit()
                else -1
            ),
            reverse=True,
        )
        if checkpoints:
            return checkpoints[0]
    return None


def wandb_name(run: RunPath) -> str:
    parts = [run.cell.model_tag, run.cell.method]
    parts.append(run.cell.train_prompt_mode)
    if run.cell.kind == "ablation":
        parts.extend(("ablations", run.cell.variant))
    if run.prompt_mode is not None:
        parts.append(prompt_label(run.prompt_mode))
    parts.append(run.run_id)
    return "/".join(parts)


def wandb_tags(run: RunPath) -> tuple[str, ...]:
    tags = [run.cell.model_tag, run.cell.method]
    tags.append(run.cell.train_prompt_mode)
    if run.cell.kind == "ablation":
        tags.append(run.cell.variant)
    if run.prompt_mode is not None:
        tags.append(prompt_label(run.prompt_mode))
    return tuple(tags)
