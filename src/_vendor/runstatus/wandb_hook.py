"""Mirror every ``wandb.log`` call (online or offline) into the status file."""
from __future__ import annotations

from .core import RunStatus


def attach_wandb(status: RunStatus, wandb_module=None):
    """Wrap ``wandb.log``; call after ``wandb.init``. Returns the module."""
    if wandb_module is None:
        import wandb as wandb_module
    run = getattr(wandb_module, "run", None)
    if run is not None:
        status.set(wandb={"id": getattr(run, "id", None), "url": getattr(run, "url", None),
                          "mode": getattr(getattr(run, "settings", None), "mode", None)})
    orig = wandb_module.log

    def log(data=None, *a, **kw):
        try:
            if isinstance(data, dict):
                step = kw.get("step", data.get("trainer/global_step", data.get("step")))
                status.log(step if isinstance(step, int) else None, **data)
        except Exception:
            pass
        return orig(data, *a, **kw)

    wandb_module.log = log
    return wandb_module
