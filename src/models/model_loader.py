"""Model loading utilities: HuggingFace model + tokenizer + LoRA + quantization
for the neuro_symbolic_t2g project.

Supports two backends:
  - Standard HuggingFace (transformers + peft + bitsandbytes)
  - Unsloth (2-5x faster training, ~50-70% less VRAM)

Set  model.use_unsloth: true  in your config YAML to enable Unsloth.
Set  model.fast_inference: true  to use vLLM-backed generation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def get_quantization_config(
    quantization: str, dtype: str = "bfloat16"
) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig based on the quantization string."""
    if quantization == "4bit":
        compute_dtype = getattr(torch, dtype, torch.bfloat16)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer(model_name: str) -> Any:
    """Load and configure the tokenizer.

    Sets pad_token to eos_token if missing, and forces padding_side="left"
    (required for batched generation).
    """
    if is_main_process():
        logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "pad_token not set → using eos_token (%r)",
            tokenizer.eos_token,
        )
    tokenizer.padding_side = "left"
    return tokenizer


# ---------------------------------------------------------------------------
# Standard HuggingFace backend
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    quantization: str = "4bit",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Any:
    """Load a causal LM with optional quantization."""
    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    quant_config = get_quantization_config(quantization, dtype=dtype)
    if is_main_process():
        logger.info(
            "Loading %s (quantization=%s, dtype=%s, device_map=%s)",
            model_name,
            quantization,
            dtype,
            device_map,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


def apply_lora(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Apply LoRA adapters to the model via PEFT."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if is_main_process():
        logger.info(
            "Applying LoRA: r=%d, alpha=%d, dropout=%s, targets=%s",
            r,
            lora_alpha,
            lora_dropout,
            target_modules,
        )

    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _load_with_transformers(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Load via standard HuggingFace transformers + PEFT."""
    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    logger.info(
        "Backend: HuggingFace (transformers + peft) — quantization=%s",
        model_cfg.get("quantization", "none"),
    )

    model = load_model(
        model_name=model_cfg["name"],
        quantization=model_cfg.get("quantization", "4bit"),
        dtype=model_cfg.get("dtype", "bfloat16"),
    )
    tokenizer = load_tokenizer(model_cfg["name"])

    if lora_cfg:
        model = apply_lora(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Unsloth backend
# ---------------------------------------------------------------------------


def _resolve_fast_inference(model_cfg: dict[str, Any]) -> bool:
    """Determine if fast_inference (vLLM) can be enabled.

    Returns True only when the config flag is set AND vLLM is installed.
    """
    requested = model_cfg.get("fast_inference", False)
    if not requested:
        return False

    try:
        import vllm  # noqa: F401
    except ImportError:
        warnings.warn(
            "fast_inference requires vllm — package not found, disabling.",
            stacklevel=2,
        )
        return False

    return True


def _load_with_unsloth(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Load model + tokenizer via Unsloth's FastLanguageModel.

    Unsloth patches the model in-place with fused kernels and handles
    LoRA + 4-bit quantization internally.

    When ``model.fast_inference`` is ``True`` and vLLM is available,
    enables vLLM-backed generation for faster GRPO rollouts.
    """
    from unsloth import FastLanguageModel

    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    quantization = model_cfg.get("quantization", "4bit")
    load_in_4bit = quantization == "4bit"

    logger.info(
        "Loading %s with Unsloth (4bit=%s, max_seq=%d)",
        model_cfg["name"],
        load_in_4bit,
        model_cfg.get("max_seq_length", 1024),
    )

    use_fast_inference = _resolve_fast_inference(model_cfg)

    fi_kwargs: dict[str, Any] = {}
    if use_fast_inference:
        fi_kwargs["fast_inference"] = True
        fi_kwargs["max_lora_rank"] = lora_cfg.get("r", 16)
        fi_kwargs["gpu_memory_utilization"] = model_cfg.get(
            "gpu_memory_utilization", 0.9
        )
        fi_kwargs["unsloth_vllm_standby"] = model_cfg.get("vllm_standby", False)
        logger.info("fast_inference=ON (vLLM backend)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_length", 1024),
        load_in_4bit=load_in_4bit,
        dtype=None,
        **fi_kwargs,
    )

    # Apply LoRA via Unsloth
    if lora_cfg:
        target_modules = lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        logger.info(
            "Unsloth LoRA: r=%d, alpha=%d, targets=%s",
            lora_cfg.get("r", 16),
            lora_cfg.get("lora_alpha", 32),
            target_modules,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0),
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=lora_cfg.get("random_state", 3407),
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """High-level loader: model + tokenizer from a config dict.

    Expected config structure::

        model:
          name: "Qwen/Qwen2.5-0.5B-Instruct"
          quantization: "4bit"
          dtype: "bfloat16"
          use_unsloth: false   # set true to use Unsloth backend
          fast_inference: false # set true for vLLM-backed generation
        lora:  # optional
          r: 16
          lora_alpha: 32
          ...
    """
    model_cfg = config["model"]
    use_unsloth = model_cfg.get("use_unsloth", False)

    if use_unsloth:
        logger.info("Backend: Unsloth")
        return _load_with_unsloth(config)

    return _load_with_transformers(config)
