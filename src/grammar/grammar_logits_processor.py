"""
GrammarLogitsProcessor for Constrained Decoding.

Implements Hugging Face ``LogitsProcessor`` subclasses that mask logits at
each generation step to enforce ASL gloss vocabulary constraints.

Two implementations are provided:
    1. ``GrammarPDALogitsProcessor`` — uses the full grammarllm PDA for
       LL(1)-style constrained generation (supports complex grammars).
    2. ``GlossVocabularyLogitsProcessor`` — lightweight, masks all tokens
       not in the ASL gloss vocabulary (simpler but less strict).

Both are compatible with Hugging Face ``model.generate()``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import LogitsProcessor

# Import grammarllm for full PDA-based constrained decoding
from grammarllm.modules.PushdownAutomaton import PushdownAutomaton
from grammarllm.modules.SimpleLogitProcessor_ import (
    MaskLogitsProcessor as GrammarLLMMaskProcessor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gloss Vocabulary Logits Processor (HF-compatible)
# ---------------------------------------------------------------------------


class GlossVocabularyLogitsProcessor(LogitsProcessor):
    """Logits processor that masks non-gloss tokens at each generation step.

    Inherits from ``transformers.LogitsProcessor`` for full compatibility
    with Hugging Face ``model.generate(logits_processor=[...])``.

    At each step, sets the logit of every token NOT in the ASL gloss vocabulary
    to ``-inf``.  EOS is always allowed so the model can terminate.

    Also provides a ``get_logit_bias_for_vllm()`` method for integrating
    with vLLM's sampling engine.

    Args:
        gloss_vocab_mask: A ``GlossVocabularyMask`` instance.
        device: Torch device for tensor operations.
    """

    def __init__(
        self,
        gloss_vocab_mask: Any,
        device: str | torch.device = "cpu",
    ) -> None:
        self.mask = gloss_vocab_mask
        self.device = device
        self.allowed_ids: set[int] = gloss_vocab_mask.token_ids

        tokenizer = gloss_vocab_mask.tokenizer
        if hasattr(tokenizer, "vocab_size"):
            self.vocab_size: int = tokenizer.vocab_size
        elif hasattr(tokenizer, "__len__"):
            self.vocab_size: int = len(tokenizer)
        else:
            self.vocab_size = 0

        self.step_count: int = 0

        logger.info(
            "GlossVocabularyLogitsProcessor initialized "
            "(allowed=%d tokens, vocab_size=%d, device=%s)",
            len(self.allowed_ids),
            self.vocab_size,
            device,
        )

    def reset(self) -> None:
        """Reset step counter for a new generation."""
        self.step_count = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply vocabulary mask to logits.

        Hugging Face ``LogitsProcessor`` interface.
        """
        self.step_count += 1

        if self.vocab_size == 0:
            self.vocab_size = scores.shape[-1]

        allowed_mask = torch.zeros(
            scores.shape[-1], dtype=torch.bool, device=scores.device
        )
        for tid in self.allowed_ids:
            if 0 <= tid < scores.shape[-1]:
                allowed_mask[tid] = True

        scores = scores.clone()
        scores[:, ~allowed_mask] = -float("inf")

        if self.step_count <= 3 or self.step_count % 10 == 0:
            logger.debug(
                "[Step %d] Allowed tokens: %d / %d",
                self.step_count,
                allowed_mask.sum().item(),
                self.vocab_size,
            )

        return scores

    def get_logit_bias_for_vllm(self) -> dict[int, float]:
        """Build a vLLM-compatible logit bias dictionary."""
        if self.vocab_size == 0:
            raise RuntimeError("vocab_size unknown; call __call__ first.")
        bias: dict[int, float] = {}
        for tid in range(self.vocab_size):
            bias[tid] = 0.0 if tid in self.allowed_ids else -100.0
        return bias

    def __repr__(self) -> str:
        return (
            f"GlossVocabularyLogitsProcessor(allowed={len(self.allowed_ids)}, "
            f"steps={self.step_count})"
        )


# ---------------------------------------------------------------------------
# Full grammarllm-based processor (for complex grammar constraints)
# ---------------------------------------------------------------------------


class GrammarPDALogitsProcessor(LogitsProcessor):
    """*EXPERIMENTAL* — Full grammar-constrained logits processor.

    Wraps ``grammarllm.modules.PushdownAutomaton`` and
    ``grammarllm.modules.MaskLogitsProcessor`` to provide true LL(1)-style
    constrained decoding.  Use this when the grammar has non-trivial
    sequential constraints beyond simple vocabulary restriction.

    .. warning::
       This processor is experimental and not used by the default training
       path.  Enable it via ``use_grammarllm_pda: true`` in the config.
       The ``GlossVocabularyLogitsProcessor`` is the recommended default.

    Inherits from ``transformers.LogitsProcessor`` for HF compatibility.

    Args:
        tokenizer: Hugging Face tokenizer.
        pda: A ``PushdownAutomaton`` instance.
        temperature: Temperature scaling (default 1.0).
    """

    def __init__(
        self,
        tokenizer: Any,
        pda: PushdownAutomaton,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pda = pda
        self._grammar_processor = GrammarLLMMaskProcessor(
            tokenizer, pda, temperature=temperature
        )
        self.step_count: int = 0

        logger.info("GrammarPDALogitsProcessor initialized with full grammarllm PDA")

    def reset(self) -> None:
        """Reset PDA, grammar processor, and step counter."""
        self.step_count = 0
        self.pda.reset()
        self._grammar_processor.reset()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply grammar-constrained mask via grammarllm's MaskLogitsProcessor."""
        self.step_count += 1
        return self._grammar_processor(input_ids, scores)

    def update_state(self, token_id: int) -> None:
        """Update the PDA state after a token is generated.

        Must be called by a streamer/callback after each token.
        """
        try:
            self.pda.next_state(token_id)
        except Exception:
            logger.error(
                "PDA state update failed for token %d. Stack: %s",
                token_id,
                self.pda.stack,
            )
            raise

    def get_valid_tokens(self) -> list[int]:
        """Get the list of currently valid token IDs from the PDA."""
        return self.pda.get_tokens()

    def is_eos(self) -> bool:
        """Check if the PDA has reached the end state (stack empty)."""
        return self.pda.eos()

    @property
    def points(self) -> list[tuple[float, float]] | None:
        """Entropy/invalid-mass trajectory points (if metrics enabled)."""
        return self._grammar_processor.points

    @property
    def preserved_mass(self) -> list[float] | None:
        """History of preserved probability mass."""
        return self._grammar_processor.preserved_mass

    def __repr__(self) -> str:
        return (
            f"GrammarPDALogitsProcessor(pda_stack={self.pda.stack[::-1]}, "
            f"steps={self.step_count})"
        )
