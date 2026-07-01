"""
GrammarLogitsProcessor for Constrained Decoding.

Implements Hugging Face ``LogitsProcessor`` subclasses that mask logits at
each generation step to enforce ASL gloss vocabulary constraints.

Two implementations are provided:
    1. ``GrammarPDALogitsProcessor`` — uses the full grammarllm PDA for
       LL(1)-style constrained generation (supports complex grammars).
    2. ``GlossVocabularyLogitsProcessor`` — lightweight, masks all tokens
       not in the ASL gloss vocabulary (simpler but less strict).

Both share the ``MaskedMassTracker`` mixin for probability mass / entropy
diagnostics, tracked on W&B during training.

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

# Import shared diagnostics mixin
from src.grammar.masked_mass_tracker import MaskedMassTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gloss Vocabulary Logits Processor (HF-compatible)
# ---------------------------------------------------------------------------


class TrieNode:
    """A node in the token-level Prefix Tree (Trie)."""

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.is_terminal: bool = False


class GlossVocabularyLogitsProcessor(LogitsProcessor, MaskedMassTracker):
    """Logits processor that enforces exact gloss sequences using a Token-level Trie.

    Inherits from ``transformers.LogitsProcessor`` for full compatibility
    with Hugging Face ``model.generate(logits_processor=[...])``.

    Uses a token-level prefix tree (Trie) compiled from the vocabulary to ensure
    the model can only generate sequences of tokens that perfectly reconstruct
    valid words from the gloss vocabulary (separated by spaces).

    Args:
        gloss_vocab_mask: A ``GlossVocabularyMask`` instance.
        device: Torch device for tensor operations.
        track_diagnostics: If True, track diagnostics (disabled by default in GRPO).
    """

    def __init__(
        self,
        gloss_vocab_mask: Any,
        device: str | torch.device = "cpu",
        track_diagnostics: bool = False,
    ) -> None:
        LogitsProcessor.__init__(self)
        MaskedMassTracker._init_masked_stats(self)

        self.mask = gloss_vocab_mask
        self.device = device
        self.tokenizer = gloss_vocab_mask.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.track_diagnostics = track_diagnostics

        # Build Token-level Trie from vocabulary
        self.root = TrieNode()
        self._build_trie(gloss_vocab_mask.vocab)

        self.vocab_size = (
            self.tokenizer.vocab_size
            if hasattr(self.tokenizer, "vocab_size")
            else len(self.tokenizer)
        )

        self.prompt_len = -1
        self.step_count = 0

        logger.info(
            "GlossVocabularyLogitsProcessor initialized with Token-level Trie "
            "(vocab_size=%d, device=%s, track_diagnostics=%s)",
            self.vocab_size,
            device,
            track_diagnostics,
        )

    def _build_trie(self, vocab: list[str]) -> None:
        """Insert all normal and space-prefixed glosses into the Trie."""
        for token in vocab:
            stripped = token.strip()
            if not stripped or stripped in {"<BOS>", "<EOS>", "<UNK>"}:
                continue

            for variant in [token, " " + token]:
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if not token_ids:
                    continue

                node = self.root
                for tid in token_ids:
                    if tid not in node.children:
                        node.children[tid] = TrieNode()
                    node = node.children[tid]
                node.is_terminal = True

    def reset(self) -> None:
        """Reset step counter, prompt length, and diagnostic metrics for a new generation."""
        self.step_count = 0
        self.prompt_len = -1
        self._reset_masked_stats()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply Token-Trie constrained mask dynamically per batch element."""
        self.step_count += 1

        if self.prompt_len < 0:
            self.prompt_len = input_ids.shape[1]

        batch_size, vocab_size_logits = scores.shape

        # Build dynamic mask for the batch
        mask = torch.zeros(
            (batch_size, vocab_size_logits),
            dtype=torch.bool,
            device=scores.device,
        )

        for i in range(batch_size):
            # Extract newly generated tokens (slice from the end of the prompt)
            gen_tokens = input_ids[i, self.prompt_len :].tolist()

            # Trace history through the Trie to determine the current state
            node = self.root
            for tok in gen_tokens:
                if tok in node.children:
                    node = node.children[tok]
                elif node.is_terminal and tok in self.root.children:
                    node = self.root.children[tok]
                else:
                    node = self.root  # Fallback on mismatch

            # Allowed tokens from the current state in the Trie
            allowed = set(node.children.keys())

            # If node is terminal or root, we can start a new gloss or generate EOS
            if node.is_terminal or node == self.root:
                allowed.update(self.root.children.keys())
                allowed.add(self.eos_token_id)

            # Apply allowed tokens to the mask
            for tid in allowed:
                if 0 <= tid < vocab_size_logits:
                    mask[i, tid] = True

        # Track masked probability mass + entropy only if diagnostics are explicitly enabled
        if self.track_diagnostics:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(scores, dim=-1)
            # Find a single allowed mask representing the root state for logging
            # (or log based on batch mean allowed mask)
            self._track_masked_stats(probs, mask.any(dim=0))

        scores = scores.clone()
        scores[~mask] = -float("inf")

        return scores

    def __repr__(self) -> str:
        return (
            f"GlossVocabularyLogitsProcessor(vocab_size={self.vocab_size}, "
            f"steps={self.step_count})"
        )


# ---------------------------------------------------------------------------
# Full grammarllm-based processor (for complex grammar constraints)
# ---------------------------------------------------------------------------


class GrammarPDALogitsProcessor(LogitsProcessor, MaskedMassTracker):
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
        LogitsProcessor.__init__(self)
        MaskedMassTracker._init_masked_stats(self)

        self.tokenizer = tokenizer
        self.pda = pda
        self._grammar_processor = GrammarLLMMaskProcessor(
            tokenizer, pda, temperature=temperature
        )

        logger.info(
            "GrammarPDALogitsProcessor initialized with full grammarllm PDA "
            "(temperature=%.2f)",
            temperature,
        )

    def reset(self) -> None:
        """Reset PDA, grammar processor, step counter, and masked mass/entropy stats."""
        self.step_count = 0
        self._reset_masked_stats()
        self.pda.reset()
        self._grammar_processor.reset()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply grammar-constrained mask via grammarllm's MaskLogitsProcessor.

        Tracks masked mass and entropy diagnostics before delegating.
        """
        probs = self._pre_process(scores)

        allowed_mask = self._build_allowed_mask(
            set(self.get_valid_tokens()), scores.shape[-1], scores.device
        )

        # Track masked probability mass + entropy (shared mixin)
        self._track_masked_stats(probs, allowed_mask)

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
