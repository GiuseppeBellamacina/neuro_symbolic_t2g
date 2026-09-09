"""
GrammarLogitsProcessor for Constrained Decoding.

Implements a Hugging Face ``LogitsProcessor`` subclass that masks logits at
each generation step to enforce ASL gloss vocabulary constraints using a
token-level Trie (prefix tree) compiled from the gloss vocabulary.

Shares the ``MaskedMassTracker`` mixin for probability mass / entropy
diagnostics, tracked on W&B during training.

Compatible with Hugging Face ``model.generate()``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import LogitsProcessor

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
        """Insert all normal and space-prefixed glosses into the Trie.

        The Trie has two root-level entry points:
        - ``self.root`` (no-space root): children are the first BPE token of
          each gloss WITHOUT a leading space. Used only at the very start of
          generation (first token after the prompt).
        - ``self.space_root`` (space root): children are the first BPE token
          of each gloss WITH a leading space (``" " + gloss``). Used to
          start a new gloss after a terminal node — this enforces whitespace
          boundaries between glosses and prevents arbitrary concatenation
          of single-BPE-token glosses (the DEBUTRECHT bug).

        See docs/T2G_PIPELINE_REVIEW.md §9.2 for the root cause analysis.
        """
        self.space_root = TrieNode()

        for token in vocab:
            stripped = token.strip()
            if not stripped or stripped in {"<BOS>", "<EOS>", "<UNK>"}:
                continue

            # Non-space variant → root
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                node = self.root
                for tid in token_ids:
                    if tid not in node.children:
                        node.children[tid] = TrieNode()
                    node = node.children[tid]
                node.is_terminal = True

            # Space-prefixed variant → space_root
            space_ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
            if space_ids:
                node = self.space_root
                for tid in space_ids:
                    if tid not in node.children:
                        node.children[tid] = TrieNode()
                    node = node.children[tid]
                node.is_terminal = True

    def reset(self) -> None:
        """Reset step counter, prompt length, and diagnostic metrics for a new generation."""
        self.step_count = 0
        self.prompt_len = -1
        self._reset_masked_stats()

    def allowed_mask_for_prefixes(
        self,
        prefixes: list[list[int]],
        vocab_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Return the ``[N, vocab_size]`` bool mask allowed after each prefix.

        This exposes the Trie transition logic used by ``__call__`` as a pure
        function of explicit token prefixes, independent of generation state.
        It exists so a teacher-forced auxiliary objective can obtain the same
        allowed set the decoder would enforce — see
        :func:`src.training.allowed_mass_loss.allowed_mass_loss`, which needs
        exactly this mask and has no other way to build it.

        Args:
            prefixes: One list of already-generated token ids per row. An empty
                list means "start of generation", which uses the bare root.
            vocab_size: Width of the returned mask (the model's logit width).
            device: Device for the returned tensor.

        Returns:
            Bool tensor of shape ``[len(prefixes), vocab_size]``.

        Raises:
            ValueError: If ``vocab_size`` is not positive.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size!r}")

        mask = torch.zeros((len(prefixes), vocab_size), dtype=torch.bool, device=device)
        for row, prefix in enumerate(prefixes):
            for token_id in self._allowed_after_prefix(list(prefix)):
                if 0 <= token_id < vocab_size:
                    mask[row, token_id] = True
        return mask

    def _allowed_after_prefix(self, gen_tokens: list[int]) -> set[int]:
        """Allowed token ids after ``gen_tokens``, mirroring ``__call__``.

        Kept as a single source of truth for the dual-root walk so the loss and
        the decoder cannot drift apart.
        """
        node = self.root
        at_start = True
        for tok in gen_tokens:
            if tok in node.children:
                node = node.children[tok]
                at_start = False
            elif node.is_terminal and tok in self.space_root.children:
                node = self.space_root.children[tok]
                at_start = False
            elif at_start and tok in self.root.children:
                node = self.root.children[tok]
                at_start = False
            else:
                node = self.root
                at_start = False

        allowed = set(node.children.keys())
        if node.is_terminal:
            allowed.update(self.space_root.children.keys())
            allowed.add(self.eos_token_id)
        if node is self.root and not gen_tokens:
            allowed.update(self.root.children.keys())
            allowed.add(self.eos_token_id)
        return allowed

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

            # Trace history through the dual-root Trie: ``root`` matcha i
            # gloss-start NON prefissati da spazio (solo primo token della
            # generazione), ``space_root`` gli start space-prefixed (nuovo
            # gloss dopo un nodo terminale). Impone i confini di whitespace e
            # impedisce concatenazioni tipo DE+B+RE+CH+T → "DEBUTRECHT".
            # See docs/T2G_PIPELINE_REVIEW.md §9.2, §10.
            node = self.root
            at_start = True  # True only for the very first generated token

            for tok in gen_tokens:
                if tok in node.children:
                    node = node.children[tok]
                    at_start = False
                elif node.is_terminal and tok in self.space_root.children:
                    # Whitespace boundary: gloss precedente terminale, si
                    # salta al child di space_root.
                    node = self.space_root.children[tok]
                    at_start = False
                elif at_start and tok in self.root.children:
                    # First token of generation — must come from root
                    node = self.root.children[tok]
                    at_start = False
                else:
                    # No valid transition. Reset to root as best-effort
                    # recovery — the mask will be very restrictive here.
                    node = self.root
                    at_start = False

            # Allowed tokens from the current state in the Trie
            allowed = set(node.children.keys())

            # If node is terminal, we can start a new gloss (via space_root)
            # or generate EOS.
            if node.is_terminal:
                allowed.update(self.space_root.children.keys())
                allowed.add(self.eos_token_id)

            # At the very start (root, no tokens generated yet), also allow
            # root children (non-space starts).
            if node == self.root and not gen_tokens:
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

    @property
    def allowed_ids(self) -> set[int]:
        """Get the set of allowed token IDs in the mask."""
        return self.mask.token_ids

    def __repr__(self) -> str:
        return (
            f"GlossVocabularyLogitsProcessor(vocab_size={self.vocab_size}, "
            f"steps={self.step_count})"
        )
