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
from grammarllm.modules.automaton import PushdownAutomaton
from grammarllm.modules.logits_processor import (
    StatelessLogitsProcessor as GrammarLLMStatelessProcessor,
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

            # Trace history through the dual-root Trie.
            #
            # The Trie has two roots:
            # - ``self.root``: non-space-prefixed gloss starts (first token
            #   of generation only).
            # - ``self.space_root``: space-prefixed gloss starts (used to
            #   begin a new gloss after a terminal node).
            #
            # This enforces whitespace boundaries: after a terminal gloss,
            # the next token MUST come from ``space_root`` (i.e. it must be
            # a space-prefixed BPE token), preventing arbitrary
            # concatenation of single-BPE-token glosses like DE+B+RE+CH+T
            # → "DEBUTRECHT". See docs/T2G_PIPELINE_REVIEW.md §9.2, §10.
            node = self.root
            at_start = True  # True only for the very first generated token

            for tok in gen_tokens:
                if tok in node.children:
                    node = node.children[tok]
                    at_start = False
                elif node.is_terminal and tok in self.space_root.children:
                    # Whitespace boundary: previous gloss is complete (node
                    # is terminal), and `tok` starts a new space-prefixed
                    # gloss. Jump to the space_root's child.
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


# ---------------------------------------------------------------------------
# Full grammarllm-based processor (for complex grammar constraints)
# ---------------------------------------------------------------------------


class GrammarPDALogitsProcessor(LogitsProcessor, MaskedMassTracker):
    """*EXPERIMENTAL* — Full grammar-constrained logits processor.

    Wraps ``grammarllm.modules.automaton.PushdownAutomaton`` and
    ``grammarllm.modules.logits_processor.StatelessLogitsProcessor`` to
    provide true LL(1)-style constrained decoding.  Use this when the grammar
    has non-trivial sequential constraints beyond simple vocabulary restriction.

    .. warning::
        This processor is experimental and not used by the default training
        path.  Enable it via ``use_grammarllm_pda: true`` in the config.
        The ``GlossVocabularyLogitsProcessor`` is the recommended default.

    .. note::
        Migration to grammarllm v0.5.0: the old ``MaskLogitsProcessor`` no
        longer exists. The new ``StatelessLogitsProcessor`` is stateless —
        it re-simulates the PDA state from the input_ids history at each
        step (with an LRU cache for O(1) amortized cost). This is a cleaner
        fit for HF ``generate()`` and adds beam-search safety. The
        ``__call__`` delegates to the new processor's ``__call__``, which
        applies the grammar mask internally.

    Inherits from ``transformers.LogitsProcessor`` for HF compatibility.

    Args:
        tokenizer: Hugging Face tokenizer.
        pda: A ``PushdownAutomaton`` instance OR a ``list[PushdownAutomaton]``
            of base templates (one per prompt in the batch). A single PDA is
            normalized to ``[pda]``. When batch_size > len(base_pdas) at
            ``__call__`` time, the list is auto-expanded by cloning the first
            PDA (mirrors ``generate_with_constraints.generate_text``).
        temperature: Temperature scaling (default 1.0). NOTE: the new
            ``StatelessLogitsProcessor`` does NOT apply temperature — it is
            handled by HF ``generate()``. Kept for API compatibility.
        track_score_history: If True, accumulate per-step logit history
            (costs one (batch, vocab) tensor per step — enable only for
            debugging/analysis, NOT production training). Default False.
    """

    def __init__(
        self,
        tokenizer: Any,
        pda: PushdownAutomaton | list[PushdownAutomaton],
        temperature: float = 1.0,
        track_score_history: bool = False,
    ) -> None:
        LogitsProcessor.__init__(self)
        MaskedMassTracker._init_masked_stats(self)

        self.tokenizer = tokenizer
        # Normalize to list of base PDA templates. Accept either a single
        # PDA (backward compat with old callers) or a list (from
        # create_grammarllm_pipeline which now returns pdas: list).
        if isinstance(pda, list):
            base_pdas = pda
            self.pda = pda[0]  # primary PDA for compat (pda.stack, get_tokens, etc.)
        else:
            base_pdas = [pda]
            self.pda = pda

        self._grammar_processor = GrammarLLMStatelessProcessor(
            tokenizer=tokenizer,
            base_pdas=base_pdas,
            sequences_per_prompt=1,
            prompt_len=0,
            temperature=temperature,
            track_score_history=track_score_history,
        )

        logger.info(
            "GrammarPDALogitsProcessor initialized with full grammarllm PDA "
            "(temperature=%.2f, stateless re-simulation + LRU cache, "
            "num_base_pdas=%d, track_score_history=%s)",
            temperature,
            len(base_pdas),
            track_score_history,
        )

    def reset(self) -> None:
        """Reset PDA, grammar processor cache, step counter, and masked mass/entropy stats."""
        self.step_count = 0
        self._reset_masked_stats()
        self.pda.reset()
        self._grammar_processor.reset()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply grammar-constrained mask via grammarllm's StatelessLogitsProcessor.

        Tracks masked mass and entropy diagnostics before delegating.
        """
        probs = self._pre_process(scores)

        # StatelessLogitsProcessor determines valid tokens from input_ids
        # history (re-simulated, with LRU cache). We track the mask here
        # for diagnostics using the PDA's current valid token set.
        # Filter out-of-range token IDs (can occur when the grammar's
        # token map includes IDs >= scores.shape[-1], e.g. Qwen2.5 has
        # vocab_size=151643 but eos_token_id=151643 — the new external
        # StatelessLogitsProcessor indexes scores[i, valid_ids] without
        # bounds checking, causing IndexError). We pre-filter here so the
        # wrapped processor never sees out-of-range IDs.
        raw_valid = self.get_valid_tokens()
        vocab_size = scores.shape[-1]
        valid_in_range = {t for t in raw_valid if 0 <= t < vocab_size}
        allowed_mask = self._build_allowed_mask(
            valid_in_range, vocab_size, scores.device
        )

        # Track masked probability mass + entropy (shared mixin)
        self._track_masked_stats(probs, allowed_mask)

        # Update prompt_len on the wrapped processor so it can extract the
        # generated-token history from input_ids correctly on first call.
        if self._grammar_processor.prompt_len == 0:
            self._grammar_processor.prompt_len = input_ids.shape[1]

        # Auto-expand base_pdas to match batch_size. StatelessLogitsProcessor
        # indexes base_pdas[prompt_idx]; if batch_size > len(base_pdas) it
        # would IndexError. This mirrors the expand logic in
        # generate_with_constraints.generate_text() and enables batched
        # constrained generation during eval (batch_size=8 with 1 base PDA).
        batch_size = scores.shape[0]
        proc = self._grammar_processor
        if len(proc.base_pdas) < batch_size:
            base_template = proc.base_pdas[0]
            while len(proc.base_pdas) < batch_size:
                proc.base_pdas.append(base_template.clone())

        # Defensive: monkey-patch the tokenizer's eos_token_id on the wrapped
        # processor's tokenizer view if it's out of range, to prevent the
        # `scores[i, self.tokenizer.eos_token_id] = 0` lines (528, 537) in
        # StatelessLogitsProcessor from raising IndexError. We point it to
        # a valid in-range ID (the last valid gloss token, or 0 as fallback).
        # This only affects the EOS-forcing path when the PDA reaches end
        # state — the actual EOS token is still emitted by HF generate()
        # via the model's own eos_token_id handling.
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and eos_id >= vocab_size:
            # Use a safe in-range sentinel for the wrapped processor's
            # internal EOS-forcing. The real EOS emission is handled by HF.
            self._grammar_processor.tokenizer_eos_fallback = (
                valid_in_range and next(iter(valid_in_range)) or 0
            )

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
    def allowed_ids(self) -> list[int]:
        """Get the list of currently allowed token IDs from the PDA."""
        return self.get_valid_tokens()

    @property
    def points(self) -> list[tuple[float, float]] | None:
        """Entropy/invalid-mass trajectory points (if metrics enabled)."""
        # StatelessLogitsProcessor doesn't expose points; return None for
        # API compatibility. Use masked_mass_stats() for diagnostics.
        return None

    @property
    def preserved_mass(self) -> list[float] | None:
        """History of preserved probability mass."""
        return None

    def __repr__(self) -> str:
        return (
            f"GrammarPDALogitsProcessor(pda_stack={self.pda.stack[::-1]}, "
            f"steps={self.step_count})"
        )
