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
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True, slots=True, eq=False)
class TrieNode:
    """Immutable, identity-hashable node in a compiled token-level prefix tree."""

    children: Mapping[int, TrieNode]
    is_terminal: bool = False


class GlossTrieStatus(Enum):
    """Status of a :class:`GlossTrieState`."""

    ACTIVE = "active"
    COMPLETE = "complete"
    INVALID = "invalid"


class GlossTrieTokenRole(Enum):
    """Semantic role of an input ID when EOS and PAD may share an ID."""

    TOKEN = "token"
    EOS = "eos"
    PAD = "pad"


@dataclass(frozen=True, slots=True)
class GlossTrieState:
    """Opaque immutable state for one dual-root Trie sequence.

    ``status`` explicitly reports invalid input and EOS completion.  ``node``
    is intentionally an implementation detail; callers should use
    :meth:`DualRootGlossTrie.allowed_token_ids` and
    :meth:`DualRootGlossTrie.advance`.
    """

    node: TrieNode
    status: GlossTrieStatus = GlossTrieStatus.ACTIVE
    at_start: bool = True


class _MutableTrieNode:
    """Construction-only counterpart of :class:`TrieNode`."""

    def __init__(self) -> None:
        self.children: dict[int, _MutableTrieNode] = {}
        self.is_terminal = False


class DualRootGlossTrie:
    """Pure state-transition API for separator-aware gloss token sequences.

    The bare root starts the first gloss.  At a terminal node, both ordinary
    continuation children (for terminal-prefix ambiguity) and children of the
    space-prefixed root are valid, as is EOS.  Methods return tuples/states and
    never allocate device tensors or full-vocabulary masks.
    """

    def __init__(
        self,
        root: TrieNode,
        space_root: TrieNode,
        eos_token_id: int,
    ) -> None:
        self.root = root
        self.space_root = space_root
        self.eos_token_id = eos_token_id

    @classmethod
    def from_vocabulary(
        cls,
        vocab: Sequence[str],
        tokenizer: Any,
    ) -> DualRootGlossTrie:
        """Compile bare and space-prefixed tokenizations of ``vocab``."""
        root = _MutableTrieNode()
        space_root = _MutableTrieNode()
        for token in vocab:
            stripped = token.strip()
            if not stripped or stripped in {"<BOS>", "<EOS>", "<UNK>"}:
                continue
            cls._insert(root, tokenizer.encode(token, add_special_tokens=False))
            cls._insert(
                space_root,
                tokenizer.encode(" " + token, add_special_tokens=False),
            )
        return cls(
            cls._freeze(root),
            cls._freeze(space_root),
            tokenizer.eos_token_id,
        )

    @staticmethod
    def _insert(root: _MutableTrieNode, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return
        node = root
        for token_id in token_ids:
            node = node.children.setdefault(token_id, _MutableTrieNode())
        node.is_terminal = True

    @classmethod
    def _freeze(cls, node: _MutableTrieNode) -> TrieNode:
        children = {
            token_id: cls._freeze(child)
            for token_id, child in sorted(node.children.items())
        }
        return TrieNode(MappingProxyType(children), node.is_terminal)

    def initial_state(self) -> GlossTrieState:
        """Return the canonical state before the first generated token."""
        return GlossTrieState(self.root)

    def allowed_token_ids(self, state: GlossTrieState) -> tuple[int, ...]:
        """Return deterministic, unique token IDs allowed from ``state``."""
        if state.status is GlossTrieStatus.COMPLETE:
            return ()

        allowed = set(state.node.children)
        if state.node.is_terminal:
            allowed.update(self.space_root.children)
            allowed.add(self.eos_token_id)
        if state.node is self.root and state.at_start:
            allowed.update(self.root.children)
            allowed.add(self.eos_token_id)
        return tuple(sorted(allowed))

    def is_invalid(self, state: GlossTrieState) -> bool:
        """Return whether ``state`` records an invalid transition."""
        return state.status is GlossTrieStatus.INVALID

    def advance(
        self,
        state: GlossTrieState,
        token_id: int,
        role: GlossTrieTokenRole = GlossTrieTokenRole.TOKEN,
    ) -> GlossTrieState:
        """Advance by one ID, returning an explicit COMPLETE/INVALID state.

        ``role`` disambiguates EOS from PAD for tokenizers where both IDs are
        equal.  Callers consuming a semantic sequence end must pass ``EOS``;
        ordinary replay (including the processor's historical post-stop
        behavior) uses ``TOKEN``, while label padding may pass ``PAD``.

        Invalid input retains production's historical root-recovery behavior:
        the returned state is marked ``INVALID`` but exposes bare-root
        continuations on the next query.
        """
        if state.status is GlossTrieStatus.COMPLETE:
            return GlossTrieState(self.root, GlossTrieStatus.INVALID, False)

        is_eos = role is GlossTrieTokenRole.EOS
        if is_eos:
            if token_id == self.eos_token_id and token_id in self.allowed_token_ids(
                state
            ):
                return GlossTrieState(state.node, GlossTrieStatus.COMPLETE, False)
            return GlossTrieState(self.root, GlossTrieStatus.INVALID, False)

        node = state.node
        if token_id in node.children:
            return GlossTrieState(node.children[token_id], at_start=False)
        if node.is_terminal and token_id in self.space_root.children:
            return GlossTrieState(
                self.space_root.children[token_id],
                at_start=False,
            )
        return GlossTrieState(self.root, GlossTrieStatus.INVALID, False)

    def state_for_tokens(self, token_ids: Sequence[int]) -> GlossTrieState:
        """Replay a token history from :meth:`initial_state`."""
        state = self.initial_state()
        for token_id in token_ids:
            # Replaying after invalid input historically resumes from root.
            if state.status is GlossTrieStatus.INVALID:
                state = GlossTrieState(state.node, at_start=state.at_start)
            state = self.advance(state, token_id)
        return state


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

        # Build the pure dual-root transition model used by generation and
        # teacher-forced consumers.
        self.trie = DualRootGlossTrie.from_vocabulary(
            gloss_vocab_mask.vocab,
            self.tokenizer,
        )
        self.root = self.trie.root
        self.space_root = self.trie.space_root

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

    def reset_generation_state(self) -> None:
        """Reset prompt-dependent state without clearing diagnostics."""
        self.step_count = 0
        self.prompt_len = -1

    reset = reset_generation_state

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

            state = self.trie.state_for_tokens(gen_tokens)
            allowed = self.trie.allowed_token_ids(state)

            # Apply allowed tokens to the mask
            for tid in allowed:
                if 0 <= tid < vocab_size_logits:
                    mask[i, tid] = True

        # Track masked probability mass + entropy only if diagnostics are explicitly enabled
        if self.track_diagnostics:
            active = self._active_rows(
                input_ids,
                self.prompt_len,
                self.eos_token_id,
                getattr(self.tokenizer, "pad_token_id", None),
            )
            self._track_masked_stats(scores, mask, active)

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
        ``StatelessLogitsProcessor`` re-simulates the PDA state from the
        ``input_ids`` history at each step (with an LRU cache for O(1)
        amortized cost). This is a cleaner
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
        track_diagnostics: bool = False,
    ) -> None:
        LogitsProcessor.__init__(self)
        MaskedMassTracker._init_masked_stats(self)

        self.tokenizer = tokenizer
        self.track_diagnostics = track_diagnostics
        # Normalize to list of base PDA templates. Accept either a single
        # PDA (for callers supplying one template) or a list (from
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
            prompt_len=-1,
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

    def reset_generation_state(self) -> None:
        """Reset PDA and prompt-dependent state without clearing diagnostics."""
        self.step_count = 0
        self.pda.reset()
        self._grammar_processor.reset()
        self._grammar_processor.prompt_len = -1

    reset = reset_generation_state

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply grammar-constrained mask via grammarllm's StatelessLogitsProcessor.

        Tracks masked mass and entropy diagnostics before delegating.
        """
        self.step_count += 1

        # Update prompt_len on the wrapped processor so it can extract the
        # generated-token history from input_ids correctly on first call.
        if self._grammar_processor.prompt_len < 0:
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

        raw_scores = scores.clone() if self.track_diagnostics else None
        filtered = self._grammar_processor(input_ids, scores)
        if raw_scores is not None:
            applied_mask = torch.isfinite(filtered)
            active = self._active_rows(
                input_ids,
                self._grammar_processor.prompt_len,
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "pad_token_id", None),
            )
            self._track_masked_stats(raw_scores, applied_mask, active)
        return filtered

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
        # API compatibility. Use get_diagnostics() for diagnostics.
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
