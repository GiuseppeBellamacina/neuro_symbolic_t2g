"""
GrammarLogitsProcessor for Constrained Decoding.

Implements a Hugging Face ``LogitsProcessor`` subclass that masks logits at
each generation step so the output belongs to the language of an
:class:`~src.grammar.output_grammar.OutputGrammar`, using a token-level Trie
(prefix tree):

* ``vocab`` (default, ASLG-PC12 / PHOENIX-2014T): any sequence of glosses of
  the closed vocabulary — the historical dual-root Trie, unchanged;
* ``source_spans`` (CoNLL-2003): ``TYPE:span`` tokens whose span is a piece of
  the prompt's own sentence, or the empty token alone — one Trie PER PROMPT;
* ``sequences`` (WOS-46985): exactly one of the allowed label sequences.

Shares the ``MaskedMassTracker`` mixin for probability mass / entropy
diagnostics, tracked on W&B during training.

Compatible with Hugging Face ``model.generate()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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
    """A node in the token-level Prefix Tree (Trie).

    Attributes:
        children: Next token id → node, continuing the current word.
        next: Where a NEW word starts once the current one ends here (its
            children are space-prefixed first tokens), or ``None`` if no word
            may follow.
        stop: EOS is allowed here (a complete output ends at this node).
    """

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.next: TrieNode | None = None
        self.stop: bool = False


_SPECIAL_GLOSSES = {"<BOS>", "<EOS>", "<UNK>"}


def _insert(start: TrieNode, token_ids: list[int]) -> TrieNode:
    """Insert a token path below ``start``; return the last node."""
    node = start
    for tid in token_ids:
        if tid not in node.children:
            node.children[tid] = TrieNode()
        node = node.children[tid]
    return node


class GlossVocabularyLogitsProcessor(LogitsProcessor, MaskedMassTracker):
    """Logits processor that enforces the output grammar with a Token-level Trie.

    Inherits from ``transformers.LogitsProcessor`` for full compatibility
    with Hugging Face ``model.generate(logits_processor=[...])``.

    Args:
        gloss_vocab_mask: A ``GlossVocabularyMask`` instance (vocabulary and
            tokenizer).
        device: Torch device for tensor operations.
        track_diagnostics: If True, track diagnostics (disabled by default in GRPO).
        grammar: Output grammar; ``None`` or mode ``vocab`` = the historical
            dual-root Trie over ``gloss_vocab_mask.vocab``.
        source_extractor: Prompt text → query sentence; required by
            ``source_spans`` (see :func:`src.utils.prompting.extract_query_source`).
    """

    def __init__(
        self,
        gloss_vocab_mask: Any,
        device: str | torch.device = "cpu",
        track_diagnostics: bool = False,
        grammar: Any = None,
        source_extractor: Callable[[str], str] | None = None,
    ) -> None:
        LogitsProcessor.__init__(self)
        MaskedMassTracker._init_masked_stats(self)

        self.mask = gloss_vocab_mask
        self.device = device
        self.tokenizer = gloss_vocab_mask.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.track_diagnostics = track_diagnostics
        self.grammar = grammar
        self.mode = grammar.mode if grammar is not None else "vocab"
        if self.mode == "source_spans" and source_extractor is None:
            raise ValueError("grammar.mode=source_spans richiede source_extractor")
        self.source_extractor = source_extractor
        # Uscita vuota (EOS come primo token): ammessa solo nel Trie storico,
        # dove lo era già; le grammatiche nuove la escludono.
        self._eos_at_start = self.mode == "vocab"

        self.root: TrieNode | None = None
        if self.mode == "vocab":
            self.root = self._build_vocab_trie(gloss_vocab_mask.vocab)
        elif self.mode == "sequences":
            self.root = self._build_sequences_trie(grammar.sequences)  # type: ignore[union-attr]
        # source_spans: un Trie per prompt, costruito al primo passo.
        self._span_roots: dict[tuple[int, ...], TrieNode] = {}
        self._row_roots: list[TrieNode] | None = None

        self.vocab_size = (
            self.tokenizer.vocab_size
            if hasattr(self.tokenizer, "vocab_size")
            else len(self.tokenizer)
        )

        self.prompt_len = -1
        self.step_count = 0

        logger.info(
            "GlossVocabularyLogitsProcessor initialized with Token-level Trie "
            "(mode=%s, vocab_size=%d, device=%s, track_diagnostics=%s)",
            self.mode,
            self.vocab_size,
            device,
            track_diagnostics,
        )

    # ── Trie construction ───────────────────────────────────────────────
    def _encode(self, texts: list[str]) -> list[list[int]]:
        """``encode`` di ogni testo; in batch quando il tokenizer lo consente
        (stessi id, ma il Trie per prompt di CoNLL ha migliaia di pezzi)."""
        if not texts:
            return []
        if callable(self.tokenizer):
            return self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]

    def _build_vocab_trie(self, vocab: list[str]) -> TrieNode:
        """Insert all normal and space-prefixed glosses into the Trie.

        The Trie has two root-level entry points:
        - ``root`` (no-space root): children are the first BPE token of
          each gloss WITHOUT a leading space. Used only at the very start of
          generation (first token after the prompt).
        - ``space_root`` (space root): children are the first BPE token
          of each gloss WITH a leading space (``" " + gloss``). Used to
          start a new gloss after a terminal node — this enforces whitespace
          boundaries between glosses and prevents arbitrary concatenation
          of single-BPE-token glosses (the DEBUTRECHT bug).

        See docs/T2G_PIPELINE_REVIEW.md §9.2 for the root cause analysis.
        """
        return self._build_items_trie(
            [t for t in vocab if t.strip() and t.strip() not in _SPECIAL_GLOSSES],
            final_items=[],
        )

    def _build_items_trie(self, items: list[str], final_items: list[str]) -> TrieNode:
        """Dual-root Trie over ``items`` (any number, any order).

        ``final_items`` may only be the WHOLE output: reachable from the
        bare root, followed only by EOS.
        """
        root, space_root = TrieNode(), TrieNode()
        for ids in self._encode(items):
            if ids:
                end = _insert(root, ids)
                end.next, end.stop = space_root, True
        for ids in self._encode([" " + t for t in items]):
            if ids:
                end = _insert(space_root, ids)
                end.next, end.stop = space_root, True
        for ids in self._encode(final_items):
            if ids:
                _insert(root, ids).stop = True
        return root

    def _build_sequences_trie(self, sequences: Any) -> TrieNode:
        """Trie over whole word sequences: word ``k+1`` may only follow the
        exact words ``1..k`` of some allowed sequence; EOS only at the end."""
        root = TrieNode()
        for seq in sorted(sequences):
            node = root
            for k, word in enumerate(seq):
                if k == 0:
                    start, text = root, word
                else:
                    if node.next is None:
                        node.next = TrieNode()
                    start, text = node.next, " " + word
                node = _insert(
                    start, self.tokenizer.encode(text, add_special_tokens=False)
                )
            node.stop = True
        return root

    def _root_for_prompt(self, prompt_ids: list[int]) -> TrieNode:
        """Trie of the prompt (``source_spans``); the shared one otherwise."""
        if self.mode != "source_spans":
            return self.root  # type: ignore[return-value]
        key = tuple(prompt_ids)
        root = self._span_roots.get(key)
        if root is None:
            # clean_up_tokenization_spaces=False: " ." e " 's" restano
            # staccati come nella frase, altrimenti i pezzi non combaciano.
            prompt = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            source = self.source_extractor(prompt)  # type: ignore[misc]
            empty = [self.grammar.empty_token] if self.grammar.empty_token else []
            root = self._build_items_trie(self.grammar.span_items(source), empty)
            self._span_roots[key] = root
        return root

    def reset(self) -> None:
        """Reset step counter, prompt length, and diagnostic metrics for a new generation."""
        self.step_count = 0
        self.prompt_len = -1
        self._row_roots = None
        self._span_roots.clear()
        self._reset_masked_stats()

    # ── Trie walk (single source of truth) ──────────────────────────────
    def _allowed(self, root: TrieNode, gen_tokens: list[int]) -> set[int]:
        """Allowed token ids after ``gen_tokens``, starting from ``root``.

        Kept as a single source of truth for the walk so the loss and the
        decoder cannot drift apart.
        """
        node = root
        for tok in gen_tokens:
            if tok in node.children:
                node = node.children[tok]
            elif node.next is not None and tok in node.next.children:
                # Whitespace boundary: word finished here, the next one
                # starts from a space-prefixed child (DEBUTRECHT guard).
                node = node.next.children[tok]
            else:
                # No valid transition. Reset to root as best-effort
                # recovery — the mask will be very restrictive here.
                node = root

        allowed = set(node.children.keys())
        if node.next is not None:
            allowed.update(node.next.children.keys())
        if node.stop:
            allowed.add(self.eos_token_id)
        if node is root and not gen_tokens and self._eos_at_start:
            allowed.add(self.eos_token_id)
        return allowed

    def allowed_mask_for_prefixes(
        self,
        prefixes: list[list[int]],
        vocab_size: int,
        device: str | torch.device = "cpu",
        prompts: list[list[int]] | None = None,
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
            prompts: Prompt token ids per row; required by ``source_spans``
                (the Trie depends on the prompt's sentence), ignored otherwise.

        Returns:
            Bool tensor of shape ``[len(prefixes), vocab_size]``.

        Raises:
            ValueError: If ``vocab_size`` is not positive, or ``prompts`` is
                missing in ``source_spans`` mode.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size!r}")
        if self.mode == "source_spans" and prompts is None:
            raise ValueError("grammar.mode=source_spans: servono i prompt per riga")

        mask = torch.zeros((len(prefixes), vocab_size), dtype=torch.bool, device=device)
        for row, prefix in enumerate(prefixes):
            root = self._root_for_prompt(list(prompts[row]) if prompts else [])
            for token_id in self._allowed(root, list(prefix)):
                if 0 <= token_id < vocab_size:
                    mask[row, token_id] = True
        return mask

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

        if self._row_roots is None or len(self._row_roots) != batch_size:
            # Il prompt di ogni riga (padding a sinistra compreso) non cambia
            # durante la generazione: la radice si sceglie una volta.
            self._row_roots = [
                self._root_for_prompt(
                    input_ids[i, : self.prompt_len].tolist()
                    if self.mode == "source_spans"
                    else []
                )
                for i in range(batch_size)
            ]

        # Build dynamic mask for the batch
        mask = torch.zeros(
            (batch_size, vocab_size_logits),
            dtype=torch.bool,
            device=scores.device,
        )

        for i in range(batch_size):
            # Extract newly generated tokens (slice from the end of the prompt)
            gen_tokens = input_ids[i, self.prompt_len :].tolist()
            for tid in self._allowed(self._row_roots[i], gen_tokens):
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
            f"GlossVocabularyLogitsProcessor(mode={self.mode}, "
            f"vocab_size={self.vocab_size}, steps={self.step_count})"
        )


def build_logits_processor(
    config: Any,
    grammar: Any,
    vocab: list[str],
    tokenizer: Any,
    device: str | torch.device = "cpu",
    track_diagnostics: bool = False,
) -> GlossVocabularyLogitsProcessor:
    """Processor del run: Trie della grammatica, frase del prompt estratta con
    il profilo del config (unico punto che conosce il template)."""
    from src.grammar.gloss_grammar import GlossVocabularyMask
    from src.utils.prompting import extract_query_source, prompt_profile_for_config

    profile = prompt_profile_for_config(config)
    return GlossVocabularyLogitsProcessor(
        GlossVocabularyMask(vocab, tokenizer),
        device=device,
        track_diagnostics=track_diagnostics,
        grammar=grammar,
        source_extractor=lambda prompt: extract_query_source(prompt, profile),
    )
