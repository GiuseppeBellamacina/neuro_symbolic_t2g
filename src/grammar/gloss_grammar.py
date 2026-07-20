"""
ASL Gloss Grammar Builder with full grammarllm integration.

Constructs an LL(1)-compatible grammar for ASL gloss generation
using the vendored ``grammarllm`` library.

Provides:
    - ``build_gloss_grammar()`` — raw grammar dict for grammarllm
    - ``create_grammarllm_pipeline()`` — *EXPERIMENTAL* full pipeline: grammar → PDA → LogitsProcessor + Streamer
    - ``GlossVocabularyMask`` — lightweight vocabulary mask (no full PDA, default path)
"""

from __future__ import annotations

import logging
import string
from typing import Any

from grammarllm import (
    generate_grammar_parameters,
    get_parsing_table_and_map_tt,
    setup_logging,
)
from grammarllm.modules.automaton import PushdownAutomaton

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gloss Grammar Production Rules
# ---------------------------------------------------------------------------


def build_gloss_grammar(
    vocab: list[str],
    tokenizer: Any,
) -> dict[str, list[str]]:
    """Build a vocabulary-constrained grammar for ASL gloss generation.

    The grammar has one non-terminal ``S*`` that generates a sequence of
    gloss tokens followed by EOS::

        S* → gloss_1 S* | gloss_2 S* | ... | gloss_N S* | <EOS>

    Each gloss token is wrapped in ``<<...>>`` for grammarllm's exact-string
    matching, which automatically handles subword tokenization.

    .. note::
       The ``<<<EOS>>>`` triple-bracket escape was **removed** because
       modern BPE tokenizers (e.g. Qwen) split it into sub-tokens
       (``<E``, ``OS``, ``>``) that conflict with other productions in
       the LL(1) parsing table.  EOS is instead added directly by
       ``get_parsing_table_and_map_tt()`` via
       ``final_grammar[("S*","RULE")].append([tokenizer.eos_token])``,
       which uses the tokenizer's native EOS token string.

    Args:
        vocab: The sorted gloss vocabulary (should include ``<BOS>``, ``<EOS>``).
        tokenizer: A Hugging Face tokenizer (unused here; kept for API compatibility).

    Returns:
        A ``grammar_dict`` in the format expected by grammarllm's
        ``ProductionRuleProcessor``::

            {
                'S*': ["<<gloss_1>> S*", ..., "<<gloss_N>> S*"],
            }
    """
    skip_tokens = {"<BOS>", "<UNK>", "<EOS>", "<PAD>"}
    gloss_tokens = [t for t in vocab if t not in skip_tokens]

    logger.info(
        "Building gloss grammar with %d tokens (skipped BOS/UNK)",
        len(gloss_tokens),
    )

    productions: list[str] = []

    for gloss in gloss_tokens:
        productions.append(f"<<{gloss}>> S*")

    # NOTE: <<<EOS>>> production removed — EOS is added by
    # get_parsing_table_and_map_tt() using the tokenizer's native
    # eos_token string, avoiding the BPE sub-token conflict.

    grammar = {"S*": productions}
    logger.info("  Grammar rules: S* → %d alternatives", len(productions))
    return grammar


# ---------------------------------------------------------------------------
# Full grammarllm pipeline factory
# ---------------------------------------------------------------------------


def create_grammarllm_pipeline(
    vocab: list[str],
    tokenizer: Any,
    temperature: float = 1.0,
    enable_logging: bool = False,
    num_return_sequences: int = 1,
    token_lookahead: bool = True,
) -> tuple[list, Any, PushdownAutomaton]:
    """*EXPERIMENTAL* — Build a complete grammarllm pipeline.

    This is the **full neuro-symbolic path**: grammar → LL(1) parsing table
    → Pushdown Automaton → (base PDAs + Streamer).

    .. warning::
        This path is experimental. The default training path uses
        ``GlossVocabularyMask`` instead (lightweight vocabulary restriction,
        no full PDA overhead). The returned ``pdas`` list and ``streamer``
        are intended for use with ``GrammarPDALogitsProcessor`` (see
        ``grammar_logits_processor.py``), which instantiates a
        ``StatelessLogitsProcessor`` internally.

    .. note::
        Migration to grammarllm v0.5.0: ``generate_grammar_parameters`` now
        returns ``(list[PushdownAutomaton], BaseStreamer)`` instead of
        ``(MaskLogitsProcessor, BaseStreamer)``. The PDA is ``pdas[0]``.
        Temperature is no longer set on a processor object — it is passed
        to ``StatelessLogitsProcessor`` at construction time (though the
        new processor ignores it; temperature is handled by HF
        ``generate()``).

    Args:
        vocab: Sorted gloss vocabulary.
        tokenizer: Hugging Face tokenizer.
        temperature: Temperature (kept for API compat; the new
            ``StatelessLogitsProcessor`` does not apply it — HF
            ``generate()`` does).
        enable_logging: If ``True``, enable grammarllm debug logging.
        num_return_sequences: Number of independent base PDA templates to
            create. For batched eval with ``batch_size=N``, pass
            ``num_return_sequences=N`` so each prompt gets its own PDA
            template (the ``GrammarPDALogitsProcessor`` will also auto-expand
            if fewer are provided). Default 1 (single-prompt training).
        token_lookahead: If ``True`` (default), enable the token-boundary
            lookahead engine — allows the model to emit native BPE merged
            tokens that span grammar terminal boundaries (e.g. Qwen can
            emit ``"IX-me"`` as one BPE token instead of ``["IX", "-me"]``).
            Set ``False`` for the legacy boundary-strict engine (A/B baseline).

    Returns:
        A tuple ``(pdas, streamer, pda)`` where:
            - ``pdas`` is a ``list[PushdownAutomaton]`` (base templates)
            - ``streamer`` is a ``BaseStreamer`` instance
            - ``pda`` is ``pdas[0]`` (the primary PDA, for convenience)

    Example::

        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> vocab = ["<BOS>", "<EOS>", "<UNK>", "IX", "MAN", "WALK"]
        >>> pdas, streamer, pda = create_grammarllm_pipeline(vocab, tokenizer)
        >>> # Use with GrammarPDALogitsProcessor(tokenizer, pdas)
    """
    if enable_logging:
        setup_logging()

    # Build the raw grammar dict
    grammar = build_gloss_grammar(vocab, tokenizer)

    # Process through grammarllm: grammar → LL(1) parsing table → token maps
    pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
        tokenizer,
        productions=grammar,
    )

    # Create base PDAs + Streamer.
    # In grammarllm v0.5.0, generate_grammar_parameters returns
    # (list[PushdownAutomaton], BaseStreamer) — NOT (logit_processor, streamer).
    # The PDA list holds independent templates (one per num_return_sequences,
    # default 1). StatelessLogitsProcessor clones them per beam/step.
    pdas, streamer = generate_grammar_parameters(
        tokenizer,
        pars_table,
        map_terminal_tokens,
        num_return_sequences=num_return_sequences,
        token_lookahead=token_lookahead,
    )

    # Primary PDA for convenience (compat with old callers that expect a pda)
    pda: PushdownAutomaton = pdas[0]

    # Note: temperature is no longer set on a processor here — it is passed
    # to StatelessLogitsProcessor (which ignores it; HF generate() applies it).
    # Kept as a no-op for API compatibility with old callers.
    _ = temperature

    logger.info(
        "GrammarLLM pipeline ready: %d gloss tokens, PDA stack=%s, "
        "num_base_pdas=%d, lookahead=%s",
        len(vocab),
        pda.stack,
        len(pdas),
        token_lookahead,
    )

    return pdas, streamer, pda


# ---------------------------------------------------------------------------
# Simple Gloss Vocabulary Mask (lightweight, no full PDA)
# ---------------------------------------------------------------------------


class GlossVocabularyMask:
    """A lightweight vocabulary mask (no full grammarllm PDA).

    Directly masks the token vocabulary to allow only ASL gloss tokens
    (plus EOS).  Used during GRPO rollouts when vocabulary restriction
    is sufficient and full LL(1) grammar parsing is unnecessary.

    For stricter sequential constraints, use ``create_grammarllm_pipeline()``.

    Attributes:
        vocab: Sorted list of gloss tokens.
        vocab_set: ``set`` of allowed tokens for fast lookup.
        token_ids: Set of allowed token IDs in the model's vocabulary.
        eos_token_id: Token ID for EOS.
    """

    def __init__(self, vocab: list[str], tokenizer: Any) -> None:
        """Initialize the vocabulary mask.

        Args:
            vocab: The sorted gloss vocabulary.
            tokenizer: A Hugging Face tokenizer.
        """
        self.vocab = vocab
        self.vocab_set: set[str] = set(vocab)
        self.tokenizer = tokenizer

        self.token_ids: set[int] = set()
        _skipped_glosses: list[str] = []

        for token in vocab:
            # ── Filter the whole gloss entry first ────────────────────
            # Skip glosses that are purely numeric (dates, codes, etc.)
            # or contain digits mixed with other chars (e.g. "T04931944").
            # These leak digit token IDs into the mask and let the model
            # generate long numeric garbage strings.
            stripped = token.strip()
            if any(c.isdigit() for c in stripped) and stripped not in {
                "<BOS>",
                "<EOS>",
                "<UNK>",
            }:
                _skipped_glosses.append(stripped)
                continue

            # Add the full token ID (if the tokenizer knows it as a single token)
            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid != tokenizer.unk_token_id:
                self.token_ids.add(tid)

            # Add the space-prefixed token ID (if it represents a single token in Qwen)
            tid_space = tokenizer.convert_tokens_to_ids(" " + token)
            if isinstance(tid_space, int) and tid_space != tokenizer.unk_token_id:
                self.token_ids.add(tid_space)

            # Add subword token IDs for both representations, but filter noisy ones aggressively.
            # Without filtering, individual character subwords (digits,
            # punctuation, lowercase letters) let the model generate garbage
            # like "c010500040005" or "-1-1-1-1-2-2".
            for token_variant in [token, " " + token]:
                sub_tokens = tokenizer.tokenize(token_variant)
                for st in sub_tokens:
                    # Decode the subword to check its surface form
                    # Strip leading space markers (like G, ▁) and literal spaces
                    raw = st.lstrip("Ġ▁ ").strip()
                    if not raw:
                        continue

                    # Block subwords containing ANY digit (catches "2022",
                    # "T04", "97", "00" etc.)
                    if any(c.isdigit() for c in raw):
                        continue

                    # Block subwords that are entirely lowercase (catches
                    # "ment", "ation", "auto", "ing" etc. that let the model
                    # invent fake glosses like AUTOPARTICIPATE, PREVIUSION)
                    if raw.islower():
                        continue

                    # Block single characters that aren't uppercase letters
                    if len(raw) == 1 and not raw.isupper():
                        continue

                    # Block pure punctuation
                    if all(c in string.punctuation for c in raw):
                        continue

                    stid = tokenizer.convert_tokens_to_ids(st)
                    if isinstance(stid, int) and stid != tokenizer.unk_token_id:
                        self.token_ids.add(stid)

        if _skipped_glosses:
            logger.info(
                "GlossVocabularyMask: skipped %d glosses containing digits "
                "(e.g. %s)",
                len(_skipped_glosses),
                _skipped_glosses[:5],
            )

        # Add EOS so the model can stop generating
        self.eos_token_id: int = tokenizer.eos_token_id
        self.token_ids.add(self.eos_token_id)

        # Add whitespace tokens so the model can separate glosses with spaces
        # (without this, it resorts to commas, dashes, or concatenation)
        for space_str in [" ", "  ", "\n"]:
            space_tokens = tokenizer.encode(space_str, add_special_tokens=False)
            for stid in space_tokens:
                self.token_ids.add(stid)

        logger.info(
            "GlossVocabularyMask: %d glosses → %d unique token IDs (inc. EOS=%d)",
            len(self.vocab),
            len(self.token_ids),
            self.eos_token_id,
        )

    def get_allowed_token_ids(self) -> list[int]:
        """Return the list of allowed token IDs."""
        return list(self.token_ids)

    def is_allowed(self, token_id: int) -> bool:
        """Check if a token ID belongs to the gloss vocabulary."""
        return token_id in self.token_ids

    def decode_to_glosses(self, token_ids: list[int]) -> list[str]:
        """Decode a list of token IDs into individual gloss tokens.

        Each token ID is decoded individually to avoid subword merging
        (e.g., "MAN" + "HOUSE" being concatenated to "MAN,HOUSE" by
        the tokenizer's sentence-level decode).
        """
        glosses: list[str] = []
        for tid in token_ids:
            if tid == self.eos_token_id:
                break  # stop at EOS
            text = self.tokenizer.decode([tid], skip_special_tokens=True).strip()
            if text:
                glosses.append(text)
        return glosses

    def __repr__(self) -> str:
        return (
            f"GlossVocabularyMask(vocab_size={len(self.vocab)}, "
            f"token_ids={len(self.token_ids)})"
        )
