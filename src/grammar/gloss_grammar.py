"""
ASL Gloss Grammar Builder with full grammarllm integration.

Constructs an LL(1)-compatible grammar for ASL gloss generation
using the vendored ``grammarllm`` library.

Provides:
    - ``build_gloss_grammar()`` — raw grammar dict for grammarllm
    - ``create_grammarllm_pipeline()`` — full pipeline: grammar → PDA → LogitsProcessor + Streamer
    - ``GlossVocabularyMask`` — lightweight vocabulary mask (no full PDA)
"""

from __future__ import annotations

import logging
from typing import Any

from grammarllm import (
    generate_grammar_parameters,
    get_parsing_table_and_map_tt,
    setup_logging,
)

from grammarllm.modules.PushdownAutomaton import PushdownAutomaton

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

    Args:
        vocab: The sorted gloss vocabulary (should include ``<BOS>``, ``<EOS>``).
        tokenizer: A Hugging Face tokenizer (unused here; kept for API compatibility).

    Returns:
        A ``grammar_dict`` in the format expected by grammarllm's
        ``ProductionRuleProcessor``::

            {
                'S*': ["<<gloss_1>> S*", ..., "<<gloss_N>> S*", "<<<EOS>>>"],
            }
    """
    skip_tokens = {"<BOS>", "<UNK>"}
    gloss_tokens = [t for t in vocab if t not in skip_tokens]

    logger.info(
        "Building gloss grammar with %d tokens (skipped BOS/UNK)",
        len(gloss_tokens),
    )

    productions: list[str] = []

    for gloss in gloss_tokens:
        productions.append(f"<<{gloss}>> S*")

    if "<EOS>" in vocab:
        productions.append("<<<EOS>>>")

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
) -> tuple[Any, Any, PushdownAutomaton]:
    """Build a complete grammarllm pipeline for gloss-constrained generation.

    This is the **full neuro-symbolic path**: grammar → LL(1) parsing table
    → Pushdown Automaton → LogitsProcessor + Streamer.

    Args:
        vocab: Sorted gloss vocabulary.
        tokenizer: Hugging Face tokenizer.
        temperature: Temperature for the logits processor (default 1.0).
        enable_logging: If ``True``, enable grammarllm debug logging.

    Returns:
        A tuple ``(logit_processor, streamer, pda)`` where:
            - ``logit_processor`` is a ``MaskLogitsProcessor`` instance
            - ``streamer`` is a ``BaseStreamer`` instance
            - ``pda`` is the ``PushdownAutomaton`` instance

    Example::

        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> vocab = ["<BOS>", "<EOS>", "<UNK>", "IX", "MAN", "WALK"]
        >>> lp, streamer, pda = create_grammarllm_pipeline(vocab, tokenizer)
        >>> output = model.generate(
        ...     logits_processor=[lp], streamer=streamer, ...
        ... )
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

    # Create PDA + LogitsProcessor + Streamer
    logit_processor, streamer = generate_grammar_parameters(
        tokenizer, pars_table, map_terminal_tokens
    )

    # The PDA is already created inside generate_grammar_parameters
    # and stored as an attribute on the logit_processor.
    pda: PushdownAutomaton = logit_processor.pda

    # Set temperature
    logit_processor.temperature = temperature

    logger.info(
        "GrammarLLM pipeline ready: %d gloss tokens, PDA stack=%s",
        len(vocab),
        pda.stack,
    )

    return logit_processor, streamer, pda


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
        for token in vocab:
            sub_tokens = tokenizer.tokenize(token)
            for st in sub_tokens:
                tid = tokenizer.convert_tokens_to_ids(st)
                if isinstance(tid, int) and tid != tokenizer.unk_token_id:
                    self.token_ids.add(tid)

            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid != tokenizer.unk_token_id:
                self.token_ids.add(tid)

        self.eos_token_id: int = tokenizer.eos_token_id
        self.token_ids.add(self.eos_token_id)

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
