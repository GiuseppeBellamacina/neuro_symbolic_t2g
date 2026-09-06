"""ASL gloss constraints backed by grammarllm or a vocabulary mask."""

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

START_NONTERMINAL = "S*"
TAIL_NONTERMINAL = "S*_TAIL"

_SKIP_TOKENS = {"<BOS>", "<UNK>", "<EOS>", "<PAD>"}


def _validate_nonterminal_names(tokenizer: Any) -> None:
    """Reject names grammarllm would confuse with concrete terminals."""
    vocab = tokenizer.get_vocab()
    for name in (START_NONTERMINAL, TAIL_NONTERMINAL):
        if name in vocab:
            raise ValueError(
                f"Nonterminal name '{name}' collides with a tokenizer token "
                f"string (id {vocab[name]}). grammarllm would misparse that "
                f"token as a nonterminal reference. Rename the tail "
                f"nonterminal (see TAIL_NONTERMINAL)."
            )


def _validate_eos_token(eos_token: str) -> None:
    """Check that EOS is usable as one literal grammar terminal."""
    if not eos_token or not eos_token.strip():
        raise ValueError(f"Invalid (empty) eos_token: {eos_token!r}")
    if "<<" in eos_token or ">>" in eos_token:
        raise ValueError(
            f"eos_token {eos_token!r} contains '<<'/'>>' and would be parsed "
            f"as a grammarllm <<tag>> instead of a literal terminal."
        )
    if any(ch.isspace() for ch in eos_token):
        raise ValueError(
            f"eos_token {eos_token!r} contains whitespace; grammarllm splits "
            f"non-tag symbols on whitespace, so it cannot be a single terminal."
        )


def build_gloss_grammar(
    vocab: list[str],
    tokenizer: Any,
) -> dict[str, list[str]]:
    """Build an LL(1) grammar with bare first and space-prefixed later glosses.

    ``S*`` emits the first gloss. ``S*_TAIL`` emits later glosses or native
    EOS, avoiding prefix/FOLLOW conflicts between adjacent bare BPE tokens.
    """
    eos_token = tokenizer.eos_token
    _validate_eos_token(eos_token)
    _validate_nonterminal_names(tokenizer)

    seen: set[str] = set()
    gloss_tokens: list[str] = []
    for token in vocab:
        if token in _SKIP_TOKENS or not token.strip() or token in seen:
            continue
        seen.add(token)
        gloss_tokens.append(token)

    logger.info(
        "Building separator-aware gloss grammar with %d tokens (skipped "
        "BOS/UNK/EOS/PAD): S* -> <<g>> %s (bare), %s -> << g>> %s | %s",
        len(gloss_tokens),
        TAIL_NONTERMINAL,
        TAIL_NONTERMINAL,
        TAIL_NONTERMINAL,
        eos_token,
    )

    grammar = {
        START_NONTERMINAL: [
            f"<<{gloss}>> {TAIL_NONTERMINAL}" for gloss in gloss_tokens
        ],
        TAIL_NONTERMINAL: [f"<< {gloss}>> {TAIL_NONTERMINAL}" for gloss in gloss_tokens]
        + [eos_token],
    }
    logger.info(
        "  Grammar rules: S* → %d alternatives, %s → %d alternatives (+EOS)",
        len(gloss_tokens),
        TAIL_NONTERMINAL,
        len(gloss_tokens),
    )
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
    """Build grammarllm PDA templates, streamer, and primary PDA."""
    if enable_logging:
        setup_logging()

    grammar = build_gloss_grammar(vocab, tokenizer)
    pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
        tokenizer,
        productions=grammar,
    )

    pdas, streamer = generate_grammar_parameters(
        tokenizer,
        pars_table,
        map_terminal_tokens,
        num_return_sequences=num_return_sequences,
        token_lookahead=token_lookahead,
    )

    pda: PushdownAutomaton = pdas[0]
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
