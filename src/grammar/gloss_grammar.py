"""
ASL Gloss vocabulary mask builder.

Provides:
    - ``GlossVocabularyMask`` — lightweight vocabulary mask for constrained
      decoding (production path).  Filters the token vocabulary to ASL gloss
      tokens (plus EOS) without a full pushdown-automaton parser.
"""

from __future__ import annotations

import logging
import string
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple Gloss Vocabulary Mask
# ---------------------------------------------------------------------------


class GlossVocabularyMask:
    """A lightweight vocabulary mask (no pushdown-automaton parsing).

    Directly masks the token vocabulary to allow only ASL gloss tokens
    (plus EOS).  Used during GRPO rollouts when vocabulary restriction
    is sufficient and full LL(1) grammar parsing is unnecessary.

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
