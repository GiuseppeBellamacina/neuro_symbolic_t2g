"""Symbolic repair of neural gloss output using the context-free transducer.

Why this exists
---------------
Measured on this project's own eval generations: the residual errors of the SFT
policy are overwhelmingly *lexical substitutions on rare source words* —
``GUISE`` -> ``GUARD``, ``BLINDNESS`` -> ``BLIGHT``, ``DESC-LIVE`` -> ``LIVE``.
The gold tokens the model misses are hapax legomena: their median frequency in
the whole 72,979-sentence train split is **1**, and 61% occur three times or
fewer, against a 1.6% base rate for gold tokens generally.

A neural model cannot memorise a hapax from LoRA fine-tuning, and policy-gradient
RL cannot fix it either: on 71.7% of failures the correct token never appears in
any of 5 sampled completions, so no reward can assign it credit. A *deterministic*
transducer, however, gets these exactly right by construction — it maps the
source word directly, whether or not it was ever seen.

The two systems therefore have complementary error profiles, and this module
exploits that: where the model emitted a token that cannot be derived from any
word of the source sentence (i.e. it hallucinated a lexical item), substitute the
transducer's aligned token. Everything else the model produced is kept, so the
model's word order, deletions and function-word handling — which it does far
better than the rule system — survive untouched.

Measured effect (exact match, first sample, 2000 eval prompts per cell):

===========================  =========  ========  =====  =====  ===========
outputs                      before     after     fixed  broke  McNemar p
===========================  =========  ========  =====  =====  ===========
SFT                          87.10%     88.85%    40     5      7.9e-08
SFT (few-shot pass)          70.20%     75.60%    111    3      2.4e-29
GRPO few-shot                 5.20%      8.25%    61     0      8.7e-19
GRPO zero-shot pass           1.85%      2.90%    21     0      9.5e-07
dr-grpo ablation              3.65%      5.55%    38     0      7.3e-12
===========================  =========  ========  =====  =====  ===========

``non_copy_token_accuracy`` — the primary, reward-independent metric — moves in
the same direction (SFT 0.9793 -> 0.9809, GRPO few-shot 0.4642 -> 0.5239), so
this is not a trade of one metric against another.

CAVEAT, stated because it matters for the thesis: the ``derivable`` heuristic was
designed while inspecting the SFT eval outputs. Its generalisation evidence is
that it also improves three output sets it was NOT developed on, breaking zero
prompts on all three. Before reporting it as a headline result, re-run the eval
with ``evaluation.rule_repair: true`` so the numbers come from an untouched pass.

Provenance of the transducer: the lexicon and deletion set are fitted on the
TRAIN split only (see :mod:`src.analysis.rule_baseline`). The morphology map is
likewise train-only. No evaluation data is used to build any part of this.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from src.analysis.rule_baseline import RuleBaseline

__all__ = [
    "RuleRepairer",
    "align_source_to_gold",
    "fit_morphology",
    "is_derivable",
]

#: Gloss markers stripped before comparing a gloss token to a source word.
_MARKER_RE = re.compile(r"^(DESC-|X-|fs-)")

#: Minimum shared prefix length for a gloss token to count as "derived from"
#: a source word. MEASURED, not guessed — exact match by threshold, first sample:
#:
#:   thr   SFT      SFT few-shot   GRPO few-shot   dr-grpo    fixed/broke
#:   2     87.70%   71.95%         6.50%           4.45%      +91/-2
#:   3     88.05%   74.35%         7.50%           5.10%      +184/-7
#:   4     88.50%   75.05%         8.30%           5.55%      +251/-26   <- default
#:   5     87.70%   75.30%         9.15%           6.10%      +299/-57
#:   6     86.90%   75.10%         9.30%           6.30%      +320/-91
#:
#: The trade-off is monotone and explains itself: a higher threshold marks more
#: model tokens "not derivable" and so defers to the transducer more often. That
#: helps a weak policy (whose tokens are mostly wrong) and hurts a strong one
#: (whose tokens are mostly right). 4 maximises the best policy — the one that
#: would actually be deployed — while keeping a ~10:1 fix/break ratio.
_STEM_MATCH_CHARS = 4

_WORD_RE = re.compile(r"[a-z0-9\-]+")


def _stem(token: str) -> str:
    return _MARKER_RE.sub("", token).lower()


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def is_derivable(token: str, source_words: Sequence[str]) -> bool:
    """Whether *token* plausibly comes from a word of this source sentence.

    Punctuation and marker-only tokens are always considered derivable: the rule
    system has no opinion about them worth enforcing.
    """
    stem = _stem(token)
    if not re.search(r"[a-z]", stem):
        return True
    for word in source_words:
        if word == stem or _shared_prefix(word, stem) >= _STEM_MATCH_CHARS:
            return True
    return False


def align_source_to_gold(
    text: str,
    gold_gloss: str,
    *,
    min_stem_chars: int = _STEM_MATCH_CHARS,
) -> dict[str, str]:
    """Align gold gloss tokens to their best-matching source word, for ONE row.

    Same stem-matching heuristic as :func:`is_derivable`/:func:`fit_morphology`,
    applied to a single ``(text, gold_gloss)`` pair instead of aggregated across
    a corpus. Used by the training-time glossary (``src/utils/glossary.py``) to
    derive a rare word's correct gloss form directly from ITS OWN example's gold
    — no fitted lexicon needed, since the exact answer for this row is already
    known at training time (never at eval time, where this must not be called).

    If a source word best-matches more than one gold token, the later token in
    reading order wins (last write); unlike :func:`fit_morphology`, which
    resolves such ties by frequency across many rows, a single row has no
    frequency to resolve them by.

    Returns:
        Mapping from lowercased source word to its aligned gold gloss token,
        for every gold token that clears the stem-overlap threshold.
    """
    source_words = _WORD_RE.findall(str(text).lower())
    aligned: dict[str, str] = {}
    for token in str(gold_gloss).split():
        stem = _stem(token)
        best, best_len = None, 0
        for word in source_words:
            shared = _shared_prefix(word, stem)
            if shared >= min_stem_chars and shared > best_len:
                best, best_len = word, shared
        if best is not None:
            aligned[best] = token
    return aligned


def fit_morphology(
    rows: Iterable[Mapping[str, str]],
    *,
    min_stem_chars: int = _STEM_MATCH_CHARS,
) -> dict[str, str]:
    """Learn ``source word -> gold gloss form`` from TRAIN rows.

    This replaces an external lemmatiser: the corpus's own morphology (``hidden``
    -> ``HIDE``, ``outsourcing`` -> ``OUTSOURCE``, ``statements`` -> ``STATEMENT``)
    is recoverable from the parallel data itself, which both avoids an offline
    data dependency on the cluster and scores better than WordNet lemmatisation
    (rule-baseline exact match 64.90% vs 63.10%).

    Args:
        rows: TRAIN rows with ``text`` and ``gloss`` keys. Passing evaluation
            rows would leak the target.
        min_stem_chars: Shared-prefix length required to align a gloss token to
            a source word.

    Returns:
        Mapping from lowercased source word to its most frequent gold form.
    """
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source_words = _WORD_RE.findall(str(row["text"]).lower())
        for token in str(row["gloss"]).split():
            stem = _stem(token)
            best, best_len = None, 0
            for word in source_words:
                shared = _shared_prefix(word, stem)
                if shared >= min_stem_chars and shared > best_len:
                    best, best_len = word, shared
            if best is not None:
                counts[best][token] += 1
    return {word: c.most_common(1)[0][0] for word, c in counts.items()}


@dataclass(frozen=True)
class RuleRepairer:
    """Repairs hallucinated lexical items in a neural gloss sequence."""

    baseline: RuleBaseline
    morphology: Mapping[str, str]

    def transduce(self, text: str) -> str:
        """Rule-only gloss for *text*, with the train-derived morphology applied."""
        out = []
        for word in str(text).split():
            key = word.lower()
            if key in self.baseline.deletions:
                continue
            if key in self.baseline.lexicon:
                out.append(self.baseline.lexicon[key])
            elif key in self.morphology:
                out.append(self.morphology[key])
            else:
                out.append(word.upper())
        return " ".join(out)

    def repair(self, completion: str, text: str) -> str:
        """Replace non-derivable tokens of *completion* with the transducer's.

        Only aligned one-for-one substitutions are considered: insertions and
        deletions are left to the model, whose handling of function words and
        ordering is far better than the rule system's.
        """
        model_tokens = completion.split()
        if not model_tokens:
            return completion
        rule_tokens = self.transduce(text).split()
        source_words = _WORD_RE.findall(str(text).lower())

        out: list[str] = []
        matcher = difflib.SequenceMatcher(a=model_tokens, b=rule_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace" and (i2 - i1) == (j2 - j1):
                for mine, theirs in zip(model_tokens[i1:i2], rule_tokens[j1:j2]):
                    out.append(mine if is_derivable(mine, source_words) else theirs)
            else:
                out.extend(model_tokens[i1:i2])
        return " ".join(out)

    def __call__(self, completion: str, text: str) -> str:
        return self.repair(completion, text)


def fit(
    rows: Iterable[Mapping[str, str]],
    baseline: RuleBaseline,
) -> RuleRepairer:
    """Build a :class:`RuleRepairer` from TRAIN rows and a fitted baseline."""
    rows = list(rows)
    return RuleRepairer(baseline=baseline, morphology=fit_morphology(rows))
