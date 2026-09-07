"""Context-free rule baseline for ASLG-PC12 text-to-gloss.

Why this exists
---------------
ASLG-PC12 is a *synthetic* corpus: the gloss side was produced from English by
rule-based methods (Moryossef et al., arXiv:2105.07476), and independent work
measures ~98% gloss/text token overlap plus a hand-written rule reproducing the
reference gloss at BLEU 96.75 (Peng et al., arXiv:2304.10844, Tables 3 and 7).

Consequence: overlap metrics on this corpus are largely saturated by a trivial
transformation, so *no* model number is interpretable without this baseline
printed next to it. Measured on the project's own evaluation prompts, this
module reaches ROUGE-L ~0.969 / glossF1 ~0.967 / BLEU-corpus ~0.891 while the
SFT model reaches 0.975 / 0.977 / 0.946 — i.e. the overlap metrics differ by
<0.01 while exact match differs by ~0.23 (0.587 vs 0.813).

The baseline is deliberately **context-free**: a unigram lexicon plus a learned
deletion set. It is near its own ceiling (lowering the deletion threshold does
not help), and the residual the model captures is dominated by *contextual*
copula deletion, which no context-free rule can express. That contrast is the
scientific point, so do not "improve" this module into a contextual model.

Protocol
--------
The lexicon and the deletion set MUST be estimated from the training split only.
`fit` enforces nothing by itself, so callers are responsible for passing train
rows; `fit_from_split` is the safe entry point.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# Defaults reproduce the audited configuration (see docs/RECOVERY_REPORT.md).
#
# Threshold provenance, because it matters for the leakage question: the grid
# {min_count 10,20,30,50} x {threshold 0.55,0.70,0.85,0.95} was scored on a
# train-INTERNAL 90/10 dev slice (seed 1234), never on test. The dev-selected
# optimum was min_count=10, threshold=0.85. Refitting on full train with either
# min_count 10 or 30 gives the SAME held-out test result (EM 0.5912, non-copy
# 0.9428), so 30 is kept as the audited value and the choice is not
# outcome-sensitive. The threshold 0.85 is the dev-selected one.
DEFAULT_MIN_COUNT = 30
DEFAULT_DELETION_THRESHOLD = 0.85


@dataclass(frozen=True)
class RuleBaseline:
    """A fitted context-free English->gloss transducer."""

    lexicon: Mapping[str, str]
    deletions: frozenset[str]

    def apply(self, text: str) -> str:
        """Map one English sentence to a gloss string."""
        out = []
        for word in str(text).split():
            key = word.lower()
            if key in self.deletions:
                continue
            out.append(self.lexicon.get(key, word.upper()))
        return " ".join(out)

    def __call__(self, text: str) -> str:
        return self.apply(text)


def _pairs(rows: Iterable[Mapping[str, str]]) -> list[tuple[list[str], list[str]]]:
    return [(str(r["text"]).split(), str(r["gloss"]).split()) for r in rows]


def fit(
    rows: Iterable[Mapping[str, str]],
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    deletion_threshold: float = DEFAULT_DELETION_THRESHOLD,
) -> RuleBaseline:
    """Estimate the lexicon and deletion set from ``rows``.

    ``rows`` must be TRAIN rows only; passing evaluation rows leaks the target.

    The lexicon counts only length-aligned pairs, where the position-wise
    correspondence is unambiguous. The deletion set collects source words whose
    mapped gloss is absent from the reference in more than ``deletion_threshold``
    of their occurrences.
    """
    if not 0.0 < deletion_threshold <= 1.0:
        raise ValueError(
            f"deletion_threshold must be in (0, 1]: {deletion_threshold!r}"
        )
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1: {min_count!r}")

    pairs = _pairs(rows)

    counts: dict[str, Counter] = defaultdict(Counter)
    for source, gloss in pairs:
        if len(source) == len(gloss):
            for word, target in zip(source, gloss):
                counts[word.lower()][target] += 1
    lexicon = {word: targets.most_common(1)[0][0] for word, targets in counts.items()}

    absent: Counter = Counter()
    total: Counter = Counter()
    for source, gloss in pairs:
        gloss_set = set(gloss)
        for word in {w.lower() for w in source}:
            total[word] += 1
            if lexicon.get(word, word.upper()) not in gloss_set:
                absent[word] += 1
    deletions = frozenset(
        word
        for word, seen in total.items()
        if seen >= min_count and absent[word] / seen > deletion_threshold
    )
    return RuleBaseline(lexicon=lexicon, deletions=deletions)


def fit_from_split(
    dataset: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    split: str = "train",
    min_count: int = DEFAULT_MIN_COUNT,
    deletion_threshold: float = DEFAULT_DELETION_THRESHOLD,
) -> RuleBaseline:
    """Fit on a named split, defaulting to ``train`` to avoid target leakage."""
    if split != "train":
        raise ValueError(
            f"refusing to fit the rule baseline on split {split!r}; "
            f"the baseline must be estimated from training data only"
        )
    return fit(
        dataset[split],
        min_count=min_count,
        deletion_threshold=deletion_threshold,
    )


def non_copy_token_accuracy(
    predictions: Sequence[str],
    sources: Sequence[str],
    references: Sequence[str],
) -> tuple[float, int, int]:
    """Accuracy restricted to reference tokens that are not source copies.

    Rationale
    ---------
    On ASLG-PC12 roughly 62% of gloss tokens are the uppercased source token, and
    the project's ROUGE-L is both case-insensitive and splits on non-alphanumeric
    characters (``rouge_score`` turns ``DESC-GOOD`` into ``['desc', 'good']``, so
    ``DESC-GOOD`` vs ``DESC-BAD`` scores 0.5). A model that merely echoes the
    English source therefore collects a large, misleading score.

    This metric scores only the reference tokens that cannot be obtained by
    uppercasing a source token, and it is case-sensitive. It is the metric that
    separates "learned to copy English" from "learned the transduction". Measured
    values on the project's evaluation prompts: rule 0.9424, SFT 0.9660,
    GRPO-only 0.4641, zero-shot+Trie 0.0432, zero-shot without constraints
    0.0006 — i.e. it reverses the apparent ROUGE-L conclusion that constrained
    decoding hurts.

    Matching is multiset-based (a reference token occurring twice must be
    produced twice), which keeps the score insensitive to word order; order is
    already covered by exact match.

    Returns:
        ``(accuracy, hits, total)``. ``accuracy`` is 0.0 when ``total`` is 0.
    """
    if not (len(predictions) == len(sources) == len(references)):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)}, "
            f"sources={len(sources)}, references={len(references)}"
        )
    hits = 0
    total = 0
    for prediction, source, reference in zip(predictions, sources, references):
        copyable = {word.upper() for word in str(source).split()}
        available = Counter(str(prediction).split())
        for token in str(reference).split():
            if token in copyable:
                continue
            total += 1
            if available[token] > 0:
                available[token] -= 1
                hits += 1
    return (hits / total if total else 0.0), hits, total
