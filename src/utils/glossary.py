"""Train-time rare-word glossary injection for GRPO prompts (opt-in).

Why gold-derived, not embeddings
---------------------------------
The population this targets is exactly the one ``src/analysis/rule_repair.py``
already characterizes: gold gloss tokens the model misses are hapax legomena
of the train split (median frequency 1; 61% occur <= 3 times, against a 1.6%
base rate for gold tokens generally). An embedding-similarity lookup would be
least reliable precisely where the glossary is meant to help — a rare word has
too little training signal to place it reliably in an embedding space, which
is the same sparsity problem the glossary exists to work around. Gold-derived
alignment has no such failure mode: at GRPO training time the gold gloss for
the CURRENT example is already available (it is what the reward scores
against), so the correct mapping for a rare source word is read directly off
this example's own gold via :func:`src.analysis.rule_repair.align_source_to_gold`
— the same stem-matching heuristic already measured and shipped for post-hoc
repair, applied one row at a time instead of fitted/aggregated.

Why the glossary is dropped on some training examples
-------------------------------------------------------
A design that always shows the glossary at training time and never at eval
time would put every evaluated prompt out of distribution relative to what
the model was trained on — the model may learn to read a block that then
vanishes, which is likely to *hurt* rather than help (see the ``glossario nel
prompt`` section of the thesis, ``docs/latex/capitoli/08_decoding_vincolato.tex``,
for the same objection raised against pure zero-shot injection). Applying
the glossary block on only a fraction of training prompts (``glossary.dropout``,
default 0.5) keeps glossary-free generation in-distribution during training
itself, so the eval-time comparison (always glossary-free, per design) tests
whether train-time exposure improved the model's OWN unaided handling of rare
words — not whether it learned to depend on a crutch that then disappears.

Where this is (and is NOT) wired in
-------------------------------------
Consumed only by ``src/training/grpo_t2g_train.py``'s dataset preparation,
for cells that set ``glossary.enabled: true`` (see
``experiments/configs/qwen25-05b/ablations/glossary/``). ``eval_t2g.py`` never
imports this module and never sets ``build_t2g_prompt``'s ``glossary_block``
argument: evaluation is always glossary-free, by construction, for every cell.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping

from src.analysis.rule_repair import align_source_to_gold

__all__ = [
    "build_example_glossary",
    "compute_word_frequencies",
    "format_glossary_block",
    "should_include_glossary",
]

_WORD_RE = re.compile(r"[a-z0-9\-]+")


def compute_word_frequencies(texts: Iterable[str]) -> Counter[str]:
    """Lowercased word frequency across a text corpus (TRAIN split only)."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_WORD_RE.findall(str(text).lower()))
    return counts


def build_example_glossary(
    text: str,
    gold_gloss: str,
    frequencies: Mapping[str, int],
    *,
    max_freq: int = 3,
) -> dict[str, str]:
    """Rare-word -> gold-gloss-form hints for ONE training example.

    A source word is "rare" (and therefore hinted) when its train-corpus
    frequency is <= ``max_freq`` — the same threshold ``rule_repair.py``
    documents as the hapax population responsible for most residual model
    errors. Only words that also align to a gold token (see
    :func:`~src.analysis.rule_repair.align_source_to_gold`) produce a hint;
    a rare word with no stem-matched gold token contributes nothing (there
    is nothing correct to hint).
    """
    aligned = align_source_to_gold(text, gold_gloss)
    return {
        word: token
        for word, token in aligned.items()
        if frequencies.get(word, 0) <= max_freq
    }


def format_glossary_block(glossary: Mapping[str, str]) -> str:
    """Render a glossary mapping as a prompt block, or ``""`` when empty."""
    if not glossary:
        return ""
    lines = "\n".join(f"{word} -> {token}" for word, token in glossary.items())
    return f"Glossary:\n{lines}"


def should_include_glossary(sample_id: str, dropout: float) -> bool:
    """Deterministic per-example dropout draw.

    Seeded off ``sample_id`` (not a shared ``Random`` instance advanced in
    iteration order) so the decision is reproducible regardless of dataset
    row order — the same reason ``sample_id`` already encodes text+gold
    elsewhere in this codebase (see ``build_t2g_dataset``).

    Args:
        sample_id: Stable per-example identifier.
        dropout: Fraction of examples that must NOT show the glossary, even
            when one was built for them. ``0.0`` always includes it, ``1.0``
            never does.
    """
    if dropout <= 0.0:
        return True
    if dropout >= 1.0:
        return False
    return random.Random(sample_id).random() >= dropout
