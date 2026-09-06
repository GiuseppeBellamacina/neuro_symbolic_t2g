#!/usr/bin/env python3
"""Regression tests for the separator-aware two-level PDA grammar (job 7198).

Root cause being regression-tested: the pre-fix flat grammar
``S* → <<gloss>> S*`` had no separator, so with real Qwen digit
tokenization (``0`` → ``['0']``, ``85`` → ``['8', '5']``, ...) a bare
sub-token could be both a gloss continuation (FIRST of a prefix-group
suffix) and the start of the next gloss (FOLLOW), crashing the LL(1) table
build with an epsilon/FOLLOW vs FIRST conflict (job 7198).

The fix under test (``build_gloss_grammar``) is the two-level grammar::

    S*      → <<g>> S*_TAIL          (first gloss, bare tokenization)
    S*_TAIL → << g>> S*_TAIL | <eos> (later glosses, leading literal space)

whose tail FOLLOW sets contain only space-prefixed sub-tokens and EOS,
token-distinct from bare continuations.

Coverage:
  1. Grammar shape: two rules, bare/space-prefixed tags, explicit native
     EOS terminal on the tail, digit glosses retained, specials skipped.
  2. Nonterminal names never collide with tokenizer token strings.
  3. Root-cause pin: the old flat pattern still raises the LL(1) conflict.
  4. Full LL(1) pipeline (grammar → parsing table → PDA) builds without
     conflict on the job 7198-style digit/punctuation regression vocab.
  5. PDA walk (automaton API): ``IX MAN WALK``, ``IX MAN 8.30``,
     ``08.30``, ``0 8.30``, ``RETAIL`` accepted; concatenations such as
     ``IX MANWALK`` rejected at the mask level.
  6. Termination semantics: EOS terminates exactly once, from ``S*_TAIL``.
  7. Generation-state check through ``GrammarPDALogitsProcessor`` (the
     processor actually used at training/eval time).
  8. Env-gated full-vocabulary production gate (~15.5K glosses): with
      ``T2G_PDA_FULL_VOCAB=1`` it hard-fails on anything but the production
      Qwen2.5-0.5B-Instruct tokenizer, requires the pre-built offline
      artifact ``data/gloss_vocab.txt`` (no dataset download — compute nodes
      are offline), checks the canonical vocab count range, builds the full
      LL(1) pipeline (exceptions propagate naturally), reports elapsed time
      and peak RSS with ``-s``, and walks a digit gloss.  Without the flag
      it is skipped in ordinary CI; with the flag it never skips.

Uses the shared session-scoped ``tokenizer`` fixture (Qwen2.5-0.5B, gpt2
fallback) from conftest.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Job 7198-style regression vocabulary: digit-prefix family (ambiguous under
# the flat grammar) + punctuation/single-letter family + core glosses.
REGRESSION_GLOSSES = [
    "IX",
    "MAN",
    "WALK",
    "0",
    "05",
    "08.30",
    "5",
    "8",
    "85",
    "8.30",
    "?",
    "??",
    "?TAT",
    "T",
    "RE",
    "RETAIL",
]
REGRESSION_VOCAB = ["<BOS>", "<EOS>", "<UNK>"] + REGRESSION_GLOSSES

ACCEPTED_SEQUENCES = [
    "IX MAN WALK",
    "IX MAN 8.30",
    "08.30",
    "0 8.30",
    "RETAIL",
    "8.30",
    "?? ?TAT",
    "WALK",
]
REJECTED_CONCATENATIONS = [
    "IX MANWALK",
    "IXMAN WALK",
    "WALKHOUSE",
]


@pytest.fixture(scope="module")
def regression_pipeline(tokenizer):
    """Full grammarllm pipeline over the job 7198 regression vocabulary."""
    from src.grammar.gloss_grammar import create_grammarllm_pipeline

    pdas, streamer, pda = create_grammarllm_pipeline(REGRESSION_VOCAB, tokenizer)
    return pdas, streamer, pda


def _walk(pda, tokenizer, text: str):
    """Feed ``text``'s token IDs through a fresh PDA clone (automaton API).

    Returns ``(walker, offending_id, accepted)``.  ``offending_id`` is the
    first token ID outside the automaton's valid set (the mask would have
    blocked it during real generation), ``None`` when fully accepted.
    """
    walker = pda.clone()
    for tid in tokenizer.encode(text, add_special_tokens=False):
        if tid not in walker.get_tokens():
            return walker, tid, False
        walker.next_state(tid)
    return walker, None, True


def _consume_eos(walker, tokenizer):
    """Consume the native EOS terminal and return the walker."""
    assert (
        tokenizer.eos_token_id in walker.get_tokens()
    ), "EOS must be a valid continuation once the grammar allows termination"
    walker.next_state(tokenizer.eos_token_id)
    return walker


# ---------------------------------------------------------------------------
# 1. Grammar construction
# ---------------------------------------------------------------------------


def test_build_gloss_grammar_two_level_shape(tokenizer):
    """Two-level grammar: bare first gloss, space-prefixed tail, explicit EOS."""
    from src.grammar.gloss_grammar import TAIL_NONTERMINAL, build_gloss_grammar

    grammar = build_gloss_grammar(REGRESSION_VOCAB, tokenizer)

    assert set(grammar) == {"S*", TAIL_NONTERMINAL}

    # First gloss alternatives: bare tags delegating to the tail.
    assert grammar["S*"] == [f"<<{g}>> {TAIL_NONTERMINAL}" for g in REGRESSION_GLOSSES]

    # Subsequent gloss alternatives: exact tags with a leading literal space,
    # plus the native tokenizer EOS as the final (plain, non-tag) terminal.
    eos = tokenizer.eos_token
    assert grammar[TAIL_NONTERMINAL] == [
        f"<< {g}>> {TAIL_NONTERMINAL}" for g in REGRESSION_GLOSSES
    ] + [eos]
    assert grammar[TAIL_NONTERMINAL][-1] == eos
    assert "<<" not in eos  # EOS must be a literal terminal, not a <<tag>>

    # Full non-special vocab kept, including digit glosses (Trie parity).
    assert any("08.30" in prod for prod in grammar["S*"])
    assert any(" 08.30>>" in prod for prod in grammar[TAIL_NONTERMINAL])
    assert "<BOS>" not in "".join(grammar["S*"])


def test_nonterminal_names_not_tokenizer_tokens(tokenizer):
    """Nonterminal names must not collide with tokenizer token strings."""
    from src.grammar.gloss_grammar import build_gloss_grammar

    vocab = tokenizer.get_vocab()
    assert "S*" not in vocab, "'S*' must not be a tokenizer token string"
    assert "S*_TAIL" not in vocab, "tail nonterminal must not be a token string"

    # Direct guard check: a tokenizer whose vocab contains the tail name must
    # be rejected loudly (grammarllm treats LHS membership as the NT test, so
    # such a collision would silently corrupt FIRST/FOLLOW).
    class _CollidingTokenizer:
        eos_token = "<|endoftext|>"

        def get_vocab(self):
            return {"S*_TAIL": 5, "<|endoftext|>": 0}

    with pytest.raises(ValueError, match="collides with a tokenizer token"):
        build_gloss_grammar(REGRESSION_VOCAB, _CollidingTokenizer())


# ---------------------------------------------------------------------------
# 2. Root-cause pin + full pipeline build (job 7198)
# ---------------------------------------------------------------------------


def test_flat_grammar_root_cause_conflict(tokenizer):
    """The pre-fix flat grammar still raises the LL(1) epsilon/FOLLOW conflict.

    Pins the root cause of job 7198: ``S* → <<g>> S*`` without separators is
    inherently token-ambiguous for the digit-prefix family.  If this stops
    failing because grammarllm changed, revisit the two-level rationale.
    """
    from grammarllm.generate_with_constraints import get_parsing_table_and_map_tt

    flat = {"S*": [f"<<{g}>> S*" for g in REGRESSION_GLOSSES]}
    with pytest.raises(ValueError, match="Conflict"):
        get_parsing_table_and_map_tt(tokenizer, productions=flat)


def test_job7198_pipeline_builds_without_conflict(regression_pipeline, tokenizer):
    """Full LL(1) pipeline builds on the job 7198 regression vocabulary."""
    from src.grammar.gloss_grammar import TAIL_NONTERMINAL

    pdas, streamer, pda = regression_pipeline
    assert isinstance(pdas, list) and pdas
    assert pdas[0] is pda and streamer is not None

    eos = tokenizer.eos_token
    bare_firsts = {tokenizer.tokenize(g)[0] for g in REGRESSION_GLOSSES}
    spaced_firsts = {tokenizer.tokenize(" " + g)[0] for g in REGRESSION_GLOSSES}

    # S* row: bare first sub-tokens + the auto-appended native EOS.
    assert set(pda.grammar["S*"]) == bare_firsts | {eos}
    assert pda.grammar["S*"][eos] == [eos]

    # Tail row: space-prefixed first sub-tokens + this module's explicit EOS.
    assert set(pda.grammar[TAIL_NONTERMINAL]) == spaced_firsts | {eos}
    assert pda.grammar[TAIL_NONTERMINAL][eos] == [eos]

    # No digit asymmetry: digit glosses contribute their sub-token terminals.
    assert "0" in pda.grammar["S*"]  # bare digit gloss start
    assert tokenizer.tokenize(" 8.30")[0] in pda.grammar[TAIL_NONTERMINAL]


# ---------------------------------------------------------------------------
# 3. PDA walk: acceptance / rejection / termination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ACCEPTED_SEQUENCES)
def test_pda_walk_accepts_gold_sequences(regression_pipeline, tokenizer, text):
    """Space-separated gloss sequences walk the PDA to an empty stack."""
    _, _, pda = regression_pipeline
    walker, offending, accepted = _walk(pda, tokenizer, text)
    assert accepted, (
        f"{text!r} rejected at token {offending!r} "
        f"({tokenizer.convert_ids_to_tokens([offending]) if offending else '-'}) "
        f"with stack {walker.stack}"
    )
    walker = _consume_eos(walker, tokenizer)
    assert walker.eos() and walker.stack == []


@pytest.mark.parametrize("text", REJECTED_CONCATENATIONS)
def test_pda_walk_rejects_concatenations(regression_pipeline, tokenizer, text):
    """Concatenated gloss strings are blocked by the PDA's valid-token set."""
    _, _, pda = regression_pipeline
    walker, offending, accepted = _walk(pda, tokenizer, text)
    assert not accepted, f"concatenated {text!r} was unexpectedly accepted"
    # The offending token was outside the valid set (i.e. the logits mask
    # would have removed it), and consuming it raises on the automaton.
    with pytest.raises(ValueError):
        walker.next_state(offending)


def test_termination_from_tail_exactly_once(regression_pipeline, tokenizer):
    """EOS terminates from S*_TAIL exactly once; no token is consumable after."""
    _, _, pda = regression_pipeline
    walker, offending, accepted = _walk(pda, tokenizer, "IX MAN")
    assert accepted and offending is None

    # Not satisfiable before EOS: the tail still expects a space-prefixed
    # gloss or EOS, so the stack is non-empty and generation continues.
    assert not walker.eos()
    assert walker.stack == ["S*_TAIL"]

    walker = _consume_eos(walker, tokenizer)
    assert walker.eos() and walker.stack == []

    # Exactly once: after EOS the grammar is exhausted — no tokens (EOS
    # included) remain consumable.
    assert walker.get_tokens() == []
    with pytest.raises(ValueError):
        walker.next_state(tokenizer.eos_token_id)


def test_first_gloss_may_terminate_immediately(regression_pipeline, tokenizer):
    """S* accepts immediate EOS, mirroring the Trie's start-state behavior."""
    _, _, pda = regression_pipeline
    assert tokenizer.eos_token_id in pda.get_tokens()
    walker = _consume_eos(pda.clone(), tokenizer)
    assert walker.eos() and walker.stack == []


# ---------------------------------------------------------------------------
# 4. Generation-state check through the production logits processor
# ---------------------------------------------------------------------------


def test_processor_mask_accepts_gold_and_forces_single_eos(
    regression_pipeline, tokenizer
):
    """GrammarPDALogitsProcessor allows gold tokens and then only EOS.

    Simulates constrained decoding state: at every step the gold token must
    survive the mask, and once EOS is emitted the mask allows EOS only
    (termination exactly once at processor level).
    """
    from src.grammar.grammar_logits_processor import GrammarPDALogitsProcessor

    pdas, _, _ = regression_pipeline
    processor = GrammarPDALogitsProcessor(tokenizer, pdas)
    width = len(tokenizer)
    ids = torch.tensor([[tokenizer.eos_token_id]])  # dummy 1-token prompt
    gold = tokenizer.encode("0 8.30", add_special_tokens=False)

    for tid in gold:
        out = processor(ids, torch.zeros(1, width))
        assert bool(out[0, tid].isfinite()), (
            f"gold token {tid} "
            f"({tokenizer.convert_ids_to_tokens([tid])}) masked at step "
            f"{ids.shape[1] - 1}"
        )
        ids = torch.cat([ids, torch.tensor([[tid]])], dim=1)

    # Grammar satisfied-but-open: EOS must be allowed after the last gloss.
    out = processor(ids, torch.zeros(1, width))
    assert bool(out[0, tokenizer.eos_token_id].isfinite())
    ids = torch.cat([ids, torch.tensor([[tokenizer.eos_token_id]])], dim=1)

    # After EOS the PDA is exhausted: the mask allows EOS only.
    out = processor(ids, torch.zeros(1, width))
    allowed = {int(i) for i in out[0].isfinite().nonzero(as_tuple=True)[0]}
    assert allowed == {tokenizer.eos_token_id}


def test_processor_mask_distinguishes_bare_and_spaced_tokens(
    regression_pipeline, tokenizer
):
    """After a complete gloss, bare continuations are masked, spaced ones not.

    This is the separator semantics at mask level: from ``IX MAN`` the bare
    ``W`` of ``IX MANWALK`` is rejected while the space-prefixed ``ĠW`` of
    ``IX MAN WALK`` is allowed.
    """
    from src.grammar.grammar_logits_processor import GrammarPDALogitsProcessor

    pdas, _, _ = regression_pipeline
    processor = GrammarPDALogitsProcessor(tokenizer, pdas)
    width = len(tokenizer)
    ids = torch.tensor([[tokenizer.eos_token_id]])  # dummy 1-token prompt

    for tid in tokenizer.encode("IX MAN", add_special_tokens=False):
        out = processor(ids, torch.zeros(1, width))
        assert bool(out[0, tid].isfinite())
        ids = torch.cat([ids, torch.tensor([[tid]])], dim=1)

    out = processor(ids, torch.zeros(1, width))
    concat_w = tokenizer.encode("IX MANWALK", add_special_tokens=False)[-2]  # bare 'W'
    spaced_w = tokenizer.encode("IX MAN WALK", add_special_tokens=False)[2]  # 'ĠW'
    assert not bool(
        out[0, concat_w].isfinite()
    ), "bare 'W' must be masked after a gloss"
    assert bool(
        out[0, spaced_w].isfinite()
    ), "spaced 'ĠW' must be allowed after a gloss"


# ---------------------------------------------------------------------------
# 5. Full-vocabulary production gate (env-gated; never skips when enabled)
# ---------------------------------------------------------------------------

# Canonical ASLG-PC12 train-split gloss vocabulary is ~15.5K entries; the
# range rejects truncated or accidentally duplicated artifacts.
FULL_VOCAB_RANGE = (14000, 17000)


def _peak_rss_mb() -> float | None:
    """Peak process RSS in MB via ``resource.getrusage`` (None if unavailable).

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; Windows has no
    ``resource`` module.
    """
    try:
        import resource
    except ImportError:
        return None
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxrss / (1024 * 1024)
    return maxrss / 1024  # Linux: kilobytes


@pytest.mark.skipif(
    os.environ.get("T2G_PDA_FULL_VOCAB") != "1",
    reason=(
        "Full ~15.5K-vocab production gate takes minutes (vendored "
        "per-terminal vocab scans); set T2G_PDA_FULL_VOCAB=1 to run it "
        "explicitly"
    ),
)
def test_full_vocabulary_pipeline_gate(tokenizer):
    """Hard production gate: full gloss vocabulary through the LL(1) pipeline.

    Manual command (compute node, offline artifact required, run with
    ``-s`` to see the metrics)::

        T2G_PDA_FULL_VOCAB=1 pytest tests/test_pda_grammar.py -k full_vocabulary -s

    Zero-skip semantics: when ``T2G_PDA_FULL_VOCAB=1`` this test either
    passes or fails — never skips.  It hard-fails on a fallback tokenizer
    (only production Qwen2.5-0.5B-Instruct is accepted), on a missing or
    empty ``data/gloss_vocab.txt`` (compute nodes are offline, so the
    artifact must be pre-built; the dataset is NOT downloaded here), on a
    vocab count outside the canonical range, or on any pipeline exception
    (which propagates naturally).
    """
    from src.grammar.gloss_grammar import TAIL_NONTERMINAL, create_grammarllm_pipeline

    # ── Gate 1: production tokenizer only (no gpt2 fallback) ────────────
    name_or_path = getattr(tokenizer, "name_or_path", "")
    assert "Qwen2.5-0.5B-Instruct" in name_or_path, (
        f"Production gate requires the Qwen2.5-0.5B-Instruct tokenizer, got "
        f"{name_or_path!r}. The conftest gpt2 fallback is not acceptable "
        f"here: digit/space tokenization differs."
    )

    # ── Gate 2: offline vocab artifact must exist and be nonempty ───────
    repo_root = Path(__file__).resolve().parent.parent
    vocab_path = repo_root / "data" / "gloss_vocab.txt"
    assert vocab_path.exists(), (
        f"Missing offline artifact {vocab_path}. Compute nodes are offline, "
        f"so the full-vocabulary gate requires the pre-built artifact "
        f"(produced by training data preparation: prepare_only). The "
        f"dataset is intentionally NOT downloaded here."
    )
    from src.datasets.aslg_dataset import load_vocabulary

    vocab = load_vocabulary(vocab_path)
    assert vocab, f"Offline artifact {vocab_path} is empty"

    # ── Gate 3: canonical vocab count range ─────────────────────────────
    lo, hi = FULL_VOCAB_RANGE
    assert lo <= len(vocab) <= hi, (
        f"Canonical gloss vocabulary must have {lo}..{hi} entries, got "
        f"{len(vocab)}. The artifact may be truncated or corrupted."
    )

    # ── Gate 4: full pipeline build, timed and RSS-reported ─────────────
    print(
        f"\n[full-vocab gate] tokenizer={name_or_path} "
        f"eos={tokenizer.eos_token!r} (id={tokenizer.eos_token_id}) "
        f"vocab_size={len(vocab)}"
    )
    start = time.perf_counter()
    pdas, streamer, pda = create_grammarllm_pipeline(vocab, tokenizer)
    elapsed = time.perf_counter() - start
    peak_mb = _peak_rss_mb()
    peak_str = f"{peak_mb:.1f}" if peak_mb is not None else "n/a (no resource module)"
    print(
        f"[full-vocab gate] pipeline build: {elapsed:.2f}s, " f"peak RSS: {peak_str} MB"
    )

    # ── Structural assertions on the built pipeline ─────────────────────
    assert streamer is not None
    assert isinstance(pdas, list) and pdas and pdas[0] is pda
    assert {TAIL_NONTERMINAL, "S*"} <= set(pda.grammar)
    assert tokenizer.eos_token in pda.grammar[TAIL_NONTERMINAL]
    assert pda.grammar[TAIL_NONTERMINAL][tokenizer.eos_token] == [tokenizer.eos_token]

    # ── Gate 6: digit-gloss walk through the automaton ──────────────────
    digit_gloss = next((g for g in vocab if any(c.isdigit() for c in g.strip())), None)
    assert digit_gloss is not None, (
        "Canonical ASLG-PC12 vocabulary must contain digit glosses "
        "(dates/codes); none found — artifact suspicious."
    )
    walker, offending, accepted = _walk(pda, tokenizer, digit_gloss)
    assert accepted, f"digit gloss {digit_gloss!r} rejected at {offending!r}"
    walker = _consume_eos(walker, tokenizer)
    assert walker.eos() and walker.stack == []
