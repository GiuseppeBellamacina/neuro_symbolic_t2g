"""Tests for ``dump_completion_examples`` alignment and prompt grouping.

Regression tests for the multi-sample misalignment bug: in evals with
``num_samples > 1`` completions/references/rouge_scores/prompts are FLAT
(one entry per completion). The best/worst table must (a) show a gold that
matches its prompt and (b) not repeat the same prompt across slots.
"""

from __future__ import annotations

import json

import pytest

from src.utils.visualization import dump_completion_examples


def _make_flat(n_prompts: int = 6, num_samples: int = 5) -> dict[str, list]:
    """Build flat, ALIGNED inputs mimicking a multi-sample eval."""
    completions: list[str] = []
    references: list[str] = []
    rouge_scores: list[float] = []
    prompts: list[str] = []
    for p in range(n_prompts):
        prompt = f"prompt {p}"
        gold = f"GOLD {p}"
        for s in range(num_samples):
            completions.append(f"PRED {p}-{s}")
            references.append(gold)
            # Distinct scores: prompt 0 is perfect, prompt 5 is terrible
            rouge_scores.append(round(1.0 - p * 0.2 - s * 0.01, 4))
            prompts.append(prompt)
    return {
        "completions": completions,
        "references": references,
        "rouge_scores": rouge_scores,
        "prompts": prompts,
    }


def test_gold_matches_prompt_in_every_entry(tmp_path):
    data = _make_flat()
    dump_completion_examples(
        data["completions"],
        data["references"],
        data["rouge_scores"],
        prompts=data["prompts"],
        n_examples=6,
        output_dir=str(tmp_path),
    )
    examples = json.loads(
        (tmp_path / "completion_examples.json").read_text(encoding="utf-8")
    )["examples"]
    assert examples, "no examples dumped"
    for ex in examples:
        prompt_num = int(ex["prompt"].split()[-1])
        assert (
            ex["gold"] == f"GOLD {prompt_num}"
        ), f"gold {ex['gold']!r} does not belong to prompt {ex['prompt']!r}"
        assert (
            ex["prediction"] == f"PRED {prompt_num}-" + ex["prediction"].split("-")[-1]
        )


def test_best_worst_use_distinct_prompts(tmp_path):
    """With num_samples>1 the 5 best slots must show 5 DIFFERENT prompts.

    Pre-fix, the top-5 by flat ROUGE-L were the 5 completions of the same
    (easiest) prompt — a useless table. Post-fix, each slot is the best
    completion of a distinct prompt.
    """
    data = _make_flat()
    dump_completion_examples(
        data["completions"],
        data["references"],
        data["rouge_scores"],
        prompts=data["prompts"],
        n_examples=10,
        output_dir=str(tmp_path),
    )
    examples = json.loads(
        (tmp_path / "completion_examples.json").read_text(encoding="utf-8")
    )["examples"]
    best = [ex for ex in examples if ex["group"] == "best"]
    worst = [ex for ex in examples if ex["group"] == "worst"]
    assert len(best) == 5 and len(worst) == 5
    assert len({ex["prompt"] for ex in best}) == 5, "best group repeats prompts"
    assert len({ex["prompt"] for ex in worst}) == 5, "worst group repeats prompts"


def test_single_sample_unchanged_semantics(tmp_path):
    """num_samples=1: one entry per prompt, best/worst by ROUGE-L as before."""
    data = _make_flat(n_prompts=4, num_samples=1)
    dump_completion_examples(
        data["completions"],
        data["references"],
        data["rouge_scores"],
        prompts=data["prompts"],
        n_examples=4,
        output_dir=str(tmp_path),
    )
    examples = json.loads(
        (tmp_path / "completion_examples.json").read_text(encoding="utf-8")
    )["examples"]
    best = [ex for ex in examples if ex["group"] == "best"]
    worst = [ex for ex in examples if ex["group"] == "worst"]
    assert {ex["index"] for ex in best} == {0, 1}
    assert {ex["index"] for ex in worst} == {2, 3}
    # Best block is ordered highest ROUGE-L first
    assert best[0]["rouge_l"] >= best[1]["rouge_l"]
    assert worst[0]["rouge_l"] <= worst[1]["rouge_l"]


def test_misaligned_per_prompt_refs_raise(tmp_path):
    """If a caller passes per-PROMPT refs with flat completions (the old
    buggy call site), the function must fail fast instead of silently
    truncating/rotating references against the wrong completions."""
    data = _make_flat(n_prompts=3, num_samples=2)
    per_prompt_refs = [f"GOLD {p}" for p in range(3)]
    with pytest.raises(ValueError, match="misaligned"):
        dump_completion_examples(
            data["completions"],
            per_prompt_refs,  # intentionally wrong length (per-prompt)
            data["rouge_scores"],
            prompts=data["prompts"],
            n_examples=4,
            output_dir=str(tmp_path),
        )
    # Same for a mismatched prompts list
    with pytest.raises(ValueError, match="prompts length"):
        dump_completion_examples(
            data["completions"],
            data["references"],
            data["rouge_scores"],
            prompts=["only one prompt"],
            n_examples=4,
            output_dir=str(tmp_path),
        )
    # Nothing should have been written
    assert not (tmp_path / "completion_examples.json").exists()
