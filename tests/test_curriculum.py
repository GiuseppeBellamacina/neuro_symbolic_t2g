#!/usr/bin/env python3
"""Test curriculum learning components.

Validates:
  1. Difficulty labels are assigned correctly by gloss length
  2. CurriculumSchedule computes correct stages and distributions
  3. CurriculumFilteredDataset filters and transitions between stages
  4. Dataset length stays constant across stage transitions
"""

from __future__ import annotations


def test_difficulty_labels(dataset):
    """Difficulty labels correspond to gloss token count."""
    from src.datasets.aslg_dataset import build_t2g_dataset

    t2g = build_t2g_dataset(dataset, split="train", max_samples=200)

    for row in t2g:
        gloss_tokens = row["completion"].split()
        n = len(gloss_tokens)
        diff = row["difficulty"]

        if n <= 5:
            assert diff == "simple", f"{n} tokens should be simple, got {diff}"
        elif n <= 15:
            assert diff == "medium", f"{n} tokens should be medium, got {diff}"
        else:
            assert diff == "hard", f"{n} tokens should be hard, got {diff}"


def test_difficulty_distribution(dataset):
    """Full dataset distribution is roughly ~9/68/22% (simple/medium/hard)."""
    from src.datasets.aslg_dataset import build_t2g_dataset

    t2g = build_t2g_dataset(dataset, split="train", max_samples=2000)
    counts = {"simple": 0, "medium": 0, "hard": 0}
    for row in t2g:
        counts[row["difficulty"]] += 1

    total = sum(counts.values())
    for label, expected_min, expected_max in [
        ("simple", 0.06, 0.15),
        ("medium", 0.55, 0.75),
        ("hard", 0.15, 0.30),
    ]:
        pct = counts[label] / total
        assert (
            expected_min <= pct <= expected_max
        ), f"{label}: expected {expected_min:.0%}-{expected_max:.0%}, got {pct:.1%}"


def test_curriculum_schedule_stages():
    """CurriculumSchedule returns correct stages at step boundaries."""
    from src.training.grpo_t2g_train import CurriculumSchedule

    schedule = CurriculumSchedule(max_steps=300)
    stage_size = 100

    # Check stage boundaries
    assert schedule.get_stage(0) == 0
    assert schedule.get_stage(stage_size - 1) == 0
    assert schedule.get_stage(stage_size) == 1
    assert schedule.get_stage(2 * stage_size - 1) == 1
    assert schedule.get_stage(2 * stage_size) == 2
    assert schedule.get_stage(500) == 2  # beyond max_steps

    assert schedule.stage_size == stage_size


def test_curriculum_schedule_distributions():
    """Each stage has the correct difficulty distribution."""
    from src.training.grpo_t2g_train import CurriculumSchedule

    schedule = CurriculumSchedule(max_steps=300)
    stage_size = schedule.stage_size

    # Stage 1 (step 0): mostly medium
    d0 = schedule.get_distribution(0)
    assert d0["simple"] == 0.10
    assert d0["medium"] == 0.65
    assert d0["hard"] == 0.25

    # Stage 2 (step 150): less simple, more hard
    d1 = schedule.get_distribution(stage_size)
    assert d1["simple"] == 0.05
    assert d1["medium"] == 0.40
    assert d1["hard"] == 0.55

    # Stage 3 (step 250): mostly hard
    d2 = schedule.get_distribution(2 * stage_size)
    assert d2["simple"] == 0.03
    assert d2["medium"] == 0.30
    assert d2["hard"] == 0.67

    # All distributions sum to 1.0
    for d in (d0, d1, d2):
        assert abs(sum(d.values()) - 1.0) < 0.01


def test_curriculum_filtered_dataset(dataset):
    """CurriculumFilteredDataset filters correctly and transitions between stages."""
    from src.datasets.aslg_dataset import build_t2g_dataset
    from src.training.grpo_t2g_train import (
        CurriculumFilteredDataset,
        CurriculumSchedule,
    )

    t2g = build_t2g_dataset(dataset, split="train", max_samples=500)

    # Sanity: dataset has all three difficulty levels
    diffs = {row["difficulty"] for row in t2g}
    assert (
        "simple" in diffs and "medium" in diffs and "hard" in diffs
    ), f"Dataset missing difficulty levels: {diffs}"

    schedule = CurriculumSchedule(max_steps=300)

    # --- Stage 1 ---
    stage1 = CurriculumFilteredDataset(t2g, schedule, stage=0)
    assert len(stage1) == len(t2g), "Dataset length stays constant"

    s1_diffs = [stage1[i]["difficulty"] for i in range(min(100, len(stage1)))]
    s1_simple_pct = sum(1 for d in s1_diffs if d == "simple") / len(s1_diffs)
    # Stage 1 should have some simple examples (target 10%, actual ~9%)
    assert s1_simple_pct > 0.05, f"Stage 1 has too few simple: {s1_simple_pct:.2%}"
    # Stage 1 should be mostly medium
    s1_medium_pct = sum(1 for d in s1_diffs if d == "medium") / len(s1_diffs)
    assert s1_medium_pct > 0.40, f"Stage 1 has too few medium: {s1_medium_pct:.2%}"

    # --- Transition to Stage 2 ---
    stage1.update_stage(1)
    s2_diffs = [stage1[i]["difficulty"] for i in range(min(100, len(stage1)))]
    s2_hard_pct = sum(1 for d in s2_diffs if d == "hard") / len(s2_diffs)
    # Stage 2 should have more hard examples
    assert s2_hard_pct > 0.20, f"Stage 2 has too few hard: {s2_hard_pct:.2%}"

    # --- Transition to Stage 3 ---
    stage1.update_stage(2)
    s3_diffs = [stage1[i]["difficulty"] for i in range(min(100, len(stage1)))]
    s3_hard_pct = sum(1 for d in s3_diffs if d == "hard") / len(s3_diffs)
    # Stage 3 should be dominated by hard
    assert s3_hard_pct > 0.30, f"Stage 3 has too few hard: {s3_hard_pct:.2%}"

    # Length stays constant across all transitions
    assert len(stage1) == len(t2g)


def test_curriculum_dataset_len_constant(dataset):
    """Dataset __len__ never changes, even with empty difficulty buckets."""
    from src.datasets.aslg_dataset import build_t2g_dataset
    from src.training.grpo_t2g_train import (
        CurriculumFilteredDataset,
        CurriculumSchedule,
    )

    # Use a tiny dataset to test edge case
    t2g = build_t2g_dataset(dataset, split="train", max_samples=30)
    schedule = CurriculumSchedule(max_steps=300)
    original_len = len(t2g)

    cds = CurriculumFilteredDataset(t2g, schedule, stage=0)
    assert len(cds) == original_len

    cds.update_stage(1)
    assert len(cds) == original_len

    cds.update_stage(2)
    assert len(cds) == original_len

    # Items are indexable
    for i in range(min(5, original_len)):
        item = cds[i]
        assert "prompt" in item
        assert "completion" in item
        assert "difficulty" in item
