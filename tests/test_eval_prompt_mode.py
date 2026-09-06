"""Prompt-controlled evaluation protocol (preregistered).

Covers:
- ``--prompt-mode`` resolution: auto preserves config behavior;
  explicit zero-shot/retrieval override retrieval.enabled for prompt
  construction/fingerprint ONLY (caller config never mutated)
- baseline-cache fingerprint separation: modes and shared
  max_new_tokens budgets split caches; identical effective prompting
  (auto+disabled vs explicit zero-shot) shares the cache legitimately
- baseline-cache NON-reuse across modes/settings
- ``--max-new-tokens`` override forces the SAME budget on the SFT
  (generation=256) and GRPO (grpo=128) config defaults
- ``mean_abs_length_error`` present in the primary eval metric block
  over the same flat completion-reference pairs
- auto backward compatibility (defaults of every new parameter)
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.eval_t2g import (  # noqa: E402
    EVAL_TEMPERATURE,
    _compute_primary_metrics,
    _effective_retrieval_cfg,
    _load_cached_baseline,
    _prompt_context_fingerprint,
    _resolve_eval_max_new_tokens,
    _resolve_prompt_mode,
    build_arg_parser,
    evaluate_checkpoint,
)

_MIN_CFG = {
    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
    "dataset": {"dataset_name": "achrafothman/aslg_pc12", "seed": 42},
}

_RETRIEVAL_ON = {**_MIN_CFG, "retrieval": {"enabled": True, "top_k": 3}}
_RETRIEVAL_OFF = {**_MIN_CFG, "retrieval": {"enabled": False}}


# ---------------------------------------------------------------------------
# Prompt mode resolution
# ---------------------------------------------------------------------------


class TestResolvePromptMode:
    def test_auto_preserves_config_behavior(self):
        assert _resolve_prompt_mode("auto", _RETRIEVAL_ON["retrieval"]) == "retrieval"
        assert _resolve_prompt_mode("auto", _RETRIEVAL_OFF["retrieval"]) == "zero-shot"
        assert _resolve_prompt_mode("auto", None) == "zero-shot"
        assert _resolve_prompt_mode("auto", {}) == "zero-shot"

    def test_explicit_overrides_config(self):
        """Explicit modes override retrieval.enabled in BOTH directions."""
        assert (
            _resolve_prompt_mode("zero-shot", _RETRIEVAL_ON["retrieval"]) == "zero-shot"
        )
        assert (
            _resolve_prompt_mode("retrieval", _RETRIEVAL_OFF["retrieval"])
            == "retrieval"
        )

    def test_invalid_mode_is_loud(self):
        with pytest.raises(ValueError, match="prompt mode"):
            _resolve_prompt_mode("few-shot", {"enabled": True})


class TestEffectiveRetrievalCfg:
    def test_no_mutation_of_caller_config(self):
        original = {"enabled": True, "top_k": 3, "backend": "tfidf"}
        snapshot = dict(original)
        zs = _effective_retrieval_cfg(original, "zero-shot")
        rt = _effective_retrieval_cfg(original, "retrieval")
        assert zs["enabled"] is False
        assert rt["enabled"] is True
        assert original == snapshot, "caller config must never be mutated"

    def test_handles_missing_section(self):
        assert _effective_retrieval_cfg(None, "zero-shot") == {"enabled": False}
        assert _effective_retrieval_cfg(None, "retrieval") == {"enabled": True}


# ---------------------------------------------------------------------------
# Fingerprint: mode + max_new_tokens separation
# ---------------------------------------------------------------------------


def _fp(cfg, prompt_mode="auto", max_new_tokens=None, num_samples=5, temperature=0.7):
    return _prompt_context_fingerprint(
        cfg,
        num_samples,
        prompt_mode=prompt_mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


class TestFingerprintSeparation:
    def test_stable_for_same_context(self):
        assert _fp(_RETRIEVAL_ON) == _fp(_RETRIEVAL_ON)

    def test_modes_split_cache_when_retrieval_on(self):
        assert _fp(_RETRIEVAL_ON, "auto") != _fp(_RETRIEVAL_ON, "zero-shot")

    def test_modes_split_cache_when_retrieval_off(self):
        assert _fp(_RETRIEVAL_OFF, "auto") != _fp(_RETRIEVAL_OFF, "retrieval")

    def test_identical_effective_prompting_shares_fingerprint(self):
        """auto+disabled and explicit zero-shot produce IDENTICAL prompts —
        sharing the cached baseline is correct, so fingerprints match."""
        assert _fp(_RETRIEVAL_OFF, "auto") == _fp(_RETRIEVAL_OFF, "zero-shot")
        assert _fp(_RETRIEVAL_ON, "auto") == _fp(_RETRIEVAL_ON, "retrieval")

    def test_max_new_tokens_splits_fingerprint(self):
        assert _fp(_MIN_CFG, max_new_tokens=256) != _fp(_MIN_CFG, max_new_tokens=128)
        assert _fp(_MIN_CFG, max_new_tokens=256) == _fp(_MIN_CFG, max_new_tokens=256)

    def test_temperature_splits_fingerprint(self):
        assert _fp(_MIN_CFG, temperature=0.7) != _fp(_MIN_CFG, temperature=1.3)

    def test_hot_training_temperature_does_not_change_eval_protocol(self):
        normal = {**_MIN_CFG, "grpo": {"temperature": 0.7}}
        hot = {**_MIN_CFG, "grpo": {"temperature": 1.3}}
        assert EVAL_TEMPERATURE == 0.7
        assert _fp(normal) == _fp(hot)

    def test_no_config_mutation(self):
        import copy

        cfg = copy.deepcopy(_RETRIEVAL_ON)
        _fp(cfg, "zero-shot", max_new_tokens=300)
        assert cfg == _RETRIEVAL_ON

    def test_backward_compatible_positional_call(self):
        """Old two-argument call signature still works (auto behavior)."""
        assert _prompt_context_fingerprint(_MIN_CFG, 5) == _fp(
            _MIN_CFG, "auto", None, 5, 0.7
        )


# ---------------------------------------------------------------------------
# Baseline cache: non-reuse across modes/settings
# ---------------------------------------------------------------------------


def _write_baseline(
    results_dir: Path,
    *,
    fingerprint: str,
    num_completions: int = 5,
    n_eval: int = 100,
) -> None:
    baseline = {
        "metrics_version": 3,
        "num_completions_per_prompt": num_completions,
        "num_samples_evaluated": n_eval,
        "test_set_size": 8109,
        "prompt_context_fingerprint": fingerprint,
    }
    (results_dir / "eval_baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )


class TestBaselineCacheNonReuse:
    def test_reuse_within_same_mode(self, tmp_path):
        _write_baseline(tmp_path, fingerprint=_fp(_RETRIEVAL_ON))
        out = _load_cached_baseline(
            tmp_path,
            num_samples=5,
            max_samples=100,
            fingerprint=_fp(_RETRIEVAL_ON),
        )
        assert out is not None

    def test_no_reuse_base_prompt_vs_retrieval(self, tmp_path):
        """A zero-shot-override baseline must NOT satisfy a retrieval-mode
        request (and vice versa) — prompts differ, generations would too."""
        _write_baseline(tmp_path, fingerprint=_fp(_RETRIEVAL_ON, "zero-shot"))
        assert (
            _load_cached_baseline(
                tmp_path,
                num_samples=5,
                max_samples=100,
                fingerprint=_fp(_RETRIEVAL_ON, "retrieval"),
            )
            is None
        )

    def test_no_reuse_across_max_new_tokens(self, tmp_path):
        """A baseline decoded with 256 tokens must not be reused for a
        forced 128-token protocol run."""
        _write_baseline(tmp_path, fingerprint=_fp(_MIN_CFG, max_new_tokens=256))
        assert (
            _load_cached_baseline(
                tmp_path,
                num_samples=5,
                max_samples=100,
                fingerprint=_fp(_MIN_CFG, max_new_tokens=128),
            )
            is None
        )

    def test_reuse_across_equivalent_modes(self, tmp_path):
        """auto+retrieval-off cache IS reusable for an explicit zero-shot
        request: the effective prompting is identical."""
        _write_baseline(tmp_path, fingerprint=_fp(_RETRIEVAL_OFF, "auto"))
        out = _load_cached_baseline(
            tmp_path,
            num_samples=5,
            max_samples=100,
            fingerprint=_fp(_RETRIEVAL_OFF, "zero-shot"),
        )
        assert out is not None

    def test_recursive_canonical_model_tree_is_fingerprint_guarded(self, tmp_path):
        current = (
            tmp_path
            / "experiments/results/qwen25-05b/sft-grpo/few-shot/eval-few-shot/run_20260905_010203"
        )
        current.mkdir(parents=True)
        nested = (
            tmp_path
            / "experiments/results/qwen25-05b/grpo/few-shot/eval-few-shot/run_20260904_010203"
        )
        other = (
            tmp_path
            / "experiments/results/qwen25-05b/sft/zero-shot/eval-zero-shot/run_20260903_010203"
        )
        nested.mkdir(parents=True)
        other.mkdir(parents=True)
        _write_baseline(nested, fingerprint="wrong")
        _write_baseline(other, fingerprint="right")
        out = _load_cached_baseline(current, 5, 100, "right")
        assert out is not None and out[2] == other


# ---------------------------------------------------------------------------
# Shared max_new_tokens resolution
# ---------------------------------------------------------------------------


class TestResolveEvalMaxNewTokens:
    SFT_CFG = {"max_completion_length": 256}
    GRPO_CFG = {"max_completion_length": 128}

    def test_override_forces_same_value_across_sft_and_grpo(self):
        """The explicit override must win over BOTH per-section defaults —
        never silently keep SFT=256 vs GRPO=128."""
        assert _resolve_eval_max_new_tokens(300, self.SFT_CFG) == 300
        assert _resolve_eval_max_new_tokens(300, self.GRPO_CFG) == 300
        assert _resolve_eval_max_new_tokens(300, self.SFT_CFG) == (
            _resolve_eval_max_new_tokens(300, self.GRPO_CFG)
        )

    def test_no_override_uses_config_default(self):
        assert _resolve_eval_max_new_tokens(None, self.SFT_CFG) == 256
        assert _resolve_eval_max_new_tokens(None, self.GRPO_CFG) == 128

    def test_no_override_missing_key_defaults_to_256(self):
        assert _resolve_eval_max_new_tokens(None, {}) == 256

    def test_invalid_override_is_loud(self):
        for bad in (0, -5):
            with pytest.raises(ValueError, match="must be > 0"):
                _resolve_eval_max_new_tokens(bad, self.GRPO_CFG)


class TestCliArgs:
    def test_prompt_mode_and_max_new_tokens_parse(self):
        args = build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--prompt-mode",
                "zero-shot",
                "--max-new-tokens",
                "300",
            ]
        )
        assert args.prompt_mode == "zero-shot"
        assert args.max_new_tokens == 300

    def test_defaults_preserve_auto_behavior(self):
        args = build_arg_parser().parse_args(["--config", "cfg.yaml"])
        assert args.prompt_mode == "auto"
        assert args.max_new_tokens is None

    def test_invalid_prompt_mode_rejected(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(
                ["--config", "cfg.yaml", "--prompt-mode", "few-shot"]
            )


# ---------------------------------------------------------------------------
# evaluate_checkpoint backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_new_parameters_have_safe_defaults(self):
        params = inspect.signature(evaluate_checkpoint).parameters
        assert params["prompt_mode"].default == "auto"
        assert params["max_new_tokens"].default is None
        # positional compatibility: the 5 original parameters keep order
        names = list(params)
        assert names[:5] == [
            "config",
            "checkpoint_path",
            "max_samples",
            "num_samples",
            "best_of_n",
        ]


# ---------------------------------------------------------------------------
# Primary eval JSON: mean_abs_length_error
# ---------------------------------------------------------------------------


class TestPrimaryMetricsLengthError:
    def test_mean_abs_length_error_in_primary_block(self, reward_setup):
        """The primary eval JSON must carry mean_abs_length_error computed
        over the SAME flat completion-reference pairs as the other primary
        metrics."""
        _vocab, bigram, token_to_idx = reward_setup
        flat_completions = ["IX MAN WALK", "DOG RUN"]
        flat_references = ["IX MAN WALK HOUSE", "CAT RUN"]
        zero_weights = {"edit_validity_reward": 0.0}

        results, *_ = _compute_primary_metrics(
            flat_completions,
            flat_references,
            [[c] for c in flat_completions],
            flat_references,
            token_to_idx=token_to_idx,
            bigram=bigram,
            reward_weights=zero_weights,
        )
        # |3-4| = 1, |2-2| = 0 → mean 0.5
        assert results["mean_abs_length_error"] == pytest.approx(0.5)

    def test_length_error_uses_gloss_normalization(self, reward_setup):
        """Think-tag wrapping must not change the word count (extraction
        before counting, same pipeline as the training diagnostic)."""
        _vocab, bigram, token_to_idx = reward_setup
        flat_completions = ["<think>x</think>IX MAN WALK", "DOG RUN"]
        flat_references = ["IX MAN WALK", "CAT RUN"]
        zero_weights = {"edit_validity_reward": 0.0}

        results, *_ = _compute_primary_metrics(
            flat_completions,
            flat_references,
            [[c] for c in flat_completions],
            flat_references,
            token_to_idx=token_to_idx,
            bigram=bigram,
            reward_weights=zero_weights,
        )
        assert results["mean_abs_length_error"] == pytest.approx(0.0)

    def test_bigram_metrics_omitted_when_artifact_unavailable(self, reward_setup):
        """Core evaluation remains usable when the optional diagnostic is absent."""
        _vocab, _bigram, token_to_idx = reward_setup
        completions = ["IX MAN WALK"]
        references = ["IX MAN WALK"]

        results, *_ = _compute_primary_metrics(
            completions,
            references,
            [completions],
            references,
            token_to_idx=token_to_idx,
            bigram=None,
            reward_weights={"edit_validity_reward": 0.0},
        )

        assert "bigram_log_prob_mean" not in results
        assert "bigram_log_prob_std" not in results


def test_generation_seeds_are_order_invariant(monkeypatch):
    from src.training import eval_t2g

    def fake_generate(*args, seed=None, **kwargs):
        return [str(seed)]

    monkeypatch.setattr(eval_t2g, "_generate_batch", fake_generate)
    a = eval_t2g._generate_protocol_completions(
        None, None, "p", None, "id-a", 42, "sampling", 5, 10
    )
    eval_t2g._generate_protocol_completions(
        None, None, "p", None, "id-b", 42, "sampling", 5, 10
    )
    again = eval_t2g._generate_protocol_completions(
        None, None, "p", None, "id-a", 42, "sampling", 5, 10
    )
    assert a == again and len(a) == 5 and len(set(a)) == 5
