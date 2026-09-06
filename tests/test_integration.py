#!/usr/bin/env python3
"""Integration tests — end-to-end coherence check.

Validates the full chain:
    data → grammar → rewards → metrics → callbacks
all produce consistent values and types.

Tests requiring the dataset are skipped if offline.
"""

from __future__ import annotations

from typing import Any


def test_sft_grpo_cells_reuse_preceding_standalone_sft(tmp_path, monkeypatch):
    """Both canonical SFT-GRPO cells resolve the standalone campaign adapter."""
    import json
    from pathlib import Path

    from src.training import sft_train
    from src.training.grpo_t2g_train import resolve_reusable_sft_adapter
    from src.utils.config import load_config

    project = Path(__file__).resolve().parent.parent
    standalone_cfg = load_config(
        project / "experiments/configs/qwen25-05b/sft/zero-shot.yaml"
    )
    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    source = model_root / "sft" / "zero-shot" / "run_20260906_120000" / "final"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = b"canonical-standalone-adapter"
    (source / "adapter_model.safetensors").write_bytes(weights)
    (source / "sft_fingerprint.json").write_text(
        json.dumps({"fingerprint": sft_train.compute_sft_fingerprint(standalone_cfg)}),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("training/local fallback must not be invoked")

    monkeypatch.setattr(sft_train, "run_sft", forbidden)

    for prompt_mode in ("zero-shot", "few-shot"):
        cfg = load_config(
            project / f"experiments/configs/qwen25-05b/sft-grpo/{prompt_mode}.yaml"
        )
        run_dir = model_root / "sft-grpo" / prompt_mode / "run_20260906_130000"
        sft_cfg = {
            **cfg,
            "training": {
                **cfg["training"],
                **cfg["sft_pretrain"]["training"],
                "output_dir": str(run_dir / "sft_pretrain"),
                "log_dir": str(tmp_path / "logs" / prompt_mode),
                "trainer": "sft",
            },
        }

        resolved = resolve_reusable_sft_adapter(sft_cfg, run_dir, model_root)
        assert resolved == (source, "sft/zero-shot")
        assert (resolved[0] / "adapter_model.safetensors").read_bytes() == weights


def test_data_to_grammar_chain(dataset, tokenizer):
    """Data → Grammar: vocab from real data builds a valid mask."""
    from src.datasets.aslg_dataset import extract_gloss_vocabulary
    from src.grammar.gloss_grammar import GlossVocabularyMask

    vocab = extract_gloss_vocabulary(dataset, split="train")
    mask = GlossVocabularyMask(vocab, tokenizer)
    assert len(mask.token_ids) > 0, f"Token IDs for {len(vocab)} glosses"
    assert mask.is_allowed(mask.eos_token_id), "EOS allowed"
    allowed = mask.get_allowed_token_ids()
    assert len(allowed) > 0, "Allowed IDs non-empty"


def test_grammar_vocab_to_edit_validity_chain(dataset):
    """Grammar vocabulary → production edit-validity reward."""
    from src.datasets.aslg_dataset import extract_gloss_vocabulary
    from src.rewards.t2g_rewards import (
        build_t2g_reward_functions,
        initialize_rewards,
    )

    vocab = extract_gloss_vocabulary(dataset, split="train")
    initialize_rewards(vocab)

    funcs, weights = build_t2g_reward_functions()
    assert len(funcs) == 1
    assert funcs[0].__name__ == "edit_validity_reward"
    assert weights == [1.0]

    gold = str(dataset["train"][0]["gloss"])
    scores = funcs[0]([gold, f"{gold} __NOT_IN_VOCAB__"], gold_gloss=[gold, gold])
    assert scores == [1.0, -1.0]


def test_rewards_to_metrics_chain(dataset):
    """Generated glosses and references produce evaluation metrics."""
    from src.utils.metrics import (
        compute_detailed_metrics,
        compute_pass_at_k,
        rouge_l_score,
    )

    completions = ["IX MAN WALK", "DOG CAT BIRD", "NOT CAN WANT"]
    references = ["IX MAN WALK", "IX MAN GO", "NOT CAN COME"]

    pass1 = compute_pass_at_k(
        [[c] for c in completions], references, k_values=(1,), threshold=0.3
    )["pass@1"]
    assert 0.0 <= pass1 <= 1.0, f"Pass@1 in [0,1], got {pass1:.4f}"

    rl = rouge_l_score(completions[0], references[0])
    assert abs(rl - 1.0) < 0.01, f"ROUGE-L perfect match = 1.0, got {rl:.4f}"

    detailed = compute_detailed_metrics(completions, references)
    assert "overall_pass_rate" in detailed, "Detailed metrics has pass_rate"


def test_callbacks_interface(reward_setup):
    """SFT callback remains constructible after reward initialization."""
    from src.training.callbacks import SFTSampleCallback

    sft_cb = SFTSampleCallback(
        tokenizer=None, model=None, dataset=None, every_n_steps=25
    )
    assert sft_cb is not None, "SFTSampleCallback created"
    assert sft_cb._every_n_steps == 25


def test_module_imports():
    """All key modules import without errors."""
    modules = [
        (
            "src.datasets.aslg_dataset",
            ["download_aslg_dataset", "extract_gloss_vocabulary", "build_t2g_dataset"],
        ),
        (
            "src.datasets.transition_matrix",
            [
                "compute_bigram_transitions",
                "load_transition_matrix",
                "transition_score",
                "sequence_score_bigram",
            ],
        ),
        ("src.grammar.gloss_grammar", ["GlossVocabularyMask"]),
        ("src.grammar.grammar_logits_processor", ["GlossVocabularyLogitsProcessor"]),
        (
            "src.rewards.t2g_rewards",
            [
                "build_t2g_reward_functions",
                "initialize_rewards",
                "edit_validity_reward",
            ],
        ),
        (
            "src.training.callbacks",
            ["CompletionSampleLogger", "CompletionSampleCallback", "SFTSampleCallback"],
        ),
        (
            "src.utils.metrics",
            [
                "compute_pass_at_k",
                "compute_reward_breakdown",
                "compute_evaluation_report",
                "bootstrap_confidence_interval",
                "bleu_sentence",
                "bleu_corpus",
            ],
        ),
        ("src.utils.visualization", ["plot_training_curves", "plot_reward_breakdown"]),
        ("src.utils.chain_monitor", []),
        ("src.utils.live_training_table", []),
        ("src.utils.show_training_log", []),
    ]
    for mod_name, attrs in modules:
        mod = __import__(mod_name, fromlist=attrs)
        for attr in attrs:
            assert hasattr(mod, attr), f"{mod_name}.{attr} exists"


# ---------------------------------------------------------------------------
# Few-shot retrieval — setup helper, GRPO dataset building (offline, no model)
# ---------------------------------------------------------------------------


def _mini_corpus(n: int = 10) -> "Any":
    """A tiny synthetic train split (text + gloss), no network.

    Sentences are topically distinct so that, under the default
    ``max_self_similarity=0.98``, any query still has valid (non-drop) few-shot
    candidates in the corpus.
    """
    from datasets import Dataset

    texts = [
        "The cat sleeps on the sofa",
        "A dog runs in the park",
        "My cat loves fresh fish",
        "The child reads a big book",
        "She drinks coffee every morning",
        "The bus arrives at the station",
        "Birds fly over the city",
        "The teacher explains the lesson",
        "We visit grandma on sunday",
        "The river flows through the valley",
    ]
    glosses = [
        "CAT SLEEP SOFA",
        "DOG RUN PARK",
        "MY CAT LOVE FISH",
        "CHILD READ BIG BOOK",
        "SHE DRINK COFFEE MORNING",
        "BUS ARRIVE STATION",
        "BIRD FLY CITY",
        "TEACHER EXPLAIN LESSON",
        "WE VISIT GRANDMA SUNDAY",
        "RIVER FLOW VALLEY",
    ]
    rows = [{"text": t, "gloss": g} for t, g in zip(texts, glosses)]
    return Dataset.from_list(rows[:n])


def _retrieval_cfg(enabled: bool = True, **overrides) -> dict:
    """A ``retrieval`` config section with sensible test defaults."""
    cfg = {
        "enabled": enabled,
        "backend": "tfidf",
        "model_name": None,
        "top_k": 3,
        "max_self_similarity": 0.98,
        "cache_path": None,
    }
    cfg.update(overrides)
    return cfg


def test_build_train_retriever_disabled_returns_none():
    """``retrieval.enabled=false`` ⇒ no retriever (zero-shot prompts)."""
    from src.training.retrieval_setup import build_train_retriever

    retriever = build_train_retriever({"train": None}, _retrieval_cfg(False))
    assert retriever is None


def test_build_train_retriever_indexes_train_and_excludes_query(tmp_path):
    """enabled=true builds a retriever over the train split; the query
    itself never appears among the retrieved examples."""
    from src.retrieval import normalize_text
    from src.training.retrieval_setup import build_train_retriever

    corpus = _mini_corpus(10)
    retriever = build_train_retriever(
        {"train": corpus},
        _retrieval_cfg(cache_path=str(tmp_path / "retriever")),
        seed=42,
    )
    assert retriever is not None
    assert retriever.backend == "tfidf"

    # An in-corpus sentence used as the query: its own row must be excluded.
    query = "The cat sleeps on the sofa"
    results = retriever.retrieve(
        query, k=3, exclude={normalize_text(query)}, max_self_similarity=0.98
    )
    assert results
    assert all(r.text != query for r in results), "query never among its examples"
    assert all(normalize_text(r.text) != normalize_text(query) for r in results)
    # Examples come from the train corpus.
    train_texts = {str(s["text"]) for s in corpus}
    assert all(r.text in train_texts for r in results)


def test_build_train_retriever_cache_roundtrip(tmp_path, monkeypatch):
    """A second call with the same cache dir loads instead of rebuilding."""
    from src.training import retrieval_setup
    from src.training.retrieval_setup import build_train_retriever

    corpus = _mini_corpus(10)
    cache_path = str(tmp_path / "retriever")
    cfg = _retrieval_cfg(cache_path=cache_path)

    build_calls = {"n": 0}
    orig_build = retrieval_setup.ExampleRetriever.build

    def counting_build(*args, **kwargs):
        build_calls["n"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(retrieval_setup.ExampleRetriever, "build", counting_build)

    r1 = build_train_retriever({"train": corpus}, cfg, seed=42)
    assert r1 is not None
    assert build_calls["n"] == 1, "first call builds the index"

    r2 = build_train_retriever({"train": corpus}, cfg, seed=42)
    assert r2 is not None
    assert build_calls["n"] == 1, "cache hit must not rebuild the index"

    query = "A dog runs in the park"
    assert [x.text for x in r1.retrieve(query, k=2)] == [
        x.text for x in r2.retrieve(query, k=2)
    ]


def test_grpo_dataset_building_with_retrieval():
    """``_prepare_t2g_dataset`` injects few-shot examples into prompts and
    keeps the ``gold_gloss`` column (reward-kwarg flow intact).

    End-to-end prompt building with a real (offline) tfidf retriever.
    """
    from datasets import DatasetDict
    from src.retrieval import ExampleRetriever
    from src.training.grpo_t2g_train import _prepare_t2g_dataset

    corpus = _mini_corpus(10)
    retriever = ExampleRetriever.build(
        [str(s["text"]) for s in corpus],
        [str(s["gloss"]) for s in corpus],
        backend="tfidf",
        seed=42,
    )

    class _FakeTokenizer:
        """No chat template → manual ChatML fallback in build_t2g_prompt."""

    config = {"dataset": {"split": "train"}}
    ds = _prepare_t2g_dataset(
        config,
        _FakeTokenizer(),
        vocab=[],
        dataset=DatasetDict({"train": corpus}),
        retriever=retriever,
        # max_self_similarity=1.0: only the exact excluded query is blocked,
        # guaranteeing every prompt gets a full 2-example few-shot block.
        retrieval_cfg={"top_k": 2, "max_self_similarity": 1.0},
    )

    assert "gold_gloss" in ds.column_names, "reward-kwarg column preserved"
    assert "prompt" in ds.column_names
    assert len(ds) == 10
    for i in range(len(ds)):
        prompt = ds[i]["prompt"]
        text = ds[i]["text"]
        assert "Examples:" in prompt, f"few-shot block missing: {prompt}"
        assert "ASL gloss:" in prompt
        assert f"Now translate:\nEnglish: {text}" in prompt, f"query missing: {prompt}"
        # Anti-leakage at prompt level: the query's own sentence appears
        # exactly once (as the query), never as an example.
        assert prompt.count(f"English: {text}") == 1, prompt
