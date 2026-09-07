# Reduced-State Structured Gloss Benchmark

The original benchmark freezes the production Qwen backbone, extracts one final-layer hidden vector at the assistant generation boundary for each source prompt, and trains only a small position-conditioned emission head. Phase B3 additionally integrates the accepted objective into manual standalone SFT pilots; it is not integrated into GRPO.

## Standalone SFT integration (Phase B3)

The manual `sft-structured` and `sft-mass-structured` pilots add structured NLL only during training. Evaluation and best-checkpoint selection use pure completion LM loss. The head consumes the LM-head input at `completion_start - 1`, before the first gold completion token.

The deterministic SFT split is finalized before graph construction. Only complete, nonempty train glosses of at most `max_gloss_length` whitespace states affect graph states, support, and counts. The LM retains every row; excluded or tokenization-truncated rows skip only structured NLL. Ordered train/eval IDs, exclusion IDs/rate, settings, and graph digests are recorded in the manifest. This is the fixed comparison policy for pilot arms.

Each checkpoint and final directory contains the ordinary loadable adapter plus `structured_head.pt`, `auxiliary_sft_config.json`, and `graph_manifest.json`. Resume and best-model restore require an exact artifact/config/manifest match. Structured modes require one process and reject packing and padding-free collation.

## Arms

All arms use exactly the same post-holdout train/dev rows and cached source features:

1. `independent`: independent position cross-entropy.
2. `uniform-support`: length-conditioned sparse structured NLL with equal
   outgoing probability on the observed support.
3. `true-weights`: length-conditioned sparse structured NLL with observed
   train transition weights.
4. `shuffled-weights`: length-conditioned sparse structured NLL with deterministic count permutation over
   the observed edge support. Keeping support fixed preserves gold-path coverage
   while disrupting learned transition strengths.

The partition terminates at exactly the observed gloss length, so the objective
is `p(y | x, L)`, not a marginal over shorter lengths. The state space is the
512 most frequent train glosses plus `<OTHER>`. BOS/EOS exist only in the graph.
The graph uses additive `alpha=0.1`; transition scores use scale `0.25`.
Dev/test rows never define vocabulary or edges.

Before capping or splitting examples, extraction computes the source length
distribution, rejects empty glosses, and deterministically excludes sequences
longer than 64. It records excluded IDs, count/fraction, maximum, and percentiles
in the feature manifest. Training never silently truncates.

## Offline execution

Online setup must cache Qwen and ASLG-PC12 first. The GPU launcher exports offline variables before Python and never downloads:

```bash
sbatch cluster/structured_probe.sh
```

The first phase creates `experiments/analysis/qwen25-05b/structured/features.npz`. Extraction uses standard Transformers (not Unsloth), `requires_grad_(False)`, inference mode, and `last_hidden_state`; it explicitly disables all-layer hidden-state output. Reusing this artifact guarantees identical backbone compute across arms. The real CLI never synthesizes features.

The second phase writes each arm to:

```text
experiments/analysis/qwen25-05b/structured/<arm>/run_<UTC timestamp>/
  head.pt
  report.json
```

Reports include exact feature/sample and graph hashes, train/dev structured NLL
and independent CE, emission accuracy, fixed-length sequence/edit diagnostics,
path coverage, runtime, CUDA peak/total memory, device, and library versions.
Extraction runtime and peak memory live separately in the feature manifest.
The shuffled report quantifies changed weights and log-weight deltas and rejects
a null shuffle. Resource thresholds produce a nonautomatic pass/fail gate.

A single-seed run is explicitly a pilot. Promotion requires `true-weights` to
beat independent, uniform-support, and shuffled-weights across all configured
seeds; this launcher does not auto-promote. These results do not tune on or
report the final test set.

The sequence diagnostics currently use positionwise emission argmax at gold length; they are not claimed as structured Viterbi decoding.
