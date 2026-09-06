# Constrained Decoding: Trie Primary, PDA Ablation

The Trie vocabulary mask is the primary decoding path. The PDA path is a manual ablation that changes the decoding formalism while inheriting the `sft-grpo/zero-shot` training recipe.

## Primary path

`grammar.enabled: true` with `use_grammarllm_pda: false` selects the Trie. It restricts generation to the gloss vocabulary and is used for the main campaign. Vocabulary validity is a system metric; it does not establish semantic correctness or a formal language guarantee.

## PDA ablation

`experiments/configs/qwen25-05b/ablations/sft-grpo-zero-pda.yaml` enables the grammarllm PDA and token lookahead. It is manual and excluded from the 12-entry default campaign.

Before launch, run the full-vocabulary gate:

```bash
PDA=1 CONFIG=experiments/configs/qwen25-05b/ablations/sft-grpo-zero-pda.yaml sbatch cluster/preflight.sh
bash cluster/run_all.sh sft-grpo-zero-pda
```

`PDA=1` executes the focused `full_vocabulary` test over the real vocabulary. If pytest or artifacts are unavailable, preflight fails loudly; do not bypass the gate or substitute a toy-vocabulary result.

## Comparison contract

Compare PDA against `sft-grpo/zero-shot.yaml` with the same initialization, train data, prompt mode, sampling budget, seed policy, and evaluation modes. Report content metrics, validity, runtime, and failures. A validity increase is not by itself a quality improvement. Do not merge PDA results into the primary 2x2 table as though PDA were another method.

Canonical PDA artifacts live under:

```text
experiments/{checkpoints,logs}/qwen25-05b/sft-grpo/zero-shot/ablations/pda/run_<timestamp>/
experiments/results/qwen25-05b/sft-grpo/zero-shot/ablations/pda/
  eval-{zero-shot,few-shot}/run_<timestamp>/
```
