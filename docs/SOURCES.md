# Accepted Sources

> **REFERENCE ONLY — NOT RUNTIME GUIDANCE.** Current operation is defined by `EXPERIMENT_DESIGN.md`, `TRAINING.md`, `EVALUATION.md`, `MARKOV_DIAGNOSTICS.md`, and `PDA.md`.

## Data and Text-to-Gloss

- ASLG-PC12 dataset: <https://huggingface.co/datasets/achrafothman/aslg_pc12>
- Othman and Jemni, statistical sign-language machine translation: <https://arxiv.org/abs/1112.0168>
- Abdullah et al., Bangla Text-to-Gloss benchmark: <https://arxiv.org/abs/2504.02293>
- Walsh, Saunders, and Bowden, *Select and Reorder*: <https://arxiv.org/abs/2404.11532>
- Guo et al., pseudo-gloss generation with few-shot LLM prompting: <https://arxiv.org/abs/2505.15438>

Published ASLG-PC12 gloss-to-English scores are not comparable to this project's English-to-gloss protocol.

## GRPO and Translation

- Shao et al., DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- Liu et al., Dr.GRPO and length bias: <https://arxiv.org/abs/2503.20783>
- Rao et al., RVLF, BLEU/ROUGE rewards for sign-language translation: <https://arxiv.org/abs/2512.07273>
- Mosquera et al., GRPO translation with Qwen2.5-0.5B: <https://arxiv.org/abs/2508.19481>
- FSA-GRPO, few-shot-aware GRPO: <https://arxiv.org/abs/2606.02615>
- RA-RFT, retrieval-augmented reinforcement fine-tuning: <https://arxiv.org/abs/2606.13680>
- TRL GRPO documentation: <https://huggingface.co/docs/trl/main/en/grpo_trainer>
- Unsloth GRPO guide: <https://unsloth.ai/blog/grpo>

## Constrained Decoding and Diagnostics

- MiniOneRec, RL with constrained discrete generation: <https://arxiv.org/abs/2510.24431>
- ARCS, grammar-constrained GRPO: <https://arxiv.org/abs/2603.29068>
- Abstract-CoT, SFT plus RL under constrained decoding: <https://arxiv.org/abs/2604.22709>
- ViterbiPlanNet DVL: <https://arxiv.org/abs/2603.04265>. This is attribution for possible future differentiable work; the current Markov probes are not DVL.
- RECIPE, verifier-scaled reward context: <https://arxiv.org/abs/2605.19976>

## Evaluation and Infrastructure

- Chen et al., HumanEval and Pass@k provenance: <https://arxiv.org/abs/2107.03374>
- Hugging Face offline mode: <https://huggingface.co/docs/transformers/installation#offline-mode>
- Weights & Biases offline mode: <https://docs.wandb.ai/guides/runs/run-modes/#offline>
- gcluster documentation: <https://gcluster.dmi.unict.it/docs/>

Pass@k with a ROUGE-L threshold remains a project diagnostic, not a standard Text-to-Gloss metric.
