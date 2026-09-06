# Interpretation Boundaries

This repository does not treat earlier campaigns as evidence for the current configuration tree. Current claims must come from canonical `qwen25-05b` artifacts evaluated under [EVALUATION.md](EVALUATION.md).

- Zero-shot and few-shot are prompting conditions, not methods.
- Vocabulary validity does not establish translation quality.
- PDA is a decoding ablation; hot is a rollout-temperature ablation.
- Markov and Viterbi analyses are diagnostics over frozen generations, not reward components or training results.
- Comparisons must preserve dataset split, sample budget, completion budget, seed policy, decoding constraint, and declared evaluation prompt mode.

Report BLEU, chrF, ROUGE-L, gloss-token F1, exact match, Pass@k criteria, and validity together. Keep run identity, resolved config, checkpoint source, and offline W&B metadata with each result.
