# Ablation Summary — conll-2003 / qwen25-15b

| Config | ROUGE-L | Valid ROUGE-L ⭐ | Pass@1 | Exact Match | Non-copy | Validity | BLEU (sent) | BLEU (corpus) | chrF2 (sent) | chrF2 (corpus) | Gloss F1 (sent) | Gloss F1 (micro) | Bigram LP | Δ ROUGE-L | Δ Valid ROUGE-L | Δ Pass@1 | Δ Exact Match | Δ Validity | Δ BLEU (sent) | Δ BLEU (corpus) | Δ chrF2 (sent) | Δ chrF2 (corpus) | Δ Gloss F1 (sent) | Δ Gloss F1 (micro) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conll-2003/qwen25-15b/baseline/zero-shot | 0.2224 | 0.2224 | 0.2295 | 0.2049 | 0.1150 | 1.0000 | 0.2124 | 0.0039 | 23.2225 | 5.9812 | 0.2071 | 0.1483 | -10.2151 | — | — | — | — | — | — | — | — | — | — | — |
| conll-2003/qwen25-15b/grpo/zero-shot | 0.4291 | 0.4291 | 0.4965 | 0.2797 | 0.2853 | 1.0000 | 0.3690 | 0.2585 | 41.6493 | 33.7953 | 0.3519 | 0.3152 | -15.8232 | +0.2067 | +0.2067 | +0.2670 | +0.0748 | +0.0000 | +0.1567 | +0.2546 | +18.4269 | +27.8141 | +0.1447 | +0.1670 |
