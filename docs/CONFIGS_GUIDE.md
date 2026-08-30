# Config T2G — Guida completa

> Schema di naming **pipeline-first**: il nome dice cosa viene addestrato.
> Tutti i config estendono `base.yaml` (o un figlio di esso) via `extends`
> con deep-merge — un config differisce dal genitore SOLO per ciò che testa
> (differenza a fattore unico).

## Matrice dei config (8 celle + base)

| Config                   | Pipeline  | Differenza da `sft-grpo`            | Scopo                                   |
| ------------------------ | -------- | ------------------------------------ | --------------------------------------- |
| `base.yaml`              | —        | —                                    | Parti comuni (modello, LoRA, reward, grammar, grpo) |
| `sft-grpo.yaml`          | SFT+GRPO | —                                    | **Pipeline principale** (controllo)     |
| `sft-only.yaml`          | SFT      | niente GRPO                          | Decomposizione: SFT da solo             |
| `grpo-only.yaml`         | GRPO     | niente SFT pre-training              | Decomposizione: GRPO senza SFT          |
| `sft-grpo-structure.yaml`| SFT+GRPO | core ×0.90 + structure 0.10          | Ablation moduli: structural_dense       |
| `sft-grpo-viterbi.yaml`  | SFT+GRPO | core ×0.90 + viterbi 0.10            | Ablation moduli: viterbi_distance       |
| `sft-grpo-soft-viterbi.yaml` | SFT+GRPO | core ×0.90 + soft_viterbi 0.10  | Ablation moduli: soft_viterbi (DVL)    |
| `sft-grpo-all-rewards.yaml` | SFT+GRPO | core ×0.80 + 3 moduli 0.20       | Ablation moduli: tutti e tre            |
| `sft-grpo-no-grammar.yaml` | SFT+GRPO | grammar.enabled: false             | Constrained decoding OFF                |
| `zero-shot.yaml`           | Eval-only | niente training, grammar OFF        | Base model da solo (lower bound)        |
| `zero-shot-grammar.yaml`   | Eval-only | niente training, grammar ON         | Base + constrained decoding (eval)      |

La **baseline zero-shot** (lower bound) non è un config: è la baseline
del `--compare` di eval.sh — calcolata una volta sola e riusata tra run
via fingerprint del contesto prompt (vedi `eval_t2g.py::_load_cached_baseline`).

## La catena di extends

```
base.yaml
  └── sft-grpo.yaml          (pipeline principale: SFT reuse + GRPO 2000 step
        │                     + curriculum + retrieval + eval set completo)
        ├── grpo-only.yaml           (sft_pretrain.enabled: false)
        ├── sft-grpo-structure.yaml  (reward: core ×0.90 + structure)
        ├── sft-grpo-viterbi.yaml    (reward: core ×0.90 + viterbi)
        ├── sft-grpo-soft-viterbi.yaml (reward: core ×0.90 + soft_viterbi)
        ├── sft-grpo-all-rewards.yaml  (reward: core ×0.80 + 3 moduli)
        └── sft-grpo-no-grammar.yaml   (grammar.enabled: false)
zero-shot.yaml               (extends base, eval-only, grammar OFF)
zero-shot-grammar.yaml        (extends base, eval-only, grammar ON)
sft-only.yaml                (extends base, trainer: sft)
```

Tutte le celle ablation ereditano iperparametri, curriculum, retrieval e
sezione `sft_pretrain` IDENTICI a `sft-grpo` → differenza a fattore unico
e riuso dell'adapter SFT via fingerprint (cross-tag).

## Design dei pesi reward (ablation moduli)

**Diluizione uniforme**: i 7 core di `sft-grpo` (translation .20, bleu .20,
gold_structure .20, order .10, verifier .10, format .10, repetition .10)
sono scalati ×0.90 mantenendo le PROPORZIONI RELATIVE identiche
(.20→.18, .10→.09) e il 10% di massa va al modulo in test. Somma = 1.0
(verificata dal validator). L'unica differenza sostanziale vs controllo è
"10% della massa reward al modulo in test".

- Singolo modulo: `core ×0.90 + modulo 0.10`
- All-three: `core ×0.80 + {structure .05, viterbi .05, soft_viterbi .10}`

I moduli sperimentali sono **v2 gold-anchored** (ricalibrati: gold → +1,
std di gruppo 0.62 vs ~0.001 pre-fix — vedi `src/rewards/t2g_rewards.py`).

## Uso

```bash
# singola cella (train + eval in catena):
CONFIG=experiments/configs/t2g/<nome>.yaml sbatch cluster/train.sh
# oppure (coda gestita):
bash cluster/run_all.sh <nome>            # es. sft-grpo-structure
bash cluster/run_all.sh --ablation        # tutte le 7 celle trainabili

# cella SFT-only (NESSUN training necessario — adapter già pronto):
CHECKPOINT=experiments/checkpoints/qwen25-05b-sft-grpo/run_*/sft_pretrain/final \
  CONFIG=experiments/configs/t2g/sft-grpo.yaml sbatch cluster/eval.sh
```

## Tag dati (output_dir)

Ogni cella scrive sotto un tag dedicato `experiments/checkpoints/<tag>/run_*`:

| Config                   | Tag dati                        |
| ------------------------ | ------------------------------- |
| `sft-grpo.yaml`          | `qwen25-05b-sft-grpo`            |
| `sft-only.yaml`          | `qwen25-05b-sft-only`           |
| `grpo-only.yaml`         | `qwen25-05b-grpo-only`          |
| `sft-grpo-structure.yaml`| `qwen25-05b-sft-grpo-structure` |
| `sft-grpo-viterbi.yaml`  | `qwen25-05b-sft-grpo-viterbi`   |
| `sft-grpo-soft-viterbi.yaml` | `qwen25-05b-sft-grpo-soft-viterbi` |
| `sft-grpo-all-rewards.yaml` | `qwen25-05b-sft-grpo-all-rewards` |
| `sft-grpo-no-grammar.yaml` | `qwen25-05b-sft-grpo-no-grammar` |

## Config eliminati (storico)

La dir `ablation/` (zero_shot, zero_shot_grammar, grpo_pda,
grpo_pda_lookahead, grpo_soft_viterbi, grpo_verifier_scaled, grpo_no_sft)
e `grpo_qwen05.yaml` sono stati eliminati: iperparametri non allineati
(LoRA r=16, 1500 step, lr=5e-6, reward v1 saturate) e non confrontabili
con la pipeline. Il confronto Trie-vs-PDA è rigenerabile in 5 righe
(`extends: sft-grpo.yaml` + `use_grammarllm_pda: true`) quando servirà.
