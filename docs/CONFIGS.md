# Config T2G — Indice

> Guida completa: `docs/CONFIGS_GUIDE.md` (matrice, extends, pesi, tag dati).

| Config                       | Pipeline  | Scopo                                |
| ---------------------------- | --------- | ------------------------------------ |
| `base.yaml`                  | —         | Parti comuni (parent, non runnable)  |
| `sft-grpo.yaml`              | SFT+GRPO  | Pipeline principale (controllo)       |
| `sft-only.yaml`              | SFT       | Decomposizione: SFT da solo           |
| `grpo-only.yaml`             | GRPO      | Decomposizione: GRPO senza SFT        |
| `sft-grpo-structure.yaml`    | SFT+GRPO  | Ablation: + structural_dense 0.10    |
| `sft-grpo-viterbi.yaml`      | SFT+GRPO  | Ablation: + viterbi_distance 0.10     |
| `sft-grpo-soft-viterbi.yaml` | SFT+GRPO  | Ablation: + soft_viterbi 0.10        |
| `sft-grpo-all-rewards.yaml`  | SFT+GRPO  | Ablation: + tutti e 3 i moduli 0.20  |
| `sft-grpo-no-grammar.yaml`   | SFT+GRPO  | Constrained decoding OFF             |
| `zero-shot.yaml`             | Eval-only | Base model, grammar OFF (lower bound) |
| `zero-shot-grammar.yaml`     | Eval-only | Base model + grammar (constrained)   |

## Convenzioni

- **Naming pipeline-first**: `sft-grpo-*` = SFT pre-training + GRPO;
  `sft-only` / `grpo-only` = decomposizione; `base` = parent.
- **extends**: ogni cella eredita da `sft-grpo.yaml` (o `base.yaml`)
  e sovrascrive SOLO ciò che testa → differenza a fattore unico.
- **Validator**: `python -m tests.validate_configs` — controlla tipi,
  somma reward weights = 1.0 (±1e-9), chiavi morte, per OGNI yaml in
  `experiments/configs/**/*.yaml`.
- **Reward weights**: diluizione uniforme del core (×0.90 singolo /
  ×0.80 all-three) — proporzioni relative invariate.
- La baseline zero-shot NON è un config: è la baseline `--compare`
  (cachata e riusata tra run/tag via fingerprint del contesto prompt).
