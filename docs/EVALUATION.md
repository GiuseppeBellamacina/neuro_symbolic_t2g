# Protocollo di Valutazione — neuro_symbolic_t2g

Versione: 1.1 (2026-09-02). Questo documento definisce il protocollo con cui
vengono prodotti e confrontati i numeri del progetto. **Dichiarare e mantenere
questo protocollo è prerequisito per ogni claim sul target BLEU 0.80** — i numeri
sono comparabili solo dentro lo stesso protocollo. La gerarchia delle metriche
(§2) segue la letteratura T2G: vedi §2b per le fonti.

## 1. Split del dataset

- **Dataset**: ASLG-PC12 (`achrafothman/aslg_pc12`), 87.710 coppie raw.
- **Deduplicazione**: PRIMA dello split, per chiave normalizzata del testo
  (lowercase + collapse whitespace + strip), prima occorrenza conservata.
  → 81.088 coppie uniche (−6.622 duplicati; le righe rimosse erano per lo più
  frasi brevi — la quota "simple" passa da ~9% a ~4.9%).
- **Split**: 90/10 train/test con `train_test_split(test_size=0.1, seed=42)`
  (HF datasets). Il test set non è MAI visto in training; nessun near-duplicato
  può attraversare gli split (dedup pre-split).
- **Nessuno split di validazione separato**: l'eval holdout di SFT (2%) è
  ricavato dal train (vedi `src/training/sft_train.py`).

## 2. Metriche — gerarchia per rilevanza

Tutte le metriche primarie sono calcolate su **tutte le completions** generate
per ogni prompt (no selezione oracolo). Implementazioni: `src/utils/metrics.py`
(sacrebleu per BLEU/chrF). L'ordine della tabella è la **gerarchia di
rilevanza** usata nel log dell'eval, nel metrics_dashboard e nelle tabelle
della tesi.

| # | Metrica | Definizione | Scala | Ruolo |
|---|---|---|---|---|
| 1 | **BLEU-4 (corpus)** | sacreBLEU corpus, refs flat allineate (v2 `metrics_version`); sentence mean riportato accanto | [0,1] | **Headline** — lo standard della letteratura T2G (confrontabile coi paper) |
| 2 | **chrF2 (corpus)** | sacrebleu CHRF2 (char F-score, β=2) | [0,100] | **Headline secondaria** — indipendente dalla tokenizzazione, àncora contro inflazione BLEU |
| 3 | **ROUGE-L** | F1 LCS (rouge_score, stemmer off), sentence mean | [0,1] | Secondaria — lineage SLT + la reward di training |
| 4 | **Gloss F1 (micro)** | F1 token-level case-insensitive | [0,1] | Diagnostica — errore a livello token |
| 5 | **Exact match** | uguaglianza stringa normalizzata | [0,1] | Diagnostica |
| 6 | **Pass@1 / Pass@k** | frazione di prompt con ≥1 completion sopra ROUGE-L 0.3 (k=1: single honest draw) | [0,1] | **Deployability** — metrica di progetto, NON letteratura (v. §2a) |
| 7 | **Validity** | frazione di completions con soli token in vocabolario gloss | [0,1] | Sistema — quantifica il contributo del constrained decoding |

### 2a. Pass@k e la soglia 0.3 — provenienza dichiarata

**Pass@k NON è una metrica della letteratura T2G**: nasce in HumanEval
(Chen et al. 2021) per il **codice**, dove "pass" = unit test binario. Il
reference repo (grpo-strict-generation) usava `check_syntax()` — check binario.
La gloss generation è open-ended: non esiste un test binario, quindi il
progetto sostituisce il check con una **proxy di similarità**:
ROUGE-L ≥ **0.3**. Il valore 0.3 è un'**euristica di progetto**
("almeno un terzo della struttura del gold recuperata"): nessun paper la
prescrive. Conseguenze:
- è **confrontabile solo dentro questo protocollo** (dichiararla sempre);
- la soglia va dichiarata in ogni tabella Pass@1/Pass@k della tesi;
- Pass@5 − Pass@1 misura quanto il modello "sa ma non è deterministico"
  (headroom del sampling a temp 0.7).

### 2b. Fonti della gerarchia (letteratura T2G)

- **BLEU-4 + chrF + valutazione umana**: Bangla T2G benchmark (Abdullah et
  al. 2025, arXiv:2504.02293) — il primo benchmark T2G dedicato. GPT-5.4:
  BLEU-4 39.26, chrF 73.75, umano 67.8%. Nota: metriche automatiche e umano
  possono disaccordare (Qwen-3 best human, GPT best BLEU).
- **BLEU + ROUGE**: Select and Reorder (Walsh, Saunders, Bowden, LREC-COLING
  2024, arXiv:2404.11532) — SOTA T2G su mDGS ("state-of-the-art BLEU and
  Rouge scores").
- **BLEU come reward e come eval**: RVLF (2025, arXiv:2512.07273 — il
  riferimento GRPO-SLT) + Mosquera et al. 2025 (GRPO su Qwen2.5-0.5B).
- **Nessun paper T2G usa Pass@k o threshold-metric**: sono del mondo
  code-generation.

### 2c. Per-difficulty breakdown

Ogni eval produce `results["difficulty_breakdown"]`: ROUGE-L / BLEU sent /
chrF sent / Pass@1 / validity per livello di difficoltà del gold (stessa
euristica del training: ≤5 token gold = simple, ≤15 = medium, >15 = hard).
Alimenta il grafico `difficulty_breakdown.png` e risponde "dove il modello
fa fatica" (monitor per-difficulty).

## 3. Decodifica in evaluation

- Generazione con lo **stesso constrained decoding** del training (Trie dual-root
  di default; PDA nei config ablation).
- **Sampling**: `num_samples` completions per prompt a temperatura 0.7
  (greedy se `num_samples=1`). Baseline e checkpoint usano **la stessa
  decodifica** in `--compare` (niente più greedy-vs-best-of-5).
- **Few-shot**: se `retrieval.enabled`, il prompt eval include gli stessi k
  esempi recuperati dal train (stesso retriever, stesso anti-leakage) —
  coerenza train/inference obbligatoria.

## 4. Selezione dei sample

- **Default: TUTTO il test set** (8.109 sample post-dedup).
- Se `evaluation.max_samples` è impostato: campionamento **random seeded**
  (`dataset.seed`), mai "primi N". Il report logga sempre
  `Evaluating N/M samples (seeded sample)`.

## 5. Reporting onesto

- Le metriche primarie NON usano mai il gold per selezionare le completions.
- Il best-of-N (selezione oracolo della migliore completion per ROUGE) è
  riportato **solo** nel blocco separato `oracle_best_of_n`, etichettato
  "NOT deployable" — è una misura di headroom, non una metrica del sistema.
- Il confronto baseline-vs-checkpoint (`comparison.json`) ha formato
  `{decoding, baseline, checkpoint, delta}` con decoding identico per i due lati.
- Ogni metrica riporta mean + CI 95% bootstrap + percentili quando applicabile.

## 6. Caveat dichiarati

1. **Constrained decoding può gonfiare BLEU**: la maschera vocabolario rende
  più probabili bigrammi gloss frequenti. Per questo chrF e gloss-F1 sono
  riportate accanto a BLEU in ogni tabella: se BLEU sale e chrF/gloss-F1 no,
  è reward hacking/metric inflation, non apprendimento.
2. **Nessun benchmark esterno per la direzione T2G**: i numeri SOTA pubblicati
   su ASLG-PC12 (Mono-SLT 89.9, TIN-SLT 84.3, STMC 82.4 BLEU-4) si riferiscono
   alla direzione **gloss→inglese** e NON sono confrontabili con i nostri
   (direzione opposta, protocollo diverso). L'unico precedente English→gloss su
   ASLG-PC12 è lo SMT del 2011. Il target BLEU 0.80 è quindi un target interno
   a questo protocollo, non un confronto con la letteratura.
3. **Split diverso dal 2011/2020-2023**: noi usiamo il 90/10 deduppato seedato,
   non lo split originale del dataset (che non ha test split nativo affidabile
   per la direzione T2G). Dichiararlo sempre nei report.
4. **chrF è case-sensitive** a livello di carattere (sacrebleu): le nostre gloss
   sono uppercase uniformi, quindi l'effetto è trascurabile, ma va dichiarato.

## 7. File di output e figure

Per ogni eval (in `experiments/results/<model>/<run_id>/`):
- `eval_<ckpt>.json` — metriche primarie + `oracle_best_of_n` + reward
  breakdown + `difficulty_breakdown`
- `generations_<ckpt>.json` — completions grezze con valid/rouge per sample
- `comparison.json` (solo `--compare`) — baseline vs checkpoint + delta

Figure (in `experiments/figures/<model>/<run_id>/`), in ordine di
rilevanza:
1. `metrics_dashboard.png` — **il grafico di confronto**: headline metrics
   (BLEU-4 corpus, chrF, ROUGE-L, Pass@1, Gloss F1, validity), baseline vs
   checkpoint, delta assoluto e % per pannello. La "one figure" della tesi.
2. `difficulty_breakdown.png` — metriche per livello di difficoltà del gold.
3. `bleu_distribution.png` / `chrf_distribution.png` / `rouge_distribution.png`
   — istogrammi per-completion delle tre metriche di contenuto, con
   overlay valid/invalid.
4. `completion_lengths.png`, `pass_at_k.png`, `error_breakdown.png`,
   `validity_pie.png`, `reward_breakdown.png`, `reward_radar.png`,
   `completion_examples.{json,html}`, `baseline_vs_grpo_comparison.png`.

Ablation cross-config: `ablation-summary` aggrega `eval_final.json` di ogni run
(preferisce `eval_final.json`; esclude `eval_baseline.json`).
