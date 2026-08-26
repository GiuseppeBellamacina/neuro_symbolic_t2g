# Protocollo di Valutazione — neuro_symbolic_t2g

Versione: 1.0 (2026-08-26). Questo documento definisce il protocollo con cui
vengono prodotti e confrontati i numeri del progetto. **Dichiarare e mantenere
questo protocollo è prerequisito per ogni claim sul target BLEU 0.80** — i numeri
sono comparabili solo dentro lo stesso protocollo.

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

## 2. Metriche

Tutte le metriche primarie sono calcolate su **tutte le completions** generate
per ogni prompt (no selezione oracolo). Implementazioni: `src/utils/metrics.py`
(sacrebleu per BLEU/chrF).

| Metrica | Definizione | Scala | Ruolo |
|---|---|---|---|
| **BLEU** | sacrebleu sentence BLEU, `effective_order=True`, smoothing `floor` 0.1; riportati sentence mean E corpus | [0,1] / [0,100] | Primaria (target 0.80) |
| **chrF** | sacrebleu CHRF2 (char F-score, β=2) | [0,100] | Primaria — indipendente dalla tokenizzazione, àncora contro inflazione BLEU |
| **Gloss-F1** | F1 token-level, case-insensitive (lowercase), precision/recall su token space-separated; micro corpus + sentence mean | [0,1] | Primaria — la metrica "sulla gloss" richiesta |
| **ROUGE-L** | F1 LCS (rouge_score, stemmer off) | [0,1] | Primaria (storica, continuità coi run precedenti) |
| **Pass@1** | frazione di sample con ≥1 completion sopra ROUGE-L 0.3 (`compute_pass_at_k(k=1)`, single honest draw) | [0,1] | Primaria |
| **Validity** | frazione di completions con soli token in vocabolario gloss | [0,1] | Diagnostica (posta ~1.0 by construction col constrained decoding) |
| **Exact match** | uguaglianza stringa normalizzata | [0,1] | Diagnostica |

**Passaggio del test**: un sample passa con ≥1 completion sopra threshold
ROUGE-L 0.3. `pass_at_k` riportato per k=1..N.

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

## 7. File di output

Per ogni eval (in `experiments/results/<model>/<run_id>/`):
- `eval_<ckpt>.json` — metriche primarie + `oracle_best_of_n` + reward breakdown
- `generations_<ckpt>.json` — completions grezze con valid/rouge per sample
- `comparison.json` (solo `--compare`) — baseline vs checkpoint + delta

Ablation cross-config: `ablation-summary` aggrega `eval_final.json` di ogni run
(preferisce `eval_final.json`; esclude `eval_baseline.json`).
