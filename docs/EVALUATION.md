# Protocollo di Valutazione — neuro_symbolic_t2g

Versione: 1.2 (2026-09-08). Questo documento definisce il protocollo con cui
vengono prodotti e confrontati i numeri del progetto. **Dichiarare e mantenere
questo protocollo è prerequisito per ogni claim sul target BLEU 0.80** — i numeri
sono comparabili solo dentro lo stesso protocollo. La gerarchia delle metriche
(§2) mette al vertice le primarie di progetto `exact_match` e
`non_copy_token_accuracy`: le metriche di overlap della letteratura sono sature
su questo corpus (evidenza in `docs/RECOVERY_REPORT.md` §9) e restano riportate
solo per confrontabilità (vedi §2b per le fonti).

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
(sacrebleu per BLEU/chrF) e `src/analysis/rule_baseline.py`
(`non_copy_token_accuracy`). L'ordine della tabella è la **gerarchia di
rilevanza** usata nel log dell'eval, nel metrics_dashboard e nelle tabelle
della tesi.

| # | Metrica | Definizione | Scala | Ruolo |
|---|---|---|---|---|
| 1 | **Exact match** | uguaglianza stringa normalizzata | [0,1] | **Headline** — separa la transduzione dalla copia dell'inglese |
| 2 | **Non-copy token accuracy** | accuratezza sui token del reference non ottenibili uppercaseando il source (case-sensitive, `src/analysis/rule_baseline.py`) | [0,1] | **Headline** — separa la transduzione dalla copia dell'inglese |
| 3 | **BLEU-4 (corpus)** | sacreBLEU corpus, refs flat allineate (v2 `metrics_version`); sentence mean riportato accanto | [0,1] | Comparabilità — standard della letteratura T2G, saturo su questo corpus |
| 4 | **chrF2 (corpus)** | sacrebleu CHRF2 (char F-score, β=2) | [0,100] | Comparabilità — indipendente dalla tokenizzazione |
| 5 | **ROUGE-L** | F1 LCS (rouge_score, stemmer off), sentence mean | [0,1] | Comparabilità — satura e difettosa (v. nota sotto); sempre accanto alla baseline a regole |
| 6 | **Gloss F1 (micro)** | F1 token-level case-insensitive | [0,1] | Diagnostica — errore a livello token |
| 7 | **Pass@1 / Pass@k** | frazione di prompt con ≥1 completion sopra ROUGE-L 0.3 (k=1: single honest draw) | [0,1] | **Deployability** — metrica di progetto, NON letteratura (v. §2a) |
| 8 | **Validity** | frazione di completions con soli token in vocabolario gloss | [0,1] | Sistema — quantifica il contributo del constrained decoding |

Le due headline sono scelte **di progetto**, non di letteratura: su ASLG-PC12 le
metriche di overlap sono sature — una regola lessicale costruita solo dal train
raggiunge ROUGE-L 0.9697 sul test completo — e ROUGE-L è anche difettosa
(case-insensitive e splitta sui non-alfanumerici, quindi `DESC-GOOD` vs
`DESC-BAD` prende 0.5). ROUGE-L va quindi riportata **solo** per confrontabilità
con la letteratura e **sempre accanto alla baseline a regole** (`rule_baseline`).
Evidenza: `docs/RECOVERY_REPORT.md` §9.

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

### 2a-bis. I DUE stimatori Pass@1 nell'output eval — quale è il riferimento

L'output dell'eval contiene DUE numeri Pass@1 che NON sono lo stesso
stimatore, ora con etichette distinte (prima condividevano l'etichetta
"Pass@1" con valori leggermente diversi — difetto di presentazione, non di
calcolo; nessun valore è stato modificato):

| Etichetta nell'output | Chiave JSON | Definizione | Campioni |
|---|---|---|---|
| `Pass@1 (first completion)` | `pass_at_1` | frazione di prompt la cui **prima** completion raggiunge ROUGE-L ≥ 0.3 (`compute_pass_at_k`, k=1) | 1 draw per prompt (N prompt) |
| `Pass@1 (all completions)` (blocco CI) | `evaluation_report.pass_at_1` | media empirica dell'indicatore ROUGE-L ≥ 0.3 su **tutte** le completions, con CI bootstrap | `num_samples` draw per prompt |

**Stimatore di riferimento: `Pass@1 (first completion)`** (`pass_at_1`,
chiave JSON `pass_at_1`). Motivazioni:

1. è l'ancora k=1 della curva Pass@k, che per definizione usa le **prime k**
   completions di ogni prompt: pass@1 e la curva restano così coerenti fra
   loro;
2. è la quantità "deployable" del protocollo: il primo draw è ciò che un uso
   reale greedy/singolo-sample otterrebbe.

Il valore nel blocco CI (`Pass@1 (all completions)`) stima la STESSA
quantità sottostante (probabilità che una singola completion campionata
superi la soglia) ma con ~`num_samples`× i campioni, quindi con varianza
minore: **usare quello quando si cita un intervallo di confidenza**. I due
numeri differiscono solo per rumore di campionamento e coincidono quando
`num_samples = 1` (test: `tests/test_eval_pass_estimators.py`).

**Nota esplicita**: NESSUNO dei due è lo stimatore combinatorio non distorto
`pass@k = 1 − C(n−c, k)/C(n, k)` di HumanEval/Codex (nel repo non esiste
alcuna implementazione del genere). Sono entrambe medie empiriche su
sottoinsiemi diversi dello stesso campione. Se in futuro si introducesse lo
stimatore combinatorio, andrebbe aggiunto come TERZA riga con etichetta
propria e bump di `METRICS_VERSION` se sostituisse una delle due.

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

- Generazione con lo **stesso constrained decoding** del training (Trie dual-root,
  l'unico path di decoding vincolato).
- **Sampling**: `num_samples` completions per prompt a temperatura 0.7
  (greedy se `num_samples=1`). Baseline e checkpoint usano **la stessa
  decodifica** in `--compare` (niente più greedy-vs-best-of-5).
- **Few-shot**: se `retrieval.enabled`, il prompt eval include gli stessi k
  esempi recuperati dal train (stesso retriever, stesso anti-leakage) —
  coerenza train/inference obbligatoria.

### 3a. Override della modalità di prompting (opt-in) e dual eval

La modalità di prompting viene dalla `retrieval.enabled` della config
(comportamento di default, invariato). Per l'eval sola esiste un override
esplicito **disattivato per default**:

- **CLI**: `--prompting {config,zero-shot,few-shot}` (default `config`).
  `zero-shot` forza il retriever a None; `few-shot` lo forza attivo e
  **abortisce** se `max_prompt_length < 512` nella config (grpo o
  generation): con un budget più corto gli esempi few-shot verrebbero
  troncati e la cella sarebbe indistinguibile dallo zero-shot. Fail loud,
  non warning. Un override ridondante (few-shot su config già few-shot) è
  equivalente al default: nessun effetto su nomi file o cache.
- **Shell**: `PROMPTING=zero-shot sbatch cluster/eval.sh` (stesso modello di
  `MAX_SAMPLES`; default: nessun flag).
- **Provenienza e distinguibilità**: la modalità effettiva e la sua
  provenienza (`config`/`cli`) sono stampate nel log, stampate nel JSON
  (`results["prompting"] = {mode, source}`) e usate nel nome dei file: un
  override che cambia modalità suffissa l'output (`eval_final__zero-shot.json`,
  `generations_final__zero-shot.json`, `eval_baseline__zero-shot.json`),
  così due eval della stessa cella in modalità diverse non si sovrascrivono.
- **Cache della baseline**: il fingerprint del contesto prompt include
  l'override SOLO quando cambia la modalità — le run di default mantengono
  fingerprint byte-identici a quelli pre-esistenti (cache valide), un
  override cambiante invalida la cache e forza la ricomputo. È la correzione
  del bug latente per cui una baseline calcolata in una modalità poteva
  essere riusata nell'altra.
- **DUAL eval (opt-in esplicito)**: valutare la stessa cella in ENTRAMBE le
  modalità misura se il modello ha interiorizzato la mappatura o se dipende
  dal prompt come stampella (cfr. celle "train few-shot / eval zero-shot").
  Attivazione: `DUAL_EVAL=1 sbatch cluster/eval.sh` o `DUAL_EVAL=1` prima di
  `run_all.sh` (la seconda passata con la modalità complementare parte DOPO
  quella primaria nello stesso job; i file hanno il suffisso `__<mode>`).
  NON è attivo nella catena di default: la matrice di celle e i tempi sono
  invariati. Nota sulla propagazione: per i tick di catena via hook bashrc la
  variabile deve essere presente anche nell'ambiente che esegue il tick
  (`export DUAL_EVAL=1`).

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
5. **Dato pre-tokenizzato: BLEU/chrF assoluti NON confrontabili con la
   letteratura.** Il gloss ASL ha la punteggiatura come token separato
   (`X-Y WILL DESC-NOT FLINCH .`): ogni riga termina legittimamente con
   `.` e sacrebleu (≤ 2.4.x) lo segnala come possibile dato dimenticato
   detokenizzato. Il warning è corretto sul dato ma NON indica un difetto
   nostro: la soppressione via `force=True` in `src/utils/metrics.py` è
   consapevole e documentata lì. Conseguenze dichiarate:
   - i confronti **FRA celle** di questo progetto restano validi (stesso
     protocollo, stesso pre-tokenizzamento, tutte le celle);
   - i valori **ASSOLUTI** di BLEU/chrF NON sono confrontabili con numeri
     pubblicati calcolati su dato detokenizzato (13a tokenizer su testo
     con punteggiatura attaccata): ogni tabella della tesi che cita valori
     assoluti deve ripetere questo caveat.

## 7. File di output e figure

Per ogni eval (in `experiments/results/<model>/<run_id>/`):
- `eval_<ckpt>.json` — metriche primarie + `oracle_best_of_n` + reward
  breakdown + `difficulty_breakdown` + stamp `prompting` (modalità e
  provenienza)
- `generations_<ckpt>.json` — completions grezze con valid/rouge per sample
- `comparison.json` (solo `--compare`) — baseline vs checkpoint + delta
- Con un override `--prompting` che cambia modalità, i file portano il
  suffisso `__<mode>` (es. `eval_final__zero-shot.json`) così le due
  modalità non si sovrascrivono (§3a)

Figure (in `experiments/figures/<model>/<run_id>/`), in ordine di
rilevanza:
1. `metrics_dashboard.png` — **il grafico di confronto**: headline metrics
   (exact match, non-copy token accuracy, BLEU-4 corpus, chrF, ROUGE-L, Pass@1, Gloss F1, validity), baseline vs
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
