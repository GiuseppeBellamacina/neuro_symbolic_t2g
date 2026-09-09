# Protocollo di Valutazione — neuro_symbolic_t2g

Versione: 1.3 (2026-09-09). Questo documento definisce il protocollo con cui
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
(sacrebleu per BLEU/chrF, `non_copy_token_accuracy`). L'ordine della tabella è
la **gerarchia di rilevanza** usata nel log dell'eval, nel metrics_dashboard e
nelle tabelle della tesi.

| # | Metrica | Definizione | Scala | Ruolo |
|---|---|---|---|---|
| 1 | **Exact match** | uguaglianza stringa normalizzata | [0,1] | **Headline** — separa la transduzione dalla copia dell'inglese |
| 2 | **Non-copy token accuracy** | accuratezza sui token del reference non ottenibili uppercaseando il source (case-sensitive, `src/utils/metrics.py`) | [0,1] | **Headline** — separa la transduzione dalla copia dell'inglese (dettagli e denominatore: §2d) |
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

### 2d. Non-copy token accuracy — definizione, denominatore e caso degenere

**Definizione** (`src/utils/metrics.py`, dichiarata in `PRIMARY_METRICS`):
accuratezza ristretta ai token del reference che NON sono ottenibili
uppercaseando un token del testo sorgente inglese. Per ogni esempio:

1. `copyable = {token_source.upper()}` sull'intero testo sorgente;
2. si scorrono i token del reference: i token in `copyable` NON contano
   (né come giudicati né come errore);
3. ogni token non copiabile è una **posizione non banale** (denominatore);
   è un hit se il token è disponibile tra quelli prodotti dalla completion,
   a **multinsieme** (un token richiesto due volte va prodotto due volte) e
   **case-sensitive** (l'echo lowercase inglese NON prende credito).

Il matching a multinsieme lo rende insensibile all'ordine: l'ordine è già
coperto da exact match. `rule_baseline.py` ri-esporta la stessa funzione
(una sola definizione viva, testata dall'equivalenza in
`tests/test_non_copy_token_accuracy.py`).

**Perché è primaria su questo corpus**: il gloss ASLG-PC12 è per il ~62% il
token inglese maiuscolizzato e le metriche di overlap (ROUGE-L, BLEU, chrF,
gloss F1) premiano la copia. Questa è l'unica metrica che valuta SOLO ciò che
la copia non può produrre. Valori storici sui prompt di eval (2000 prompt,
**9350 posizioni non banali**):

| sistema | ROUGE-L | non-copy |
|---|---|---|
| base zero-shot senza Trie | 0,3648 | **0,0006** |
| base zero-shot con Trie | 0,1373 | **0,0432** |
| GRPO da base | 0,5998 | 0,4641 |
| baseline a regole | 0,9685 | 0,9424 |
| SFT | 0,9749 | 0,9660 |

Le prime due righe sono il punto: su ROUGE-L il vincolo simbolico "peggiora",
sulla non-copy il vincolo migliora di ~70x.

**Denominatore obbligatorio**: l'eval serializza tre chiavi nel blocco
primario del JSON (`eval_*.json`) — `non_copy_token_accuracy` (valore),
`non_copy_token_hits` e `non_copy_token_total` (denominatore: numero di
posizioni non banali). Il valore non è giudicabile senza il denominatore: su
quante posizioni si basa determina la sua stabilità. Ogni tabella della tesi
che cita la metrica riporta anche il denominatore.

**Caso degenere (insieme non banale vuoto)**: se NESSUN token del reference è
non copiabile (es. gloss identico al source maiuscolizzato), la funzione
restituisce `(accuracy=0.0, hits=0, total=0)`. La scelta documentata è
accuracy 0.0 (valore neutro, nessuna posizione valutabile): il chiamante deve
leggere `total` per distinguere "nessuna posizione valutabile" da "nessuna
posizione corretta". Non accade mai sui dati reali del progetto (il gloss
contiene sempre simboli non-inglesi come IX, fs-JOHN, DESC-*).

**Dove viene calcolata**: in `eval_t2g._compute_primary_metrics`, sul blocco
primario onesto (tutte le completions per prompt), con il testo sorgente
preso dalla colonna `text` del dataset (`flat_sources`); è stampata nel log
e nell'output dell'eval accanto a exact match, e va anche nel blocco
`oracle_best_of_n` quando presente. Non entra in `comparison.json`
(compare_keys): le baseline cachate pre-esistenti non hanno la chiave e un
confronto con 0.0 lato baseline sarebbe fuorviante.

## 3. Decodifica in evaluation

- **Invocazione**: `python -m src.training.eval_t2g --config <file.yaml>`
  (sul cluster: `CONFIG=<file.yaml> sbatch cluster/eval.sh`, con
  `CHECKPOINT=<path>` opzionale per forzare un checkpoint specifico).
  **Tutti i knob comportamentali vivono nella sezione `evaluation:` del
  config** — niente flag CLI oltre a `--config`/`--checkpoint` (che
  identificano COSA valutare), niente variabili d'ambiente. La tabella dei
  knob è commentata in `experiments/configs/qwen25-05b/base.yaml`.
- Generazione con lo **stesso constrained decoding** del training (Trie dual-root,
  l'unico path di decoding vincolato).
- **Sampling**: `num_samples` completions per prompt a temperatura 0.7
  (greedy se `num_samples=1`). Baseline e checkpoint usano **la stessa
  decodifica** in compare mode (niente più greedy-vs-best-of-5).
- **Few-shot**: se `retrieval.enabled`, il prompt eval include gli stessi k
  esempi recuperati dal train (stesso retriever, stesso anti-leakage) —
  coerenza train/inference obbligatoria.

### 3a. Modalità di prompting, dual eval e deduzione delle modalità

La modalità di prompting viene dalla `retrieval.enabled` della config
(comportamento di default, invariato). L'override esplicito e il dual eval
sono knob del config (sezione `evaluation:`):

- **Override**: `evaluation.prompting: config|zero-shot|few-shot`
  (default `config` — deriva da `retrieval.enabled`). `zero-shot` forza il
  retriever a None; `few-shot` lo forza attivo e **abortisce** se
  `max_prompt_length < 512` nella config (grpo o generation): con un budget
  più corto gli esempi few-shot verrebbero troncati e la cella sarebbe
  indistinguibile dallo zero-shot. Fail loud, non warning. Un override
  ridondante (few-shot su config già few-shot) è equivalente al default:
  nessun effetto su nomi file o cache. La validazione
  `prompting: few-shot ⇒ max_prompt_length >= 512` è replicata in
  `tests/validate_configs.py`.
- **Deduzione delle modalità eval** (storico di cluster/eval.sh, ora in
  `eval_t2g.py`): `compare` (baseline base-model + checkpoint) si deduce da
  `training.output_dir`; le celle eval-only (`baseline/*`, senza output_dir e
  senza checkpoint) valutano solo il base model **senza dichiarare nulla**.
  `evaluation.compare` / `evaluation.eval_baseline_only` (default `null`)
  sovrascrivono la deduzione quando dichiarati.
- **Provenienza e distinguibilità**: la modalità effettiva e la sua
  provenienza sono stampate nel log, stampate nel JSON
  (`results["prompting"] = {mode, source}` con source `config` (derivata),
  `config-override` (forzata) o `config-dual` (passata complementare)) e
  usate nel nome dei file: una passata con modalità diversa da quella
  implicita nella config suffissa l'output (`eval_final__zero-shot.json`,
  `generations_final__zero-shot.json`, `eval_baseline__zero-shot.json`),
  così due eval della stessa cella in modalità diverse non si sovrascrivono.
- **Cache della baseline**: il fingerprint del contesto prompt include la
  modalità SOLO quando differisce da quella implicita nella config — le run
  di default mantengono fingerprint byte-identici a quelli pre-esistenti
  (cache valide), un cambio di modalità (override o dual) invalida la cache
  e forza la ricomputo, e ogni modalità ha il suo file
  `eval_baseline__<mode>.json`. È la correzione del bug latente per cui una
  baseline calcolata in una modalità poteva essere riusata nell'altra.
- **DUAL prompting — ATTIVO DI DEFAULT sulle celle addestrabili**
  (`evaluation.dual_prompting: true` in base.yaml): la cella è valutata in
  ENTRAMBE le modalità (quella della config + la complementare), per misurare
  se il modello ha interiorizzato la mappatura o se dipende dal prompt come
  stampella (celle "train few-shot / eval zero-shot" e viceversa). La seconda
  passata parte DOPO quella primaria nello stesso processo; i file hanno il
  suffisso `__<mode>` e la cache baseline della modalità complementare è
  separata. **Nessun artefatto della passata primaria viene riscritto**: il
  suffisso copre anche `comparison.json` (`comparison__<mode>.json` — così
  `ablation-summary` continua a leggere i delta della modalità della cella)
  e un eventuale `evaluation.output` esplicito; le figure vanno in una
  **sottodirectory per modalità** (`figures/<model>/<run>/<mode>/`). La
  passata primaria non cambia alcun nome file. **Costo: l'eval raddoppia**
  (~25 min per passata a 5000 prompt;
  al primo giro la passata complementare valuta anche la SUA baseline del
  base model, poi cachata). Le celle `baseline/*` lo disattivano
  (`dual_prompting: false`): sono già una griglia esplicita di prompting
  (`zero-shot` / `zero-shot-no-grammar` / `few-shot`) e il dual le
  duplicherebbe. NOTA: la passata complementare few-shot NON è soggetta al
  vincolo `max_prompt_length >= 512` (vale solo per `evaluation.prompting`):
  a eval time i prompt non vengono troncati (`max_prompt_length` è un knob di
  TRAINING) e la misura del cross-prompting è deliberata — sempre marcata
  `source: config-dual` e con suffisso `__<mode>`.
- **Checkpoint incompleto**: `final` viene scritto solo a training
  completato. Se l'eval usa un `checkpoint-<step>` intermedio (training
  interrotto/TIMEOUT/in corso), il JSON dei risultati porta
  `checkpoint_incomplete: true` + `checkpoint_step` — un eval su modello
  parziale non è mai indistinguibile da uno su modello completo (§7).

## 4. Selezione dei sample

- **Default: TUTTO il test set** (8.109 sample post-dedup).
- Se `evaluation.max_samples` è impostato: campionamento **random seeded**
  (`dataset.seed`), mai "primi N". Il report logga sempre
  `Evaluating N/M samples (seeded sample)`.

## 4-bis. Resume dello stato parziale (walltime-safe)

Su cluster condiviso il ritmo di generazione varia fino a 6x senza preavviso
(1,86 s/prompt con GPU libera, 11,67 s/prompt con GPU contesa) e il JSON dei
risultati esiste solo a fine passata: un TIMEOUT azzera ore di lavoro (accaduto
su sft/zero-shot: kill a 1964/3000 dopo 6,5 h). Il rimedio, attivo di default:

- **Salvataggio periodico**: ogni `evaluation.resume_every` prompt (default
  100; `<= 0` disattiva il meccanismo) la passata scrive gli accumulatori
  (completions, references, sample_ids, texts, difficulties — allineati per
  indice) in `<results_dir>/resume_state_<soggetto>[__<mode>].json`.
  Scrittura **atomica** (`.tmp` + `os.replace`): un kill durante la scrittura
  lascia lo stato precedente valido, mai un file corrotto. Formato JSON
  indentato, pochi MB anche a 2000 prompt × 5 completions — ispezionabile a
  mano in debug.
- **Ripresa**: al rilancio lo stato è caricato SOLO se passa TUTTA la
  validazione: `state_schema_version`, `metrics_version`, `max_samples`,
  `num_samples` (completions per prompt), modalità di prompting, checkpoint
  valutato, `prompt_context_fingerprint` (§3a), dimensione del test set e del
  campione, lunghezze coerenti degli accumulatori, e infine l'**allineamento
  degli sample_id** con quelli che il campione deterministico (§4) produce per
  le stesse posizioni. Un qualunque scarto ⇒ ripartenza da zero con avviso
  che indica il campo non corrispondente: mai riprendere da uno stato di cui
  non si è certi, perché produrrebbe metriche su un insieme misto di prompt.
- **Due passate del dual**: uno stato per SOGGETTO (baseline base-model vs
  checkpoint) e per modalità (`__<mode>` nel nome, stesso contratto degli
  altri artefatti, §3a) — la seconda passata non può riprendere lo stato
  della prima.
- **Pulizia**: a passata completata lo stato è rimosso; la sua presenza deve
  significare solo "questa passata è incompleta". Il nome non inizia con
  `eval_`, quindi `ablation_summary` e `campaign_report` non lo raccolgono mai
  come risultato (§7).
- **Osservabilità**: all'avvio con ripresa il log dichiara prompt recuperati e
  rimanenti (`RESUMED partial eval state: N/M ...`); il JSON finale porta
  `resumed_from: {resumed, resume_count, recovered_prompts}` — una passata
  prodotta in più rilanci ha le STESSE metriche di una completata in un colpo
  solo (gli accumulatori sono identici), ma la provenienza è dichiarata.

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
- `eval_<ckpt>.json` — metriche primarie (incluse `non_copy_token_accuracy`,
  `non_copy_token_hits`, `non_copy_token_total`, §2d) + `oracle_best_of_n` +
  reward
  breakdown + `difficulty_breakdown` + stamp `prompting` (modalità e
  provenienza) + eventuale stamp di incompletezza: un eval su un checkpoint
  intermedio `checkpoint-<step>` (invece di `final`) porta
  `checkpoint_incomplete: true` e `checkpoint_step`, così il risultato di una
  cella interrotta non è mai indistinguibile da quello di un modello completo
  (§3a)
- `generations_<ckpt>.json` — completions grezze con valid/rouge per sample
- `resume_state_<soggetto>[__<mode>].json` — SOLO mentre la passata è in
  corso o interrotta: stato parziale per il resume da walltime (§4-bis).
  Rimosso a passata completata; mai raccolto come risultato (il nome non
  matcha i pattern `eval_*.json` di ablation_summary/campaign_report)
- `comparison.json` (solo compare mode) — baseline vs checkpoint + delta;
  con modalità di prompting effettiva diversa da quella della config la
  passata dual scrive `comparison__<mode>.json` e non tocca il file della
  primaria (§3a)
- Con una passata in modalità diversa da quella implicita nella config
  (override `evaluation.prompting` o dual prompting), i file portano il
  suffisso `__<mode>` (es. `eval_final__zero-shot.json`) così le due
  modalità non si sovrascrivono (§3a); le figure vanno in
  `experiments/figures/<model>/<run_id>/<mode>/`

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

### 7a. Campaign report — confronti appaiati cross-fattore

`python -m src.analysis.campaign_report` (alias `campaign-report`) è
complementare ad `ablation-summary`: mentre quello produce la tabella piatta
per config, questo appaia i run che differiscono per **un solo fattore
sperimentale** (metodo di addestramento, prompting, decodifica vincolata) e
riporta il delta di ogni metrica — le righe che servono per compilare la
matrice di ablazione. Da lanciare a fine campagna o quando il cluster termina
una chain; gira senza GPU sui file locali. Output in `experiments/figures/`:
`campaign_report.json`, `campaign_report.md`,
`campaign_pairwise_deltas.png`, `campaign_matrix.png`.

Regole dichiarate (riprodotte in ogni report generato):
- **Selezione dei run**: per ogni *tipologia* (combinazione dei fattori
  dedotti, non il nome della directory) viene preso il run più recente;
  eval_final prevale sui checkpoint intermedi; `eval_baseline.json` è un run
  `method=baseline` (modello senza checkpoint nel contesto della cella).
- **Deduzione dei fattori**: dal percorso, dallo stamp `prompting` del JSON e
  dal suffisso `__<mode>` dei file (dual/override). `reward_stack` e
  `rl_objective` NON sono deducibili (vivono nel config risolto, non nel
  payload): restano `unknown` e appaiono come caveat su ogni coppia — mai
  assunti uguali in silenzio.
- **Confronto più importante**: la stessa cella valutata in entrambe le
  modalità (dual, stesso checkpoint) — misura se il modello ha interiorizzato
  o dipende dal prompt come stampella; marcato 🔴 nel report.
- **Avvertenze obbligatorie**: `num_samples_evaluated` e `metrics_version` di
  ogni run con avviso esplicito se discordanti; data di ogni run (dalla run
  dir, fallback mtime dichiarato); JSON malformati/parziali segnalati senza
  sollevare; run con `checkpoint_incomplete: true` esclusi e listati.
- **Soglia di rumore 0.02** (metriche in scala [0,1]): la dispersione fra due
  esecuzioni della stessa cella arriva a 0.0161 ROUGE-L contro un CI di
  ±0.0044, quindi i delta sotto 0.02 sono marcati NOISE e non vanno
  interpretati. Un delta di 0.003 non è un risultato.
- **Nessuna metrica ricalcolata**: il modulo legge; una chiave assente nel
  JSON produce un delta assente, non un'euristica.
