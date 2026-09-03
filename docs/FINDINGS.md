# Findings della Campagna — Decomposizione + Ablation

**Data**: settembre 2026
**Protocollo**: eval 2000 prompt × 5 sample, T=0.7, metriche v2 (vedi `EVALUATION.md`)
**Fonte dati**: `experiments/results/qwen25-05b-*/run_*/` + `experiments/logs/*/run_*/output.log`

---

## Finding 1 — GRPO-from-SFT degrada il modello SFT (IL finding)

### Il dato

| Cella | Init GRPO | ROUGE-L test | BLEU-c | Pass@1 | Reward train (media) |
|---|---|---|---|---|---|
| sft-only | — | **0.979** | 0.955 | 0.999 | — |
| sft-grpo | SFT (3 ep) | 0.501 | 0.217 | 0.798 | 0.930 |
| grpo-only | base | 0.608 | 0.335 | 0.928 | 0.062 |

GRPO da base **migliora** (0.466→0.608); GRPO da SFT **distrugge** (0.979→0.501, −49%).
Il fix ricetta 3-epoche NON ha risolto: il degrado non dipende dall'allineamento SFT.

### Meccanismo (dai log di training, 400 step-record per run)

Tre fatti convergenti, estratti dalle righe `step=… reward=… kl=…` di
`run_20260902_151330` (sft-grpo) vs `run_20260901_233552` (grpo-only):

1. **Reward saturo sul train dal primo step.** sft-grpo parte a reward 0.886
   (step 5) e media 0.930 su 2000 step. Su prompt di train il modello SFT
   produce completion quasi-perfette (i primi sample loggati hanno tutti i
   componenti = +1.00). GRPO non ha margine di miglioramento → il segnale
   di apprendimento non c'è.

2. **Collasso dei gruppi a vantaggio nullo.** `frac_reward_zero_std` media
   **0.578** (max 1.00): nel ~58% dei gruppi G=5 i sample hanno reward
   identici → advantage = 0 → gradiente nullo. La policy SFT è troppo
   deterministica: all'eval, l'85.9% dei prompt produce 5/5 completions
   identiche anche a T=0.7. In grpo-only `frac_reward_zero_std` = 0.029:
   c'è sempre segnale, e l'ottimizzazione è reale (reward 0.023→0.197,
   KL media 0.81).

3. **Il segnale residuo è distorto e il danno cresce con la difficoltà.**
   Il ~42% di gruppi con varianza spinge il policy su una direzione
   sistematica: lunghezza media generata 11.48 parole vs gold 12.26
   (−0.71; no-grammar addirittura −2.07), entropia iniettata (prompt con
   5/5 identici: 85.9% → **4.5%**), e degrado che cresce con la lunghezza
   del gold:

   | Bucket len(gold) | 1–3 | 4–6 | 7–9 | 10+ |
   |---|---|---|---|---|
   | sft-only | 0.884 | 0.971 | 0.983 | 0.980 |
   | sft-grpo | 0.693 | 0.637 | 0.575 | **0.467** |

   L'`oracle_best_of_n` di sft-grpo è 0.623: nemmeno la selezione oracolo
   tra i suoi sample recupera — il danno è nel modo del policy, non in
   qualche sample sfortunato.

### Interpretazione (per la tesi)

Il paradigma SFT→RL presuppone che il policy iniziale lasci margine di
reward sul train set (es. DeepSeekMath, dove RL-from-SFT funziona perché il
task non è memorizzabile a saturazione). Con un dataset "traducibile" come
ASLG-PC12 (87K coppie, SFT a 0.979), GRPO si trova con **reward saturato +
gruppi degeneri a vantaggio nullo**: i gradienti residui ottimizzano le
poche direzioni rimaste (soprattutto accorciare/semplificare, che alza i
componenti structure/format) a spese della generalizzazione sui prompt
lunghi. Risultato: RL non è "sempre utile dopo SFT" — **è utile solo se il
reward sul train è lontano dal saturo**. Controprova naturale: grpo-only,
partendo da reward 0.023, migliora su ogni metrica.

### Esperimento di controllo (`sft-grpo-hotrollout`, in coda)

Config `experiments/configs/t2g/sft-grpo-hotrollout.yaml`: differenza a
fattore unico vs sft-grpo — `grpo.temperature: 1.3` (era 0.7). La rollout
temperature più alta rompe il determinismo della policy SFT e ripristina
la varianza intra-gruppo. L'adapter SFT è riusato (fingerprint invariata
→ nessun retrain SFT, solo GRPO+eval ~3.5h).

**Previsioni registrate** (scritte nel config header PRIMA del run):

| # | Previsione | Valore sft-grpo (riferimento) |
|---|---|---|
| P1 | `frac_reward_zero_std` media **< 0.15** | 0.578 |
| P2 | reward a step 5 ~ 0.70–0.80 (il campionamento caldo abbassa il reward) | 0.886 |
| P3 | se il collasso zero-advantage è la CAUSA: ROUGE-L finale **> 0.55** e % prompt 5/5 identici a eval > 4.5% | 0.501 / 4.5% |
| P4 | se invece la causa è il bias di lunghezza: ROUGE-L finale ~ 0.50 (con P1/P2 confermate) | 0.501 |

P3 o P4 chiudono la causalità in entrambi i rami: o il degrado è
mediato dal determinismo (e si riduce), o resta e la causa va cercata
nella direzione del segnale residuo.

**Nota storica**: SFT 1-epoca + GRPO è già stato provato (run 20260901,
documentato in sft-grpo.yaml) e sottoperformava persino grpo-only —
l'undertraining dell'SFT NON è la manopola giusta: lascia un init più
debole senza dare a GRPO un segnale migliore. Per questo la cella di
controllo agisce sulla temperatura dei rollout, non sulla ricetta SFT.

---

## Finding 2 — Constrained decoding: garanzia di validità, non di contenuto

### Trade-off quantificato (stesso policy sft-grpo, stesso test set, cambia solo il decoding)

| Decoding | Validity | ROUGE-L | chrF-c | Pass@1 |
|---|---|---|---|---|
| Trie ON | 99.0% | 0.501 | 44.7 | 0.798 |
| Grammar OFF | 77.4% | **0.588** | **50.6** | 0.903 |

Il vincolo costa **−0.087 ROUGE-L (−15% rel.) e −5.9 chrF-c**, e compra
**+21.6 punti di validità**. La penalità cresce con la lunghezza del gold
(Δ ROUGE OFF−ON per bucket: +0.045, +0.034, +0.068, **+0.097** su 10+):
più lunga la sequenza, più spesso il Trie forza una continuazione valida
ma subottimale (errore che si propaga).

### Il caso zero-shot (senza conoscenza del task)

| Cella | Validity | ROUGE-L |
|---|---|---|
| zero-shot no grammar | 3.7% | 0.365 |
| zero-shot + grammar | 90.6% | **0.137** |

Qui la grammatica **peggiora anche il contenuto**: senza conoscenza, il
vincolo forza formati validi cancellando l'unica cosa che il modello sa
fare (copiare token dell'input). Il constrained decoding amplifica ciò
che il policy già sa, non aggiunge conoscenza.

### Il caso sft-only (policy competente)

Sft-only + Trie = validity **100%** e ROUGE 0.979: quando il policy ha
la competenza, il vincolo è **quasi gratuito** (−0.01~0.02) e dà la
garanzia formale gratis.

### Narrativa decisa per la tesi

> Il constrained decoding va presentato come **garanzia formale di
> validità con costo di contenuto dipendente dalla competenza del
> policy**: costo ~zero per un modello competente (sft-only), −15% per un
> policy degradato (sft-grpo), distruttivo senza competenza (zero-shot).
> La scelta deployment è: grammar-ON quando la validità è un hard
> requirement; grammar-OFF + post-filter (Finding 3) quando domina il
> contenuto e il budget di inferenza lo permette.
>
> L'asse Trie-vs-PDA (cella `sft-grpo-pda`, pronta) completa il capitolo:
> stesso policy, due formalismi di vincolo a confronto.

---

## Finding 3 — Post-filter validity: simulazione e verdetto

### Simulazione dai dati esistenti (sg-no-grammar, 5 sample/prompt, T=0.7)

Strategia deployable: genera senza grammatica, filtra le completion
non valide (check vocabolario, O(1)), tieni la prima valida; resample
solo se nessuna è valida.

| Quantità | Valore |
|---|---|
| Validità per-sample | 77.4% |
| Copertura ≥1 valida su 5 sample | **95.6%** |
| Copertura con resample fino a k=10 | ~100% (1−0.226¹⁰) |
| E[sample] per prompt (1/p) | ~1.3 |
| ROUGE-L first-valid (deployable) | **0.614** |
| ROUGE-L best-valid (oracolo, ceiling) | 0.691 |
| ROUGE-L senza filtro | 0.588 |

**Confronto diretto**: post-filter beats grammar-ON sulla stessa
linea GRPO — **0.614 vs 0.501 ROUGE (+22% rel.) con entrambe a
validità ~100%**, al costo di ~1.3 sample/prompt (vs 1 del
constrained decoding).

### Perché NON ritrainiamo

Il post-filter è la strategia giusta **per un policy con tensione
contenuto-validità** — cioè le celle GRPO. Ma la campagna ha già un
punto che domina tutto lo spazio: **sft-only + Trie = 0.979 ROUGE,
validity 100%, 1 sample greedy, zero post-processing**. Nessun
ritraining (grammar-off + post-filter) può competere: il suo ceiling
è il contenuto del policy, e il miglior policy resta quello SFT.

Inoltre il post-filter ha un range di applicabilità: richiede
p(valida) per-sample sufficiente. Su zero-shot (p=0.037) la copertura
a 20 sample è 53% — infeasible. È una **leva di deployment**, non un
fix di training.

### Verdetto

1. **Nessun retraining.** La configurazione migliore della campagna è
   già sft-only (+ Trie all'inferenza).
2. Il post-filter entra nella tesi come **alternativa quantificata**:
   "per deployment dove il contenuto domina e il budget di inferenza
   cresce di ~30%, grammar-off + validity-filter supera il constrained
   decoding di +0.11 ROUGE sulla stessa policy" — dimostrato per
   simulazione sui dati, senza run aggiuntive.
3. Se si vuole una cella empirica a supporto (opzionale):
   `sft-grpo-no-grammar` già esiste; basta aggiungere al report eval
   la metrica `post_filtered` (first-valid + fallback) — nessun
   training nuovo.

---

## Tabella riassuntiva campagna (v2 metrics)

| Cella | ROUGE-L | BLEU-c | chrF-c | Validity | Pass@1 |
|---|---|---|---|---|---|
| zero-shot | 0.365 | 0.001 | 1.2 | 3.7% | 58.5% |
| zero-shot-grammar | 0.137 | 0.027 | 13.2 | 90.6% | 18.2% |
| **sft-only** | **0.979** | **0.955** | **97.8** | **100%** | **99.9%** |
| grpo-only | 0.608 | 0.335 | 55.7 | 99.2% | 92.9% |
| sft-grpo | 0.501 | 0.217 | 44.7 | 99.0% | 79.8% |
| sg-structure | 0.511 | 0.222 | 45.4 | 99.4% | 79.9% |
| sg-viterbi | 0.510 | 0.222 | 45.4 | 99.2% | 80.3% |
| sg-soft-viterbi | 0.505 | 0.219 | 44.7 | 98.9% | 79.3% |
| sg-all-rewards | 0.513 | 0.226 | 45.5 | 99.0% | 79.8% |
| sg-no-grammar | 0.588 | 0.204 | 50.6 | 77.4% | 90.3% |
| sg-pda | *(in coda)* | | | | |
| sg-hotrollout | *(in coda)* | | | | |

I moduli simbolici aggiuntivi (structural/viterbi/soft) hanno effetto
≤ ±0.01 sul policy GRPO: coerente con Finding 1 — senza segnale di
apprendimento, nessun reward può esprimersi.

---

## Script di riproduzione

Le tre analisi sono riproducibili da:

- Curve train: parse `output.log` → `step=… reward=… kl=… frac_reward_zero_std=…`
- Post-filter: `experiments/results/qwen25-05b-sft-grpo-no-grammar/run_*/generations_final.json`
  (raggruppare per `text` = prompt, campi `valid`/`rouge_l` per sample)
- Difficoltà: bucket su `len(gold_gloss.split())`
