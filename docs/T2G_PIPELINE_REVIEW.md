# Revisione Pipeline T2G — SFT → Rewards → GRPO → Constrained Decoding

> Documento di analisi tecnica su `neuro_symbolic_t2g/`. Non copre `grammarllm/`
> o `grpo-strict-generation/` (fuori scope, per richiesta esplicita).
> Data: luglio 2026.

## 1. Riassunto esecutivo

| Problema                                                                                                                    | Causa individuata                                                                                                                                                                      | Stato                                             |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Crash in Fase 2 (GRPO) — `RuntimeError: output with shape [8, 14, 1, 64] doesn't match the broadcast shape [8, 14, 73, 64]` | `unsloth==2026.3.17` pinnato in `pyproject.toml` è troppo vecchio: non contiene i fix upstream per la gestione di `position_ids` durante il decode incrementale su `transformers>=5.0` | ✅ **Fix applicato** (bump versione)              |
| Bug di sincronizzazione nel Trie di `GlossVocabularyLogitsProcessor`                                                        | Il fallback su mismatch scarta il token invece di ri-testarlo contro la radice                                                                                                         | ⚠️ Documentato, **non ancora corretto** (vedi §4) |
| Qualità generale bassa                                                                                                      | Multi-fattoriale: vedi §5                                                                                                                                                              | ⚠️ Raccomandazioni fornite                        |

---

## 2. Causa del crash: versione Unsloth obsoleta

### 2.1 Traceback e riproduzione del bug

Il crash avviene durante il rollout di generazione in GRPO (dentro
`_patched_generate` in `src/training/grpo_t2g_train.py`), quando Unsloth
esegue il suo kernel di inferenza rapida per Llama-family
(`LlamaAttention_fast_forward_inference`, usato anche da Qwen2/Qwen3 via
architettura condivisa). L'errore esatto:

```
RuntimeError: output with shape [8, 14, 1, 64] doesn't match the
broadcast shape [8, 14, 73, 64]
```

si verifica sulla riga `Qn *= cos`, all'interno dell'applicazione del RoPE
in-place durante un singolo step di decode (batch=8, 14 head, head_dim=64).

### 2.2 Root cause

A partire da `transformers>=5.0`, durante la generazione incrementale
(KV-cache), `position_ids` **non** viene più passato come tensore
`[batch, 1]` (la sola posizione del token corrente) ma si accumula come
`[batch, full_seq_len]` (tutte le posizioni viste finora — nel caso
osservato, 73).

Il codice upstream corrente di Unsloth (branch `main`,
`unsloth/models/llama.py`, dentro `LlamaAttention_fast_forward_inference`)
gestisce esplicitamente questo caso:

```python
if position_ids.dim() == 1:
    position_ids = position_ids[:, None]
# Transformers 5.x accumulates position_ids as [batch, full_seq_len] across
# decode steps; single-token inference only needs the last position.
if position_ids.shape[-1] > 1:
    position_ids = position_ids[:, -1:]
position_ids = position_ids.to(Qn.device)
...
cos = cos[position_ids].unsqueeze(1)...   # ora ha seq-dim = 1, non 73
...
Qn *= cos          # <- riga del crash originale, ora broadcast-compatibile
```

Senza questo troncamento, `cos[position_ids]` produce un tensore con
seq-dim pari alla lunghezza accumulata (73) invece di 1, e il successivo
`Qn *= cos` (in-place, su `Qn` di shape `[8, 14, 1, 64]`) fallisce nel
broadcast — esattamente l'errore osservato.

Lo stesso pattern (`Qn *= cos` dentro `*_fast_forward_inference`) è
condiviso da `llama.py`, `qwen3.py`, `cohere.py`, `falcon_h1.py`,
`gemma2.py`, `granite.py` — quindi il bug non è specifico di un'architettura,
ma della gestione generica di `position_ids` durante il decode con
`transformers>=5.0`.

### 2.3 Perché la versione pinnata (`2026.3.17`) è interessata

Il fix non è isolabile a un singolo commit con un messaggio esplicito (il
testo del commento non compare in nessun titolo di PR), ma è stato
introdotto/consolidato attraverso una serie di fix RoPE/transformers-5.x
successivi al pin di marzo, tra cui:

- **1 apr 2026** — `#4752` _"Fix forward compatibility with transformers 5.x"_
- **11 giu 2026** — `#6197` _"Fix Llama 3.1+ rope scaling dropped on the FastLanguageModel path"_
- **12 giu 2026** — `#6223` _"Stop false RoPE 'default' warning and fix rope drift gate on transformers 5"_
- **6 lug 2026** — `#6907` _"Fix llama3 RoPE scaling dropped on transformers v5"_
- **7 lug 2026** — `#6925` _"Guard RoPE scaling against the transformers v5 buffer blank"_

Tutti questi commit sono **successivi** al 17 marzo 2026 (data del pin
`unsloth==2026.3.17`). Il codice attuale su `main` contiene il fix
verificato (`position_ids[:, -1:]`); la ricerca sistematica nella cronologia
commit di `unsloth/models/llama.py` confirma che l'area RoPE/decode è stata
oggetto di patch continue per tutta la primavera-estate 2026 — segno che
`transformers` 5.x era (ed è) un target in movimento per Unsloth.

### 2.4 Fix applicato

In `neuro_symbolic_t2g/pyproject.toml`:

```toml
# prima
"unsloth==2026.3.17",

# dopo
"unsloth>=2026.7.1",
```

> ⚠️ **Correzione**: un primo tentativo aveva fissato `>=2026.7.7`,
> dedotto dalle date dei commit visti nella cronologia GitHub del branch
> `main`. Quella versione **non esiste ancora su PyPI** — i commit su
> GitHub non sono automaticamente rilasciati come package. L'installazione
> sul cluster ha fallito con `No matching distribution found for
unsloth>=2026.7.7` (l'ultima versione pubblicata al 2026-07-07 è
> `2026.7.1`). Corretto a `>=2026.7.1`. **Lezione**: quando si fissa una
> versione minima da bug-fix osservati su GitHub, verificare sempre che
> la versione sia effettivamente pubblicata su PyPI (`pip index versions
<pkg>`), non solo che il commit esista nel repository.

`unsloth_zoo` non è stato pinnato esplicitamente: è una dipendenza
transitiva di `unsloth` e verrà risolta automaticamente alla versione
compatibile.

**Azione richiesta sul cluster**: rieseguire l'installazione
(`pip install --user -e . --upgrade`) prima del prossimo run, poiché il
pin è cambiato solo nel file di progetto — l'ambiente Python esistente sul
cluster ha ancora la versione vecchia installata finché non si aggiorna.

> **Nota di follow-up**: se dopo l'upgrade a `2026.7.1` il crash dovesse
> ripresentarsi, significa che il fix `position_ids[:, -1:]` non è ancora
> incluso in quella release (i commit più recenti osservati, come #6907
> del 6 luglio e #6925 del 7 luglio, potrebbero non essere ancora nel
> tarball pubblicato). In tal caso verificare `pip index versions
unsloth` per una release più recente, oppure installare direttamente da
> Git (`unsloth @ git+https://github.com/unslothai/unsloth.git@main`) come
> soluzione ponte.

---

## 3. Coerenza SFT → Rewards → GRPO

Verificata l'intera catena di responsabilità:

- **Prompting**: `src/utils/prompting.py` centralizza `build_t2g_prompt()`,
  usato identicamente da `_prepare_sft_dataset` (SFT) e
  `_prepare_t2g_dataset` (GRPO). ✅ Nessuna incoerenza: stesso
  `SYSTEM_PROMPT`, stesso `chat_template`, stesso fallback manuale ChatML.
- **Estrazione testo**: `src/utils/text_utils.py` (`extract_gloss_text`,
  `extract_user_text`) è condiviso da rewards, callback e metriche. ✅
  Nessuna duplicazione logica divergente.
- **Caricamento modello / merge adapter**: `src/models/model_loader.py`
  usa due percorsi diversi tra SFT (`_load_with_transformers`, con
  `PeftModel.from_pretrained` + `merge_and_unload()` espliciti) e GRPO via
  Unsloth (`_load_with_unsloth`, che passa direttamente la cartella
  dell'adapter SFT a `FastLanguageModel.from_pretrained` e lascia a Unsloth
  la risoluzione PEFT). Il codice di `unsloth/models/loader.py` (verificato
  a monte) rileva correttamente `adapter_config.json` +
  `base_model_name_or_path` e applica `PeftModel.from_pretrained` +
  merge internamente — quindi il comportamento è **funzionalmente
  equivalente**, non un bug. ⚠️ Resta comunque un punto di fragilità
  architetturale: i due percorsi di caricamento (`transformers` puro vs.
  `Unsloth`) non sono testati per bit-identical output; piccole differenze
  di dtype/quantizzazione nel merge potrebbero introdotre drift silenzioso
  tra il modello valutato "sulla carta" dopo SFT e quello effettivamente
  usato come punto di partenza per GRPO. **Raccomandazione**: loggare un
  checksum (es. media/std dei pesi di un layer campione) subito dopo il
  merge in entrambi i percorsi, per verificarne l'equivalenza empirica.
- **Reward functions** (`src/rewards/t2g_rewards.py`): la combinazione
  pesata (`translation=0.40, gold_structure=0.40, format=0.10,
repetition=0.10`) è coerente con le docstring e il design (Viterbi puro
  è correttamente lasciato a peso 0/experimental perché degenera in loop
  senza emission probabilities — commento esplicito nel codice). Nessuna
  incoerenza di segno o normalizzazione tra le componenti: tutte restituiscono
  valori in range compatibili (`[0,1]` o `[-1,1]` per la repetition).

**Conclusione**: la catena SFT→rewards→GRPO è internamente coerente. Il
problema di qualità non risiede in una rottura di questa catena, ma
altrove (vedi §5).

---

## 4. Constrained decoding: bug nel Trie di fallback

File: `src/grammar/grammar_logits_processor.py`,
`GlossVocabularyLogitsProcessor.__call__`:

```python
node = self.root
for tok in gen_tokens:
    if tok in node.children:
        node = node.children[tok]
    elif node.is_terminal and tok in self.root.children:
        node = self.root.children[tok]
    else:
        node = self.root  # Fallback on mismatch
```

**Problema**: quando né `tok in node.children` né
`(node.is_terminal and tok in self.root.children)` sono veri, il codice
torna alla radice (`self.root`) **senza ri-processare `tok`** contro la
radice. Questo significa che il token realmente emesso (e già presente in
`input_ids`) viene "perso" nella ricostruzione dello stato del Trie: al
prossimo step il Trie ripartirà da uno stato che non riflette l'ultimo
token generato, potenzialmente permettendo (o vietando) token in modo
scorretto per uno o più step successivi, finché la storia non si
resincronizza casualmente.

In pratica questo può accadere ogni volta che il modello genera un token
di spaziatura/punteggiatura imprevisto o una sequenza gloss che non è un
prefisso esatto registrato nel Trie (es. per via di un tokenizzatore BPE
che spezza diversamente lo stesso gloss in contesti diversi). L'effetto
netto è una graduale "deriva" dello stato del vincolo rispetto alla
sequenza reale, che può manifestarsi come output apparentemente
grammaticali step-by-step ma che in realtà seguono un vincolo starato.

**Fix raccomandato** (da applicare, non ancora fatto in questa sessione
per rimanere focalizzati sulla diagnosi del crash prioritario):

```python
node = self.root
for tok in gen_tokens:
    if tok in node.children:
        node = node.children[tok]
    elif node.is_terminal and tok in self.root.children:
        node = self.root.children[tok]
    elif tok in self.root.children:
        # Fallback: il token stesso può iniziare un nuovo gloss dalla radice
        node = self.root.children[tok]
    else:
        node = self.root  # Nessun match: stato indefinito, riparti da vuoto
```

Questo garantisce che un token valido come inizio di un nuovo gloss non
venga scartato solo perché non segue immediatamente un nodo terminale.

---

## 5. Analisi qualità generale (bassa qualità delle generazioni)

Cause candidate, in ordine di impatto stimato:

1. **Crash silenzioso/degradazione da Unsloth obsoleto (§2)** — se il
   training arrivava a completare comunque alcuni step prima del crash (o
   se in versioni precedenti del codice il crash non si manifestava ma
   RoPE produceva comunque posizioni leggermente scorrette), è plausibile
   che **anche i run "riusciti"** abbiano sofferto di posizionamento RoPE
   corrotto in fase di generazione durante il training, con reward
   rumorosi/non correlati alla vera qualità linguistica. Questo è il primo
   sospetto da verificare dopo il fix: la qualità potrebbe migliorare
   semplicemente eliminando questa fonte di rumore.
2. **Bug del Trie (§4)** — deriva di stato del vincolo può permettere
   occasionalmente token fuori vocabolario o vietare token legittimi,
   introducendo rumore nei rollout usati per il GRPO.
3. **Reward `gold_structure` basato su bigrammi** — è un proxy strutturale
   ragionevole ma resta un modello del secondo ordine (bigram); non
   catturerà errori di ordine delle glosse a lungo raggio. Il peso 0.40
   condiviso con `translation` (ROUGE-L) è sensato come bilanciamento, ma
   ROUGE-L stesso è un proxy lessicale debole per l'ordine sintattico ASL.
   **Raccomandazione**: considerare l'aggiunta di una metrica basata su
   edit-distance sull'ordine delle glosse (es. word-level Levenshtein
   normalizzato) come reward aggiuntivo o in sostituzione parziale di
   ROUGE-L, che è pensato per riassunti in linguaggio naturale, non per
   sequenze di gloss brevi e altamente strutturate.
4. **Quantizzazione 4-bit + merge**: il modello SFT viene salvato come
   adapter LoRA su base 4-bit; il caricamento per GRPO ri-carica e fa
   merge. Ogni merge di un adapter LoRA su pesi quantizzati introduce un
   piccolo errore di ri-quantizzazione. Non è stato possibile verificare
   empiricamente la magnitudo di questo drift in questa sessione (nessun
   accesso al cluster), ma è un sospetto secondario coerente con
   "funzionava meglio prima delle modifiche".
5. **Modifiche recenti dell'utente**: poiché il problema è stato descritto
   come una regressione ("prima funzionava"), e la causa del crash è stata
   isolata in una dipendenza esterna (non nel codice del progetto), è
   probabile che le "modifiche fatte" dall'utente non siano la causa
   diretta del crash — a meno che tali modifiche non abbiano
   involontariamente aggiornato l'ambiente (es. rigenerazione
   dell'ambiente cluster con una versione più recente di `transformers`,
   che ha esposto il bug latente in Unsloth 2026.3.17). Si raccomanda di
   verificare, se possibile, la history di `transformers.__version__`
   installata sul cluster nelle settimane recenti.

---

## 6. Checklist di verifica raccomandata (post-fix, sul cluster)

1. `pip install --user -e . --upgrade --retries 10 --timeout 60` per
   aggiornare Unsloth alla nuova versione minima.
2. Rilanciare la Fase 2 (GRPO) e verificare che il crash su
   `LlamaAttention_fast_forward_inference` non si ripresenti.
3. Monitorare le metriche W&B di `masked_mass` / `entropy` /
   `allowed_entropy` esposte da `CompletionSampleCallback` nei primi step
   per verificare che il constrained decoding stia effettivamente
   vincolando come previsto (nessuna deriva anomala).
4. Se la qualità resta bassa dopo il fix del crash, applicare il fix del
   Trie (§4) come secondo intervento e ripetere un run di confronto A/B.
5. Considerare l'aggiunta di un reward basato su edit-distance sull'ordine
   delle glosse, come discusso in §5.3.

---

## 7. File modificati in questa sessione

- `neuro_symbolic_t2g/pyproject.toml` — bump `unsloth==2026.3.17` →
  `unsloth>=2026.7.1` (versione più recente pubblicata su PyPI al
  2026-07-07).
- `neuro_symbolic_t2g/docs/T2G_PIPELINE_REVIEW.md` — questo documento
  (nuovo).

Nessun'altra modifica al codice è stata applicata in questa sessione. Il
bug del Trie (§4) è documentato ma **non corretto**, per mantenere il
focus sulla causa primaria del crash come richiesto; può essere applicato
come intervento successivo se l'utente lo desidera.
