# Revisione Pipeline T2G — SFT → Rewards → GRPO → Constrained Decoding

> Documento di analisi tecnica su `neuro_symbolic_t2g/`. Non copre `grammarllm/`
> o `grpo-strict-generation/` (fuori scope, per richiesta esplicita).
> Data: luglio 2026.

## 1. Riassunto esecutivo

| Problema                                                                                                                                      | Causa individuata                                                                                                                                                                                                                                                                                                                                                                                                                                         | Stato                                                                                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Crash in Fase 2 (GRPO) — `RuntimeError: output with shape [8, 14, 1, 64] doesn't match the broadcast shape [8, 14, 73, 64]`                   | `unsloth==2026.3.17` pinnato in `pyproject.toml` è troppo vecchio: non contiene i fix upstream per la gestione di `position_ids` durante il decode incrementale su `transformers>=5.0`                                                                                                                                                                                                                                                                    | ✅ **Fix applicato e confermato** (bump a `2026.7.1`, run cluster passato oltre questo punto — vedi §2.4)                                      |
| Crash in GRPO Training (step 7, ricorrente) — `RuntimeError: self and mat2 must have the same dtype, but got Half and Float` in `matmul_lora` | **Causa reale**: `unsloth_zoo.rl_replacements.grpo_accumulated_loss` legge la env var raw `ACCELERATE_MIXED_PRECISION` (mai settata dal progetto) e di default usa `torch.float16` per l'autocast del forward GRPO, in conflitto con i pesi reali in bfloat16. Il primo workaround tentato (`_align_lora_dtype_to_base`, §2.5) affrontava un sintomo statico errato e **non ha risolto il crash** (ricorso identico confermato da un secondo run cluster) | ✅ **Fix corretto applicato** (env var + monkeypatch diretto su `trainer._autocast_dtype`, vedi §2.6) — da verificare sul prossimo run cluster |
| Bug di sincronizzazione nel Trie di `GlossVocabularyLogitsProcessor`                                                                          | Il fallback su mismatch scarta il token invece di ri-testarlo contro la radice                                                                                                                                                                                                                                                                                                                                                                            | ✅ **Fix implementato** (vedi §4)                                                                                                              |
| Qualità generale bassa                                                                                                                        | Multi-fattoriale: vedi §5                                                                                                                                                                                                                                                                                                                                                                                                                                 | ✅ **Raccomandazioni implementate** (checksum merge §3, reward edit-distance §5.3)                                                             |

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

### 2.4 ✅ Confermato: il crash originale è risolto

Run sul cluster (Job ID 6126, 2026-07-07) con `unsloth==2026.7.1`,
`transformers==5.3.0`: **SFT pre-training completato senza errori**
(1250/1250 step) e **GRPO Step 1–6 completati** (data prep, model
loading, constrained-decoding setup, dataset, reward functions, config).
Il rollout di generazione (`_patched_generate`) non ha più prodotto il
`RuntimeError: output with shape [...] doesn't match the broadcast shape
[...]` — il bump di versione ha risolto il problema originale.

### 2.5 🆕 Nuovo crash (diverso): dtype mismatch in `matmul_lora` durante GRPO

Con la Fase 2 ora funzionante fino a **STEP 7 (GRPO Training)**, è emerso
un **secondo crash, distinto e successivo**, al primo step di training
(dentro il forward compilato di `grpo_accumulated_loss`):

```
RuntimeError: self and mat2 must have the same dtype, but got Half and Float
    File ".../unsloth/kernels/utils.py", line 1107, in matmul_lora
        out.addmm_(XA, B.to(dtype), alpha = s)
```

Traceback: `LoraLayer.forward → LoRA_MLP.apply → matmul_lora`, invocato
dentro il **gradient-checkpointing recompute** di Unsloth
(`unsloth_zoo/gradient_checkpointing.py:UnslothCheckpointFunction`)
durante `LlamaDecoderLayer_fast_forward → self.mlp(hidden_states) →
apply_lora_mlp_swiglu`.

**Causa**: bug upstream noto e **ancora non risolto** in Unsloth —
[issue #4891](https://github.com/unslothai/unsloth/issues/4891)
("`RuntimeError: self and mat2 must have the same dtype (Half and
BFloat16) in matmul_lora during GRPO training with 4-bit quantization`",
aperta 2026-04-07) con relativa
[PR #4918](https://github.com/unslothai/unsloth/pull/4918) **ancora in
review, non mergiata** al 2026-07-07 (3 mesi dopo l'apertura).

Causa radice (dalla discussione della issue): con base model in bnb-4bit
e attivazioni in bfloat16, `fast_dequantize` può restituire il peso base
dequantizzato nel dtype embedded nel `quant_state` del checkpoint (spesso
float16 di default), mentre i pesi degli adapter LoRA A/B appena creati
seguono il `dtype` richiesto al load (bfloat16). Il kernel fuso
`matmul_lora` fa quindi `out.addmm_(XA, B.to(dtype), alpha=s)` dove `out`
(dal matmul sul peso base) è float16 ma `B.to(dtype)` è bfloat16 →
crash. Questo si manifesta specificamente con `GRPOTrainer` (chunked
loss via `grpo_accumulated_loss`) combinato con il gradient checkpointing
"smart" di Unsloth — l'SFT pre-training, che usa la stessa configurazione
LoRA (`use_gradient_checkpointing="unsloth"`), **non** attiva questo
percorso di codice e infatti ha completato senza errori.

**Fix applicato** (workaround locale, dato che upstream non ha ancora
mergiato una fix): in `src/models/model_loader.py`, nuova funzione
`_align_lora_dtype_to_base()` chiamata subito dopo la creazione degli
adapter LoRA (sia nel path Unsloth `_load_with_unsloth` sia nel path
HuggingFace standard `apply_lora`). Cammina tutti i moduli `LoraLayer` e
forza il dtype di `lora_A`/`lora_B` a corrispondere al dtype del peso del
layer base sottostante, eliminando la disallineamento dtype alla radice
prima che `matmul_lora` venga invocato.

```python
def _align_lora_dtype_to_base(model: Any) -> None:
    for module in model.modules():
        if isinstance(module, LoraLayer):
            target_dtype = module.get_base_layer().weight.dtype
            for adapter_dict in (module.lora_A, module.lora_B):
                for sub in adapter_dict.values():
                    if sub.weight.dtype != target_dtype:
                        sub.to(target_dtype)
```

Questo fix è **difensivo e a costo quasi zero** (un singolo cast one-shot
dopo la creazione degli adapter, non ripetuto ad ogni forward) e diventa
un no-op automatico quando/se PR #4918 verrà mergiata in una release
futura di Unsloth.

**⚠️ AGGIORNAMENTO — questo fix NON ha risolto il problema.** Un secondo
run cluster (stesso job, versioni identiche) ha mostrato **lo stesso
identico crash**, alla stessa riga esatta (`utils.py:1107`,
`out.addmm_(XA, B.to(dtype), alpha = s)`), con lo stesso traceback
strutturale — questa volta al GRPO **step 7** (dopo che gli step 1-6
erano completati correttamente). Il fatto che il crash sia
**perfettamente identico** nonostante il fix di `_align_lora_dtype_to_base`
fosse attivo indica che la diagnosi era **incompleta**: quel fix
correggeva un possibile mismatch _statico_ nei parametri salvati, ma il
crash reale è causato da uno stato _dinamico_ — il contesto di autocast
usato durante il forward pass stesso. La causa radice effettiva è
descritta in §2.6. Il fix di questa sezione **rimane in codice** come
misura difensiva aggiuntiva (è innocuo), ma non è la fix risolutiva.

---

### 2.6 🔧 Causa radice reale: `ACCELERATE_MIXED_PRECISION` mai settata → autocast float16 in GRPO

**Scoperta**: analizzando il codice sorgente di `unsloth-zoo` (funzione
`grpo_accumulated_loss` in `unsloth_zoo/rl_replacements.py`, materializzata
localmente come `unsloth_compiled_cache/UnslothGRPOTrainer.py` sul
cluster — visibile nel traceback dell'utente), si trova questa
inizializzazione pigra (eseguita una sola volta, al primo step di
training):

```python
if not hasattr(trainer, '_autocast_dtype'):
    trainer._autocast_dtype = (
        torch.float16
        if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16'
        else torch.bfloat16
    )
    if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1':
        trainer._autocast_dtype = None
```

Questo `trainer._autocast_dtype` viene poi usato per avvolgere **tutte**
le forward pass del calcolo della loss GRPO in un
`torch.amp.autocast(device_type=..., dtype=trainer._autocast_dtype)`, che
raggiunge `LoRA_MLP.forward` (decorato `@torch_amp_custom_fwd`) →
`matmul_lora`, dove `dtype = X.dim()`/`X.dtype` viene catturato
dall'input effettivamente autocastato.

**Il problema**: questo codice legge la env var **raw**
`ACCELERATE_MIXED_PRECISION` direttamente da `os.environ`, bypassando
completamente l'API propria di HF Accelerate
(`AcceleratorState().mixed_precision`). Tracciando la catena
`GRPOConfig(bf16=True)` → `TrainingArguments.__post_init__` (che imposta
`self.mixed_precision = "bf16"` come **attributo Python**, non env var)
→ `Trainer.create_accelerator_and_postprocess()` (che passa
`mixed_precision=self.args.mixed_precision` **direttamente** al
costruttore `Accelerator(mixed_precision="bf16")`) — si scopre che
**questo percorso non scrive mai** `os.environ["ACCELERATE_MIXED_PRECISION"]`.
Quella env var viene impostata _solo_ dal CLI `accelerate launch` o da
DeepSpeed (confermato leggendo il codice sorgente di
`huggingface/accelerate`, funzioni `prepare_simple_launcher_cmd_env`,
`prepare_multi_gpu_env`, `prepare_deepspeed_cmd_env` in
`accelerate/utils/launch.py`) — nessuno dei due è usato da questo
progetto, che lancia lo script direttamente con
`python -m src.training ...` (confermato via `grep_search`: zero
occorrenze di `ACCELERATE_MIXED_PRECISION` in tutto il workspace).

**Conseguenza**: la env var non è mai settata → il default `'fp16'` in
`os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16')` viene sempre
usato → `trainer._autocast_dtype = torch.float16` → il forward pass GRPO
viene eseguito sotto autocast **float16**, in conflitto con i pesi reali
del modello (bfloat16, sia base sia adapter LoRA) → il kernel fuso
`matmul_lora` produce un mismatch di dtype in `out.addmm_(...)`.

**Fix applicato** in `src/training/grpo_t2g_train.py`:

1. **Fix della causa radice**: subito dopo la costruzione di
   `grpo_config` (prima della creazione di `GRPOTrainer`), impostare
   esplicitamente la env var:

   ```python
   os.environ["ACCELERATE_MIXED_PRECISION"] = (
       "bf16" if grpo_config.bf16 else "fp16"
   )
   ```

2. **Difesa aggiuntiva**: subito dopo la costruzione di `GRPOTrainer`,
   impostare direttamente l'attributo sull'istanza del trainer, per
   rendere il controllo pigro di `grpo_accumulated_loss`
   (`if not hasattr(trainer, '_autocast_dtype')`) un no-op
   indipendentemente da eventuali timing/caching della propagazione
   della env var:

   ```python
   trainer._autocast_dtype = (
       torch.bfloat16 if grpo_config.bf16 else torch.float16
   )
   ```

Entrambe le fix sono minime, non invasive, e coerenti con lo stile già
usato nel file per altri workaround (`_patched_generate`). Il fix di
§2.5 (`_align_lora_dtype_to_base`) **rimane attivo** come ulteriore
livello di difesa (innocuo, costo quasi zero).

**Da verificare sul cluster**: rieseguire il training GRPO. Se il crash
persiste nonostante entrambi i fix, il prossimo passo sarebbe ispezionare
direttamente il file `unsloth_compiled_cache/UnslothGRPOTrainer.py`
materializzato sul nodo del cluster per confermare che la logica
corrisponda esattamente a quella qui documentata (potrebbe esistere uno
scostamento di versione tra `unsloth-zoo` installato e quanto letto da
GitHub `main`).

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

  ✅ **Implementato**: `_log_merge_checksum(model, label)` in
  `src/models/model_loader.py`, chiamata subito dopo `merge_and_unload()`
  in entrambi i path (`_load_with_transformers` con `label="transformers"`
  e `_load_with_unsloth` con `label="unsloth"`). Sceglie il primo peso
  `q_proj`/`k_proj`/`v_proj`/`o_proj` non-LoRA trovato in
  `model.named_parameters()` (fallback: primo parametro con >1 dimensione)
  e loggia `mean`/`std`/`dtype`/`numel` con prefisso `[merge-checksum:<label>]`.
  Puramente diagnostico (non solleva mai eccezioni). **Come verificare sul
  cluster**: confrontare i due log emessi durante SFT (`transformers`) e
  durante GRPO (`unsloth`) per lo stesso adapter — `mean`/`std` dovrebbero
  coincidere (a meno di un piccolo errore di ri-quantizzazione 4-bit, vedi
  §5 punto 4).

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

**✅ Fix implementato** in `src/grammar/grammar_logits_processor.py`,
`GlossVocabularyLogitsProcessor.__call__`:

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

**Verifica**: `tests/test_grammar.py` sezione 2
(`GlossVocabularyLogitsProcessor`) passa integralmente (7/7) dopo il fix;
nessuna regressione osservata. Le uniche sezioni fallite nella test suite
(§5 masked-mass tracking, §6 PDA grammar) sono preesistenti e indipendenti
dal Trie (mancata attivazione di `track_diagnostics=True` nel test, e un
import rotto in `grammarllm/`, fuori scope).

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

   ✅ **Implementato**: nuova funzione `gloss_order_reward(completion,
gold_gloss)` in `src/rewards/t2g_rewards.py`, che calcola la distanza
   di Levenshtein **word-level** (non a caratteri) tra la sequenza gloss
   generata e quella gold, normalizzata come
   `1 - edit_distance / max(len(gen), len(gold))` → `1.0` per match
   esatto (ordine e contenuto), `0.0` per sequenze completamente diverse
   o input vuoti. Registrata in `build_t2g_reward_functions` sotto la
   nuova chiave di peso `weight_gloss_order` (opt-in: peso `0.0` di
   default, quindi **nessun cambio di comportamento se non richiamata
   esplicitamente** — il default di `build_t2g_reward_functions()` senza
   argomenti resta a 4 funzioni, invariato). Abilitata esplicitamente nei
   config `experiments/configs/t2g/grpo_qwen05.yaml` e `.../sft.yaml` con
   pesi ribilanciati:
   `weight_translation=0.30, weight_gold_structure=0.35,
weight_gloss_order=0.15, weight_format=0.10, weight_repetition=0.10`
   (somma 1.0) — il peso sottratto a `weight_translation` (0.40→0.30) è
   parzialmente compensato dal nuovo reward, come suggerito sopra
   ("aggiuntivo o in sostituzione parziale di ROUGE-L").

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
4. Il fix del Trie (§4) è già applicato — ripetere un run di confronto A/B
   se la qualità resta bassa dopo il fix del crash, per isolarne l'impatto
   specifico.
5. Il reward basato su edit-distance sull'ordine delle glosse (§5.3) è
   già abilitato nei config GRPO/SFT di default — monitorare su W&B il
   nuovo componente `gloss_order_reward` separatamente dagli altri per
   verificarne il contributo reale, e confrontare con un run A/B a
   `weight_gloss_order: 0.0` (ripristinando i pesi originali) se necessario.
6. Confrontare i log `[merge-checksum:transformers]` e
   `[merge-checksum:unsloth]` (§3) emessi rispettivamente durante SFT e
   GRPO per lo stesso adapter, per verificare l'assenza di drift
   significativo tra i due backend di caricamento.

---

## 7. File modificati in questa sessione

- `neuro_symbolic_t2g/pyproject.toml` — bump `unsloth==2026.3.17` →
  `unsloth>=2026.7.1` (versione più recente pubblicata su PyPI al
  2026-07-07). **Confermato**: ha risolto il crash originale su
  `position_ids` (§2.4).
- `neuro_symbolic_t2g/src/models/model_loader.py` — aggiunta funzione
  `_align_lora_dtype_to_base()`, chiamata dopo la creazione degli adapter
  LoRA in entrambi i path (`_load_with_unsloth` e `apply_lora`).
  Workaround difensivo per il bug upstream Unsloth #4891 (dtype mismatch
  in `matmul_lora` durante GRPO) — vedi §2.5. **Nota**: da solo non ha
  risolto il crash ricorrente; rimane attivo come difesa aggiuntiva.
- `neuro_symbolic_t2g/src/training/grpo_t2g_train.py` — **(nuovo, questa
  sessione)** due fix per la causa radice reale del crash dtype in GRPO
  (§2.6):
  1. `os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16" if grpo_config.bf16 else "fp16"`,
     impostato subito dopo la costruzione di `grpo_config` (prima di
     `GRPOTrainer`).
  2. `trainer._autocast_dtype = torch.bfloat16 if grpo_config.bf16 else torch.float16`,
     impostato direttamente sull'istanza del trainer subito dopo la sua
     costruzione, come difesa aggiuntiva indipendente dalla propagazione
     della env var.
- `neuro_symbolic_t2g/docs/T2G_PIPELINE_REVIEW.md` — questo documento.

Il bug del Trie (§4) è documentato ma **non corretto**, per mantenere il
focus sulla causa primaria del crash come richiesto; può essere applicato
come intervento successivo se l'utente lo desidera.

**Stato al 2026-07-07**: due crash distinti individuati durante lo stesso
run. Il primo (position_ids) è risolto e confermato dal bump di versione
Unsloth. Il secondo (dtype mismatch in `matmul_lora` durante GRPO) ha
richiesto **due iterazioni di diagnosi**: il primo workaround
(`_align_lora_dtype_to_base`, §2.5) si è rivelato insufficiente — un
secondo run cluster ha mostrato il crash ricorrere in modo identico. La
causa radice effettiva è stata isolata in `unsloth-zoo`
(`grpo_accumulated_loss` legge `ACCELERATE_MIXED_PRECISION` da
`os.environ` invece di usare l'API propria di Accelerate, e la env var
non è mai settata da questo progetto) e corretta con un fix a doppio
livello in `grpo_t2g_train.py` (§2.6) — **da confermare sul prossimo run
cluster**.

---

## 8. Miglioramenti implementati (sessione successiva, 2026-07-07)

Su richiesta esplicita, le tre raccomandazioni lasciate in sospeso nella
sessione precedente (§§3, 4, 5.3) sono state implementate:

- `neuro_symbolic_t2g/src/grammar/grammar_logits_processor.py` — fix del
  bug di sincronizzazione del Trie (§4): aggiunto il ramo
  `elif tok in self.root.children: node = self.root.children[tok]` nel
  fallback di `GlossVocabularyLogitsProcessor.__call__`, così che un
  token che può iniziare un nuovo gloss dalla radice non venga scartato
  solo perché il nodo precedente non era terminale. Verificato con
  `tests/test_grammar.py` (sezione 2, 7/7 pass, nessuna regressione).
- `neuro_symbolic_t2g/src/models/model_loader.py` — nuova funzione
  `_log_merge_checksum(model, label)` (§3), chiamata subito dopo
  `merge_and_unload()` in `_load_with_transformers` (`label="transformers"`)
  e in `_load_with_unsloth` (`label="unsloth"`). Loggia mean/std/dtype di
  un layer di attenzione campione per rilevare drift tra i due backend di
  caricamento. Puramente diagnostico, non solleva mai eccezioni.
- `neuro_symbolic_t2g/src/rewards/t2g_rewards.py` — nuova funzione
  `gloss_order_reward(completion, gold_gloss)` (§5.3): distanza di
  Levenshtein word-level normalizzata tra gloss generato e gold,
  sensibile all'ordine (a differenza di ROUGE-L e dei reward a bigrammi).
  Registrata in `build_t2g_reward_functions` sotto la chiave opt-in
  `weight_gloss_order` (default `0.0` — nessun cambio di comportamento se
  non richiamata esplicitamente; `build_t2g_reward_functions()` senza
  argomenti resta a 4 funzioni come prima).
- `neuro_symbolic_t2g/experiments/configs/t2g/grpo_qwen05.yaml` e
  `.../sft.yaml` — abilitato `weight_gloss_order: 0.15` con pesi
  ribilanciati (`weight_translation: 0.40→0.30`,
  `weight_gold_structure: 0.40→0.35`, `weight_format`/`weight_repetition`
  invariati a `0.10` ciascuno; somma pesi = 1.0).
- `neuro_symbolic_t2g/docs/T2G_PIPELINE_REVIEW.md` — questo documento,
  aggiornato per marcare le tre raccomandazioni come implementate.

**Verifica eseguita in locale** (nessun accesso GPU/cluster in questa
sessione): `tests/test_rewards.py` → 60/61 pass (l'unico fallimento,
`Viterbi Distance Reward: Raw < 0.0`, è preesistente e non correlato a
queste modifiche); `tests/test_grammar.py` → 55/62 pass (fallimenti
preesistenti in sezioni non toccate: masked-mass tracking con
`track_diagnostics` non abilitato nel test, e un import rotto in
`grammarllm/`, fuori scope per questo progetto). Uno script ad-hoc ha
confermato il comportamento atomico di `gloss_order_reward` (match
perfetto=1.0, riordino/parziale in valori intermedi coerenti, input vuoti
=0.0) e che `build_t2g_reward_functions()` di default resta a 4 funzioni
(nessuna regressione), mentre con `weight_gloss_order>0` produce 5
funzioni con pesi che sommano a 1.0.

**Non ancora verificato sul cluster**: l'equivalenza empirica dei
checksum di merge tra i due backend (§3) e l'impatto reale del nuovo
reward sulla qualità delle generazioni (§5.3) — richiede un run GRPO
completo.
