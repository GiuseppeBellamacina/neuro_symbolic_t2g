# Config Reference — chiave per chiave, verificata sul codice

Riferimento completo delle chiavi YAML di `experiments/configs/qwen25-05b/`.
Ogni voce riporta: tipo, valori ammessi, default (e DOVE è definito: `base.yaml`
o fallback nel codice), a cosa serve, e **chi la legge** con `file:riga` verificate
su `src/**/*.py`. Il tipo e i valori ammessi sono dedotti dal codice che consuma
la chiave e dai vincoli di `tests/validate_configs.py`, non dai valori di esempio.

Fonti citate (abbreviazioni usate sotto):

| Abbrev. | File |
| --- | --- |
| `bootstrap` | `src/training/__main__.py` |
| `grpo_train` | `src/training/grpo_t2g_train.py` |
| `sft_train` | `src/training/sft_train.py` |
| `eval` | `src/training/eval_t2g.py` |
| `loader` | `src/models/model_loader.py` |
| `aslg` | `src/datasets/aslg_dataset.py` |
| `retrieval_setup` | `src/training/retrieval_setup.py` |
| `retriever` | `src/retrieval/example_retriever.py` |
| `rewards` | `src/rewards/t2g_rewards.py` |
| `aux` | `src/training/auxiliary_sft_trainer.py` |
| `validator` | `tests/validate_configs.py` |
| `TRL grpo_config` | `.venv/.../trl/trainer/grpo_config.py` (0.24.0) |
| `TRL grpo_trainer` | `.venv/.../trl/trainer/grpo_trainer.py` (0.24.0) |

---

## Come funziona l'ereditarietà

Ogni config di cella dichiara `extends: <path>` verso il padre (tipicamente
`base.yaml`). La risoluzione avviene in `src/utils/config.py`:

- `resolve_config` (`src/utils/config.py:87`) risolve la catena ricorsivamente,
  con percorsi relativi alla directory del file che dichiara `extends`
  (`src/utils/config.py:143`); rileva i cicli (`src/utils/config.py:122-124`).
- Il merge è **profondo e ricorsivo** (`_deep_merge`, `src/utils/config.py:56-71`):
  i dict annidati vengono fusi chiave per chiave, **liste e scalari sono
  sostituiti** (mai concatenati). Il figlio vince sempre.
- Il dict fuso **non contiene mai** la chiave `extends` (`src/utils/config.py:128`):
  i trainer non la vedono mai.
- Un file senza `extends` equivale a un `yaml.safe_load` puro
  (`src/utils/config.py:96-97`).

**Regola d'organizzazione**: i valori comuni a tutte le celle (modello, LoRA,
dataset, stack di reward, sezione `evaluation`, `grammar`, `retrieval`,
`wandb`) stanno in `base.yaml`; ogni figlio dichiara **solo le proprie
differenze** (tipicamente `training.max_steps`/`learning_rate`/`output_dir`,
il prompting via `retrieval.enabled`, e gli ablation knob).
`base.yaml` è un template di ereditarietà, non eseguibile: il validator lo salta
(`validator:549-554`) e ogni cella lo estende.

Conseguenza operativa: le celle eval-only (`baseline/*`) ereditano da
`base.yaml` anche `training.max_steps`/`num_train_epochs`, quindi **la presenza
di questi step non distingue più nulla**: è la presenza di
`training.output_dir` a segnare una cella addestrabile (vedi `training.output_dir`).

---

## `model` (base.yaml:30-36)

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `model.name` | str | hub id HF o path locale | nessuno (obbligatoria, `validator:66`) | Identità del base model: load tokenizer/modello, fingerprint SFT, fingerprint contesto prompt eval. | `loader:357` (transformers), `loader:487,499` (unsloth), `eval:845,865` (eval base model), `sft_train:282` (fingerprint), `eval:306` (fingerprint baseline) |
| `model.num_gpus` | int | int ≥ 1 | `1` (fallback codice `bootstrap:97`) | Se > 1 il bootstrap **disabilita Unsloth** (incompatibile con multi-GPU) prima di qualunque import. Nessun data parallel reale: restare a 1. | `bootstrap:97-99` |
| `model.quantization` | str \| null | `"4bit"` \| `"8bit"` \| `"none"`/null | `"4bit"` (fallback `loader:344,476`) | Seleziona la quantizzazione BitsAndBytes: `"4bit"` → nf4 + double quant; `"8bit"` → load_in_8bit; altro/null → nessuna. **Con un adapter_path fornito viene forzata a `"none"`** per permettere il merge in memoria (`loader:345-349,479-485`). | `loader:344` (transformers), `loader:476-477` (unsloth), `loader:172-186` (`get_quantization_config`) |
| `model.dtype` | str | nome attributo `torch` (es. `bfloat16`, `float16`) | `"bfloat16"` (fallback `loader:358-359`) | Compute dtype del load e del quant config **solo nel path transformers**. Il path Unsloth passa `dtype=None` (auto-detect) e **lo ignora** (`loader:502`). | `loader:358-359` (transformers); `sft_train:282` (fingerprint) |
| `model.use_unsloth` | bool | true \| false | `False` (fallback `loader:580`) | Se true: importa Unsloth PRIMA di torch/transformers/trl (`bootstrap:105-109`, ordine richiesto per le ottimizzazioni) e carica il modello via `FastLanguageModel`. | `bootstrap:105-109`, `loader:580-585` |
| `model.max_seq_length` | int | int | `2048` (fallback `loader:500`) | Lunghezza massima della sequenza **solo nel path Unsloth** (`FastLanguageModel.from_pretrained`). Il path transformers la ignora. Attenzione: NON è `training.max_seq_length`, che è un'altra chiave (vedi sotto). Non è in `TYPE_CONSTRAINTS` del validator. | `loader:495-500` |
| `model.fast_inference` | — | — | — | **Non letta da nessuno.** Appare solo nella documentazione (README, CLUSTER.md): nessun config la dichiara e nessun modulo la consuma. Difetto documentale, non di config. | — |

---

## `lora` (base.yaml:39-52)

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `lora.r` | int | int | `16` (fallback `loader:376,539`) | Rank degli adapter LoRA. | `loader:376` (transformers), `loader:539` (unsloth); fingerprint `sft_train:288` |
| `lora.lora_alpha` | int | int | `32` (fallback `loader:377,540`) | Scaling α degli adapter. | `loader:377`, `loader:540`; fingerprint `sft_train:289` |
| `lora.lora_dropout` | float | float ∈ [0,1] | **diverso per backend**: `0.05` nel path transformers (`loader:378`), `0` nel path unsloth (`loader:541`) | Dropout sugli adapter. `base.yaml` dichiara `0.05`, quindi il valore esplicito copre entrambi i path. | `loader:378` (transformers), `loader:541` (unsloth) |
| `lora.target_modules` | list[str] | nomi di moduli linear del modello | transformers: `["q_proj","k_proj","v_proj","o_proj"]` (`loader:267-268`); unsloth: i 7 progetti (`loader:514-525`) | Dove montare gli adapter. **Le liste vengono SOSTITUITE dal merge**, mai fuse: un figlio che rideclara la lista perde gli elementi del padre. | `loader:379` (transformers), `loader:514-525` (unsloth); fingerprint `sft_train:290-293` |
| `lora.task_type` | str | `"CAUSAL_LM"` (qualsiasi stringa accettata, ma il solo valore sensato) | `"CAUSAL_LM"` (fallback `loader:380`) | Tipo di task PEFT. **Letto solo dal path transformers**: il path Unsloth lo ignora. | `loader:380` |
| `lora.random_state` | int | int | `3407` (fallback `loader:544`) | Seed di init degli adapter. **Letto solo dal path Unsloth**: nel path transformers PEFT non riceve alcun seed per l'init LoRA. | `loader:544` (unsloth); fingerprint `sft_train:292` |

---

## `dataset` (base.yaml:61-69)

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `dataset.dataset_name` | str | — | — | **Non letta funzionalmente.** L'id del dataset HF è hardcoded (`aslg:39`). La chiave entra SOLO nei fingerprint (provenienza): `sft_train:298` (fingerprint SFT), `eval:307` (fingerprint contesto prompt) e nella lista chiavi richieste del validator (`validator:67`). Cambiarla NON cambia il dataset scaricato. | `sft_train:298`, `eval:307` (solo fingerprint) |
| `dataset.dataset_cache` | str | path | `"data/aslg_pc12"` (`aslg:40,121`) | Directory cache HF del corpus scaricato. | `grpo_train:614`, `sft_train:599`, `eval:774` |
| `dataset.vocab_path` | str | path | `"data/gloss_vocab.txt"` (fallback `eval:776`) | Vocabolario glossa (il Trie e le reward lo caricano da qui). Cache con sidecar meta su seed/train_size (`grpo_train:624-631`). | `grpo_train:603,624-631`, `sft_train:587`, `eval:776` |
| `dataset.bigram_matrix_path` | str | path | `"data/bigram_transition.npy"` (fallback `eval:778`) | Matrice di transizione bigram per le reward strutturali e il bigram log-prob dell'eval. | `grpo_train:604,634-641`, `sft_train:588`, `eval:778` |
| `dataset.split` | str | split HF (atteso `"train"`) | `"train"` (fallback `grpo_train:326`) | Split usato per il **training** e per la costruzione del vocabolario/bigram. **L'eval lo ignora**: valuta sempre `dataset["test"]` (`eval:881`). | `grpo_train:326`, `sft_train:205` |
| `dataset.max_samples` | int \| null | int ≥ 1 \| null | `null` (`base.yaml:67`) | **TRONCA il train set alle prime N righe** (`aslg:293-294`: `rows = rows[:max_samples]`): serve solo per debug veloce. Null = tutto. Non influenza l'eval. ⚠️ Omonimia pericolosa con `evaluation.max_samples`, che ha significato OPPOSTO (sottocampiona i prompt di valutazione). | `grpo_train:327`, `sft_train:206` → troncamento in `aslg:293-294` |
| `dataset.seed` | int | int | `42` (fallback un po' ovunque, es. `grpo_train:593`) | Seed dell'intera pipeline: split deterministico 90/10 del corpus (`download_aslg_dataset`), seed random/numpy/torch, sottocampionamento seeded dell'eval, seed del retriever. Cambiare il seed invalida le cache vocab/bigram (sidecar meta, `grpo_train:400-444`). | `grpo_train:593,613-614`, `sft_train:577,598-599`, `eval:774,884`, `retrieval_setup` (seed param) |
| `dataset.thinking` | bool | true \| false | `false` (`base.yaml:69`) | **Inerte funzionalmente.** Entra solo nel fingerprint SFT (`sft_train:298`); la pulizia dei tag ` imp assaila`. Non c'è un mode "thinking" attivabile nel codice. | `sft_train:298` (solo fingerprint) |

---

## `training`

Sezione comune: `base.yaml:79-116`; i knob di durata/directory sono per-cella.

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `training.trainer` | str | `"sft"` \| `"grpo"` (default) | `"grpo"` (fallback `bootstrap:121`) | Router del bootstrap: `sft` → `sft_train.main`, altrimenti `grpo_t2g_train.main`. Legto PRIMA di importare torch, per scegliere il trainer. | `bootstrap:75,121-127`; `validator:196` |
| `training.per_device_train_batch_size` | int | int | GRPO: `1` (`grpo_train:165`); SFT: `4` (`sft_train:722`) | Batch micro per dispositivo. In GRPO con `num_generations: 8` e accum 8, il batch effettivo 8 = **1 prompt × 8 generazioni**. | `grpo_train:165`, `sft_train:722` |
| `training.per_device_eval_batch_size` | int | int | `8` (`sft_train:723`) | Batch dell'eval heldout **dell'SFT**. Il trainer GRPO non lo legge. | `sft_train:723` |
| `training.gradient_accumulation_steps` | int | int | GRPO: `8`; SFT: `4` | Accumulo gradiente. ⚠️ In TRL 0.24 `steps_per_generation = gradient_accumulation_steps` quando non impostato (`TRL grpo_config:648-650`): l'accumulo NON aumenta i prompt per passo. | `grpo_train:166`, `sft_train:724` |
| `training.lr_scheduler_type` | str | scheduler HF (es. `"cosine"`) | `"cosine"` | Scheduler del learning rate. | `grpo_train:168`, `sft_train:726` |
| `training.optim` | str | optim HF (es. `"paged_adamw_8bit"`) | `"paged_adamw_8bit"` | Ottimizzatore. | `grpo_train:170`, `sft_train:728` |
| `training.weight_decay` | float | float | `0.1` | Weight decay. | `grpo_train:171`, `sft_train:729` |
| `training.max_grad_norm` | float | float > 0 | GRPO: `0.1`; SFT: `1.0` | Clipping della norma del gradiente. NB: i fallback dei due trainer DIVERGONO. | `grpo_train:172`, `sft_train:730` |
| `training.bf16` | bool | true \| false | `True` | Mixed precision. Il valore pilota anche il workaround `ACCELERATE_MIXED_PRECISION` (`grpo_train:929`) e il dtype autocast (`grpo_train:1038,1059`). | `grpo_train:173,929,1038`, `sft_train:731` |
| `training.gradient_checkpointing` | bool | true \| false | `False` nel codice (`grpo_train:179`), `true` in `base.yaml:87` | Ricomputa le attivazioni nel backward (~20% più lento, molta meno VRAM). Essential con `num_generations: 8` su GPU 22GB. | `grpo_train:179`, `sft_train:738` |
| `training.logging_steps` | int | int | GRPO: `5`; SFT: `10` | Cadenza di logging delle metriche. | `grpo_train:180`, `sft_train:732,992` |
| `training.save_steps` | int | int | GRPO: `100`; SFT: `200` (base.yaml: `500`) | Cadenza dei checkpoint. `base.yaml` alza a 500 per ridurre l'I/O su NFS. | `grpo_train:181`, `sft_train:733` |
| `training.save_total_limit` | int | int | GRPO: `3`; SFT: `2` | Quanti checkpoint conservare (resume dopo TIMEOUT). | `grpo_train:182`, `sft_train:734` |
| `training.max_seq_length` | int | int | `768` (fallback `sft_train:735-737`) | `SFTConfig.max_length` (rinominata da `max_seq_length` in TRL 0.20+): troncamento del pack SFT. **Il GRPO non la legge** — la lunghezza dei rollout è governata da `grpo.max_prompt_length` + `grpo.max_completion_length`. Attenzione: diversa da `model.max_seq_length`. | `sft_train:735-737` |
| `training.max_steps` | int | int | `1500` (fallback `grpo_train:164`); `base.yaml:108` dichiara `5000` | **Governà SOLO il GRPO**: `GRPOConfig.max_steps`. ⚠️ È **1 PROMPT PER PASSO**, non 8: in TRL 0.24 `steps_per_generation = grad_accum`, quindi gli 8 micro-batch producono 8 rollout dello stesso prompt (`num_generations: 8`). Verificato su log reale: `epoch=0.02740514` a 2000 passi = 2000/72979. L'SFT **ignora** `max_steps` (`SFTConfig` non lo riceve, `sft_train:717-756`). | `grpo_train:164,1004` |
| `training.num_train_epochs` | int \| float | > 0 | `3` (fallback `sft_train:721`); `base.yaml:116` = `3` | **Governà SOLO l'SFT** (`SFTConfig.num_train_epochs`): passate sul train post-holdout. Il GRPO non lo legge. ⚠️ Viene EREDITATO anche dalle celle eval-only da `base.yaml` — la sua presenza non segnala nulla. | `sft_train:721` |
| `training.learning_rate` | float | float | GRPO: `5e-6`; SFT: `2e-5` | Learning rate (per-cella, dichiarato nei figli). | `grpo_train:167`, `sft_train:725` |
| `training.warmup_steps` | int | int ≥ 0 | GRPO: `50` (`grpo_train:139-143`); SFT: `100` (`sft_train:727`) | Warmup lineare in step. | `grpo_train:139-143,169`, `sft_train:727` |
| `training.warmup_ratio` | float | float ∈ [0,1] | — | **Non letta da nessuno.** È in `TYPE_CONSTRAINTS` (`validator:120`) ma nessun trainer la consuma: se la dichiari viene validata e poi ignorata. Difetto (dichiarabile ma inerte). | — |
| `training.seed` | int | int | fallback `dataset.seed` | Seed del trainer (non in base.yaml). | `grpo_train:161-163`, `sft_train:720` |
| `training.eval_fraction` | float | float ∈ [0,1] | `0.02` (`sft_train:220`) | Frazione del train SFT tenuta come holdout per eval_loss e early stopping. `<= 0` → eval disattivata ed early stopping disattivato (`sft_train:165-166,711-715`). | `sft_train:220-224` |
| `training.eval_steps` | int | int | `200` (`sft_train:751`) | Cadenza dell'eval heldout SFT. | `sft_train:751` |
| `training.early_stopping_patience` | int | int | `3` (`sft_train:983`) | Early stopping sull'eval_loss (solo se eval attiva). | `sft_train:981-985` |
| `training.sft_sample_every_n_steps` | int | int | `100` (`sft_train:993`) | Cadenza dei sample generati dal callback SFT. | `sft_train:988-995` |
| `training.output_dir` | str | path | nessuno (obbligatoria sulle celle addestrabili, `validator:68`) | Root dei checkpoint di run (`run_<timestamp>/`). ⚠️ **SEGNALE di tipo cella**: la sua ASSENZA dichiara una cella di sola valutazione — lo deduce l'eval (`eval:2307`, `eval:1253-1284`), la guardia del bootstrap (exit 2, `bootstrap:76-94`) e il validator (`validator:196-201`). | `grpo_train:529-547`, `sft_train:659` + i consumatori del segnale sopra |
| `training.log_dir` | str | path | nessuno (obbligatoria, `validator:68`) | Root dei log (tensorboard env var, wandb dir, output.log). | `grpo_train:529-548,156`, `sft_train:660,709` |
| `training.run_timestamp` | str | — | scritto a runtime (`grpo_train:571`) | Chiave **interna** iniettata dal trainer, mai dichiarare. | `grpo_train:149-152,571` |

---

## `grpo` (base.yaml:125-148)

Letti da `_build_grpo_config` / `_grpo_objective_kwargs` (`grpo_train:127-273`).

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `grpo.num_generations` | int | int; il batch effettivo (`per_device_bs × grad_accum`) deve essere divisibile | `4` nel codice (`grpo_train:184`); TRL default `8` (`TRL grpo_config:306-307`); `base.yaml:126` = `8` | G = completioni campionate per prompt, base dell'advantage group-relative. | `grpo_train:184` |
| `grpo.max_completion_length` | int | int | `256` (`grpo_train:185`); `base.yaml:127` = `128` | Tetto di token generati nei rollout. Entra anche nei `generation_kwargs` dei rollout (`grpo_train:473-476`, default lì `128`). | `grpo_train:185,473-476` |
| `grpo.max_prompt_length` | int | int | `256` (`grpo_train:186`); TRL default `512` (`TRL grpo_config:300-301`); `base.yaml:145` = `768` | I prompt dei rollout vengono **troncati** a questa lunghezza (`TRL grpo_trainer:1136,1175,1277`) — è qui che gli esempi few-shot vengono tagliati in silenzio se il valore è < 512. | `grpo_train:186,824`; guardia eval `eval:2290-2293` |
| `grpo.beta` | float | float ≥ 0 | `0.04` (fallback `grpo_train:187`); TRL default `0.0` = nessuna KL (`TRL grpo_config:479-481`) | Coefficiente KL verso la reference policy (0 = reference non caricata). | `grpo_train:187` |
| `grpo.temperature` | int \| float | float > 0 | `0.7` (fallback `grpo_train:188`); TRL default `1.0` (`TRL grpo_config:343-344`) | Temperatura di sampling dei rollout GRPO. **Non influisce sull'eval**, che hardcode 0.7 (`eval:980`). | `grpo_train:188` |
| `grpo.loss_type` | str | `grpo` \| `bnpo` \| `dr_grpo` \| `dapo` | **TRL 0.24: `dapo`** (`TRL grpo_config:538-539`) | Normalizzazione della loss RL: `dapo` = token totali del batch globale; `dr_grpo` = costante `B × max_completion_length` (rimuove il bias di lunghezza differenziale); `bnpo` = token del batch locale; `grpo` = lunghezza di sequenza (sconsigliato). Se ASSENTE vale il default TRL — è deliberato (commento `base.yaml:118-124`): tutti i run storici hanno girato sotto `dapo`. | `grpo_train:228-235` (validazione fail-loud) |
| `grpo.scale_rewards` | str \| bool | `group` \| `batch` \| `none` (bool: `True`→`group`, `False`→`none`) | **TRL 0.24: `group`** (`TRL grpo_config:526-527`) | Scala dell'advantage. ⚠️ **`none` NON disattiva la centratura**: TRL sottrae SEMPRE la media di gruppo (`TRL grpo_trainer:1493`) e con `none` salta solo la divisione per la std (`TRL grpo_trainer:1508-1509`). `A = R − mean(R)` è l'advantage non distorto di Dr-GRPO. | `grpo_train:237-248` |
| `grpo.mask_truncated_completions` | bool | true \| false | **TRL 0.24: `False`** (`TRL grpo_config:557`) | Esclude dalla loss i token dei completamenti troncati (la metà "filtering" di DAPO; la metà "soft punishment" non esiste in TRL 0.24). | `grpo_train:250-256` |
| `grpo.epsilon` | float | float > 0 | **TRL 0.24: `0.2`** (`TRL grpo_config:490-492`) | Epsilon del clipping PPO. | `grpo_train:258-262` |
| `grpo.epsilon_high` | float | float ≥ `grpo.epsilon` (nel check, epsilon defaulta a 0.2) | TRL default `None` → = `epsilon` (`TRL grpo_config:502-506`) | Clip-higher di DAPO (il paper raccomanda 0.28). Deve essere **≥ epsilon**, altrimenti fail-loud prima dell'avvio. | `grpo_train:264-271` |

NB: questi cinque knob sono deliberatamente NON in `base.yaml`: ometterli
preserva esattamente i default TRL 0.24 sotto cui sono stati prodotti i
risultati storici (`grpo_train:191-212`).

---

## `generation` (presente solo in `sft/zero-shot.yaml:70-74`)

La sezione ha **precedenza su `grpo`** nella risoluzione del config di
generazione: `config.get("generation", config.get("grpo", {}))` —
`eval:770`, `grpo_train:473` e `grpo_train:574`. Non dichiararla su una cella
GRPO (soprascriverebbe i knob `grpo` con i fallback del codice).

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `generation.max_completion_length` | int | int | `256` (fallback `eval:987` e `grpo_train:475`) | `max_new_tokens` della generazione in **eval** (`eval:770,987`) e dei `generation_kwargs` dei rollout GRPO se la sezione esiste. | `eval:770,987`; `grpo_train:473-476` |
| `generation.max_prompt_length` | int | int | — | Non serve al decoding eval (i prompt NON vengono troncati a eval time). È letta SOLO come **fallback** (dopo `grpo`) nella guardia anti-troncamento few-shot dell'eval (`eval:2290-2292`) e nei check cross-section del validator (`validator:301-303,373-375`). ⚠️ In questi check `grpo` vince su `generation` (ordine opposto al gen_cfg). | `eval:2290-2292`; `validator:301-303,373-375` |
| `generation.temperature` | int \| float | float > 0 | — | **Non letta dall'eval**, che hardcode 0.7 quando campiona (`eval:980-989`). Viene letta dal trainer GRPO solo se una cella GRPO dichiara `generation` (`grpo_train:574→188`): su celle SFT è **inerte** (difetto: dichiarata in `sft/zero-shot.yaml:73`, mai consumata lì). | `grpo_train:188` (solo via risoluzione `generation > grpo`) |

---

## `reward` (base.yaml:158-165)

Lette da `build_t2g_reward_functions` (`rewards:956-1040`) per il GRPO e dalla
mappa di breakdown metrico dell'eval (`eval:1015-1024`). Una componente con
peso 0 non viene nemmeno costruita. **Se la sezione è assente del tutto**, il
fallback è `{translation: 0.40, gold_structure: 0.40, format: 0.10,
repetition: 0.10}` (`rewards:956-962`) — ma in pratica la sezione è sempre
presente in `base.yaml`.

Vincoli (`validator`):
- Le chiavi `weight_*` presenti devono **sommare esattamente 1.0 (±1e-9)**
  (`validator:247-262`). È una convenzione del progetto, non un requisito TRL.
- `edit_validity_oov_weight` ∈ [0.0, 0.6] (`validator:291-296`). Il codice
  accetta [0,1] (`rewards:485-486`), ma sopra 0.6 esiste un attacco dimostrato:
  a 0.75 la spazzura in vocabolario supera un quasi-corretto con un OOV e
  **l'ordinamento si inverte** (`tests/test_edit_validity_reward.py`).

| Chiave | Tipo | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- |
| `reward.weight_translation` | float | 0.0 (peso attivo: `base.yaml:159` = 0.20) | ROUGE-L simmetrico vs gold. | `rewards:968-973`; breakdown eval `eval:1017` |
| `reward.weight_bleu` | float | 0.0 (base: 0.20) | BLEU-4 sacrebleu (effective_order + smoothing floor) vs gold. Se > 0, l'assenza di `sacrebleu` crasha a config time, non a runtime (`rewards:984`). | `rewards:976-986`; `eval:1018` |
| `reward.weight_gold_structure` | float | 0.0 (base: 0.20) | Bigram log-prob del generato confrontato col gold come baseline, con penalità OOV e di mismatch di lunghezza. | `rewards:989-994`; `eval:1019` |
| `reward.weight_verifier_scaled` | float | 0.0 (base: 0.10) | ROUGE-L × confidenza strutturale (RECIPE-inspired). | `rewards:998-1003`; `eval:1020` |
| `reward.weight_gloss_order` | float | 0.0 (base: 0.10) | Levenshtein word-level normalizzato vs gold (segnale di ORDINE). | `rewards:1007-1010`; `eval:1021` |
| `reward.weight_format` | float | 0.0 (base: 0.10) | Solo token di glossa in vocabolario (satura ~0.998 nei run storici). | `rewards:1031-1034`; `eval:1022` |
| `reward.weight_repetition` | float | 0.0 (base: 0.10) | Penalità loop degeneri (token/trigram uniqueness). | `rewards:1037-1040`; `eval:1023` |
| `reward.weight_edit_validity` | float | 0.0 (`rewards:1020`) | Opt-in: edit similarity con termine di validità continuo. Dove la validità ≈ 1 (sotto Trie) equivale a `0.5 × gloss_order + 0.5`: con `scale_rewards='none'` DIMEZZA l'advantage. Solo in `ablations/rewards/edit-validity.yaml`. ⚠️ Non compare nella reward breakdown dell'eval (`eval:1016-1024` non la mappa). | `rewards:1020-1028` |
| `reward.edit_validity_oov_weight` | float | 0.5 (`rewards:1022-1023`) | Peso del termine di validità (frazione di token in vocabolario). `0.0` riproduce esattamente `gloss_order_reward` (scala storica). | `rewards:1022-1023,1025`; vincolo `validator:291-296` |

---

## `grammar` (base.yaml:177-183)

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `grammar.enabled` | bool | true \| false | `True` (fallback `grpo_train:785`, `eval:869`) | Attiva il constrained decoding (Trie dual-root sul vocabolario glossa) nei rollout GRPO e nella generazione eval. `false` = generazione libera (ablation `decoding/no-grammar`). Entra nel fingerprint della baseline eval (`eval:311-313`). | `grpo_train:785-799`, `eval:869-878` |
| `grammar.track_diagnostics` | bool | true \| false | `false` (`base.yaml:183`) | **DIFETTO — non letta da nessuno.** La chiave è tipata dal validator (`validator:165`) e dichiarata in `base.yaml` e in `ablations/objectives/sft-allowed-mass.yaml`, ma il config non viene mai consumato: `GlossVocabularyLogitsProcessor` è costruito con `track_diagnostics=False` implicito sia nel GRPO (`grpo_train:796-798`) sia nell'eval (`eval:872-875`), e il circuito allowed-mass dell'SFT non lo passa (`sft_train:891-895`). La telemetria della massa mascherata NON si attiva via config, qualunque cosa dichiari il YAML. | — (parametro interno `src/grammar/grammar_logits_processor.py:61`, mai alimentato dal config) |

---

## `evaluation` (base.yaml:203-254)

L'UNICA fonte dei knob comportamentali dell'eval: l'interfaccia CLI è solo
`--config` + `--checkpoint` (`eval:2231-2243`). La risoluzione è in
`eval:2260-2269`. Il validator ammette **solo** le chiavi elencate qui sotto
(set chiuso, `validator:78-92`): una chiave extra è un errore loud, non un
silenzioso knob ignorato.

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `evaluation.batch_size` | int | int | `8` (`base.yaml:204`) | **DIFETTO — non letta.** `eval_t2g.py` non legge questa chiave da nessuna parte: la generazione è per-prompt (`num_return_sequences`, `eval:981-990`), non c'è batching dei prompt. Dichiarata, tipata (`validator:139`) e inerte. | — |
| `evaluation.max_samples` | int \| null | int ≥ 1 \| null | nessuna (null = test set intero); `base.yaml:205` = `3000` | Numero di **PROMPT** valutati, sottocampionati dal test set con seed deterministico (`seeded_sample_indices`, `eval:883-889`) — mai i primi N. ⚠️ Omonimia invertita con `dataset.max_samples` (che tronca il TRAIN). Cambiando il valore la cache baseline non è riutilizzata (`eval:355-360`) e i numeri NON sono confrontabili con celle valutate a N diverso. | `eval:2261` → `eval:714,883-885` |
| `evaluation.num_samples` | int | int ≥ 1 | `1` (fallback `eval:2262`); `base.yaml:206` = `5` | Generazioni per prompt. `> 1` attiva il sampling (`do_sample`, `eval:913`). | `eval:2262,913` |
| `evaluation.best_of_n` | bool | true \| false | `False` (`eval:2263`) | Best-of-N **ORACOLO** (usa il gold per scegliere la completion migliore): esce nel blocco separato `oracle_best_of_n`, mai nelle metriche primarie. Richiede `num_samples > 1`, altrimenti disattivata con warning (`eval:914-919`). | `eval:2263,913-928,1131-1159` |
| `evaluation.plot` | bool | true \| false | `False` (fallback `eval:2264`); `base.yaml:214` = `true` | Genera le figure plotnine (11-12 per eval). ⚠️ La modalità compare la forza a True comunque (`eval:1362-1363`). | `eval:2264,1362-1363,1818` |
| `evaluation.compare` | bool \| null | true \| false \| null | `null` = deduzione (`eval:1279-1284`) | Eval baseline (cachata o da `baseline_json`) + checkpoint, con grafici e `comparison.json`. Con null: true sulle celle che dichiarano `training.output_dir`, false sulle eval-only. Il valore esplicito vince sulla deduzione. | `eval:1274-1284,2307-2312`; guardia senza checkpoint `eval:2318-2326` |
| `evaluation.eval_baseline_only` | bool \| null | true \| false \| null | `null` = deduzione (`eval:1274-1278`) | Eval del SOLO base model (celle `baseline/*`). Con null: true solo senza checkpoint E senza `training.output_dir`. Se true e compare true insieme, vince eval_baseline_only (warning, `eval:2356-2360`). | `eval:1274-1278,2307-2312` |
| `evaluation.force_baseline_eval` | bool | true \| false | `False` (`eval:2265`) | Ricalcola la baseline del base model anche se una `eval_baseline.json` compatibile è in cache. | `eval:2265,1461` |
| `evaluation.dual_prompting` | bool | true \| false | `False` (fallback `eval:2269`); `base.yaml:242` = `true` | Valuta la cella in ENTRAMBE le modalità di prompting (config + complementare): **raddoppia il tempo**; la passata complementare non è soggetta al vincolo `max_prompt_length ≥ 512` (misura deliberata del cross-prompting, file con suffisso `__<mode>`). | `eval:2269,2448-2469` |
| `evaluation.prompting` | str | `config` \| `zero-shot` \| `few-shot` | `"config"` (fallback `eval:1245`); `base.yaml:246` = `"config"` | `"config"` deriva la modalità da `retrieval.enabled`; gli altri due la FORZANO per questa eval sola. `"few-shot"` richiede `grpo.max_prompt_length` (o `generation.max_prompt_length`) ≥ 512, altrimenti **parser.error** a runtime (`eval:2289-2301`). | `eval:1245-1250,2289-2301` |
| `evaluation.output` | str \| null | path | `null` = nome derivato dal checkpoint (`eval:1785-1786`) | Path esplicito del results JSON (ex flag `--output`). Con dual/override riceve il suffisso `__<mode>`. | `eval:2266,1775-1786` |
| `evaluation.baseline_pass_at1` | float \| null | float ∈ [0,1] \| null | `null` | Pass@1 del baseline per il plot `baseline_vs_grpo` (ex flag CLI). | `eval:2267,1978-1984` |
| `evaluation.baseline_json` | str \| null | path \| null | `null` | JSON di baseline esterno per i confronti (ex flag `--baseline-json`); di solito superfluo grazie alla cache automatica. | `eval:2268,1447-1460,1854-1857,1992-1995` |

---

## `retrieval` (base.yaml:265-271)

Lette da `build_train_retriever` (`retrieval_setup:56-152`) e, per l'eval,
da `eval:794-818`. Corpus = SOLO il train split, con anti-leakage per query
(query esclusa + near-duplicati sopra `max_self_similarity` rimossi).

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `retrieval.enabled` | bool | true \| false | `False` (`retrieval_setup:94-96`); `base.yaml:266` = `false` | Attiva il few-shot: recupera `top_k` esempi text→gloss dal train e li inietta nel prompt. Determina anche la modalità di prompting implicita dell'eval (`_config_prompting_mode`, `eval:261-272`). ⚠️ Con true serve `max_prompt_length ≥ 512` (il runtime avverte sotto 768, `grpo_train:824-832`; il validator blocca sotto 512, `validator:300-308`): altrimenti gli esempi vengono troncati in silenzio e la cella few-shot è indistinguibile da una zero-shot. | `retrieval_setup:94-96`; override eval `eval:794-798` |
| `retrieval.backend` | str | `tfidf` \| `minilm` | `"tfidf"` (`retrieval_setup:98`) | Backend di similarità: tfidf (deterministico, scikit-learn) o minilm (sentence-transformers, extra opzionale). Valore non ammesso → `ValueError` dal builder. | `retrieval_setup:98,134-141`; set ammessi `retriever:64` |
| `retrieval.model_name` | str \| null | hub id \| null | `null` (`retrieval_setup:99`) | Modello di embedding, SOLO per il backend `minilm` (default `sentence-transformers/all-MiniLM-L6-v2`, `retriever:74`). Ignorato da tfidf. | `retrieval_setup:99` |
| `retrieval.top_k` | int | int ≥ 1 | `3` (`retrieval_setup:34,100`) | Numero di esempi few-shot per query (meno se i filtri anti-leakage li riducono). | `retrieval_setup:100`; `eval:808`; `grpo_train:331` |
| `retrieval.max_self_similarity` | float | float ∈ [0,1] | `0.98` (`retrieval_setup:35,101-103`) | I candidati con similarità alla query sopra la soglia vengono scartati (anti-leakage dei near-duplicati). | `retrieval_setup:101-103`; `eval:809`; `grpo_train:332` |
| `retrieval.cache_path` | str | path | `"data/retriever_index"` (`retrieval_setup:33,48-53`) | Indice del retriever su disco; ricaricato se il sidecar meta (version/backend/model/seed/n) è coerente, altrimenti ricostruito. | `retrieval_setup:48-53,110-128` |

---

## `wandb` (base.yaml:274-277)

| Chiave | Tipo | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- |
| `wandb.project` | str | `"neuro-symbolic-t2g"` (fallback nei tre entrypoint) | Progetto wandb (offline sul cluster). | `grpo_train:948,956`; `sft_train:811,819`; `eval:2135` |
| `wandb.run_name` | str | fallback nome del modello (`eval:2111-2112`) | Prefisso del run name (il trainer appende il timestamp: `grpo_train:148-152`, `sft_train:704-705`). | `grpo_train:148`; `sft_train:704`; `eval:2110-2112` |
| `wandb.tags` | list[str] | `["T2G","grpo"]` (GRPO, `grpo_train:959`), `["T2G","sft"]` (SFT, `sft_train:822`), `["T2G","eval"]` (eval, `eval:2115`) | Tag del run; l'eval aggiunge `eval`/`baseline`/`compare` automaticamente. **Le liste vengono sostituite dal merge**, non fuse: un figlio che dichiara `tags` riscrive quelli di `base.yaml`. | `grpo_train:950-951,959`; `sft_train:813-814,822`; `eval:2115-2126` |

---

## `sft_pretrain` (celle `sft-grpo/*`; assente in `base.yaml`)

Letta dal Step 1.5 del GRPO (`grpo_train:651-760`).

| Chiave | Tipo | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- |
| `sft_pretrain.enabled` | bool | `False` (`grpo_train:652`) | Abilita la Phase 0 di SFT prima del GRPO. | `grpo_train:652` |
| `sft_pretrain.reuse_adapter` | bool | `True` (`grpo_train:700`) | Se esiste un adapter SFT con la stessa fingerprint (model/lora/dataset/iperparametri SFT/system prompt, `sft_train:306-325`), salta la Phase 0 e riusa l'adapter (anche cross-tag, con copia locale). `--force-sft` bypassa. | `grpo_train:700,712-749` |
| `sft_pretrain.adapter_path` | str \| null | `null` | Path esplicito a un adapter da riusare, senza ricerca per fingerprint. Se la directory è incompleta viene ignorato con warning. | `grpo_train:699-711` |
| `sft_pretrain.training.*` | mapping | — | Iperparametri della Phase 0, **fusi dentro `config["training"]`** prima di chiamare `run_sft` (`grpo_train:668-687`): quindi le chiavi valide sono quelle della sezione `training` SFT (`num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `warmup_steps`, `max_seq_length`, `eval_fraction`, `early_stopping_patience`, ...). | `grpo_train:672`; consumo effettivo in `sft_train:717-756,981-985` |
| `sft_pretrain.output_dir` / `.log_dir` | str | `<training.output_dir>/sft_pretrain` (idem log) | Override delle directory della Phase 0. | `grpo_train:673-684` |

NB: l'SFT pre-training è sempre zero-shot (mai few-shot, `grpo_train:808`); il
few-shot vale solo per i rollout GRPO e per l'eval.

---

## `auxiliary_objective` (celle `ablations/objectives/*`; assente in `base.yaml`)

Letta da `resolve_auxiliary_config` (`aux:76-131`) e consumata in
`sft_train:877-954`. Opt-in: sezione assente o peso 0 ⇒ SFT stock,
**bit-identico** al path standard (garantito da test). Vincoli trasversali:
`training.packing`/`padding_free` vietati (`aux:123-130`), richiede
`WORLD_SIZE=1` (`aux:175-189`).

| Chiave | Tipo | Valori ammessi | Default | A cosa serve | Letta da |
| --- | --- | --- | --- | --- | --- |
| `auxiliary_objective.allowed_mass.weight` | float | float ≥ 0 (finite) | `0.0` = disattivato | Log-marginale sull'insieme ammesso dal Trie (partial-label learning / MML), teacher-forced, sommato alla NLL. | `aux:55-64,98-101`; `sft_train:947` |
| `auxiliary_objective.allowed_mass.warmup_steps` | int | int ≥ 0 | `0` | Rampa lineare del peso. | `aux:67-73,101-104`; `sft_train:948` |
| `auxiliary_objective.structured.weight` | float | float ≥ 0 | `0.0` = disattivato | NLL strutturata su spazio di stati ridotto (CRF sparsa source-conditioned). | `aux:55-64,98-101`; `sft_train:952` |
| `auxiliary_objective.structured.warmup_steps` | int | int ≥ 0 | `0` | Rampa lineare del peso. | `aux:67-73`; `sft_train:953` |
| `auxiliary_objective.structured.top_k` | int | int ≥ 1 | `512` (`aux:106-118`) | Stati ordinari del grafo (+1 stato OTHER per la coda). | `aux:106-118`; `aux:167` (build_structured_graph) |
| `auxiliary_objective.structured.alpha` | float | float ≥ 0 | `0.1` | Smoothing sulle transizioni osservate nel train. | `aux:106-118`; `aux:168` |
| `auxiliary_objective.structured.shuffled_control` | bool | true \| false | `False` (`aux:119-121`) | Permuta le transizioni: controllo negativo del GATE 2 (il grafo vero non deve battere quello shuffled). | `aux:119-121,170-171` |

---

## Errori frequenti (sintomo → causa)

- **La cella few-shot produce risultati identici alla zero-shot.**
  `grpo.max_prompt_length` (o `generation.max_prompt_length`) < 512 con
  `retrieval.enabled: true`: TRL **tronca i prompt** dei rollout
  (`TRL grpo_trainer:1136,1175,1277`) e gli esempi few-shot vengono tagliati in
  silenzio. Runtime l'eval fallisce loud (`eval:2289-2301`); sotto 768 il
  training avverte senza bloccare (`grpo_train:824-832`).

- **Ho messo 500 in `evaluation.max_samples` e il training è diventato piccolo.**
  Le due chiavi omonime hanno significato opposto: `dataset.max_samples` TRONCA
  il train set (`aslg:293-294`), `evaluation.max_samples` sottocampiona i
  prompt di VALUTAZIONE (`eval:883-889`). Confonderli distrugge un esperimento.

- **8 passi di accumulo ma il training consuma 1 prompt per passo.**
  In TRL 0.24 `steps_per_generation = gradient_accumulation_steps`
  (`TRL grpo_config:648-650`): gli 8 micro-batch producono 8 rollout dello
  STESSO prompt. `training.max_steps` conta quindi prompt visti, non batch:
  5000 step = 5000 prompt ≈ 6,9% del train. Alzare batch o accum NON cambia i
  prompt per passo.

- **Ho cambiato `training.max_steps` ma l'SFT fa sempre 3 epoche** (o viceversa:
  cambiato `num_train_epochs` e il GRPO non si muove). `max_steps` governa SOLO
  il GRPO (`grpo_train:164`), `num_train_epochs` SOLO l'SFT (`sft_train:721`);
  ciascun trainer ignora l'altro knob. E per via dell'ereditarietà, entrambi
  compaiono anche nelle celle eval-only senza significato.

- **`scale_rewards: 'none'` e l'advantage è comunque centrato.**
  Corretto: TRL sottrae SEMPRE la media di gruppo (`TRL grpo_trainer:1493`);
  `none` salta solo la divisione per la std (`TRL grpo_trainer:1508-1509`).
  "Reward grezze" non esiste in TRL 0.24.

- **Il validator rifiuta i reward weights.** Le chiavi `weight_*` devono sommare
  1.0 ± 1e-9 (`validator:247-262`) e `edit_validity_oov_weight` deve stare in
  [0.0, 0.6] (`validator:291-296`): a 0.75 la reward preferisce ripetizioni di
  token legali a una traduzione quasi corretta con un OOV.

- **`evaluation.prompting: few-shot` e l'eval esce con parser.error.**
  Manca il budget: `max_prompt_length` (in `grpo` o `generation`) < 512 o
  assente (`eval:2289-2301`).

- **`sbatch cluster/train.sh` su una cella baseline esce con 2.**
  La cella non dichiara `training.output_dir`: è una cella eval-only per
  costruzione (`bootstrap:76-94`). Va lanciata con `cluster/eval.sh`.

- **compare senza `--checkpoint` esce con errore.**
  Guardia anti-autoconfronto: un compare valuta il base model due volte
  (`eval:2318-2326`). Passare `--checkpoint` o impostare
  `evaluation.eval_baseline_only: true`.

- **La cache della baseline non viene riusata dopo un cambio di eval.**
  Compatibilità richiesta su metric version, fingerprint del contesto prompt,
  `num_samples` e `max_samples` (`eval:349-360`): cambiare QUALUNQUE knob che
  altera le generazioni (retrieval, grammar, num_samples, prompting) invalida
  la cache — deliberato, non un bug. E i numeri fra celle con `max_samples`
  diversi non sono confrontabili comunque.

- **`best_of_n: true` ma nessun blocco oracle nei risultati.**
  Richiede `num_samples > 1`: altrimenti viene disattivato con warning
  (`eval:914-919`).

- **`dual_prompting: true` su una cella baseline la valuta due volte.**
  Le celle `baseline/*` sono già una griglia esplicita di prompting e dichiarano
  `dual_prompting: false`; se lo riattivi duplichi il lavoro.

- **Ho impostato `grammar.track_diagnostics: true` ma nessun diagnostic appare.**
  La chiave non è letta da nessuno (vedi la sezione `grammar`): la telemetria
  non si attiva via config, oggi.

- **Ho impostato `evaluation.batch_size` / `training.warmup_ratio` e non cambia
  nulla.** Chiavi dichiarate e tipate ma mai lette dal codice (vedi le sezioni
  rispettive).

- **Un adapter fornito fa sparire la quantizzazione.** Deliberato: per fondere
  l'adapter in memoria la quantizzazione viene disattivata
  (`loader:345-349,479-485`).

- **Ho dichiarato `generation:` su una cella GRPO e i knob `grpo:` non valgono
  più.** La sezione `generation` ha precedenza su `grpo` nella risoluzione
  (`eval:770`, `grpo_train:473,574`): `grpo.num_generations`/`beta`/... sparirebbero
  nei fallback. Non mescolare le sezioni.

- **Le liste `lora.target_modules` / `wandb.tags` hanno perso elementi del
  padre.** Il merge sostituisce liste e scalari (`src/utils/config.py:70`),
  non le concatena: un figlio che rideclara una lista la riscrive intera.
