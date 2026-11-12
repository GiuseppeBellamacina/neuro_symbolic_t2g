# Prossimi esperimenti

Stato al 2026-11-12. Vincolo di risorse: **mezza A40, 24 GB di VRAM** (shard da
circa 22 GB), un solo job alla volta. Quello che segue è da riprendere **dopo**
la coda attuale (sezione 1), non prima.

## 1. In coda sul cluster (25 job, circa 24 ore)

Cosa decide ciascuno. Config in `experiments/configs/`.

| Esperimento | Config | Come leggerlo |
|---|---|---|
| SFT valutato senza Trie (0,5B, 3 dataset) | `<ds>/qwen25-05b/sft/zero-shot-no-trie` | Se validity ed exact match restano uguali, dopo l'SFT il Trie è ridondante. Risultato in `run_*/no-trie/`. |
| WOS 1,5B, SFT con lr 1e-5 | `wos-46985/qwen25-15b/ablations/sft/lr-1e-5` | Se l'exact match sale verso 0,762 (lo 0,5B), l'anomalia WOS era di iperparametri: provare anche 5e-6. Se resta ~0,735, è la taglia. |
| CoNLL 0,5B, rollout più caldi (T=1,3) | `conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot` | Confronto con GRPO zero-shot: exact match 0,030, F1 sulle entità tipate 0,13. Se sale, il limite è davvero l'esplorazione. |
| GRPO dal base a 1,5B su PHOENIX e WOS | `<ds>/qwen25-15b/grpo/zero-shot` | Su CoNLL il 1,5B esplora meglio dello 0,5B (F1 0,13 → 0,26): si vede se vale anche altrove. |
| **Quanta SFT prima del GRPO** (CoNLL) | `conll-2003/qwen25-05b/ablations/sft-amount/{sft,sft-grpo}-{ep005,ep015,ep05}` | Curva exact match contro passi di SFT: 0 passi (GRPO dal base), circa 43, 129, 430, 2 580 (SFT completo, 0,781). L'incremento del GRPO sopra l'SFT a ogni punto dice da quanta supervisione il GRPO decolla. |

## 2. Da decidere dopo la coda

Ordine di preferenza.

1. **GRPO più lungo su SFT→GRPO** (CoNLL 0,5B). Oggi il GRPO fa 5 000 passi
   e vede solo il 12–36 % del train: il +2–4 punti sull'SFT potrebbe essere un
   tetto di durata. Config: `training.max_steps: 15000` su `sft-grpo/zero-shot`.
   Costo circa 9 ore, una sola cella.
2. **Qwen2.5-3B.** Entra nei 24 GB (QLoRA 4 bit). Dice se l'incremento del GRPO
   resta invariante a una terza taglia e se l'esplorazione migliora in modo
   monotono (F1 sulle entità tipate del GRPO dal base: 0,13 a 0,5B, 0,26 a
   1,5B). Celle: CoNLL SFT, SFT→GRPO, GRPO dal base, più baseline; circa 10 ore.
   Prima di lanciare le lunghe, far girare il solo SFT e controllare la VRAM.
   Pesi: scaricare in locale e caricare nella cache del cluster, come per l'1,5B
   (il cluster non raggiunge Internet; tre repository: `Qwen/...`,
   `unsloth/...`, `unsloth/...-unsloth-bnb-4bit`, verificare lo SHA-256).
3. **Altre famiglie di modelli** (oggi un limite dichiarato: solo Qwen2.5).
   Vedi sezione 3.
4. **Reward senza ROUGE, con il tipo scomposto.** Solo se il test sui rollout
   caldi dà un segnale.

**Escluso:** `num_generations: 16`. Non entra nella VRAM disponibile.

## 3. Altre famiglie: cosa va modificato

Il codice è quasi tutto indipendente dal modello (LoRA su `q_proj…down_proj`,
nome del modello nel tag, cache offline). Le dipendenze vere sono tre.

1. **`src/utils/prompting.py::extract_query_source`** ha scritto a mano il formato
   ChatML (`<|im_start|>user\n … <|im_end|>`). Serve al Trie per frase di CoNLL
   per ricavare la frase dagli id del prompt. Con un template diverso
   (Llama 3: `<|start_header_id|>`, Gemma: `<start_of_turn>`) il vincolo si
   romperebbe in silenzio. Va resa indipendente dal template, per esempio
   rendendo il template con una frase sentinella e cercandola. Lo stesso vale
   per il ripiego ChatML di `build_t2g_prompt` e per `callbacks.py`.
2. **Il Trie a doppia radice** assume che `" parola"` e `"parola"` diano token
   diversi, come nei BPE a livello di byte (Qwen, SmolLM2, Llama 3). Nei
   tokenizer SentencePiece (Gemma, Llama 2, Mistral) il prefisso `▁` e lo spazio
   iniziale automatico cambiano il comportamento: va verificato con un test, e
   probabilmente ritoccata la costruzione delle due radici.
3. **Cache offline e Unsloth.** I tre repository da scaricare cambiano nome per
   ogni famiglia (`setup.sh --models-only` e `T2G_MODELS`), e Unsloth deve
   supportare il modello.

Candidati, dal meno al più costoso:

- **Qwen3-1.7B**: stesso tokenizer e ChatML. Quasi nulla da cambiare, tranne il
  flag di ragionamento (`dataset.thinking` esiste già).
- **SmolLM2-1.7B-Instruct**: ChatML e BPE a livello di byte, aperto. Da
  controllare il tokenizer con i test del Trie.
- **Gemma**: SentencePiece, quindi il lavoro sul Trie (punto 2). Per ultimo.
- **Llama 3.2**: i pesi sono con accesso a richiesta; il cluster non scarica da
  solo. Da evitare finché non serve.

Lavoro preliminare, indipendente dalla scelta: rendere `extract_query_source`
indipendente dal template e parametrizzare `tests/test_output_grammar.py` su più
tokenizer.
