# Prossimi esperimenti

Stato al 2026-11-17. Vincolo di risorse: **mezza A40, 24 GB di VRAM**, un solo
job alla volta. Config in `experiments/configs/`.

## 1. Risultati dell'ultimo giro (CoNLL-2003 salvo dove indicato)

Exact match; tra parentesi F1 sulle entità tipate. Run in
`experiments/results/<dataset>/<modello>/<cella>/run_*`.

| Esperimento | Risultato | Lettura |
|---|---|---|
| GRPO lungo, SFT completo (0,5B) | 5 000 passi 0,781 → 15 000 passi 0,791 (0,85) | Più passi dopo SFT completo danno ~+1 punto: il tetto è del metodo, non della durata. |
| GRPO lungo, SFT da 0,05 epoche (0,5B) | 5 000 passi 0,441 → 15 000 passi 0,528 (0,58) | Con poca SFT più tempo aiuta (+9) ma resta lontano dall'SFT completo (0,745). Il GRPO non sostituisce la supervisione. |
| Replica a seme 1, `sft-grpo-ep005` | 0,445 contro 0,441 (seme 42) | Il +30 punti del GRPO dopo poca SFT è riproducibile (SFT riaddestrato a seme 42, poi GRPO a seme 1). |
| Entity-F1 + rollout caldi, GRPO dal base | 0,053 (0,12) | Peggio dei soli rollout caldi (0,123, 0,17): la reward sul tipo non aiuta, nemmeno con più esplorazione. Voce chiusa. |
| Qwen2.5-3B, SFT / SFT→GRPO / GRPO dal base | 0,813 / 0,844 / 0,298 | L'incremento del GRPO sull'SFT è ~+3 punti a ogni taglia (0,5B +3,6, 1,5B +3,6, 3B +3,1). L'SFT cresce con la taglia (0,745, 0,778, 0,813). Il GRPO dal base sale con la taglia (0,03, 0,28, 0,30) ma resta lontanissimo dall'SFT. |
| WOS 1,5B, SFT con lr 4e-5 | 0,796 (lr 2e-5: 0,734; 1e-5: 0,523) | L'anomalia del 1,5B era di iperparametri: con lr 4e-5 supera lo 0,5B (0,762). L'ottimo può essere ancora più in alto. |

Giro precedente (curva SFT prima del GRPO, SFT senza Trie, rollout caldi, GRPO dal
base a 1,5B): vedi la relazione, capitolo 12.

## 2. Qwen2.5-3B su WOS e PHOENIX, WOS lr 8e-5 (finiti il 2026-11-17)

Tutti i 16 job completati, nessun errore. Exact match su WOS, ROUGE-L su PHOENIX.

| Taglia | WOS SFT | WOS SFT→GRPO | WOS GRPO dal base | PHOENIX SFT | PHOENIX SFT→GRPO | PHOENIX GRPO dal base |
|---|---|---|---|---|---|---|
| 0,5B | 0,762 | 0,787 (+2,5) | 0,169 | 0,485 | 0,527 (+4,2) | 0,203 |
| 1,5B | 0,734 | 0,761 (+2,7) | 0,127 | 0,502 | 0,542 (+4,1) | 0,310 |
| 3B | 0,764 | 0,794 (+3,0) | 0,123 | 0,542 | 0,571 (+2,9) | 0,293 |

- **SFT→GRPO è la cella migliore a ogni taglia, su tutti e tre i compiti**, con un
  incremento sull'SFT di +2,5/+4 punti che non dipende dalla taglia. Il GRPO dal
  base resta lontanissimo dall'SFT ovunque.
- **GRPO dal base e taglia:** su PHOENIX sale (0,203 → 0,310) e poi si ferma
  (0,293 a 3B); su WOS **scende** con la taglia (0,169 → 0,127 → 0,123). L'ipotesi
  «un modello più grande esplora meglio» regge su CoNLL e in parte su PHOENIX,
  non su WOS.
- **WOS, curva del learning rate (1,5B):** 1e-5 0,523 · 2e-5 0,734 · 4e-5 0,796 ·
  8e-5 0,794. Satura intorno a 4e-5. L'«anomalia» del 1,5B era un SFT
  sottoaddestrato a 2e-5. **Attenzione:** lo 0,5B e il 3B sono ancora a 2e-5, e il
  3B (0,764) è piatto rispetto allo 0,5B (0,762): probabilmente lo stesso effetto.
  Il confronto di scala su WOS a lr 2e-5 non è un confronto di taglia.

## 2a. Prossimo test consigliato (non lanciato)

WOS SFT a lr 4e-5 anche per 0,5B e 3B (`ablations/sft/lr-4e-5`, circa 4 ore in
tutto). Dice se su WOS la taglia conta davvero una volta tolto l'effetto del
learning rate. Chiude l'ultima cella aperta del confronto di scala.

## 2bis. Ancora aperto

1. **Altre famiglie di modelli**: non lanciate, perché richiedono lavoro sul
   codice prima (sezione 3).
2. **Reward senza ROUGE, con il tipo scomposto**: il risultato di entity-F1 con
   rollout caldi è negativo, quindi la voce perde priorità.
3. **WOS lr 5e-6**: escluso, 1e-5 è peggiorato.

**Escluso:** `num_generations: 16`. Non entra nella VRAM disponibile.

## Nota operativa: checkpoint mancanti

Una cella che dipende da un checkpoint (eval di sola valutazione, `adapter_path`
esplicito) non fallisce se manca: la catena accoda prima il train della cella
madre e il trainer riaddestra la fase SFT (CLUSTER.md). Tutte le celle con
`adapter_path` fissano `sft_pretrain.training.seed: 42`. La cache HF del cluster
deve contenere anche il repository in minuscolo (`unsloth/qwen2.5-3b-instruct`,
creato da `setup.sh --models-only`) per caricare un modello non quantizzato con
adapter.

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
