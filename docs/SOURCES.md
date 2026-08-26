# Fonti e Bibliografia — neuro_symbolic_t2g

Data ultima verifica: 2026-08-26. Tutti gli URL sono stati raggiunti e verificati
(fetch reale: arXiv API / pagine HTML, ACL Anthology, docs TRL, blog Unsloth).
Ricerca condotta via arXiv API e ricerca web; Semantic Scholar non consultabile
(HTTP 429 persistente) — nessuna fonte è citata senza verifica diretta.

Legenda dell'uso: **[implementato]** = ha influito su codice/config del progetto;
**[ispirazione]** = motivazione del design; **[calibrazione]** = aspettative
sui risultati; **[futuro]** = candidata per esperimenti successivi;
**[contesto]** = letteratura di settore.

---

## 1. Paper di partenza dell'utente (pre-ricerca)

| Fonte | Uso nel progetto |
|---|---|
| **arXiv:2603.04265** — ViterbiPlanNet DVL (differentiable Viterbi) | **[implementato]** Ispirazione delle reward `soft_viterbi_distance_reward` (log-partition function forward-backward, `src/rewards/t2g_rewards.py`) e `viterbi_distance_reward` (argmax Viterbi diversificato). Pesi default 0.0 nei config (moduli da ablation). |
| **arXiv:2605.19976** — RECIPE (verifier-scaled reward) | **[implementato]** Ispirazione della `verifier_scaled_reward` (confidence multiplier: ROUGE × gold-structure, `src/rewards/t2g_rewards.py:798-851`). NOTA: la formula implementata differisce dalla doc storica (vedi docs/REWARDS.md) — semplificata in quality×structure. |

## 2. T2G / Sign Language Translation con LLM

| Fonte | Uso nel progetto |
|---|---|
| **arXiv:2512.07273** — RVLF: Reinforcing Vision-language Framework (Rao, Zhou, Zhou, Huang, Escalera, Wan, 2025). Primo GRPO nella SLT: reward = BLEU (fidelity) + ROUGE (completeness); +5.1 BLEU-4 CSL-Daily, +1.11 PHOENIX-2014T. | **[implementato]** Riferimento principale della `bleu_reward` (sostituisce la citazione inesistente "T2G-Reasoner 2025" in docstring `src/rewards/t2g_rewards.py`, README, TRAINING.md, RESEARCH_REPORT, commenti yaml). Dimostra che reward BLEU+ROUGE in GRPO funzionano per la sign language. |
| **arXiv:2508.19481** — Spanish→Wayuunaiki dictionary-guided MT (Mosquera, Robles, Rodriguez, Manrique, 2025). GRPO con reward similarità BLEU su **Qwen2.5-0.5B** (il nostro stesso modello): +3.37 BLEU sul SOTA. | **[implementato]** Secondo riferimento della `bleu_reward`; **[calibrazione]** prova di fattibilità diretta: una 0.5B impara MT via GRPO con reward BLEU. |
| **arXiv:2504.02293** — Bangla Text-to-Gloss (Abdullah et al., 2025). Primo benchmark T2G per lingua segnata; mBART competitivo vs LLM chiuso. | **[contesto]** Il T2G è trattabile con modelli piccoli; leva sui dati sintetici. |
| **arXiv:2404.11532** — Select and Reorder (Walsh, Saunders, Bowden, LREC-COLING 2024). T2G mDGS: selezione gloss + riordino, SOTA BLEU/Rouge. | **[contesto]** Valida la visione "T2G ≈ selezione+riordino dal vocabolario" che il nostro constrained decoding implementa. |
| **arXiv:2505.15438** — Pseudo Gloss Generation (Guo et al., 2025). Genera pseudo-gloss con LLM few-shot ICL. | **[contesto]** Uso di esempi text→gloss in-context per SLT. |
| **arXiv:2405.10718** — SignLLM (Fang et al., 2024); **arXiv:2404.00925** — LLMs are Good Sign Language Translators (Gong et al., CVPR 2024); **arXiv:2407.01394** — Gloss2Text (Fayyazsanavi et al., 2024); **arXiv:2501.09754** — Lost in Translation, Found in Context (Jang et al., CVPR 2025); **arXiv:2305.17714** — Gloss-based Baseline (Moryossef et al., 2023); **arXiv:1112.0168** — Statistical SL MT (Othman & Jemni, 2011). | **[contesto]** Quadro letteratura SLT/LLM e origine storica dei gloss ASLG-PC12. |
| ulteriori **[contesto]** SLT: arXiv:2405.04164 (Sign2GPT, ICLR 2024), arXiv:2408.10593 (SpaMo, NAACL 2025), arXiv:2411.16789 (MMSLT, ICCV 2025), arXiv:2509.00030 (SignBind-LLM), arXiv:2607.27614 (DualAnchor), arXiv:2608.09006 (SignLlama), arXiv:2507.23575 (BeyondGloss, BMVC 2025), arXiv:2412.16524 (LLaVA-SLT), arXiv:2410.19586 (DivSLT). | Letteratura correlata; nessun impatto diretto sul codice. |

## 3. GRPO per MT / generation con modelli piccoli

| Fonte | Uso nel progetto |
|---|---|
| **arXiv:2606.21413** — CAT-Translate (Jinnai, 2026). SFT 2 stadi + Multi-Objective GRPO su 0.8B–7B. | **[calibrazione]** Il template "SFT → GRPO multi-reward su modelli <1B" è consolidato (è la nostra pipeline). |
| **arXiv:2605.15976** — Reference-Free RL MT (Garcia-Estrada, Escolano, Fonallosa, 2026). GRPO su NLLB 600M/1.3B: guadagni maggiori dove il baseline è più debole. | **[calibrazione]** Il nostro baseline zero-shot è debole → molto spazio per il RL. |
| **arXiv:2608.10812** — MiLMMT-46 (Han, Gao, Fu, Luan, 2026). Interpolazione lineare checkpoint SFT↔RL come àncora anti-drift. | **[futuro]** Candidata tecnica post-prima-run contro il drift del RL. |
| **[contesto]** altri MT-RL: arXiv:2605.14366 (Semantic Rewards, ACL 2026 Findings), arXiv:2604.04839 (MERIT), arXiv:2602.14028 (GRRM — ranking intra-gruppo), arXiv:2602.03352 (PEGRL — post-editing ausiliario), arXiv:2602.03102 (C-GRPO — MBR distillato in GRPO), arXiv:2601.06307 (MTQE idiomi). | Design alternative di reward per MT; nessun impatto diretto. |

## 4. Varianti GRPO e stabilità (beta/temperature/lr/G)

| Fonte | Uso nel progetto |
|---|---|
| **arXiv:2402.03300** — DeepSeekMath (Shao et al., 2024). Origine di GRPO, β default 0.04. | **[implementato]** Giustifica `beta=0.04` in `base.yaml` (riallineamento del config, ex beta=0). |
| **arXiv:2503.20783** — Dr.GRPO (Liu et al., 2025). Bias di lunghezza di GRPO; loss unbiased; `scale_rewards=False` evita bias di difficoltà. | **[implementato]** Motivazione della vigilanza sul bias lunghezza (anti-hacking delle reward su output corti, fix del 2026-08); **[futuro]** `loss_type`/`scale_rewards` in trl. |
| **arXiv:2503.14476** — DAPO (Yu et al., 2025). Clip decoupled, dynamic sampling, loss token-level. | **[contesto]** Quadro delle varianti; il vecchio commento "beta=0 DAPO-style" nei config derivava da qui. |
| **arXiv:2508.13023** — G²RPO (Guo, Deng, Cheng, Tang, 2025). **Guided** GRPO: iniezione adattiva di step ground-truth nei rollout per SLM deboli. | **[implementato]** Correzione bibliografica: il curriculum 3-stage del progetto era attribuito a "G²RPO-A 2026" (fonte inesistente con quel contenuto); ora dichiarato project-original in `grpo_t2g_train.py` (nota a riga ~384), RESEARCH_REPORT e yaml. Il paper reale resta **[futuro]** come tecnica guida. |
| **arXiv:2606.02615** — FSA-GRPO (Zheng, Wang, Fan, Jin, Hasegawa-Johnson, 2026). GRPO few-shot aware: demo nel prompt durante i rollout + reward per l'uso delle demo. | **[ispirazione]** Valida il nostro few-shot retrieval nei rollout GRPO (`src/training/retrieval_setup.py`, `src/utils/prompting.py`); **[futuro]** reward "uso corretto dell'esempio". |
| **arXiv:2606.13680** — RA-RFT (Xiao, Ma et al., 2026). Retrieval-augmented RL: retriever distillato + esempi recuperati nei prompt durante RL (+7.1 avg@32 su Qwen3-1.7B). | **[ispirazione]** Prova che retrieval-few-shot nel loop RL aiuta i modelli piccoli — base del modulo `src/retrieval/`. |
| **arXiv:2606.02313** — EG-GRPO (Chen, Li et al., 2026). Rollout online arricchiti da expert few-shot. | **[ispirazione]** Pattern "rollout = prompt condizionati da esempi". |
| **arXiv:2607.21626** — Discrete Action Space GRPO (Filatov et al., 2026). Su Qwen-0.5B GRPO vanilla collassa (entropia 0.35→0.03 in 60 step); si salva vincolando lo spazio d'azione. | **[calibrazione]** Supporta la tesi che il constrained decoding (spazio d'azione ristretto) **stabilizza** GRPO su 0.5B. |
| **arXiv:2606.22189** — L20-Edu-135M (Li, 2026). RLVR su 135M fa calare l'accuracy. | **[calibrazione]** Limite inferiore di scala: sotto una certa taglia il RL non aiuta (0.5B è sopra la soglia per i precedenti, ma da tenere d'occhio). |
| **[contesto]** ulteriori: arXiv:2605.25604 (DVAO), arXiv:2605.11538 (Covariance-Aware GRPO, ACL 2026), arXiv:2604.13515 (SFT-GRPO overlap), arXiv:2607.02869 (Reward Granularity, Qwen 0.5B), arXiv:2606.19990 (Reward as Agent). | Dinamica/stabilità GRPO. |
| **TRL docs** — https://huggingface.co/docs/trl/main/en/grpo_trainer e paper index (https://github.com/huggingface/trl/blob/main/docs/source/paper_index.md) | **[implementato]** Valori pratici GRPOConfig (num_generations=8, scale_rewards, loss_type) usati per il riallineamento config; NOTA: la 0.24 installata può esporre un sottoinsieme delle opzioni — verificare sul cluster. |
| **Unsloth blog** — https://unsloth.ai/blog/grpo | **[implementato]** β=0.04 e group=8 negli esempi; conferma VRAM accessibile per GRPO+LoRA su modelli piccoli. |

## 5. Reward hacking (GRPO con reward rule-based)

| Fonte | Uso nel progetto |
|---|---|
| **arXiv:2509.22047** — MO-GRPO (Ichihara, Jinnai et al., TACL). Multi-reward GRPO ottimizza una reward a scapito delle altre; fix: reweighting automatico per varianza del gruppo. | **[futuro]** candidato #1 per il round-2 di tuning delle nostre 7 reward pesate. |
| **arXiv:2608.11669** — Rubric Dropout (Yang, Guo et al., 2026). Dropout 30–50% dei criteri a ogni step, condiviso nel gruppo. | **[futuro]** fix a basso costo contro l'hacking multi-reward. |
| **arXiv:2607.09492** — Multimodal Reward Hacking (Yao et al., 2026). GRPO è l'algoritmo più resistente tra GRPO/RLOO/DAPO; verifier affidabili > keyword checks. | **[calibrazione]** La scelta GRPO + reward deterministiche benigne è difendibile; le nostre metriche strutturali (bigram/viterbi) vanno trattate come "keyword-like" → vigilanza (già fixata con anti-hacking su output corti). |
| **arXiv:2608.17423** — Prism-GRPO (Deng et al., 2026). Gruppi same-outcome → advantage nulla; split per qualità. | **[futuro]** Se `frac_reward_zero_std` sarà alto nei run (output identici per via della maschera), questo è il fix documentato. |
| **arXiv:2606.30789** — Predictable GRPO (Ghosh et al., 2026). G agisce da "temperatura del rumore" (varianza ~1/G). | **[calibrazione]** Motivo per G=8 (compromesso varianza/VRAM). |
| **arXiv:2506.02355** — Rewarding the Unlikely (He, Fried, Welleck, 2025). Rank bias: GRPO rinforza il probabile, trascura il raro → distribution sharpening. | **[futuro]** Rilevante se la BLEU-reward porta a ripetere solo le gloss più frequenti. |
| **[contesto]** arXiv:2605.30451 (VeriGate), arXiv:2606.19818 (UARM), arXiv:2510.18924 (Noise-corrected GRPO), arXiv:2606.21053 (OPRL, Interspeech 2026), arXiv:2606.27291 (Reward Design, KDD 2026 WS). | Altre mitigazioni dell'hacking. |
| **TRL reward utils** (`get_repetition_penalty_reward`, cosine-scaled reward — paper index sopra) | **[implementato]** Precedente per la nostra `gloss_repetition_reward` (implementazione propria, range [-1,1]). |

## 6. Constrained decoding + RL

| Fonte | Uso nel progetto |
|---|---|
| **arXiv:2510.24431** — MiniOneRec (Kong, Sheng et al., 2025). RL + constrained decoding + reward ibride su Qwen 0.5B→7B (vocabolario chiuso di Semantic ID). | **[ispirazione]** Dimostra che decoding vincolato a vocabolario ristretto **convive** col RL — nucleo della nostra architettura Trie mask + GRPO. |
| **arXiv:2603.29068** — ARCS (Pathak, 2026). GRPO + grammar-constrained decoding (validità 100%); normalizzazione advantage per-topologia. | **[futuro]** Se la maschera cambia la distribuzione delle reward tra prompt, la per-condition normalization è il rimedio documentato. |
| **arXiv:2604.22709** — Abstract-CoT (Ramji, Naseem, Astudillo, 2026). SFT warm-up + RL sotto constrained decoding su vocabolario riservato. | **[ispirazione]** Pattern SFT→RL sotto vincolo = la nostra pipeline (SFT phase 0 → GRPO con Trie mask). |
| **[contesto]** arXiv:2606.03954 (VLESA), arXiv:2605.21993 (ECPO), arXiv:2401.16979 (Re3val, EACL 2023 Findings). | Combinazioni RL+constraint in altri domini. |
| **Gap dichiarato**: nessuno studio sistematico su interferenza maschera-grammaticale × advantage estimation in GRPO. | **[calibrazione]** Potenziale contributo originale del progetto (da esplorare coi dati dei run). |

## 7. ASLG-PC12 — calibrazione del target

| Fonte | Uso nel progetto |
|---|---|
| **arXiv:2304.10844** — Mono-SLT (Peng et al., 2023): BLEU-4 **89.90**, ROUGE-L 97.19. **arXiv:2204.05953** — TIN-SLT (Cao et al., NAACL 2022 Findings): 84.29/95.39. **arXiv:2004.00588** — STMC-Transformer (Yin & Read, COLING 2020): 82.41/95.87. | **[calibrazione]** ATTENZIONE: sono numeri della direzione **gloss→inglese**, NON English→gloss. La direzione T2G non ha leaderboard pubblicati moderni → il target BLEU 0.80 va dichiarato col nostro protocollo (split deduppato, tokenizzazione, case). |
| **arXiv:1112.0168** (Othman & Jemni, 2011) | **[contesto]** Unico precedente T2G su ASLG-PC12 (SMT statistico). |
| **Dataset**: https://huggingface.co/datasets/achrafothman/aslg_pc12 | **[implementato]** Dataset del progetto (loader `src/datasets/aslg_dataset.py`, dedup normalizzato dal 2026-08). |

## 8. Fonti tecniche non-paper

| Fonte | Uso |
|---|---|
| https://gcluster.dmi.unict.it/docs/ (sezione Jobs Management / QoS) | **[implementato]** Vincoli cluster: 1 job per utente (inclusi pending → escluso self-chaining/array job), QoS time limits (gpu-small 4h, medium 6h, large/xlarge 12h), divieto processi su login node — input per il redesign della chain a tick (`cluster/chain_tick.sh`, `_lib.sh`, hook bashrc). |
| https://huggingface.co/docs/trl (GRPOTrainer/GRPOConfig, via context7) | **[implementato]** Contratto kwargs reward (colonne dataset → reward fn), `completion_only_loss` SFT, valori default. |
| https://unsloth.ai/blog/grpo | **[implementato]** Iperparametri GRPO+LoRA su piccoli modelli; VRAM. |
| grammarllm (libreria vendored del collega, v0.5.0 + backport — vedi `grammarllm/VENDORED_STATUS.md`) | **[implementato]** PDA LL(1) + StatelessLogitsProcessor per il path ablation `grpo_pda*`. |

## 9. Correzioni bibliografiche applicate (2026-08-26)

1. **"T2G-Reasoner (2025)"** — citazione rimossa ovunque (docstring `bleu_reward`,
   README.md, TRAINING.md, docs/RESEARCH_REPORT.md, commenti in
   base.yaml/grpo_optimal.yaml/grpo_experimental_all.yaml): la fonte non risulta
   esistente (0 risultati su arXiv e web). Sostituita con RVLF (arXiv:2512.07273)
   e Mosquera et al. (arXiv:2508.19481), entrambe verificate.
2. **"G²RPO-A (2026)"** come curriculum learning — attribuzione errata: il paper
   reale G²RPO (arXiv:2508.13023) è *Guided* GRPO, non un curriculum. Il
   curriculum 3-stage è ora dichiarato **project-original** in
   `src/training/grpo_t2g_train.py` (nota nel codice), RESEARCH_REPORT e yaml.
   Il vero G²RPO resta citato come tecnica **[futuro]**.
3. Distribuzione difficoltà curriculum aggiornata ai valori post-dedup
   (4.9/71.2/23.9 — era 9.3/68.4/22.2 pre-dedup, vedi RESEARCH_REPORT §11.4).
