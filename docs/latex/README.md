# Relazione tecnico-scientifica (LaTeX)

Relazione autocontenuta del progetto `neuro_symbolic_t2g`, in italiano.

## Compilazione

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Le tre passate di `pdflatex` servono a stabilizzare indice, riferimenti incrociati
e citazioni. Prima della seconda passata `bibtex` genera la bibliografia da
`bibliografia.bib`.

In alternativa, con `latexmk`:

```bash
latexmk -pdf main.tex
```

## Struttura

```
docs/latex/
  main.tex              preambolo, abstract, indice, \input dei capitoli
  bibliografia.bib      voci BibTeX
  capitoli/
    01_introduzione.tex
    02_fondamenti_teorici.tex
    03_dataset.tex
    04_architettura_modello.tex
    05_sft.tex
    06_grpo.tex
    07_reward.tex
    08_decoding_vincolato.tex
    09_obiettivi_ausiliari.tex
    10_metriche_valutazione.tex
    11_risultati.tex
    12_infrastruttura_cluster.tex
    13_codebase.tex
    14_conclusioni.tex
```

Ogni file in `capitoli/` inizia con `\chapter{...}` e non contiene preambolo né
`\begin{document}`: sono frammenti inclusi da `main.tex`.

## Dipendenze (pacchetti LaTeX)

Tutti presenti in una distribuzione TeX Live completa o in `texlive-latex-extra`:

| Pacchetto | Uso |
|---|---|
| `inputenc`, `fontenc` | codifica UTF-8, font T1 |
| `babel` (italian) | sillabazione e nomi delle sezioni in italiano |
| `amsmath`, `amssymb` | ambienti matematici, simboli |
| `booktabs` | tabelle (`\toprule`, `\midrule`, `\bottomrule`) |
| `listings` | blocchi di codice |
| `graphicx` | inclusione di immagini |
| `xcolor` | colori (usato dalla macro `\todo`) |
| `geometry` | margini |
| `hyperref` | collegamenti interni |

Su Debian/Ubuntu:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-lang-italian
```

## Convenzioni

- **Lingua**: italiano. Restano in inglese i termini tecnici consolidati
  (reward, loss, token, rollout, checkpoint, gloss, dataset, policy, advantage,
  logits).
- **Numeri**: solo valori misurati. Dove un valore non è disponibile si usa la
  macro `\todo{...}`, che lo segnala in rosso nel PDF, invece di stimarlo.
- **Notazione**: le macro `\vocab`, `\allowed`, `\policy`, `\reference` sono
  definite in `main.tex` e vanno usate in luogo delle rispettive espressioni
  esplicite, per coerenza fra capitoli.
- **Riferimenti**: ogni capitolo definisce una label `cap:<nome>` subito dopo
  `\chapter`; le sezioni citate altrove definiscono `sec:<nome>`.

## Nota sugli errori del language server

Un language server LaTeX segnala `Undefined reference` sui singoli file dei
capitoli. È atteso: le label risolte stanno in altri file e vengono unite solo
in fase di compilazione di `main.tex`. Non sono errori reali.
