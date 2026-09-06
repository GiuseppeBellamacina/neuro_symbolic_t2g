# Manuale LaTeX italiano

Il documento canonico è modulare: `main.tex`, quindici capitoli tematici in
`chapters/`, appendice in `appendix/` e bibliografia in `references.bib`.
`docs/SOURCES.md` resta il registro completo delle fonti; la bibliografia contiene
il sottoinsieme citato. Non include sorgenti esterni, così il build è riproducibile
dalla directory `docs/report`.

## Build consigliato

```bash
cd docs/report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternativa con `pdflatex` e `bibtex`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Richiede pacchetti comuni di TeX Live: `babel` (italian), `lmodern`,
`microtype`, `geometry`, `amsmath`, `booktabs`, `longtable`, `listings`,
`hyperref` e `cleveref`.

Il manuale descrive il protocollo corrente senza dichiarare metriche della
campagna. Prima di pubblicare risultati, verificare sempre gli artifact e la
configurazione risolta nel repository.

Non incorporare figure della campagna finché non esistono risultati canonici
verificati; si veda `figures/README.md`.
