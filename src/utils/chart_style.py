"""Colori e inchiostri dei grafici di riepilogo (ablation summary, campaign report).

Un solo posto per i valori, così i grafici restano coerenti fra loro. Sono i toni
della palette di riferimento in modalità chiara: questi grafici sono PNG statici
per la relazione e i report, non pagine con tema.

* Una sola serie per pannello -> un solo colore (``SERIES_1``).
* Grandezza su una griglia (heatmap) -> una rampa a un solo tono, dal chiaro allo
  scuro (``sequential_cmap``). Mai una rampa multi-tono: fa leggere confini che
  nei dati non ci sono.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"

#: Rampa sequenziale blu, step 100 -> 700.
SEQUENTIAL_BLUE = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)


def sequential_cmap():
    """Colormap matplotlib della rampa blu (import pigro di matplotlib)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)


def ink_on(hex_fill: str) -> str:
    """Testo leggibile sopra un riempimento: bianco sugli step scuri, inchiostro
    sugli altri. Soglia sulla luminanza relativa WCAG."""

    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    h = hex_fill.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    luminance = 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    # Punto in cui il contrasto con il bianco eguaglia quello con INK.
    return "#ffffff" if luminance < 0.18 else INK
