"""ieee_style.py -- Matplotlib style matched to the IEEEtran document font.

WHY THIS FILE EXISTS
--------------------
The figures produced earlier did not match the typography of the manuscript.
Three defects, in order of how visible they are on the printed page:

  1. WRONG TYPEFACE.  IEEEtran typesets the body in Times Roman.  The figures
     asked for "DejaVu Serif", which is a different serif with wider glyphs and
     lower stroke contrast.  Placed next to the caption, the mismatch reads as
     two different documents.  Mathematical symbols were worse: the mathtext
     set was "dejavuserif", so an axis label $N_{pop}$ used glyphs unrelated to
     the $N_{pop}$ in the running text.

  2. WRONG SCALE.  Figures were drawn 9.4-9.8 in wide and then included at
     \\linewidth.  In a two-column IEEEtran layout a single column is 3.5 in, so
     the figure was scaled down by 2.7x -- an 8.5 pt tick label landed on the
     page at about 3.1 pt.  A figure must be drawn at the width it is printed
     at, and its point sizes must then be read literally.

  3. RASTER OUTPUT.  PNG at 190 dpi cannot carry a font; the glyphs are pixels.
     Vector PDF embeds the Times outlines, so the figure text and the caption
     text are the same shapes at any zoom.

The rule this module enforces: figure point sizes are DOCUMENT point sizes.
A tick label declared at 7 pt here prints at 7 pt.

WIDTHS (IEEEtran, journal, two-column)
--------------------------------------
    COL   3.50 in   -- \\columnwidth, for `figure`
    PAGE  7.16 in   -- \\textwidth,   for `figure*`

COLOURS
-------
Okabe-Ito: distinguishable under every common form of colour blindness and
still separable in greyscale print.  Each method keeps one fixed
(colour, linestyle, marker) triple across every figure of both papers, so a
reader who learns the key once can read all of them.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- widths of the IEEEtran two-column journal layout, in inches ------------
COL = 3.50
PAGE = 7.16

# --- point sizes, read literally because figures are drawn at print size ----
PT_TICK = 7.0
PT_LABEL = 8.0
PT_TITLE = 8.0
PT_LEGEND = 7.0
PT_NOTE = 6.5

# --- Okabe-Ito assignment, fixed per method across both papers -------------
METHOD_STYLE = {
    "ga": {"c": "#000000", "ls": "-", "m": "o"},
    "ga_gnn": {"c": "#0072B2", "ls": "--", "m": "s"},
    "ga_gnn_policy": {"c": "#D55E00", "ls": "-", "m": "^"},
    "policy": {"c": "#009E73", "ls": ":", "m": "*"},
    "random": {"c": "#7F7F7F", "ls": "-.", "m": "v"},
}
RED = "#C0392B"        # constraint thresholds and violation markers only
GREY = "#555555"       # annotations that must not compete with the data


def use_ieee() -> None:
    """Install the style.  Call once before creating any figure."""
    plt.rcParams.update({
        # -- typeface: Times, to match the IEEEtran body text ---------------
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L",
                       "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        # STIX is metric-compatible with Times, so $x$ in a label and $x$ in
        # the running text are the same glyph.
        "mathtext.fontset": "stix",
        "font.size": PT_TICK,

        # -- embed real font outlines in the PDF (TrueType, not Type 3) -----
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # -- IEEE print: white ground, full box, inward ticks ---------------
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,

        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "axes.labelsize": PT_LABEL,
        "axes.titlesize": PT_TITLE,
        "axes.titleweight": "normal",
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": True,
        "axes.spines.right": True,

        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",

        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": PT_TICK,
        "ytick.labelsize": PT_TICK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,

        "text.color": "black",
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.fontsize": PT_LEGEND,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.28,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.1,

        "lines.linewidth": 1.1,
        "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.5,

        "figure.dpi": 110,
    })


def legend_below(fig, ax, ncol: int = 3, y: float = 0.0, handles=None,
                 labels=None):
    """One shared legend under the figure.

    An in-axes legend always covers part of a curve here: the anytime curves
    span nearly the full frame in both directions.  Placing it outside removes
    the choice between seeing the key and seeing the data.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    return fig.legend(handles, labels, loc="lower center", ncol=ncol,
                      bbox_to_anchor=(0.5, y), fontsize=PT_LEGEND,
                      frameon=True, edgecolor="black", framealpha=1.0,
                      fancybox=False)


def save(fig, out_dir: str, name: str, pdf: bool = False):
    """Write a 600 dpi PNG, which is what the manuscript includes.

    Vector PDF would be the better format -- it carries the embedded Times
    outlines rather than pixels -- but the toolchain used to build these
    manuscripts silently dropped PDF figures, producing a document with no
    graphics at all.  PNG at 600 dpi is indistinguishable in print and cannot
    fail that way, so it is the default.  Pass pdf=True to emit both if your
    compiler handles PDF graphics; then point \\includegraphics at the .pdf.
    """
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, name + ".png")
    fig.savefig(png, dpi=600)
    if pdf:
        fig.savefig(os.path.join(out_dir, name + ".pdf"))
    plt.close(fig)
    print("  saved %s.png%s" % (name, " (+.pdf)" if pdf else ""))
    return png


def check_font() -> str:
    """Report which serif family matplotlib actually resolved.

    `font.serif` is a preference list, not a guarantee.  If Times New Roman is
    absent the fallback is silent, and the figures would once again not match
    the manuscript -- so the regeneration scripts print this.
    """
    from matplotlib.font_manager import findfont, FontProperties
    path = findfont(FontProperties(family="serif"))
    return os.path.basename(path)
