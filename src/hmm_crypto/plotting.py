"""Report-quality figures (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical palette, assigned in a FIXED order (never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Regimes are ordered from calmest to most volatile; each tier owns a fixed slot.
REGIME_ORDER = ["tres_calme", "calme", "normal", "agite", "turbulent", "crise", "crise_extreme", "unique"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False,
})


def regime_color(label: str) -> str:
    return PALETTE[REGIME_ORDER.index(label) % len(PALETTE)] if label in REGIME_ORDER else "#8a8984"


def _save(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_regimes(close: pd.Series, regimes: pd.Series, title: str = "", path=None):
    """Log-price coloured by regime (points), with the regime strip underneath."""
    regimes = regimes.reindex(close.index).dropna()
    close = close.loc[regimes.index]
    labels = [lab for lab in REGIME_ORDER if lab in set(regimes)] + sorted(
        set(regimes) - set(REGIME_ORDER))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [5, 0.6], "hspace": 0.05})
    ax.plot(close.index, close, color="#b9b8b2", lw=0.8, zorder=1)
    for lab in labels:
        m = regimes == lab
        ax.scatter(close.index[m], close[m], s=6, color=regime_color(lab), label=lab,
                   zorder=2, linewidths=0)
    ax.set_yscale("log")
    ax.set_ylabel("Prix (échelle log)")
    ax.set_title(title or "Régimes de marché identifiés par le HMM", loc="left")
    ax.legend(loc="upper left", markerscale=3, ncol=len(labels))
    # strip
    codes = pd.Categorical(regimes, categories=labels).codes
    cmap = matplotlib.colors.ListedColormap([regime_color(lab) for lab in labels])
    ax2.imshow(codes[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(labels) - 1,
               extent=[matplotlib.dates.date2num(close.index[0]),
                       matplotlib.dates.date2num(close.index[-1]), 0, 1])
    ax2.set_yticks([])
    ax2.grid(False)
    ax2.xaxis_date()
    return _save(fig, path)


def plot_equity(equities: dict[str, pd.Series], title: str = "", path=None):
    """Equity curves (log scale) + drawdowns. Keys = legend names, in fixed colour order."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    for (lab, eq), c in zip(equities.items(), PALETTE):
        ax.plot(eq.index, eq, color=c, label=lab)
        ax.annotate(f"{lab}  ×{eq.iloc[-1]:.2f}", (eq.index[-1], eq.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd * 100, color=c, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Valeur du portefeuille (base 1)")
    ax.set_title(title or "Backtest hors-échantillon (walk-forward)", loc="left")
    ax.legend(loc="upper left")
    ax2.set_ylabel("Drawdown (%)")
    return _save(fig, path)


def plot_model_selection(sel: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for col, c in [("bic", PALETTE[0]), ("aic", PALETTE[1])]:
        ax.plot(sel.index, sel[col], marker="o", ms=6, color=c, label=col.upper())
        k = sel[col].idxmin()
        ax.scatter([k], [sel.loc[k, col]], s=120, facecolors="none", edgecolors=c, lw=2, zorder=3)
    ax.set_xlabel("Nombre d'états cachés K")
    ax.set_ylabel("Critère (plus bas = meilleur)")
    ax.set_title("Sélection du nombre de régimes", loc="left")
    ax.set_xticks(sel.index)
    ax.legend()
    return _save(fig, path)


def plot_transition_matrix(A: np.ndarray, labels: list[str], path=None):
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#2a78d6", "#0d3b75"])
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.1 * len(labels) + 1.5))
    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("état t+1")
    ax.set_ylabel("état t")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center",
                    color="white" if A[i, j] > 0.5 else INK, fontsize=10)
    ax.set_title("Matrice de transition estimée", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_convergence(log_likelihoods: list[float], path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, color=PALETTE[0])
    ax.set_xlabel("Itération de Baum-Welch")
    ax.set_ylabel("log-vraisemblance")
    ax.set_title("Convergence de l'algorithme EM", loc="left")
    return _save(fig, path)
