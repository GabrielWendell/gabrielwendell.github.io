"""Reproduce the figures in "Topological Representation of Stellar Spectra".

The spectrum is deliberately pedagogical: it is a continuum-normalized blend
of Gaussian absorption lines near the Mg I b triplet, convolved with a Gaussian
kernel that acts as an *effective* instrumental/velocity broadening.  It is not
intended to replace a radiative-transfer synthesis code.

Dependencies
------------
numpy, scipy, matplotlib

Run
---
python code/topological_stellar_spectra.py

The script writes three PNG files to the package's ``assets`` directory and
prints the summary metrics quoted in the accompanying article.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d


C_KMS = 299_792.458
LAMBDA_REF = 5175.0
NOISE_SIGMA = 0.004
ROBUST_CUTOFF = 0.030

BG = "#070b17"
AX_BG = "#0e1628"
FG = "#e6edf7"
MUTED = "#a4acc4"
GRID = "#26324d"
CYAN = "#5ac8fa"
MAGENTA = "#c084fc"
GOLD = "#fbbf24"
CORAL = "#fb7185"
GREEN = "#34d399"


@dataclass(frozen=True)
class Line:
    species: str
    wavelength: float
    depth: float
    sigma: float


LINE_LIST = (
    Line("Fe I", 5162.27, 0.055, 0.075),
    Line("Fe I", 5166.28, 0.090, 0.080),
    Line("Mg I", 5167.32, 0.300, 0.105),
    Line("Fe I", 5171.60, 0.065, 0.075),
    Line("Mg I", 5172.68, 0.420, 0.115),
    Line("Fe I", 5180.27, 0.055, 0.070),
    Line("Mg I", 5183.60, 0.500, 0.125),
    Line("Fe I", 5184.80, 0.085, 0.080),
    Line("Ti I", 5188.69, 0.050, 0.075),
)


def synthetic_spectrum(
    wavelength: np.ndarray,
    effective_fwhm_kms: float,
    *,
    noise_sigma: float = NOISE_SIGMA,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return noisy normalized flux, noiseless flux, and the imposed continuum.

    The input line list is convolved with a Gaussian whose FWHM is expressed in
    velocity units.  This is a transparent broadening proxy, not a detailed
    rotational kernel or atmosphere calculation.
    """

    absorption = np.zeros_like(wavelength)
    for line in LINE_LIST:
        z = (wavelength - line.wavelength) / line.sigma
        absorption += line.depth * np.exp(-0.5 * z * z)

    intrinsic_flux = np.clip(1.0 - absorption, 0.02, None)
    dlambda = float(np.median(np.diff(wavelength)))
    sigma_lambda = LAMBDA_REF * effective_fwhm_kms / C_KMS / 2.354820045
    sigma_pixels = sigma_lambda / dlambda
    clean_normalized = gaussian_filter1d(
        intrinsic_flux, sigma=sigma_pixels, mode="nearest"
    )

    x = (wavelength - wavelength.mean()) / np.ptp(wavelength)
    continuum = 1.0 + 0.018 * x + 0.012 * (x * x - 1.0 / 12.0)
    rng = np.random.default_rng(seed)
    observed = continuum * clean_normalized + rng.normal(
        0.0, noise_sigma, wavelength.size
    )
    normalized = observed / continuum
    return normalized, clean_normalized, continuum


def lower_star_h0(
    values: np.ndarray,
    *,
    terminal_level: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute degree-zero lower-star persistence on a one-dimensional path.

    Vertices enter in increasing function-value order, and an edge enters when
    both endpoints are active.  At a merge, the younger component dies (elder
    rule).  The sole essential component is capped at ``terminal_level`` for
    spectroscopy-friendly display; its row is marked in the returned Boolean
    array.

    Returns
    -------
    pairs:
        Array with columns ``birth, death, birth_index, death_index``.
    essential:
        Boolean mask identifying the terminally capped class.
    """

    f = np.asarray(values, dtype=float)
    n = f.size
    parent = np.arange(n)
    birth = f.copy()
    birth_index = np.arange(n)
    active = np.zeros(n, dtype=bool)
    records: list[tuple[float, float, int, int]] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def older_root(a: int, b: int) -> tuple[int, int]:
        key_a = (birth[a], birth_index[a])
        key_b = (birth[b], birth_index[b])
        return (a, b) if key_a <= key_b else (b, a)

    order = np.argsort(f, kind="stable")
    for i in order:
        active[i] = True
        parent[i] = i
        birth[i] = f[i]
        birth_index[i] = i

        for j in (i - 1, i + 1):
            if j < 0 or j >= n or not active[j]:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            old, young = older_root(ri, rj)
            death = max(f[i], f[j])
            records.append((birth[young], death, birth_index[young], i))
            parent[young] = old
            parent[ri] = old
            parent[rj] = old

    root = find(int(np.argmin(f)))
    records.append((birth[root], terminal_level, birth_index[root], -1))
    pairs = np.asarray(records, dtype=float)
    persistence = pairs[:, 1] - pairs[:, 0]
    positive = persistence > 1e-12
    pairs = pairs[positive]
    essential = pairs[:, 3] < 0
    return pairs, essential


def beta0_curve(
    pairs: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Evaluate the Betti-0 curve from half-open persistence intervals."""

    births = pairs[:, 0, None]
    deaths = pairs[:, 1, None]
    return np.sum((births <= levels[None, :]) & (levels[None, :] < deaths), axis=0)


def summary_metrics(
    wavelength: np.ndarray,
    normalized: np.ndarray,
    clean: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, float | int]:
    persistence = pairs[:, 1] - pairs[:, 0]
    robust = persistence >= ROBUST_CUTOFF
    robust_p = persistence[robust]
    return {
        "equivalent_width_A": float(trapezoid(1.0 - clean, wavelength)),
        "maximum_depth": float(1.0 - clean.min()),
        "robust_components": int(robust.sum()),
        "total_robust_persistence": float(robust_p.sum()),
        "largest_persistence": float(robust_p.max(initial=0.0)),
        "all_finite_or_capped_pairs": int(pairs.shape[0]),
        "flux_min_noisy": float(normalized.min()),
    }


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.55)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax._left_title.set_color(FG)
    ax._right_title.set_color(FG)


def add_line_labels(ax: plt.Axes, clean_flux: np.ndarray, wavelength: np.ndarray) -> None:
    for k, line in enumerate(LINE_LIST):
        idx = int(np.argmin(np.abs(wavelength - line.wavelength)))
        y = clean_flux[idx]
        y_text = max(0.405, y - 0.075 - 0.035 * (k % 2))
        ax.annotate(
            f"{line.species}\n{line.wavelength:.2f}",
            xy=(line.wavelength, y),
            xytext=(line.wavelength, y_text),
            ha="center",
            va="top",
            fontsize=7.0,
            color=MUTED,
            arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.7},
        )


def plot_diagram(
    ax: plt.Axes,
    pairs: np.ndarray,
    essential: np.ndarray,
    *,
    title: str,
) -> None:
    persistence = pairs[:, 1] - pairs[:, 0]
    robust = persistence >= ROBUST_CUTOFF
    lo = min(0.44, float(pairs[:, 0].min()) - 0.015)
    hi = 1.015
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=MUTED, alpha=0.65)
    ax.scatter(
        pairs[~robust, 0],
        pairs[~robust, 1],
        s=10,
        color=MUTED,
        alpha=0.28,
        linewidths=0,
        label="low persistence",
    )
    finite_robust = robust & ~essential
    ax.scatter(
        pairs[finite_robust, 0],
        pairs[finite_robust, 1],
        s=38,
        color=MAGENTA,
        edgecolors="#f3e8ff",
        linewidths=0.45,
        alpha=0.95,
        label=f"persistence ≥ {ROBUST_CUTOFF:.3f}",
    )
    if np.any(essential):
        ax.scatter(
            pairs[essential, 0],
            pairs[essential, 1],
            marker="D",
            s=48,
            color=GOLD,
            edgecolors="#fff7cc",
            linewidths=0.5,
            label="terminally capped class",
            zorder=5,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Birth flux")
    ax.set_ylabel("Death flux")
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    style_axis(ax)


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    jumps = np.diff(padded)
    starts = np.flatnonzero(jumps == 1)
    stops = np.flatnonzero(jumps == -1)
    return list(zip(starts, stops, strict=True))


def save_overview(
    out: Path,
    wavelength: np.ndarray,
    normalized: np.ndarray,
    clean: np.ndarray,
    pairs: np.ndarray,
    essential: np.ndarray,
) -> None:
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.65, 1.0]}
    )
    fig.patch.set_facecolor(BG)
    ax0.plot(wavelength, normalized, color=CYAN, lw=0.65, alpha=0.72, label="noisy")
    ax0.plot(wavelength, clean, color=FG, lw=1.25, label="noiseless")
    ax0.axhline(1.0, color=MUTED, ls="--", lw=0.9, alpha=0.65)
    ax0.set_xlim(wavelength[0], wavelength[-1])
    ax0.set_ylim(0.40, 1.035)
    ax0.set_xlabel(r"Wavelength $\lambda$ [Å]")
    ax0.set_ylabel("Continuum-normalized flux")
    ax0.set_title("A  Synthetic Mg I b-region spectrum", loc="left", fontsize=11, weight="bold")
    add_line_labels(ax0, clean, wavelength)
    style_axis(ax0)
    ax0.legend(frameon=False, labelcolor=FG, fontsize=8, loc="lower left")

    plot_diagram(ax1, pairs, essential, title="B  Degree-zero persistence diagram")
    ax1.legend(frameon=False, labelcolor=FG, fontsize=7.6, loc="upper left")

    fig.suptitle(
        "From absorption troughs to a topological representation",
        color=FG,
        fontsize=15,
        weight="bold",
        x=0.055,
        y=0.995,
        ha="left",
    )
    fig.text(
        0.055,
        0.93,
        "Prominent minima lie far from the diagonal; noise-generated minima cluster near it.",
        color=MUTED,
        fontsize=9.3,
    )
    fig.tight_layout(rect=(0.035, 0.03, 0.99, 0.89))
    fig.savefig(out, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_filtration(
    out: Path,
    wavelength: np.ndarray,
    clean: np.ndarray,
) -> None:
    # The final level is infinitesimally above the normalized continuum so that
    # round-off values at 1.0 cannot split the terminal connected component.
    thresholds = (0.68, 0.96, 1.00001)
    colors = (CORAL, MAGENTA, CYAN)
    fig, axes = plt.subplots(4, 1, figsize=(11.6, 8.6), sharex=True)
    fig.patch.set_facecolor(BG)

    ax0 = axes[0]
    ax0.plot(wavelength, clean, color=FG, lw=1.45)
    for threshold, color in zip(thresholds, colors, strict=True):
        ax0.axhline(threshold, color=color, ls="--", lw=1.0, alpha=0.9)
    ax0.set_ylabel("Normalized flux")
    ax0.set_ylim(0.40, 1.02)
    ax0.set_title("A  The spectrum and three filtration levels", loc="left", fontsize=11, weight="bold")
    style_axis(ax0)

    for panel, threshold, color in zip(axes[1:], thresholds, colors, strict=True):
        mask = clean <= threshold
        regions = contiguous_regions(mask)
        panel.plot(wavelength, clean, color=MUTED, lw=0.85, alpha=0.52)
        panel.axhline(threshold, color=color, ls="--", lw=1.0)
        for start, stop in regions:
            x0 = wavelength[start]
            x1 = wavelength[min(stop - 1, wavelength.size - 1)]
            panel.axvspan(x0, x1, ymin=0.03, ymax=0.96, color=color, alpha=0.28)
            panel.hlines(
                threshold,
                x0,
                x1,
                color=color,
                linewidth=4.0,
                zorder=4,
            )
        panel.text(
            0.012,
            0.12,
            rf"$S_{{{threshold:.2f}}}=\{{\lambda:f(\lambda)\leq {threshold:.2f}\}}$   "
            + rf"$\beta_0={len(regions)}$",
            transform=panel.transAxes,
            color=FG,
            fontsize=9,
            bbox={"facecolor": BG, "edgecolor": GRID, "alpha": 0.88, "pad": 5},
        )
        panel.set_ylim(0.40, 1.02)
        panel.set_ylabel("Flux")
        style_axis(panel)

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [Å]")
    fig.suptitle(
        "Sublevel-set filtration: components are born and then merge",
        color=FG,
        fontsize=15,
        weight="bold",
        x=0.06,
        y=0.995,
        ha="left",
    )
    fig.text(
        0.06,
        0.955,
        "Colored intervals are the connected components present at each flux threshold.",
        color=MUTED,
        fontsize=9.3,
    )
    fig.tight_layout(rect=(0.045, 0.035, 0.99, 0.925))
    fig.savefig(out, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_comparison(
    out: Path,
    wavelength: np.ndarray,
    cases: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0))
    fig.patch.set_facecolor(BG)
    levels = np.linspace(0.42, 1.0, 260)

    for row, case in enumerate(cases):
        normalized = np.asarray(case["normalized"])
        clean = np.asarray(case["clean"])
        pairs = np.asarray(case["pairs"])
        essential = np.asarray(case["essential"])
        metrics = case["metrics"]
        label = str(case["label"])
        color = CYAN if row == 0 else MAGENTA

        ax = axes[row, 0]
        ax.plot(wavelength, normalized, color=color, lw=0.55, alpha=0.58)
        ax.plot(wavelength, clean, color=FG, lw=1.15)
        ax.axhline(1.0, color=MUTED, ls="--", lw=0.8, alpha=0.6)
        ax.set_xlim(wavelength[0], wavelength[-1])
        ax.set_ylim(0.42, 1.035)
        ax.set_xlabel(r"Wavelength $\lambda$ [Å]")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"{chr(65 + row * 3)}  {label}", loc="left", fontsize=11, weight="bold")
        style_axis(ax)

        ax = axes[row, 1]
        plot_diagram(
            ax,
            pairs,
            essential,
            title=f"{chr(66 + row * 3)}  Persistence diagram",
        )

        ax = axes[row, 2]
        persistence = pairs[:, 1] - pairs[:, 0]
        robust_pairs = pairs[persistence >= ROBUST_CUTOFF]
        beta = beta0_curve(robust_pairs, levels)
        ax.step(levels, beta, where="post", color=color, lw=1.6)
        ax.axvline(1.0, color=MUTED, ls="--", lw=0.8, alpha=0.55)
        ax.set_xlabel("Flux threshold")
        ax.set_ylabel(r"$\beta_0$")
        ax.set_title(
            f"{chr(67 + row * 3)}  Robust Betti-0 curve",
            loc="left",
            fontsize=11,
            weight="bold",
        )
        ax.set_xlim(levels[0], levels[-1])
        style_axis(ax)
        ax.text(
            0.05,
            0.93,
            "Robust bars: " + str(metrics["robust_components"]) + "\n"
            + r"$\sum p_i$: " + f"{metrics['total_robust_persistence']:.3f}" + "\n"
            + "Max persistence: " + f"{metrics['largest_persistence']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=FG,
            fontsize=8.6,
            bbox={"facecolor": BG, "edgecolor": GRID, "alpha": 0.9, "pad": 6},
        )

    fig.suptitle(
        "Broadening changes the line topology while nearly preserving equivalent width",
        color=FG,
        fontsize=15,
        weight="bold",
        x=0.045,
        y=0.995,
        ha="left",
    )
    fig.text(
        0.045,
        0.955,
        "A Gaussian broadening proxy blends neighboring troughs, reducing the number and total persistence of robust components.",
        color=MUTED,
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.995, 0.925))
    fig.savefig(out, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    assets = package_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    wavelength = np.linspace(5158.0, 5191.0, 4200)
    configurations = (
        ("Narrow-line case · FWHM 18 km s⁻¹", 18.0, 11),
        ("Broadened case · FWHM 85 km s⁻¹", 85.0, 23),
    )
    cases: list[dict[str, object]] = []

    for label, fwhm, seed in configurations:
        normalized, clean, continuum = synthetic_spectrum(
            wavelength,
            fwhm,
            noise_sigma=NOISE_SIGMA,
            seed=seed,
        )
        pairs, essential = lower_star_h0(normalized, terminal_level=1.0)
        metrics = summary_metrics(wavelength, normalized, clean, pairs)
        cases.append(
            {
                "label": label,
                "fwhm": fwhm,
                "normalized": normalized,
                "clean": clean,
                "continuum": continuum,
                "pairs": pairs,
                "essential": essential,
                "metrics": metrics,
            }
        )

    save_overview(
        assets / "stellar-spectrum-topology-overview.png",
        wavelength,
        np.asarray(cases[0]["normalized"]),
        np.asarray(cases[0]["clean"]),
        np.asarray(cases[0]["pairs"]),
        np.asarray(cases[0]["essential"]),
    )
    save_filtration(
        assets / "stellar-spectrum-sublevel-filtration.png",
        wavelength,
        np.asarray(cases[0]["clean"]),
    )
    save_comparison(
        assets / "stellar-spectrum-broadening-comparison.png",
        wavelength,
        cases,
    )

    print("Synthetic experiment metrics")
    print(f"noise_sigma={NOISE_SIGMA:.4f}")
    print(f"robust_cutoff={ROBUST_CUTOFF:.4f}")
    for case in cases:
        print(f"\n{case['label']}")
        for key, value in case["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
