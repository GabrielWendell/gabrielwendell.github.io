"""Reproducible example for the blog post
"A Topological Representation of Exoplanet Transits".

The script generates a limb-darkened transit with batman, constructs a
circular delay-coordinate embedding of one phase-folded orbital cycle, computes
Vietoris--Rips persistence with Ripser.py, and saves a four-panel summary figure.

Install dependencies with

    python -m pip install numpy matplotlib scikit-learn batman-package ripser persim
"""

from __future__ import annotations

from pathlib import Path

import batman
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from persim import plot_diagrams
from ripser import ripser
from sklearn.decomposition import PCA


SEED = 7
OUTPUT = Path(__file__).with_name("topological_exoplanet_transit.png")


def make_transit(
    time: np.ndarray,
    *,
    period: float = 3.0,
    radius_ratio: float = 0.10,
    scaled_semimajor_axis: float = 12.0,
    inclination: float = 87.6,
    limb_darkening: tuple[float, float] = (0.30, 0.20),
) -> np.ndarray:
    """Evaluate a circular, quadratic-limb-darkened transit model."""
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = period
    params.rp = radius_ratio
    params.a = scaled_semimajor_axis
    params.inc = inclination
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(limb_darkening)
    params.limb_dark = "quadratic"

    model = batman.TransitModel(params, time)
    return model.light_curve(params)


def circular_delay_embedding(
    signal: np.ndarray,
    *,
    dimension: int = 25,
    lag: int = 2,
    stride: int = 3,
) -> np.ndarray:
    """Create a periodic delay embedding of one phase-folded orbital cycle.

    Modular indexing joins the last phase sample to the first. This is
    appropriate only because ``signal`` represents one complete periodic cycle.
    It should not be used blindly on an isolated, non-periodic transit segment.
    """
    signal = np.asarray(signal, dtype=float)
    scale = signal.std()
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("The signal must have non-zero finite variance.")

    standardized = (signal - signal.mean()) / scale
    starts = np.arange(0, standardized.size, stride)
    offsets = lag * np.arange(dimension)
    indices = (starts[:, None] + offsets[None, :]) % standardized.size
    return standardized[indices]


def persistence(signal: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the point cloud and its H0/H1 persistence diagrams."""
    cloud = circular_delay_embedding(signal)
    result = ripser(cloud, maxdim=1, n_perm=min(300, len(cloud)))
    return cloud, result["dgms"]


def dominant_h1(diagrams: list[np.ndarray]) -> float:
    """Length of the most persistent one-dimensional class."""
    h1 = diagrams[1]
    return float(np.max(h1[:, 1] - h1[:, 0])) if h1.size else 0.0


def main() -> None:
    rng = np.random.default_rng(SEED)

    period = 3.0  # days
    n_samples = 1_200
    time = np.linspace(-period / 2, period / 2, n_samples, endpoint=False)
    phase = time / period

    clean_flux = make_transit(time, period=period)
    sigma_ppm = 200.0
    noisy_flux = clean_flux + rng.normal(0.0, sigma_ppm * 1e-6, n_samples)

    clean_cloud, clean_diagrams = persistence(clean_flux)
    noisy_cloud, noisy_diagrams = persistence(noisy_flux)

    # Fit one projection to both clouds so their geometry is directly comparable.
    pca = PCA(n_components=2)
    pca.fit(np.vstack([clean_cloud, noisy_cloud]))
    clean_2d = pca.transform(clean_cloud)
    noisy_2d = pca.transform(noisy_cloud)
    embedded_phase = phase[::3]

    noise_grid_ppm = np.array([0, 50, 100, 200, 500, 1_000, 2_000])
    h1_scores = []
    for sigma in noise_grid_ppm:
        trial = clean_flux + rng.normal(0.0, sigma * 1e-6, n_samples)
        _, diagrams = persistence(trial)
        h1_scores.append(dominant_h1(diagrams))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    fig.patch.set_facecolor("#050914")
    for ax in axes.flat:
        ax.set_facecolor("#0d1224")

    ax = axes[0, 0]
    ax.plot(phase, clean_flux, color="#5ac8fa", lw=2.2, label="noise-free")
    ax.scatter(
        phase,
        noisy_flux,
        s=5,
        color="#d5a6ff",
        alpha=0.42,
        linewidths=0,
        label=f"{sigma_ppm:.0f} ppm noise",
    )
    ax.set_xlim(-0.065, 0.065)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Relative flux")
    ax.set_title("Limb-darkened synthetic transit")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    norm = Normalize(vmin=-0.5, vmax=0.5)
    ax.plot(clean_2d[:, 0], clean_2d[:, 1], color="#5ac8fa", lw=1.4, alpha=0.8)
    points = ax.scatter(
        noisy_2d[:, 0],
        noisy_2d[:, 1],
        c=embedded_phase,
        cmap="twilight",
        norm=norm,
        s=13,
        alpha=0.78,
        linewidths=0,
    )
    fig.colorbar(points, ax=ax, label="Orbital phase", shrink=0.82)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Circular delay embedding (PCA view)")

    ax = axes[1, 0]
    plot_diagrams(
        noisy_diagrams,
        ax=ax,
        show=False,
        lifetime=False,
        legend=True,
    )
    ax.set_title("Vietoris–Rips persistence")
    ax.grid(alpha=0.12)

    ax = axes[1, 1]
    ax.plot(noise_grid_ppm, h1_scores, marker="o", color="#5ac8fa", lw=2.2)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_xlabel("White-noise amplitude [ppm]")
    ax.set_ylabel(r"Dominant $H_1$ persistence")
    ax.set_title("Survival of the main loop")
    ax.grid(alpha=0.18)

    fig.suptitle(
        "From an exoplanet transit to a persistent topological signature",
        fontsize=16,
        fontweight="bold",
        color="white",
    )
    fig.savefig(OUTPUT, dpi=220, facecolor=fig.get_facecolor())

    print(f"Saved: {OUTPUT}")
    print(f"Clean dominant H1 persistence: {dominant_h1(clean_diagrams):.3f}")
    print(f"Noisy dominant H1 persistence: {dominant_h1(noisy_diagrams):.3f}")


if __name__ == "__main__":
    main()
