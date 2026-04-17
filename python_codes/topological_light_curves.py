from __future__ import annotations

from pathlib import Path

import numpy as np
import smplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


OUTDIR = Path("blog_assets/topological-light-curves")
OUTDIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_active_light_curve(
    n_samples: int = 900,
    t_min: float = 0.0,
    t_max: float = 60.0,
    period: float = 5.5,
    phase: float = 0.5,
    flare_amplitude: float = 1.2,
    flare_center: float = 33.5,
    flare_rise: float = 0.45,
    flare_decay: float = 1.8,
    flare_support: float = 6.0,
    noise_std: float = 0.08,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Build the synthetic active light curve used in the blog post:
    periodic baseline + harmonic + asymmetric flare + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(t_min, t_max, n_samples)

    baseline = (
        0.75 * np.sin(2.0 * np.pi * t / period)
        + 0.18 * np.sin(4.0 * np.pi * t / period + phase)
    )

    flare = np.where(
        t <= flare_center,
        flare_amplitude * np.exp(-(flare_center - t) / flare_rise),
        flare_amplitude * np.exp(-(t - flare_center) / flare_decay),
    )
    flare[np.abs(t - flare_center) > flare_support] = 0.0

    noise = noise_std * rng.normal(size=t.size)
    x = baseline + flare + noise
    x = (x - x.mean()) / x.std()

    params = {
        "period": period,
        "phase": phase,
        "flare_amplitude": flare_amplitude,
        "flare_center": flare_center,
        "flare_rise": flare_rise,
        "flare_decay": flare_decay,
        "flare_support": flare_support,
        "noise_std": noise_std,
    }
    return t, x, params


def delay_embedding(x: np.ndarray, tau: int = 12, m: int = 3) -> np.ndarray:
    """
    Takens-style delay embedding.

    X_i = [x_i, x_{i+tau}, ..., x_{i+(m-1)tau}]
    """
    if m < 2:
        raise ValueError("Embedding dimension m must be at least 2.")

    n = len(x) - (m - 1) * tau
    if n <= 0:
        raise ValueError("Time series is too short for the chosen tau and m.")

    return np.column_stack([x[j : j + n] for j in range(0, m * tau, tau)])


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Full Euclidean distance matrix."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def recurrence_matrix(X: np.ndarray, quantile: float = 0.06) -> tuple[np.ndarray, float]:
    """
    Binary recurrence matrix using a distance-quantile threshold.
    """
    D = pairwise_distances(X)
    iu = np.triu_indices_from(D, k=1)
    eps = float(np.quantile(D[iu], quantile))
    R = (D <= eps).astype(float)
    return R, eps


def plot_light_curve(
    t: np.ndarray,
    x: np.ndarray,
    flare_center: float,
    outfile: Path,
) -> None:
    plt.figure(figsize=(11, 5.2), dpi=150)
    plt.plot(t, x, lw=1.5, color="k")
    plt.axvspan(flare_center - 1.5, flare_center + 3.2, color="red", alpha=0.15)
    plt.title("Synthetic active light curve: periodic baseline + transient flare")
    plt.xlabel("Time")
    plt.ylabel("Relative flux")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def plot_delay_embedding_2d(X: np.ndarray, outfile: Path) -> None:
    c = np.arange(len(X))
    plt.figure(figsize=(6.5, 6.5), dpi=150)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=c,
        cmap="viridis",
        s=18,
        linewidths=0,
        alpha=0.95,
    )
    plt.title("Delay embedding")
    plt.xlabel(r"$x(t)$")
    plt.ylabel(r"$x(t+\tau)$")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def plot_delay_embedding_3d(X: np.ndarray, outfile: Path) -> None:
    c = np.arange(len(X))
    fig = plt.figure(figsize=(7.4, 6.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=c,
        cmap="viridis",
        s=12,
        depthshade=True,
    )
    ax.set_title("3D delay embedding")
    ax.set_xlabel(r"$x(t)$", labelpad=8)
    ax.set_ylabel(r"$x(t+\tau)$", labelpad=8)
    ax.set_zlabel(r"$x(t+2\tau)$", labelpad=8)
    ax.view_init(elev=32, azim=-61)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_recurrence_matrix(R: np.ndarray, outfile: Path) -> None:
    plt.figure(figsize=(6.6, 6.2), dpi=150)
    plt.imshow(
        R,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    plt.title("Recurrence structure of the embedded trajectory")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def main() -> None:
    t, x, params = generate_synthetic_active_light_curve()

    # Delay embedding used in the post
    X = delay_embedding(x, tau=12, m=3)

    # Recurrence structure
    R, eps = recurrence_matrix(X, quantile=0.06)

    plot_light_curve(
        t=t,
        x=x,
        flare_center=params["flare_center"],
        outfile=OUTDIR / "synthetic_light_curve.png",
    )
    plot_delay_embedding_2d(X, OUTDIR / "delay_embedding_2d.png")
    plot_delay_embedding_3d(X, OUTDIR / "delay_embedding_3d.png")
    plot_recurrence_matrix(R, OUTDIR / "recurrence_matrix.png")

    print("Saved figures to:", OUTDIR)
    print("Embedding shape:", X.shape)
    print("Recurrence threshold epsilon:", eps)


if __name__ == "__main__":
    main()