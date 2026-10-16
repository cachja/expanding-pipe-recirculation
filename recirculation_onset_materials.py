#!/usr/bin/env python3
"""Reproduce the onset-comparison figures used in the manuscript.

The script is self-contained: the Peixinho--Besnard onset data are embedded
below, but an external ``peixinho2013.txt`` file in the same directory may be
used instead. The external file must contain two columns::

    Re_d    sin(alpha)

Two publication figures are generated as vector PDFs:

* ``01_semilogy_sinalpha_vs_Re.pdf``
* ``02_loglog_sinalpha_vs_Re_with_endpoint_inset.pdf``

The nonlinear-parameter normalization in the code is done in a way that the
direct BVP returns ``k_raw`` and the plotted radius-based quantity is simply
``K_R = pi * k_raw``.

* Figure 01: published data, fixed-exponent fit, finite-angle nonlinear BVP,
  and nonlinear slender prediction; no right-angle continuation.
* Figure 02: all data and all four theoretical/fit curves.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import NullFormatter, NullLocator
from scipy.integrate import solve_ivp
from scipy.optimize import root


# =============================================================================
# Paths and figure-content options
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "peixinho2013.txt"
OUT_DIR = HERE

# Available keys: "data", "fit", "bvp", "slender", "right_angle".
PLOT_CONTENT = {
    "01_semilogy": {
        "data": True,
        "fit": True,
        "bvp": True,
        "slender": True,
        "right_angle": False,
    },
    "02_loglog_inset": {
        "data": True,
        "fit": True,
        "bvp": True,
        "slender": True,
        "right_angle": True,
    },
}


# =============================================================================
# Numerical configuration and normalization
# =============================================================================

DATA_RE_TO_RADIUS_RE = 0.5

# Base resolution over the slender/intermediate range and added resolution
# between the second published data point and the right-angle endpoint.
N_CURVE = 72
N_RIGHT_ANGLE = 61
RIGHT_ANGLE_DISTANCE = 1.0e-7

# Endpoint representation and inset limits.
ENDPOINT_RE_PROXY = 1.0e-16
ENDPOINT_SIN_ALPHA = 1.0
MAIN_RE_MIN = 1.0e-3

# Nonlinear slender value and BVP settings, solving for K_RAW instead of K/pi.
# We store the exact value as init. guess, but later we recompute it on the fly.
K0_RAW = 2.21035460
K0_RADIUS = np.pi * K0_RAW

BVP_SETTINGS = dict(
    rtol=2.0e-9,
    atol=2.0e-11,
    max_step=5.0e-3,
    eps=2.0e-8,
)


# =============================================================================
# Plot styles (unchanged)
# =============================================================================

STYLE_BVP = dict(
    color="#1769AA",
    lw=2.35,
    ls="-",
    zorder=7,
    label=r"Finite-angle nonlinear radial solve",
)
STYLE_DATA_FIT = dict(
    color="#C23B33",
    lw=1.85,
    ls="--",
    zorder=8,
    label=r"Fit to Peixinho & Besnard (2013)",
)
STYLE_SLENDER = dict(
    color="#2F8F4E",
    lw=2.05,
    ls=":",
    zorder=9,
    label=r"Nonlinear slender prediction $K=K_0$",
)
STYLE_RIGHT_ANGLE = dict(
    color="#D18400",
    lw=1.65,
    ls="-.",
    zorder=10,
    label=r"First-order right-angle continuation",
)
STYLE_DATA = dict(
    s=26,
    facecolors="none",
    edgecolors="black",
    linewidths=0.95,
    zorder=12,
    label="Peixinho & Besnard (2013)",
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)


# =============================================================================
# Source data
# =============================================================================

def embedded_data():
    """Return the Peixinho--Besnard 2013 onset dataset used in the manuscript."""
    return np.array(
        [
            [ENDPOINT_RE_PROXY, 1.0],
            [20, 0.52199], [30, 0.3544], [50, 0.21759],
            [70, 0.15779], [100, 0.11224], [150, 0.0762],
            [200, 0.0579], [250, 0.04679], [300, 0.03931],
            [350, 0.03393], [400, 0.02987], [450, 0.02669],
            [500, 0.02413], [600, 0.02028], [700, 0.0175],
            [800, 0.01541], [900, 0.01377], [1000, 0.01245],
            [1100, 0.01137], [1200, 0.01046], [1300, 0.00969],
            [1400, 0.00903], [1500, 0.00845], [1600, 0.00795],
            [1700, 0.0075], [1800, 0.0071], [1900, 0.00674],
            [2000, 0.00642], [2100, 0.00613], [2200, 0.00586],
            [2300, 0.00562], [2400, 0.0054], [2450, 0.00529],
            [2500, 0.00519], [3000, 0.00436], [3500, 0.00376],
            [4000, 0.00331], [4500, 0.00296], [5000, 0.00268],
            [5500, 0.00244], [6000, 0.00225], [6500, 0.00208],
            [7000, 0.00194], [7500, 0.00182], [8000, 0.00171],
            [8500, 0.00161], [9000, 0.00153], [9500, 0.00145],
            [10000, 0.00138], [10500, 0.00132], [11000, 0.00126],
            [11500, 0.00121], [12000, 0.00116], [12500, 0.00112],
            [13000, 0.00107], [13500, 0.00104], [14000, 0.001],
        ],
        dtype=float,
    )


def load_data(path):
    """Load ``Re_d`` and ``sin(alpha)`` from ``path`` or embedded data."""
    arr = np.loadtxt(path, comments="#", dtype=float) if path.exists() else embedded_data()
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("peixinho2013.txt must contain Re_d and sin(alpha)")

    re_d = arr[:, 0].astype(float)
    sin_alpha = arr[:, 1].astype(float)
    mask = (
        np.isfinite(re_d)
        & np.isfinite(sin_alpha)
        & (re_d >= 0.0)
        & (sin_alpha > 0.0)
        & (sin_alpha <= 1.0)
    )
    re_d = re_d[mask]
    sin_alpha = sin_alpha[mask]

    # Represent the exact Re_d=0 endpoint by a positive machine-scale proxy
    # only where logarithmic plotting requires it.
    endpoint = np.isclose(sin_alpha, 1.0, rtol=0.0, atol=1.0e-12)
    re_d[endpoint] = ENDPOINT_RE_PROXY

    order = np.argsort(re_d)
    return re_d[order], sin_alpha[order]


def refined_sin_alpha_grid(sin_alpha):
    """Build the original refined grid, including right-angle clustering."""
    sa_finite = sin_alpha[sin_alpha < 1.0]
    if sa_finite.size == 0:
        raise ValueError("At least one point with sin(alpha) < 1 is required")

    sa_second = float(np.max(sa_finite))
    sa_main = np.geomspace(float(np.min(sa_finite)), sa_second, N_CURVE)
    u = np.linspace(0.0, 1.0, N_RIGHT_ANGLE)
    sa_right = sa_second + (1.0 - sa_second) * (1.0 - (1.0 - u) ** 3)
    sa_right[-1] = 1.0 - RIGHT_ANGLE_DISTANCE
    sa_num = np.unique(np.concatenate([sa_main, sa_right]))
    sa_all = np.concatenate([sa_num, [1.0]])
    return sa_num, sa_all


# =============================================================================
# Analytical curves and nonlinear BVP
# =============================================================================

def first_order_right_angle_K_radius(alpha):
    """Right-angle first-order small-Re continuation for ``K_R``."""
    alpha = np.asarray(alpha, dtype=float)
    s = np.cos(alpha)
    numerator = 70.0 * s * (1.0 + s) * (2.0 * s + 1.0) ** 2
    denominator = 32.0 * s**3 + 79.0 * s**2 + 75.0 * s + 24.0
    return numerator / denominator


def nonlinear_slender_K0_radius(alpha):
    """Leading nonlinear slender result ``K_R(alpha)=K0``."""
    return np.full_like(np.asarray(alpha, dtype=float), K0_RADIUS, dtype=float)


def radius_K_to_diameter_Re(k_radius, sin_alpha):
    """Convert ``K_R=Re_R sin(alpha)`` to diameter-based ``Re_d``."""
    return np.asarray(k_radius) / (
        DATA_RE_TO_RADIUS_RE * np.asarray(sin_alpha)
    )


def solve_full_angle(
    alpha,
    initial=(5.18643, 2.21035),
    *,
    rtol=2.0e-9,
    atol=2.0e-11,
    max_step=5.0e-3,
    eps=2.0e-8,
):
    """Solve the finite-angle critical BVP by two-parameter shooting."""
    s = np.cos(alpha)
    d = 1.0 - s
    y_end = 1.0 - eps

    def mismatch(par):
        lam, k_raw = par

        def rhs(y, state):
            g, p, integral = state
            A = (1.0 - y) * (1.0 + s + d * y)
            return (
                p / A,
                lam - 6.0 * d * g - k_raw * g * g,
                g,
            )

        sol = solve_ivp(
            rhs,
            (0.0, y_end),
            (0.0, 0.0, 0.0),
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not sol.success:
            return np.array([1.0e6, 1.0e6])

        g, p, integral = sol.y[:, -1]
        integral_at_one = integral + eps * g
        return np.array([integral_at_one - 1.0, p])

    answer = root(
        mismatch,
        np.asarray(initial, dtype=float),
        method="hybr",
        tol=1.0e-10,
    )
    residual = mismatch(answer.x)
    if not answer.success or np.linalg.norm(residual, ord=np.inf) > 2.0e-7:
        raise RuntimeError(
            f"BVP solve failed at alpha={alpha:.8e}: "
            f"{answer.message}; mismatch={residual}"
        )
    return float(answer.x[0]), float(answer.x[1])


def continue_full_angle_branch(alphas):
    """Continue the nonlinear critical branch from small to larger angle."""
    alphas = np.asarray(alphas, dtype=float)
    order = np.argsort(alphas)
    k_raw = np.empty_like(alphas)
    guess = (5.18643, 2.21035)

    for index in order:
        lam, k_value = solve_full_angle(
            float(alphas[index]),
            initial=guess,
            **BVP_SETTINGS,
        )
        k_raw[index] = k_value
        guess = (lam, k_value)
    return k_raw


# =============================================================================
# Fits, plotting helpers, and output
# =============================================================================

def fixed_inverse_sine_fit(re_d, sin_alpha):
    """OLS fit ``Re_d=C_d/sin(alpha)``, excluding the exact endpoint."""
    mask = re_d >= MAIN_RE_MIN
    x = 1.0 / sin_alpha[mask]
    return np.dot(x, re_d[mask]) / np.dot(x, x)


def plot_data(ax, re_d, sin_alpha, min_re=0.0):
    """Plot the published onset data as transparent ring markers."""
    mask = re_d >= min_re
    ax.scatter(re_d[mask], sin_alpha[mask], **STYLE_DATA)


def curve_entries():
    """Return curve metadata in the original plotting order."""
    return {
        "right_angle": ("right_angle_Re_d", STYLE_RIGHT_ANGLE),
        "fit": ("data_fit_Re_d", STYLE_DATA_FIT),
        "bvp": ("bvp_Re_d", STYLE_BVP),
        "slender": ("slender_Re_d", STYLE_SLENDER),
    }


def plot_selected_curves(ax, sin_alpha, curves, selection):
    """Plot selected positive curve segments and allow axes to clip them."""
    for name, (key, style) in curve_entries().items():
        if not selection.get(name, False):
            continue
        values = np.asarray(curves[key], dtype=float)
        mask = (
            np.isfinite(values)
            & np.isfinite(sin_alpha)
            & (values > 0.0)
            & (sin_alpha > 0.0)
        )
        ax.plot(values[mask], sin_alpha[mask], **style)


def save_pdf(fig, filename, use_tight_layout=True):
    """Save a vector PDF without changing the original figure geometry."""
    path = OUT_DIR / filename
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def save_source_data(re_d, sin_alpha_data, sin_alpha_curve, curves):
    """Write the numerical data underlying the publication figures."""
    curve_path = OUT_DIR / "source_data_theoretical_curves.csv"
    np.savetxt(
        curve_path,
        np.column_stack(
            [
                sin_alpha_curve,
                curves["data_fit_Re_d"],
                curves["bvp_Re_d"],
                curves["slender_Re_d"],
                curves["right_angle_Re_d"],
            ]
        ),
        delimiter=",",
        header=(
            "sin_alpha,Re_d_fixed_exponent_fit,"
            "Re_d_finite_angle_nonlinear_radial_solve,"
            "Re_d_nonlinear_slender_prediction,"
            "Re_d_first_order_right_angle_continuation"
        ),
        comments="",
        fmt="%.12g",
    )

    data_path = OUT_DIR / "source_data_peixinho_besnard_2013.csv"
    np.savetxt(
        data_path,
        np.column_stack([sin_alpha_data, re_d]),
        delimiter=",",
        header="sin_alpha,Re_d_onset_data",
        comments="",
        fmt="%.12g",
    )

    print(f"wrote {curve_path}")
    print(f"wrote {data_path}")


# =============================================================================
# Figure generation
# =============================================================================

def make_semilogy_figure(re_d, sa, sa_curve, curves):
    """Generate Figure 01 with independently configurable content."""
    selection = PLOT_CONTENT["01_semilogy"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plot_selected_curves(ax, sa_curve, curves, selection)
    if selection.get("data", False):
        plot_data(ax, re_d, sa)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{Re}_d$")
    ax.set_ylabel(r"$\sin\alpha$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_pdf(fig, "01_semilogy_sinalpha_vs_Re.pdf")


def make_loglog_inset_figure(re_d, sa, sa_curve, curves, max_re_plot):
    """Generate Figure 02, preserving the original panel and inset layout."""
    selection = PLOT_CONTENT["02_loglog_inset"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # Main panel: same curves, order, styles, and endpoint treatment as before.
    for name, (key, style) in curve_entries().items():
        if not selection.get(name, False):
            continue
        values = np.asarray(curves[key], dtype=float)
        mask = (
            np.isfinite(values)
            & (values > 0.0)
            & np.isfinite(sa_curve)
            & (sa_curve > 0.0)
        )
        x_values = values[mask]
        y_values = sa_curve[mask]
        if name == "bvp":
            x_values = np.r_[x_values, ENDPOINT_RE_PROXY]
            y_values = np.r_[y_values, ENDPOINT_SIN_ALPHA]
        ax.plot(x_values, y_values, **style)

    if selection.get("data", False):
        plot_data(ax, re_d, sa)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-1, max_re_plot)
    ax.set_ylim(0.8 * np.min(sa[sa > 0.0]), 1.05)
    ax.set_xlabel(r"$\mathrm{Re}_d$")
    ax.set_ylabel(r"$\sin\alpha$")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)

    # Right-angle endpoint inset: original position and dimensions.
    ax_inset = ax.inset_axes([0.075, 0.335, 0.38, 0.36])
    inset_entries = (
        ("right_angle", "right_angle_Re_d", STYLE_RIGHT_ANGLE),
        ("bvp", "bvp_Re_d", STYLE_BVP),
    )
    for name, key, style in inset_entries:
        if not selection.get(name, False):
            continue
        values = np.asarray(curves[key], dtype=float)
        mask = (
            np.isfinite(values)
            & (values > 0.0)
            & np.isfinite(sa_curve)
            & (sa_curve >= 0.98)
        )
        ax_inset.plot(
            np.r_[values[mask], ENDPOINT_RE_PROXY],
            np.r_[sa_curve[mask], ENDPOINT_SIN_ALPHA],
            color=style["color"],
            lw=1.45,
            ls=style["ls"],
            zorder=style["zorder"],
        )

    ax_inset.set_xscale("log")
    ax_inset.set_xlim(0.7e-16, 1.4e-14)
    ax_inset.set_ylim(0.98, 1.001)
    ax_inset.set_xticks([1.0e-16, 1.0e-15, 1.0e-14])
    ax_inset.set_xticklabels([r"$10^{-16}$", r"$10^{-15}$", r"$10^{-14}$"])
    ax_inset.xaxis.set_minor_locator(NullLocator())
    ax_inset.xaxis.set_minor_formatter(NullFormatter())
    ax_inset.set_yticks([0.98, 0.99, 1.00])
    ax_inset.tick_params(
        axis="both", which="major", labelsize=7.2,
        direction="out", length=3, width=0.8,
    )
    ax_inset.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax_inset.tick_params(axis="y", which="minor", left=False, right=False)
    ax_inset.set_title("Right-angle endpoint", fontsize=8.5, pad=3)
    ax_inset.grid(False, which="both")
    ax_inset.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.65)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")

    # Exact endpoint marker and annotation.
    ax_inset.scatter(
        [ENDPOINT_RE_PROXY],
        [ENDPOINT_SIN_ALPHA],
        s=38,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=12,
        clip_on=False,
    )
    ax_inset.annotate(
        r"exact endpoint: $\mathrm{Re}=0$",
        xy=(ENDPOINT_RE_PROXY, ENDPOINT_SIN_ALPHA),
        xycoords="data",
        xytext=(2.0e-16, 0.9975),
        textcoords="data",
        fontsize=7.7,
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="-", color="0.25", lw=0.75,
            shrinkA=2, shrinkB=3,
        ),
        zorder=13,
    )

    # Original guide-line geometry between the main panel and inset.
    connector_upper = ConnectionPatch(
        xyA=(1.0e-1, 1.0),
        coordsA=ax.transData,
        xyB=(1.0, 1.0),
        coordsB=ax_inset.transAxes,
        color="0.40",
        lw=0.8,
        alpha=0.85,
        zorder=2,
        clip_on=False,
    )
    connector_lower = ConnectionPatch(
        xyA=(1.0e-1, 1.0),
        coordsA=ax.transData,
        xyB=(0.0, 1.0),
        coordsB=ax_inset.transAxes,
        color="0.40",
        lw=0.8,
        alpha=0.85,
        zorder=2,
        clip_on=False,
    )
    fig.add_artist(connector_upper)
    fig.add_artist(connector_lower)

    ax.legend(
        loc="lower left",
        fontsize=8.2,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.85",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.97)
    save_pdf(
        fig,
        "02_loglog_sinalpha_vs_Re_with_endpoint_inset.pdf",
        use_tight_layout=False,
    )


def main():
    """Load data, evaluate all requested curves, and write both PDFs."""
    re_d, sa = load_data(DATA_FILE)
    sa_curve_num, sa_curve = refined_sin_alpha_grid(sa)
    alpha_curve_num = np.arcsin(sa_curve_num)
    alpha_curve = np.arcsin(sa_curve)

    lambda0, k0_raw_fresh = solve_full_angle(0.0, initial=(5.18643, K0_RAW), **dict(BVP_SETTINGS, eps=2.0e-10))
    K0_fresh = np.pi * k0_raw_fresh

    C_d_fit = fixed_inverse_sine_fit(re_d, sa)
    k0_R = nonlinear_slender_K0_radius(alpha_curve)
    k_right_angle_R = first_order_right_angle_K_radius(alpha_curve)

    # Evaluate the nonlinear BVP once if either publication figure requests it.
    bvp_requested = any(
        options.get("bvp", False) for options in PLOT_CONTENT.values()
    )
    if bvp_requested:
        k_raw_num = continue_full_angle_branch(alpha_curve_num)
        k_bvp_R_num = np.pi * k_raw_num
        k_bvp_R = np.concatenate([k_bvp_R_num, [0.0]])
    else:
        k_bvp_R = np.full_like(sa_curve, np.nan)

    curves = {
        "bvp_Re_d": radius_K_to_diameter_Re(k_bvp_R, sa_curve),
        "data_fit_Re_d": C_d_fit / sa_curve,
        "slender_Re_d": radius_K_to_diameter_Re(k0_R, sa_curve),
        "right_angle_Re_d": radius_K_to_diameter_Re(k_right_angle_R, sa_curve),
    }
    max_re_plot = 1.05 * max(
        np.max(re_d),
        *(np.nanmax(values) for values in curves.values()),
    )

    print(f"data source                             : {DATA_FILE if DATA_FILE.exists() else 'embedded'}")
    print(f"fixed-exponent fit coefficient C_1      : {DATA_RE_TO_RADIUS_RE*C_d_fit:.10f}")
    print(f"nonlinear slender coefficient K0        : {K0_fresh:.10f}")

    save_source_data(re_d, sa, sa_curve, curves)

    make_semilogy_figure(re_d, sa, sa_curve, curves)
    make_loglog_inset_figure(re_d, sa, sa_curve, curves, max_re_plot)


if __name__ == "__main__":
    main()
