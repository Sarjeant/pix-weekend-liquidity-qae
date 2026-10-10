# charts.py - the four figures on the project page, drawn from the data and saved results.
# Each function saves one PNG into the project (Qollab lists files the run writes).
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import circuits
import data
import estimate

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"
RECORD_FRIDAY, RECORD_TUESDAY = "2026-09-04", "2026-09-08"  # record weekend, Monday 7 Sep a holiday


def _style(ax, title, xlabel, ylabel):
    """Apply the shared look: recessive grid and axes, title states the finding."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def record_weekend_drawdown():
    """Return the Friday-to-Tuesday drawdown of the 4 to 8 September 2026 holiday weekend."""
    b = data.load_balances(start="2026-09-01")
    return float(1 - b.loc[RECORD_TUESDAY] / b.loc[RECORD_FRIDAY])


def chart1_balances(path="chart1_balances.png"):
    """Friday and Monday Conta PI balances since January 2025."""
    w = data.weekend_drawdowns(data.load_balances())
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fri = np.array(w["friday"], dtype="datetime64[D]")
    ax.plot(fri, w["friday_balance"] / 1e9, color=BLUE, linewidth=2, label="Friday")
    ax.plot(fri, w["monday_balance"] / 1e9, color=ORANGE, linewidth=2, label="Following Monday")
    _style(ax, "Pix participants prefund on Friday; the weekend draws it down",
           "Weekend (Friday date)", "Total Conta PI balance (R$ billion)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def chart2_drawdowns(buffer=0.10, path="chart2_drawdowns.png"):
    """Histogram of weekend drawdowns with the buffer and the record holiday weekend marked."""
    dd = data.weekend_drawdowns(data.load_balances())["drawdown"].to_numpy() * 100
    rec = record_weekend_drawdown() * 100
    p95 = float(np.percentile(dd, 95))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.hist(dd, bins=np.arange(-12, 18, 1.5), color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.axvline(buffer * 100, color=INK, linewidth=2)
    ax.text(buffer * 100 + 0.3, ax.get_ylim()[1] * 0.92, f"{buffer:.0%} buffer\n9.3% of weekends breach",
            color=INK, fontsize=9, va="top")
    ax.axvline(rec, color=ORANGE, linewidth=2, linestyle="--")
    ax.text(rec - 0.3, ax.get_ylim()[1] * 0.6, f"4 to 8 Sep 2026\n(record, 4 days): {rec:.1f}%",
            color=INK2, fontsize=9, ha="right")
    _style(ax, f"86 weekends since January 2025: 95th percentile drawdown {p95:.1f}%",
           "Weekend drawdown: 1 - Monday / Friday balance (%)", "Number of weekends")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def chart3_amplification(rows, p_true, path="chart3_amplification.png"):
    """Measured flag probability against depth: IonQ noise model vs ideal theory and the fit."""
    depth = [r for r in rows if r.get("kind") == "depth" and r["shots"] >= 2000]
    f = estimate.fit(rows)
    mm = np.arange(0, 24)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(mm, [circuits.theory(p_true, m) for m in mm], color=INK2, linewidth=1.5,
            marker="o", markersize=3, label="Ideal theory")
    ax.plot(mm, estimate.model(f["theta"], f["eta"], mm), color=BLUE, linewidth=2, linestyle="--",
            label=f"Fit: {f['eta']*100:.2f}% signal kept per step")
    x = np.array([r["m"] for r in depth])
    y = np.array([r["ones"] / r["shots"] for r in depth])
    err = np.sqrt(y * (1 - y) / np.array([r["shots"] for r in depth]))
    ax.errorbar(x, y, yerr=1.96 * err, fmt="o", color=ORANGE, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label="IonQ Forte Enterprise 1 noise model")
    _style(ax, "Amplification survives the IonQ noise model to 23 steps (70 two-qubit gates)",
           "Grover steps m", "P(breach flag = 1)")
    ax.set_ylim(-0.02, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def chart4_precision(points, p_true, data_se, path="chart4_precision.png"):
    """Standard error against model calls: Monte Carlo line vs the three quantum results."""
    calls = np.logspace(3, 6, 50)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    ax.plot(calls, np.sqrt(p_true * (1 - p_true) / calls), color=INK2, linewidth=2,
            label="Classical Monte Carlo")
    for label, n, se in points:
        ax.plot(n, se, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        mc = np.sqrt(p_true * (1 - p_true) / n)
        ax.annotate(f"{label}\n{mc / se:.1f}x smaller error", (n, se), textcoords="offset points",
                    xytext=(-12, 10), ha="right", fontsize=8.5, color=INK)
    ax.plot([], [], "o", color=BLUE, label="Quantum amplitude estimation")
    ax.axhline(data_se, color=ORANGE, linewidth=1.5, linestyle="--",
               label=f"Uncertainty from 86 weekends of data (+/-{data_se:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 6e-2)
    _style(ax, "Same model calls, 2 to 5 times smaller error than Monte Carlo",
           "Model calls (shots x (2m + 1), log scale)", "Standard error of breach probability (log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path