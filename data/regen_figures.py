"""Regenerate the two analysis figures with blog-friendly panel names and
cleaner styling.

Outputs:
  blog/assets/divergence_vs_top1.png   — KL(warp || base) bar chart, 2 panels
                                          (LogitMLP pass@1 + LogitMLP pass@k)
  blog/assets/per_token_T_hist.png     — Histograms of per-token best-fit T,
                                          2 panels (one per variant)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("/home/timothygao/extra_layer_rlvr")
PV = ROOT / "analysis_qwen_pl_warp" / "per_variant"
OUT = ROOT / "blog" / "assets"

# Blog-friendly variant labels for the panels.
LABEL = {
    "pl_warp_dapo":  "LogitMLP (DAPO / pass@1-trained)",
    "pl_warp_passk": "LogitMLP (pass@k-trained)",
}
COLOR = {
    "pl_warp_dapo":  "#2b6cb0",   # blue
    "pl_warp_passk": "#c7541d",   # orange (the blog accent)
}


def fig_divergence():
    """Two-panel KL(warp || base) by base top-1 prob bin."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    edges = np.array([0, .5, .7, .85, .95, .98, .995, 1.0001])
    # Build human-readable bin labels: "0.5–0.7", "0.85–0.95", "≥0.995" etc.
    bin_labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > 1.0:
            bin_labels.append(f"≥ {lo:.3g}")
        else:
            bin_labels.append(f"{lo:.3g}–{hi:.3g}")

    for ax, var in zip(axes, ("pl_warp_dapo", "pl_warp_passk")):
        d = torch.load(PV / f"{var}.pt", weights_only=False, map_location="cpu")
        top1_b = d["per_pos"]["top1_prob_base"].numpy()
        kl = d["per_pos"]["kl_method_base"].numpy()

        means, p95s, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (top1_b >= lo) & (top1_b < hi)
            n = int(sel.sum())
            v = kl[sel] if n else np.zeros(1)
            means.append(float(v.mean()))
            p95s.append(float(np.quantile(v, 0.95)))
            ns.append(n)

        # Combine bin label and sample size into a single tick label
        tick_labels = [f"{lbl}\n(n={n:,})" for lbl, n in zip(bin_labels, ns)]

        x = np.arange(len(bin_labels))
        c = COLOR[var]
        ax.bar(x - 0.18, means, 0.34, color=c, label="mean KL", zorder=3)
        ax.bar(x + 0.18, p95s,  0.34, color=c, alpha=0.40, label="95th percentile",
               edgecolor=c, linewidth=0.8, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 0.5)
        ax.grid(True, axis="y", which="major", alpha=0.25, zorder=0)
        ax.set_xlabel("base distribution's top-1 probability", fontsize=10.5,
                      labelpad=8)
        if var == "pl_warp_dapo":
            ax.set_ylabel("KL(warp ‖ base)   [nats, log scale]", fontsize=10.5)
        ax.set_title(LABEL[var], fontsize=12, pad=8)
        ax.legend(fontsize=9, frameon=False, loc="upper right")

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("LogitMLP barely touches confident positions; it acts mainly on moderate-confidence tokens",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = OUT / "divergence_vs_top1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("→", out)


def fig_per_token_T():
    """Histograms of per-token best-fit T for each LogitMLP variant.

    Vectorized: for each record we evaluate KL against the entire T_GRID
    in one matrix operation rather than looping over T values."""
    T_GRID = torch.linspace(0.2, 3.0, 141)  # 141 points = 0.02 step

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)

    for ax, var in zip(axes, ("pl_warp_dapo", "pl_warp_passk")):
        d = torch.load(PV / f"{var}.pt", weights_only=False, map_location="cpu")
        records = [r for r in d["full_dist_records"] if r["is_active"]]

        # Stack into (n_active, V) tensors so we can vectorize the T sweep.
        logp_b = torch.stack([r["logp_base"] for r in records])    # (N, V)
        logp_m = torch.stack([r["logp_method"] for r in records])  # (N, V)
        p_m = torch.softmax(logp_m, dim=-1)
        log_p_m = (p_m + 1e-20).log()
        # KL(p_m || softmax(logp_b/T)) = sum p_m * (log p_m - log_softmax(logp_b/T))
        # log_softmax(logp_b / T) = (logp_b / T) - logsumexp(logp_b / T)
        # We compute for each T in T_GRID:
        T_per = torch.empty(len(records))
        # do in chunks of T to avoid blowing memory (N*V*|T| can be big)
        Tlist = T_GRID.tolist()
        kl_chunks = []
        for T in Tlist:
            scaled = logp_b / T  # (N, V)
            ls = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)  # log softmax
            kl = (p_m * (log_p_m - ls)).sum(dim=-1)  # (N,)
            kl_chunks.append(kl)
        kl_mat = torch.stack(kl_chunks, dim=-1)  # (N, |T|)
        best_idx = kl_mat.argmin(dim=-1)
        T_per = T_GRID[best_idx].numpy()
        c = COLOR[var]
        ax.hist(T_per, bins=60, color=c, alpha=0.78, edgecolor=c, linewidth=0.6,
                label="per-token best-fit T")
        ax.axvline(1.0, color="#000", ls=":", lw=1.0, label="T = 1 (identity)")

        ax.set_xlabel("temperature T  that best matches the warped distribution",
                      fontsize=10.5)
        if var == "pl_warp_dapo":
            ax.set_ylabel("number of active tokens", fontsize=10.5)
        ax.set_title(LABEL[var], fontsize=12, pad=8)
        ax.set_xlim(0.2, 2.6)
        # Make room above the bars for the sharpen/flatten labels
        ymax_cur = ax.get_ylim()[1]
        ax.set_ylim(0, ymax_cur * 1.28)
        ax.grid(True, axis="y", which="major", alpha=0.25, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # sharpen / flatten zones, placed in the top quarter
        ann_y = ax.get_ylim()[1] * 0.94
        ax.text(0.55, ann_y, "sharpen  (T < 1)",
                fontsize=10, color="#444", ha="center", style="italic")
        ax.text(1.75, ann_y, "flatten  (T > 1)",
                fontsize=10, color="#444", ha="center", style="italic")
        # legend below, with a clean handles list
        ax.legend(fontsize=9, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, 0.86))

    fig.suptitle("Active LogitMLP tokens want very different temperatures",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = OUT / "per_token_T_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("→", out)


if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    if "divergence" in sys.argv or len(sys.argv) == 1:
        fig_divergence()
    if "per_token_T" in sys.argv or len(sys.argv) == 1:
        fig_per_token_T()
