import csv, glob, math, os, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
METHODS = {"ppo": "PPO (baseline)", "orion": "PPO + ORION"}
COLORS = {"ppo": "#6b7280", "orion": "#2563eb"}
EM = [("success", "Success", 1), ("spl", "SPL", 1), ("distance_to_goal", "Distance to goal (m)", 0)]
OUT = "results"

def runs_for(m):
    ns = sorted(os.path.basename(p)[:-4] for p in glob.glob("logs/eval/%s_s*.log" % m))
    return [n for n in ns if re.fullmatch(m + r"_s\d+", n)]

def parse_eval(path):
    v = {}
    for line in open(path):
        r = re.search(r"Average episode (\w+): (-?[\d.]+)", line)
        if r: v[r.group(1)] = float(r.group(2))
    return v

def welch(a, b):
    if len(a) < 2 or len(b) < 2: return None, None, None
    va, vb, na, nb = st.variance(a), st.variance(b), len(a), len(b)
    se2 = va / na + vb / nb
    if se2 == 0: return None, None, None
    t = (st.mean(b) - st.mean(a)) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except ImportError:
        p = None
    return t, df, p

def scalars(logdir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0}); ea.Reload()
    if tag not in ea.Tags().get("scalars", []): return None, None
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])

def smooth(y, k=15):
    if len(y) < k: return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")

def band(m, tag):
    cs = []
    for run in runs_for(m):
        x, y = scalars("tb/runs/" + run, tag)
        if x is not None and len(x) > 1: cs.append((x, smooth(y)))
    if not cs: return None
    g = np.linspace(0, min(c[0].max() for c in cs), 200)
    ys = np.stack([np.interp(g, x, y) for x, y in cs])
    return g, ys.mean(0), ys.std(0), len(cs)

def fmt(xs):
    if not xs: return "n/a"
    return "%.3f +/- %.3f" % (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0)

def main():
    os.makedirs(OUT, exist_ok=True)
    rows, per = [], {m: {k: [] for k, _, _ in EM} for m in METHODS}
    for m in METHODS:
        for run in runs_for(m):
            v = parse_eval("logs/eval/%s.log" % run)
            if "success" not in v: print("WARNING: no metrics in", run); continue
            rows.append(dict(run=run, method=m, **v))
            for k, _, _ in EM: per[m][k].append(v[k])
    with open(OUT + "/eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "method"] + [k for k, _, _ in EM], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    L = ["# ORION results (held-out val split)", "",
         "Seeds: PPO = %d, ORION = %d. Mean +/- std across seeds." % (len(per["ppo"]["success"]), len(per["orion"]["success"])), "",
         "| Method | " + " | ".join(n + (" (higher better)" if hi else " (lower better)") for _, n, hi in EM) + " |",
         "|---|" + "---|" * len(EM)]
    for m, lab in METHODS.items():
        L.append("| %s | " % lab + " | ".join(fmt(per[m][k]) for k, _, _ in EM) + " |")
    L += ["", "## Welch t-test (ORION vs PPO)", "", "| Metric | Diff in mean | t | df | p |", "|---|---|---|---|---|"]
    for k, n, _ in EM:
        a, b = per["ppo"][k], per["orion"][k]; t, df, p = welch(a, b)
        if t is None: L.append("| %s | n/a | n/a | n/a | n/a |" % n); continue
        L.append("| %s | %+.3f | %.2f | %.1f | %s |" % (n, st.mean(b) - st.mean(a), t, df, "%.3f" % p if p is not None else "(no scipy)"))
    L += ["", "## Per-run values", "", "| Run | " + " | ".join(n for _, n, _ in EM) + " |", "|---|" + "---|" * len(EM)]
    for r in rows: L.append("| %s | " % r["run"] + " | ".join("%.4f" % r[k] for k, _, _ in EM) + " |")
    open(OUT + "/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (k, n, hi) in zip(axes, EM):
        for i, m in enumerate(METHODS):
            xs = per[m][k]
            if not xs: continue
            ax.bar(i, st.mean(xs), color=COLORS[m], alpha=0.35, width=0.6)
            ax.scatter([i] * len(xs), xs, color=COLORS[m], zorder=3, s=28)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["PPO", "ORION"])
        ax.set_title(n + (" (higher is better)" if hi else " (lower is better)"), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Held-out val evaluation: bars = mean, dots = individual seeds", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT + "/eval_comparison.png", dpi=160); plt.close(fig)

    try:
        tags = [("metrics/success", "Training success"), ("metrics/spl", "Training SPL"), ("metrics/distance_to_goal", "Distance to goal (m)")]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, (tag, title) in zip(axes, tags):
            for m, lab in METHODS.items():
                b = band(m, tag)
                if b is None: continue
                x, mu, sd, n = b
                ax.plot(x / 1e3, mu, color=COLORS[m], label="%s (n=%d)" % (lab, n))
                ax.fill_between(x / 1e3, mu - sd, mu + sd, color=COLORS[m], alpha=0.18, linewidth=0)
            ax.set_title(title, fontsize=10); ax.set_xlabel("Environment frames (thousands)")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Training curves: mean +/- std across seeds (smoothed)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT + "/learning_curves.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for tag, lab, sty in [("learner/aux_orion_pred_mean", "Predicted P(success)", "-"), ("learner/aux_orion_target_mean", "TD(lambda) success target", "--")]:
            b = band("orion", tag)
            if b is None: continue
            x, mu, sd, _ = b
            ax.plot(x / 1e3, mu, sty, color=COLORS["orion"], label=lab)
            ax.fill_between(x / 1e3, mu - sd, mu + sd, color=COLORS["orion"], alpha=0.12, linewidth=0)
        ax.set_xlabel("Environment frames (thousands)"); ax.set_title("ORION success head: prediction vs target", fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(OUT + "/orion_head.png", dpi=160); plt.close(fig)
    except ImportError:
        print("tensorboard not importable: skipped training-curve plots")
    print("\nWrote", ", ".join(sorted(os.listdir(OUT))), "in results/")

main()
