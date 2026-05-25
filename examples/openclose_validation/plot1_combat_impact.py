import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from conninfpy import fisher_r_to_z
from examples.openclose_validation.openclose_loader import OpenCloseDataset

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
HARM_PATH = RESULTS / "openclose_harmonized.npz"

def _ts_to_fisher_z(ts: np.ndarray) -> np.ndarray:
    n, _, N = ts.shape
    out = np.empty((n, N, N), dtype=np.float64)
    for s in range(n):
        c = np.corrcoef(ts[s].T)
        np.fill_diagonal(c, 0.0)
        out[s] = np.clip(c, -0.9999, 0.9999)
    return fisher_r_to_z(out)

def _stack_pooled(Y_ihb_open, Y_ihb_close, Y_china_open, Y_china_close):
    blocks = [Y_ihb_open, Y_ihb_close, Y_china_open, Y_china_close]
    cohorts = ["IHB", "IHB", "China", "China"]
    states = [1.0, 0.0, 1.0, 0.0]
    N = blocks[0].shape[1]
    iu = np.triu_indices(N, k=1)
    flats = [b[:, iu[0], iu[1]] for b in blocks]
    Y_flat = np.concatenate(flats, axis=0)
    cohort = np.concatenate([np.array([c] * b.shape[0]) for c, b in zip(cohorts, blocks)])
    state = np.concatenate([np.full(b.shape[0], v) for v, b in zip(states, blocks)])
    return Y_flat, cohort, state, iu, N

def _residualize(Y_flat: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_full = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.linalg.pinv(X_full) @ Y_flat
    return Y_flat - X_full @ beta

def _cohort_f_per_edge(resid_flat: np.ndarray, cohort: np.ndarray) -> np.ndarray:
    n_edges = resid_flat.shape[1]
    f_vec = np.zeros(n_edges)
    groups_by_cohort = [resid_flat[cohort == c] for c in np.unique(cohort)]
    for e in range(n_edges):
        f, _ = stats.f_oneway(*(g[:, e] for g in groups_by_cohort))
        f_vec[e] = f if np.isfinite(f) else 0.0
    return f_vec

def _to_matrix(vec: np.ndarray, iu, N: int) -> np.ndarray:
    M = np.zeros((N, N)); M[iu] = vec; return M + M.T

def _draw_block_boundaries(ax, labels: np.ndarray) -> None:
    boundaries = np.where(labels[:-1] != labels[1:])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", ls="--", lw=0.4, alpha=0.4)
        ax.axvline(b, color="black", ls="--", lw=0.4, alpha=0.4)

def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    if not HARM_PATH.exists(): return
    ihb = OpenCloseDataset.load("ihb"); china = OpenCloseDataset.load("china")
    Y_ihb_open_raw = _ts_to_fisher_z(ihb.open_ts); Y_ihb_close_raw = _ts_to_fisher_z(ihb.close_ts)
    Y_china_open_raw = _ts_to_fisher_z(china.open_ts); Y_china_close_raw = _ts_to_fisher_z(china.close_ts)
    h = np.load(HARM_PATH)
    Y_ihb_open_h = h["Y_ihb_open_harm"]; Y_ihb_close_h = h["Y_ihb_close_harm"]
    Y_china_open_h = h["Y_china_open_harm"]; Y_china_close_h = h["Y_china_close_harm"]
    Y_raw, cohort, state, iu, N = _stack_pooled(Y_ihb_open_raw, Y_ihb_close_raw, Y_china_open_raw, Y_china_close_raw)
    Y_harm, _, _, _, _ = _stack_pooled(Y_ihb_open_h, Y_ihb_close_h, Y_china_open_h, Y_china_close_h)
    state_X = state.reshape(-1, 1); resid_raw = _residualize(Y_raw, state_X); resid_harm = _residualize(Y_harm, state_X)
    f_raw = _cohort_f_per_edge(resid_raw, cohort); f_harm = _cohort_f_per_edge(resid_harm, cohort)
    net_labels = np.asarray(ihb.net_labels, dtype=int); order = np.argsort(net_labels, kind="stable"); labels_ord = net_labels[order]
    M_raw = _to_matrix(f_raw, iu, N)[np.ix_(order, order)]; M_harm = _to_matrix(f_harm, iu, N)[np.ix_(order, order)]
    vmax = float(np.percentile(np.concatenate([f_raw, f_harm]), 99))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    
    df_box = pd.DataFrame({"F": np.concatenate([f_raw, f_harm]), "Stage": ["Before ComBat"] * len(f_raw) + ["After ComBat"] * len(f_harm)})
    sns.boxplot(data=df_box, x="Stage", y="F", hue="Stage", palette={"Before ComBat": "#e08020", "After ComBat": "#2080c0"}, ax=axes[0, 0], fliersize=1, legend=False)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_title(f"ANOVA F-distribution\nmean F {f_raw.mean():.2f} → {f_harm.mean():.2f}")
    
    df_b = pd.DataFrame({"Cohort": np.concatenate([cohort, cohort]), "Mean residual FC": np.concatenate([resid_raw.mean(axis=1), resid_harm.mean(axis=1)]), "Stage": ["Before ComBat"] * len(cohort) + ["After ComBat"] * len(cohort)})
    sns.boxplot(data=df_b, x="Cohort", y="Mean residual FC", hue="Stage", palette={"Before ComBat": "#e08020", "After ComBat": "#2080c0"}, ax=axes[0, 1])
    axes[0, 1].set_title("Per-cohort mean residual FC")
    
    im_c = axes[1, 0].imshow(M_raw, cmap="YlOrRd", vmin=0, vmax=vmax); _draw_block_boundaries(axes[1, 0], labels_ord); axes[1, 0].set_title("Cohort F BEFORE ComBat")
    fig.colorbar(im_c, ax=axes[1, 0], shrink=0.8)
    
    im_d = axes[1, 1].imshow(M_harm, cmap="YlOrRd", vmin=0, vmax=vmax); _draw_block_boundaries(axes[1, 1], labels_ord); axes[1, 1].set_title("Cohort F AFTER ComBat")
    fig.colorbar(im_d, ax=axes[1, 1], shrink=0.8)
    
    plt.suptitle("Site-Effect Neutralization Diagnostic (IHB + China Pooled)\nMethod: ComBat (Cohort = Batch, State = Preserved)", fontsize=16, fontweight='bold')
    out_path = PLOTS / "plot1_combat_impact.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
