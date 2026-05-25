import numpy as np
from pathlib import Path
from conninfpy import AtlasInfo, analyze

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "diagnosis"

def run_variants():
    data = np.load(DATA_FILE, allow_pickle=True)
    Y = data["connectivity_z"]
    group = data["group"].astype(float)
    confounds = np.column_stack([data["age"].astype(float), data["sex"].astype(float), data["mean_fd"].astype(float)])
    sites = data["site"]
    net_labels = data["net_labels"]
    
    # Atlas for CSV export
    network_order = list(data["network_order"])
    atlas = AtlasInfo(labels=[str(x) for x in data["roi_names"]], 
                      networks=[network_order[i] for i in net_labels.astype(int)])

    common = dict(Y=Y, interest=group, confounds=confounds, sites=sites, 
                  harmonize="nuisance_only", fisher_z=False, n_permutations=500, rng=42)

    # 1. NI-TFNBS
    print("Running NI-TFNBS...")
    out = analyze(**common, method="ni_tfnbs", net_labels=net_labels)
    out.to_csv(OUT_DIR / "dx_ni_tfnbs_edges.csv", atlas=atlas)

    # 2. FBC-TFNBS
    print("Running FBC-TFNBS...")
    out = analyze(**common, method="fbc_tfnbs", net_labels=net_labels)
    out.to_csv(OUT_DIR / "dx_fbc_tfnbs_edges.csv", atlas=atlas)

    # 3. cNBS
    print("Running cNBS...")
    out = analyze(**common, method="cnbs", net_labels=net_labels)
    out.to_csv(OUT_DIR / "dx_cnbs_edges.csv", atlas=atlas)

if __name__ == "__main__":
    run_variants()
