"""
Example: generate a `weight_map` with `create_prior_weights` and run
`get_tfce_score_scipy` (weighted TFCE) with and without priors.

Run:

    python examples/weighted_tfce_example.py

This script constructs a small synthetic `t_stats` connectivity matrix and a
`weight_map` based on simple network labels. It prints TFCE score matrices
for unweighted and weighted TFCE so you can compare the effect of the priors.
"""

import numpy as np
from conninfpy.utils import create_prior_weights
from conninfpy.tfnos import get_tfce_score_scipy


def make_synthetic_t_stats():
    """Create a small symmetric t-statistics matrix (zero diagonal).

    Construction aims to produce suprathreshold edges within two small
    communities so TFCE detects clusters.
    """
    N = 6
    t = np.zeros((N, N), dtype=float)

    # Create stronger edges inside community 1 (nodes 0-2) and community 2 (nodes 3-4)
    t[0, 1] = t[1, 0] = 3.0
    t[0, 2] = t[2, 0] = 2.8
    t[1, 2] = t[2, 1] = 2.6

    t[3, 4] = t[4, 3] = 2.5

    # Add a weak inter-community edge
    t[2, 3] = t[3, 2] = 1.2

    # Node 5 isolated except a small spurious edge
    t[4, 5] = t[5, 4] = 1.0

    np.fill_diagonal(t, 0.0)
    return t


def main():
    np.set_printoptions(precision=3, suppress=True)

    # Labels for each node (same length as matrix dimension)
    # Here nodes 0-2 belong to network 1, 3-4 to network 2, node 5 is network 3
    labels = np.array([1, 1, 1, 2, 2, 3])

    # Build weight_map with higher weights inside network 1 and 2
    weight_map = create_prior_weights(labels, boost_factor=3.0)

    t_stats = make_synthetic_t_stats()

    print("t_stats:\n", t_stats)
    print("\nweight_map:\n", weight_map)

    # Parameters for TFCE-like scoring
    e = 0.5
    h = 2.0
    n = 20
    start_thres = 1.65

    # Unweighted TFCE
    tfnos_unweighted = get_tfce_score_scipy(t_stats, e, h, n, start_thres=start_thres)

    # Weighted TFCE using the prior weight_map
    tfnos_weighted = get_tfce_score_scipy(t_stats, e, h, n, start_thres=start_thres, weight_map=weight_map)

    print("\nTFNOS (unweighted):\n", np.round(tfnos_unweighted, 3))
    print("\nTFNOS (weighted):\n", np.round(tfnos_weighted, 3))

    # Quick comparison: show difference
    diff = tfnos_weighted - tfnos_unweighted
    print("\nWeighted - Unweighted:\n", np.round(diff, 3))


if __name__ == '__main__':
    main()
