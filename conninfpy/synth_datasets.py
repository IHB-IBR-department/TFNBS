"""Synthetic connectivity datasets for benchmarking and testing."""
import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix


__all__ = [
    "generate_fc_matrices",
    "generate_multisite_glm_dataset",
    "ModularDatasetGenerator",
]


def generate_multisite_glm_dataset(
    n_subjects: int = 30,
    N: int = 100,
    n_sites: int = 3,
    effect_size: float = 0.0,
    site_shift_sigma: float = 0.2,
    corr_site_interest: float = 0.0,
    n_signal_edges: int = 0,
    base_corr_sparsity: float = 0.9,
    seed: int | None = None,
) -> dict:
    """Multi-site GLM connectivity dataset, sized to mimic a Schaefer-100 study.

    Built for the v2.1 full-pipeline calibration tests
    (`tests/test_full_pipeline.py`) and for the matrix-level phase of the
    SC-prior validation work in
    `Projects/NetworkStatistics/_wiki/pseudo_real_validation.md`.

    Output is in **Fisher-z** units already (so callers should pass
    ``fisher_z=False`` to ``analyze()``). Site effects are additive on
    Fisher-z; signal is linear in the regressor of interest at a fixed
    set of signal edges.

    Parameters
    ----------
    n_subjects : int, default 30
        Total subjects, evenly distributed across ``n_sites``.
    N : int, default 100
        Number of ROIs. Default 100 mimics Schaefer-100.
    n_sites : int, default 3
        Number of distinct sites; each contributes a fixed symmetric
        Fisher-z offset on all edges.
    effect_size : float, default 0.0
        Slope of the linear-in-interest perturbation at signal edges.
        Set to ``0.0`` for an H₀ dataset (no group / regressor effect).
    site_shift_sigma : float, default 0.2
        Scale of the per-site additive Fisher-z offset. Each site's
        offset is drawn from ``N(0, σ_site²)`` once per simulation.
    corr_site_interest : float in [0, 1], default 0.0
        Population correlation between ``sites`` (as an integer code)
        and the ``interest`` regressor. ``0.0`` = independent (the
        H₀-friendly regime); ``0.6`` = the regime where unrestricted
        permutation after harmonisation leaks Type-I.
    n_signal_edges : int, default 0
        Number of edges carrying the planted signal. Ignored when
        ``effect_size == 0``. Edges are drawn uniformly from the upper
        triangle (excluding diagonal).
    base_corr_sparsity : float, default 0.9
        ``alpha`` for ``make_sparse_spd_matrix`` controlling baseline
        connectivity sparsity (higher → sparser).
    seed : int, optional
        Reproducibility handle.

    Returns
    -------
    dict with keys
        - ``"Y"`` — ``(n_subjects, N, N)`` Fisher-z connectivity tensor,
          symmetric with zero diagonal.
        - ``"interest"`` — ``(n_subjects,)`` regressor of interest.
        - ``"sites"`` — ``(n_subjects,)`` integer site labels.
        - ``"signal_mask"`` — ``(N, N)`` boolean upper-triangle mask of
          true positive edges (all False when ``effect_size == 0`` or
          ``n_signal_edges == 0``).

    Notes
    -----
    Designed for end-to-end exercise of
    ``analyze(Y, interest=..., sites=...)``: it triggers the auto-
    preserve, auto-strata, ComBat-with-preserve, Freedman-Lane with
    within-block exchangeability path in a single call. Calibration
    tests use ``effect_size=0.0`` (H₀); strata-vs-no-strata sanity
    tests use ``corr_site_interest > 0`` to bring the leak into view.

    Examples
    --------
    >>> data = generate_multisite_glm_dataset(
    ...     n_subjects=24, N=20, n_sites=3,
    ...     site_shift_sigma=0.3, corr_site_interest=0.6, seed=0,
    ... )
    >>> data["Y"].shape
    (24, 20, 20)
    >>> data["sites"].shape, data["interest"].shape
    ((24,), (24,))
    """
    if not 0.0 <= corr_site_interest <= 1.0:
        raise ValueError(
            f"corr_site_interest must be in [0, 1], got {corr_site_interest}."
        )
    if n_subjects < n_sites:
        raise ValueError(
            f"Need at least one subject per site (n_subjects={n_subjects}, "
            f"n_sites={n_sites})."
        )

    rng = np.random.default_rng(seed)

    # ---- Site assignment (round-robin) ----
    sites = np.tile(np.arange(n_sites), n_subjects // n_sites + 1)[:n_subjects]

    # ---- Regressor of interest with target corr(site, interest) ----
    # Project site labels to mean-zero, unit-variance to make the
    # correlation parameter well-defined.
    site_codes = sites.astype(np.float64)
    site_centered = (site_codes - site_codes.mean()) / max(site_codes.std(), 1e-12)
    noise = rng.standard_normal(n_subjects)
    interest = (
        corr_site_interest * site_centered
        + np.sqrt(max(1.0 - corr_site_interest ** 2, 0.0)) * noise
    )

    # ---- Base covariance + per-subject sampled correlation ----
    base_cov = make_sparse_spd_matrix(
        N, alpha=base_corr_sparsity, norm_diag=True,
        random_state=rng.integers(0, 2**31 - 1),
    )
    n_obs_for_corr = max(N + 1, 200)  # ensure stable per-subject correlation
    Y = np.zeros((n_subjects, N, N), dtype=np.float64)
    for i in range(n_subjects):
        samples = rng.multivariate_normal(
            mean=np.zeros(N), cov=base_cov, size=n_obs_for_corr,
        )
        Y[i] = np.corrcoef(samples.T)

    # Fisher r→z (clip to avoid arctanh blow-up at |r|→1)
    Y = np.clip(Y, -0.9999, 0.9999)
    Y = np.arctanh(Y)
    for i in range(n_subjects):
        np.fill_diagonal(Y[i], 0.0)

    # ---- Per-site additive shift (symmetric, zero diagonal) ----
    if site_shift_sigma > 0:
        iu = np.triu_indices(N, k=1)
        site_shifts = np.zeros((n_sites, N, N), dtype=np.float64)
        for s in range(n_sites):
            shift_upper = rng.standard_normal(iu[0].size) * site_shift_sigma
            site_shifts[s][iu] = shift_upper
            site_shifts[s] = site_shifts[s] + site_shifts[s].T
        for i in range(n_subjects):
            Y[i] += site_shifts[sites[i]]

    # ---- Signal injection ----
    signal_mask = np.zeros((N, N), dtype=bool)
    if effect_size != 0.0 and n_signal_edges > 0:
        iu = np.triu_indices(N, k=1)
        chosen = rng.choice(iu[0].size, size=n_signal_edges, replace=False)
        signal_rows = iu[0][chosen]
        signal_cols = iu[1][chosen]
        for i in range(n_subjects):
            delta = effect_size * interest[i]
            Y[i, signal_rows, signal_cols] += delta
            Y[i, signal_cols, signal_rows] += delta
        signal_mask[signal_rows, signal_cols] = True
        signal_mask[signal_cols, signal_rows] = True

    # Re-zero diagonal (site shifts and signal already preserve it)
    for i in range(n_subjects):
        np.fill_diagonal(Y[i], 0.0)

    return {
        "Y": Y,
        "interest": interest,
        "sites": sites,
        "signal_mask": signal_mask,
    }


def generate_fc_matrices(N,  effect_size, mask=None, n_samples_group1=50, n_samples_group2=50,
                         repeated_measures=False, seed=None):
    """
    Generate synthetic functional connectivity correlation matrices for groupwise comparisons
    or repeated measures.

    Parameters
    ----------
    N : int
        Number of ROIs (regions of interest), i.e. an ``N x N`` matrix.
    effect_size : float
        Magnitude of correlation difference between groups.
    mask : np.ndarray, optional
        Binary mask ``(N, N)`` to apply correlation differences.
    n_samples_group1 : int
        Number of matrices in group 1 (default: 50).
    n_samples_group2 : int
        Number of matrices in group 2 (default: 50).
    repeated_measures : bool
        If True, generate within-subject repeated-measures (paired) data,
        otherwise independent groups (default: False).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    group1 : np.ndarray
        Array of FC matrices for group 1, shape ``(n_samples_group1, N, N)``.
    group2 : np.ndarray
        Array of FC matrices for group 2, shape ``(n_samples_group2, N, N)``.
    (base_cov, mod_cov) : tuple of np.ndarray
        Original covariance matrix and modified covariance matrix with the
        ``effect_size`` introduced; used for group 1 and group 2 matrices
        respectively.

    Examples
    --------
    >>> N = 6; e = 0.2; mask = np.zeros((N, N))
    >>> mask[0:2, 0:2] = 1; mask[2:4, 2:4] = -1
    >>> g1, g2, (c1, c2) = generate_fc_matrices(N, e, mask, 5, 10, seed=0)
    >>> g1.shape == (5, 6, 6)
    True
    >>> g2.shape == (10, 6, 6)
    True
    >>> np.allclose(c1, c1.T)
    True
    """

    if seed is not None:
        np.random.seed(seed)
    if mask is None:
        mask = np.zeros((N, N))
        N_pos_block = N//3
        mask[:N_pos_block, :N_pos_block] = 1
        N_neg_block = N//3
        mask[N_pos_block:N_pos_block+N_neg_block, N_pos_block:N_pos_block+N_neg_block] = -1

    # Generate base covariance matrix
    base_cov = make_sparse_spd_matrix(N, alpha=0.8, norm_diag=True, random_state=seed)

    # Create modified covariance for the second condition or group
    mod_cov = base_cov.copy()
    mod_cov[mask == 1] += effect_size + np.abs(np.random.normal(0,0.05, mod_cov[mask == 1].shape)) # Increase correlations in masked regions
    mod_cov[mask == -1] -= effect_size - np.abs(np.random.normal(0,0.1, mod_cov[mask == 1].shape)) # Decrease correlations in masked regions
    np.fill_diagonal(mod_cov, 1.0)
    mod_cov = (mod_cov+mod_cov.T)/2


    def enforce_spd(matrix, eps=1e-6):
        """ Ensures a matrix is symmetric positive definite (SPD) by adjusting eigenvalues. """
        eigvals, eigvecs = np.linalg.eigh(matrix)       # Get eigenvalues & eigenvectors
        eigvals[eigvals < eps] = eps                    # Replace negative eigenvalues with small positive value
        return eigvecs @ np.diag(eigvals) @ eigvecs.T   # Reconstruct SPD matrix


    # Enforce SPD property
    mod_cov = enforce_spd(mod_cov)


    # Generate sample correlation matrices with variability
    def generate_samples(cov_matrix, n_samples):
        return np.array([np.corrcoef(np.random.multivariate_normal(np.zeros(N), cov_matrix, size=N).T)
                         for _ in range(n_samples)])

    if repeated_measures:
        # Generate paired data (same subjects measured twice)
        group1 = generate_samples(base_cov, n_samples_group1)
        group2 = generate_samples(mod_cov, n_samples_group1)  # Same number as group1
    else:
        # Generate independent groups
        group1 = generate_samples(base_cov, n_samples_group1)
        group2 = generate_samples(mod_cov, n_samples_group2)

    return group1, group2, (base_cov, mod_cov)


class ModularDatasetGenerator:
    """
    A generator for synthetic functional connectivity data with a modular (block) structure.
    
    This class simulates brain connectivity matrices where nodes are organized into 
    distinct functional modules (e.g., Visual, DMN, Motor). It allows for the 
    injection of specific topological effects (within-module or between-module changes)
    to simulate pathological conditions.
    """

    def __init__(
        self, 
        N: int, 
        n_modules: int = 5, 
        intra_corr: float = 0.6, 
        inter_corr: float = 0.1, 
        noise_level: float = 0.05, 
        seed: int = None
    ):
        """
        Initialize the generator.

        Parameters
        ----------
        N : int
            Number of nodes (Regions of Interest).
        n_modules : int
            Number of functional modules (networks).
        intra_corr : float
            Base correlation strength between nodes within the same module.
        inter_corr : float
            Base correlation strength between nodes of different modules.
        noise_level : float
            Standard deviation of Gaussian noise added to the base covariance structure.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.N = N
        self.n_modules = n_modules
        self.rng = np.random.default_rng(seed)
        self.noise_level = noise_level
        
        # Assign nodes to modules (evenly distributed)
        # e.g., for N=10, n=2: [0, 1, 0, 1, ...] sorted to [0, 0, ..., 1, 1, ...]
        self.labels = np.sort(np.arange(N) % n_modules)
        
        # Create the underlying ground-truth covariance for "Healthy" controls
        self.base_cov = self._create_base_covariance(intra_corr, inter_corr, noise_level)

    def _create_base_covariance(self, intra: float, inter: float, noise: float) -> np.ndarray:
        """
        Constructs a modular covariance matrix.
        """
        cov = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    cov[i, j] = 1.0
                elif self.labels[i] == self.labels[j]:
                    # Within-module connection
                    val = intra + self.rng.normal(0, noise)
                    cov[i, j] = val
                else:
                    # Between-module connection
                    val = inter + self.rng.normal(0, noise)
                    cov[i, j] = val
        
        # Symmetrize
        cov = (cov + cov.T) / 2
        
        # Clip to ensure values are roughly within correlation bounds before SPD enforcement
        cov = np.clip(cov, -0.95, 0.95)
        np.fill_diagonal(cov, 1.0)
        
        # Ensure the matrix is mathematically valid (Symmetric Positive Definite)
        return self._enforce_spd(cov)

    def _enforce_spd(self, matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Projects a matrix onto the cone of Symmetric Positive Definite matrices.
        Replaces negative eigenvalues with a small epsilon and normalizes the diagonal.
        """
        # 1. Symmetrize
        matrix = (matrix + matrix.T) / 2
        
        # 2. Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # 3. Clip negative eigenvalues
        eigvals[eigvals < eps] = eps
        
        # 4. Reconstruct
        reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # 5. Rescale to ensure 1.0 on diagonal (Correlation matrix property)
        d = np.sqrt(np.diag(reconstructed))
        d_inv = 1.0 / d
        # D^-1 * M * D^-1
        corr = reconstructed * np.outer(d_inv, d_inv)
        
        # Final safety clip
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def get_mask_within_module(self, module_idx: int) -> np.ndarray:
        """
        Returns a binary mask for all edges WITHIN a specific module.
        """
        mask = np.zeros((self.N, self.N), dtype=int)
        nodes = np.where(self.labels == module_idx)[0]
        
        # Create block mask using broadcasting indices
        # We exclude the diagonal (self-connections)
        for i in nodes:
            for j in nodes:
                if i != j:
                    mask[i, j] = 1
        return mask

    def get_mask_between_modules(self, module_idx_A: int, module_idx_B: int) -> np.ndarray:
        """
        Returns a binary mask for all edges connecting module A and module B.
        """
        mask = np.zeros((self.N, self.N), dtype=int)
        nodes_A = np.where(self.labels == module_idx_A)[0]
        nodes_B = np.where(self.labels == module_idx_B)[0]
        
        for i in nodes_A:
            for j in nodes_B:
                mask[i, j] = 1
                mask[j, i] = 1  # Symmetrize
        return mask

    def generate_data(
        self, 
        effect_mask: np.ndarray, 
        effect_size: float, 
        n_samples_g1: int = 50, 
        n_samples_g2: int = 50,
        time_points: int = 200
    ):
        """
        Generates sample correlation matrices for two groups.
        
        Group 1 is sampled from the base modular covariance.
        Group 2 is sampled from a modified covariance (base + effect).

        Parameters
        ----------
        effect_mask : np.ndarray
            Effect mask matrix of shape (N, N).

            - 0 means "no effect" for that edge.
            - Non-zero values scale the effect magnitude per edge.
            - The sign of the value controls effect direction (positive/negative).

            The matrix is treated as undirected: it will be symmetrized internally
            and its diagonal will be set to 0.
        effect_size : float
            Magnitude of the effect (Cohen's d-like shift in correlation). 
            Positive values increase correlation, negative values decrease it.
        n_samples_g1 : int
            Number of subjects in Group 1 (Control).
        n_samples_g2 : int
            Number of subjects in Group 2 (Test).
        time_points : int
            Length of the simulated BOLD time-series. Higher values reduce sampling noise.

        Returns
        -------
        g1_data : np.ndarray (n_samples_g1, N, N)
            Correlation matrices for Group 1.
        g2_data : np.ndarray (n_samples_g2, N, N)
            Correlation matrices for Group 2.
        labels : np.ndarray (N,)
            Vector of module assignments for nodes.
        """
        effect_mask = np.asarray(effect_mask, dtype=np.float64)
        if effect_mask.shape != (self.N, self.N):
            raise ValueError(f"effect_mask must have shape ({self.N}, {self.N}).")

        # Enforce undirected mask semantics.
        effect_mask = (effect_mask + effect_mask.T) / 2
        np.fill_diagonal(effect_mask, 0.0)

        # Group 1: Base Covariance
        cov_g1 = self.base_cov.copy()
        
        # Group 2: Modified Covariance
        cov_g2 = self.base_cov.copy()
        
        # Add random noise to the effect to simulate biological variability
        # This prevents the effect from being a perfect constant shift
        effect_noise = self.rng.normal(0, self.noise_level, size=cov_g2.shape)
        effect_noise = (effect_noise + effect_noise.T) / 2
        
        # Apply effect: Delta = (Target_Shift + Noise) * Mask
        # - For binary masks (0/1), this reduces to a constant mean shift on masked edges.
        # - For signed/weighted masks, this enables mixed-sign and graded effects.
        update_vals = (effect_size + effect_noise) * effect_mask
        cov_g2 += update_vals
        
        # Re-enforce SPD property after modification
        cov_g2 = self._enforce_spd(cov_g2)
        
        # Helper to generate subjects by sampling time-series
        def sample_subject_matrices(true_cov, n_subs):
            matrices = []
            for _ in range(n_subs):
                # Generate time series: shape (time_points, N)
                ts = self.rng.multivariate_normal(
                    mean=np.zeros(self.N), 
                    cov=true_cov, 
                    size=time_points
                )
                # Compute Pearson correlation: shape (N, N)
                # rowvar=False means columns are variables (nodes)
                corr = np.corrcoef(ts, rowvar=False)
                matrices.append(corr)
            return np.array(matrices)

        g1_data = sample_subject_matrices(cov_g1, n_samples_g1)
        g2_data = sample_subject_matrices(cov_g2, n_samples_g2)
        
        return g1_data, g2_data, self.labels
