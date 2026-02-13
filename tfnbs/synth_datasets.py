import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix


def generate_fc_matrices(N,  effect_size, mask=None, n_samples_group1=50, n_samples_group2=50,
                         repeated_measures=False, seed=None):
    """
    Generate synthetic functional connectivity correlation matrices for groupwise comparisons
    or repeated measures.

    Parameters:
        N (int): Number of ROIs (regions of interest) i.e. an N*N matrix.
        effect_size (float): Magnitude of correlation difference between groups.
        mask (np.ndarray, optional): Binary mask (N*N) matrix to apply correlation differences.
        n_samples_group1 (int): Number of matrices in group 1 (default: 50).
        n_samples_group2 (int): Number of matrices in group 2 (default: 50).
        repeated_measures (bool): If True, generate within-subject repeated measures data
                                i.e. Paired else independently (default = False).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        group1 (np.ndarray): Array of with functional connectivity matrics of group1 as (n_samples_group1, N, N).
        group2 (np.ndarray): Array of with functional connectivity matrics of group2 as (n_samples_group2, N, N).
        (base_cov, mod_cov): Original Covariance matrix and Modified covariance matrix with effect_size introduced
                             for group1 and group2 matrices. 

    >>> N = 6; e = 0.2; mask = np.zeros((N, N))
    >>> mask[0:2, 0:2] = 1; mask[2:4, 2:4] = -1
    >>> g1, g2, (c1,c2) = generate_fc_matrices(N, e, mask, 5, 10, seed = 0)
    >>> g1.shape == (5, 6, 6)
    True
    >>> g2.shape == (10, 6, 6)
    True
    >>> np.allclose(c1,c1.T)
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
