import numpy as np

def draw_module_boundaries(ax, labels: np.ndarray) -> None:
    """Overlay dashed lines at module boundaries on an imshow axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    labels : ndarray
        The module labels (integer) per node.
    """
    boundaries = np.where(labels[:-1] != labels[1:])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.axvline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.25)
