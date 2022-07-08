import numpy as np


def compute_coverage(observations: np.ndarray) -> np.ndarray:
    """
    For observations, return the cumulative space coverage.

    :param observations: Observations as (num_timesteps, n_envs, obs)
    :type observations: np.ndarray
    :rtype: 1D np.ndarray as (num_timesteps,)
    """
    cells = np.floor(observations)
    seen_cells = []
    counts = np.zeros(observations.shape[0])
    for t, cell in enumerate(cells):
        is_new = True
        for seen_cell in seen_cells:
            if (seen_cell == cell).all():
                is_new = False
        if is_new:
            seen_cells.append(cell)
        counts[t] = len(seen_cells)
    return counts
