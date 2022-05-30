import numpy as np


def cumulative_cells(observations):
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
