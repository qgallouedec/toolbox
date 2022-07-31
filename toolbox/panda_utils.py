import numpy as np


def cumulative_object_coverage(observations):
    object_pos = observations[:, :, 4:7]
    num_timesteps, num_envs = observations.shape[0], observations.shape[1]
    cells = np.floor(object_pos * 10)  # NUM_TIMESTEPS  x NUM_ENV x 3
    seen_cells = []
    counts = np.zeros(num_timesteps)
    for t in range(num_timesteps):
        for env_idx in range(num_envs):
            is_new = True
            cell = cells[t, env_idx]
            for seen_cell in seen_cells:
                if (seen_cell == cell).all():
                    is_new = False
                    break
            if is_new:
                seen_cells.append(cell)
        counts[t] = len(seen_cells)
    return counts
