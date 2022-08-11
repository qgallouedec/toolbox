import numpy as np


def compute_coverage(observations):
    num_timesteps, num_envs = observations.shape[0], observations.shape[1]
    cells = np.floor(observations * 10)
    unique, cell_uids = np.unique(np.squeeze(cells), axis=0, return_inverse=True)
    seen_uids = []
    counts = np.zeros(num_timesteps, dtype=np.int32)
    for t in range(num_timesteps):
        cell_uid = cell_uids[t]
        if not cell_uid in seen_uids:
            seen_uids.append(cell_uid)
        counts[t] = len(seen_uids)
    return counts
