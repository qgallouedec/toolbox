import gym
import numpy as np


def highest_so_far(infos):
    n_envs = infos[0].shape[0]
    object_z = np.array([[info[env_idx]["object_z"] for env_idx in range(n_envs)] for info in infos])
    highest = np.maximum.accumulate(object_z, 0).max(1)
    return highest


class FetchWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        object_pos = self.env.sim.data.get_site_xpos("object0")
        info["object_z"] = object_pos[2]
        return obs, reward, done, info


def cumulative_object_coverage(observations):
    object_pos = observations[:, :, 8:11]
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
