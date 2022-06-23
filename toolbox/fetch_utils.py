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
