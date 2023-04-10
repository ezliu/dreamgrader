import collections
import cv2
cv2.ocl.setUseOpenCL(False)
import gymnasium as gym
from gym import spaces
import numpy as np
import torch
import meta_exploration


class MultiEpisodeWrapper(gym.Wrapper):
    """Allows for outer episodes (trials in RL^2) consisting of multiple inner
    episodes by subsuming the intermediate dones.

    Dones are already labeled by the InstructionState.
    """

    def __init__(self, env, episodes_per_trial=2):
        super().__init__(env)
        assert isinstance(env, meta_exploration.InstructionWrapper)

        self._episodes_so_far = 0
        self._episodes_per_trial = episodes_per_trial

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if done:
            self._episodes_so_far += 1
            # Need to copy reward from previous state
            next_state = self.env.reset()._replace(
                    prev_reward=next_state.prev_reward, done=done)

        trial_done = self._episodes_so_far == self._episodes_per_trial
        return next_state, reward, trial_done, info

    def reset(self):
        self._episodes_so_far = 0
        state = super().reset()
        return state

    def render(self):
        return self.env.render()


# Adapted from gym baselines
class WarpFrame(gym.ObservationWrapper):
    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, -1)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = collections.deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        # TODO: Think about how to not duplicated memory for subsequent frames
        return np.concatenate(self.frames, axis=-1)


class ActionRepeatWrapper(gym.Wrapper):
    """Executes each action num_repeat times.

    Can be used to reduce memory by reducing the episode horizon.
    """

    def __init__(self, env, num_repeat=3):
        super().__init__(env)
        self._num_repeat = num_repeat

    def step(self, action):
        total_reward = 0
        for _ in range(self._num_repeat):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                return state, total_reward, True, info
        return state, total_reward, done, info