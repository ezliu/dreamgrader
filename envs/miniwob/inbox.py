import os
import ast
import csv
import json
import torch
import itertools
import collections

import torch
import gymnasium as gym
import numpy as np
from PIL import Image
from gym import spaces

import render
import meta_exploration
from envs.miniwob.wrappers import InboxScreenshotWrapper, InboxQAWrapper, WarpScreenshot, RestrictedActionWrapper
from miniwob.envs.miniwob_envs import EmailInboxEnv
from envs.miniwob.constants import NUM_INSTANCES


class InstructionWrapper(meta_exploration.InstructionWrapper):
    def __init__(self, env, exploration_trajectory, seed=None,
                 test=False, first_episode_no_instruction=False,
                 first_episode_no_optimization=False,
                 fixed_instructions=False, exploitation=False):
        super().__init__(
            env,
            exploration_trajectory,
            seed=seed, test=test,
            first_episode_no_instruction=first_episode_no_instruction,
            first_episode_no_optimization=first_episode_no_optimization,
            fixed_instructions=fixed_instructions)
        self._exploitation = exploitation
        if exploitation:
            self.action_space = spaces.Discrete(2)

    def _instruction_observation_space(self):
        return gym.spaces.Box(np.array([0]), np.array([1]), dtype=np.int)

    def _reward(self, instruction_state, action, original_reward):
        return original_reward, False

    def _generate_instructions(self, test=False):
        return np.array([0]) # dummy unused instruction

    def step(self, action):
        if self._exploitation:
            done = True
            reward = int(self.env._env.current_question[1] == action)
            # Take dummy action, since existing action may be out of
            # bounds
            # Bypass parent class
            state, _, _, info = self.env.step(0)
            return state, reward, done, info
        # Bypass parent class
        return self.env.step(action)


class InboxMetaEnv(meta_exploration.MetaExplorationEnv):
    MAX_STEPS = 10
    NUM_TRAIN = 400
    NUM_TEST = 100

    def __init__(self, env_id, _):
        super().__init__(env_id, EmailInboxObservation)
        self._steps = 0
        
        env = EmailInboxEnv(num_instances=NUM_INSTANCES)
        env = InboxScreenshotWrapper(env)
        env = InboxQAWrapper(env)
        env = WarpScreenshot(env)
        env = RestrictedActionWrapper(env)
        self.observation_space = gym.spaces.Dict({
            "observation": env.observation_space,
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([500]),
                dtype=np.int)
        })
        self._env = env
        self._env.reset()
        self.action_space = self._env.action_space

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    @classmethod
    def load_config(cls, config=None):
        pass

    @classmethod
    def env_ids(cls):
        return list(range(cls.NUM_TRAIN)), list(range(cls.NUM_TRAIN, cls.NUM_TRAIN + cls.NUM_TEST))

    @property
    def env_id(self):
        return self._env.current_question[1]

    def _step(self, action):
        state, reward, done, _, info = self._env.step(action)
        self._steps += 1
        done = done if self._steps < type(self).MAX_STEPS else [True]*NUM_INSTANCES
        return state, reward, done, info

    def _reset(self):
        self._steps = 0
        np.random.seed(self._env_id)
        obs, _ = self._env.reset(seed=self._env_id)
        return obs

    def render(self, mode=None):
        env_render = self._env.render()
        image = render.Render(Image.fromarray(env_render))
        image.write_text("Underlying env ID: {}".format(self._env_id))
        image.write_text(f"Q: {self._env.current_question[0]}")
        image.write_text(f"A: {self._env.current_question[1]}")
        return image


"""
def cpu(self):
    if self.observation.is_cuda:
        return self._replace(
            observation=self.observation.cpu().pin_memory())
    return self

def cuda(self):
    if self.observation.is_cuda:
        return self
    return self._replace(
        observation=self.observation.cuda(non_blocking=True))
"""

class EmailInboxObservation:
    def __init__(self, observation):
        observation["screenshot"] = torch.tensor(observation["screenshot"])
        self._observation = observation

    @property
    def is_cuda(self):
        return self._observation["screenshot"].is_cuda

    @property
    def screenshot(self):
        return self._observation["screenshot"]

    @property
    def question(self):
        return self._observation["question"]

    def cpu(self):
        # Hacky way to accomodate cpu/cuda switching in observation buffer
        self._observation["screenshot"] = self._observation["screenshot"].cpu()
        return self

    def pin_memory(self):
        self._observation["screenshot"] = self._observation["screenshot"].pin_memory()
        return self

    def cuda(self, **kwargs):
        self._observation["screenshot"] = self._observation["screenshot"].cuda(**kwargs)
        return self