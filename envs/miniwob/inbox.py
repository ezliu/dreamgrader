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
from envs.miniwob.wrappers import InboxScreenshotWrapper, InboxQAWrapper, WarpScreenshot, RestrictedActionWrapper, InboxDOMWrapper
from miniwob.envs.miniwob_envs import EmailInboxEnv
from envs.miniwob.constants import NUM_INSTANCES, TASK_HEIGHT, TASK_WIDTH


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
        self.env.exploitation = exploitation
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
            done = [True] * len(action)
            reward = []
            for a, label in zip(action, self.env_id):
                reward.append((a == label).item())
            # Take dummy action, since existing action may be out of
            # bounds
            # Bypass parent class
            state, _, _, info = self.env.step([0] * len(action))
            return state, reward, done, info
        # Bypass parent class
        return self.env.step(action)


class InboxMetaEnv(meta_exploration.MetaExplorationEnv):
    MAX_STEPS = 6
    NUM_TRAIN = 1000000
    NUM_TEST = 1000

    def __init__(self, env_id, _):
        super().__init__(env_id, EmailInboxObservation)
        self._steps = 0
        
        env = EmailInboxEnv(num_instances=NUM_INSTANCES)
        env = InboxScreenshotWrapper(env)
        env = InboxQAWrapper(env, env_id)
        env = InboxDOMWrapper(env)
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
        self.exploitation = False

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
        return list(zip(*self._env.current_question))[1]

    def _step(self, action):
        # Hack to speed up env during exploitation (don't need to actually take steps)
        if not self.exploitation:
            state, reward, done, _, info = self._env.step(action)
        else:
            state = [{
                "screenshot": np.zeros((TASK_HEIGHT, TASK_WIDTH, 1)),
                "question": "None",
                "dom": "None"
            } for _ in range(NUM_INSTANCES)]
            reward = [0] * NUM_INSTANCES
            info = [None] * NUM_INSTANCES
            done = [True] * NUM_INSTANCES
        self._steps += 1
        done = done if self._steps < type(self).MAX_STEPS else [True]*NUM_INSTANCES
        return state, reward, done, info

    def _reset(self):
        # old hack but messes up evaluation of correct answer
        self._steps = 0
        """if not self.exploitation:
            obs, _ = self._env.reset(seed=self._env_id)
        else:
            obs = [{
                "screenshot": np.zeros((TASK_HEIGHT, TASK_WIDTH, 1)),
                "question": "None"
            } for _ in range(NUM_INSTANCES)]"""
        obs, _ = self._env.reset(seed=self._env_id)
        return obs

    def render(self, mode=None):
        env_render = self._env.render()
        imgs = []
        for i in range(NUM_INSTANCES):
            img = Image.fromarray(env_render[i])
            img = render.Render(img)
            img.write_text("Underlying env ID: {}".format(self._env_id[i]))
            img.write_text(f"Q: {self._env.current_question[i][0]}")
            img.write_text(f"A: {self._env.current_question[i][1]}")
            imgs.append(img)
        return imgs
    
    @property
    def underlying_env_id(self):
        return self._env_id

    def set_underlying_env_id(self, id):
        self._env_id = id
        self._env.set_qa_env_ids(id)


class EmailInboxObservation:
    def __init__(self, observation):
        if not isinstance(observation["screenshot"], torch.Tensor):
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
    
    @property
    def dom(self):
        return self._observation["dom"]

    def cpu(self):
        # Hacky way to accomodate cpu/cuda switching in observation buffer
        return EmailInboxObservation({
            "screenshot": self._observation["screenshot"].detach().cpu(),
            "question": self._observation["question"],
            "dom": self._observation["dom"]
        })

    def pin_memory(self):
        return EmailInboxObservation({
            "screenshot": self._observation["screenshot"].pin_memory(),
            "question": self._observation["question"],
            "dom": self._observation["dom"]
        })

    def cuda(self, **kwargs):
        return EmailInboxObservation({
            "screenshot": self._observation["screenshot"].cuda(**kwargs),
            "question": self._observation["question"],
            "dom": self._observation["dom"]
        })