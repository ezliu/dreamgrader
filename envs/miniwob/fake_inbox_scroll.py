import os
import ast
import csv
import json
import torch
import itertools
import collections

import pandas as pd
import torch
from torchvision.io import read_image
import gymnasium as gym
import numpy as np
from PIL import Image
from gym import spaces

import render
import meta_exploration
from envs.miniwob.inbox import EmailInboxObservation
from envs.miniwob.constants import NUM_INSTANCES, TASK_HEIGHT, TASK_WIDTH, ASCII_CHARSET, TEXT_MAX_LENGTH

# Actions
SCROLL_DOWN = 0
SCROLL_UP = 1
CLICK_UP = 2
CLICK_MID = 3
CLICK_DOWN = 4

# States
INBOX_UP = 0
INBOX_MID = 1
INBOX_DOWN = 2
EMAIL_1 = 3
EMAIL_2 = 4
EMAIL_3 = 5
EMAIL_4 = 6
EMAIL_5 = 7
EMAIL_6 = 8
EMAIL_7 = 9


TRANSITIONS = {
    INBOX_UP: {
        SCROLL_DOWN: INBOX_MID,
        CLICK_UP: EMAIL_1,
        CLICK_MID: EMAIL_2,
        CLICK_DOWN: EMAIL_3
    },
    INBOX_MID: {
        SCROLL_DOWN: INBOX_DOWN,
        SCROLL_UP: INBOX_UP,
        CLICK_UP: EMAIL_3,
        CLICK_MID: EMAIL_4,
        CLICK_DOWN: EMAIL_5
    },
    INBOX_DOWN: {
        SCROLL_UP: INBOX_MID,
        CLICK_UP: EMAIL_5,
        CLICK_MID: EMAIL_6,
        CLICK_DOWN: EMAIL_7
    }
}



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
                reward.append(1 if (a == label).item() else -0.1)
            # Take dummy action, since existing action may be out of
            # bounds
            # Bypass parent class
            state, _, _, info = self.env.step([0] * len(action))
            return state, reward, done, info
        # Bypass parent class
        return self.env.step(action)


class FakeInboxScrollMetaEnv(meta_exploration.MetaExplorationEnv):
    MAX_STEPS = 4
    NUM_TRAIN = 22000
    NUM_TEST = 4000
    CLICK_LOCATIONS = 6
    DATA_DIR = "./data_envs_scroll"

    def __init__(self, env_id, _):
        super().__init__(env_id, EmailInboxObservation)
        self._steps = 0
        self.cur_states = [0 for _ in range(NUM_INSTANCES)]
 
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Sequence(
                gym.spaces.Dict({
                    'screenshot': gym.spaces.Box(low=0, high=255, shape=(TASK_HEIGHT, TASK_WIDTH, 1), dtype=np.uint8),
                    'question': gym.spaces.Text(min_length=0, max_length=TEXT_MAX_LENGTH, charset=ASCII_CHARSET)
                })
            ),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([500]),
                dtype=np.int)
        })
        self.action_space = gym.spaces.Discrete(self.CLICK_LOCATIONS)
        self.exploitation = False
        self.df = pd.read_csv(os.path.abspath(f"{self.DATA_DIR}/inbox_samples.csv"))
        self._questions = [self.df.iloc[idx, 1] for idx in env_id]
        self._labels = [int(self.df.iloc[idx, 2]) for idx in env_id]

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
        return self._labels

    @property
    def questions(self):
        return self._questions


    def _get_next_state(self, cur_state, action):
        if cur_state in [0, 1] and action == 0:
            return cur_state + 1
        if cur_state in [1, 2] and action == 1:
            return cur_state - 1
        if 

    def _step(self, action):
        if self.exploitation:
            state = [{
                "screenshot": np.zeros((TASK_HEIGHT, TASK_WIDTH, 1)),
                "question": "None",
                "dom": "None"
            } for _ in range(NUM_INSTANCES)]
            reward = [0] * NUM_INSTANCES
            info = [None] * NUM_INSTANCES
            done = [True] * NUM_INSTANCES
        else:
            self.cur_states = [a+1 if c == 0 else c for a, c in zip(action, self.cur_states)]
            state = [{
                "screenshot": read_image(f"{self.DATA_DIR}/inboxes/{idx}-{self.cur_states[i]-1}.png").permute(1, 2, 0).cuda(),
                "question": self._questions[i],
                "dom": "None"
            } for i, (idx, a) in enumerate(zip(self._env_id, action))]
            reward = [0] * NUM_INSTANCES
            info = [None] * NUM_INSTANCES
            done = [False] * NUM_INSTANCES
            self._steps += 1
            done = done if self._steps < type(self).MAX_STEPS else [True]*NUM_INSTANCES
        return state, reward, done, info

    def _reset(self):
        # old hack but messes up evaluation of correct answer
        self._steps = 0
        self.cur_states = [0 for _ in range(NUM_INSTANCES)]
        obs = [{
            "question": self._questions[i],
            "dom": None,
            "screenshot": read_image(f"./data_envs/inboxes/{idx}.png").permute(1, 2, 0).cuda()
        } for i, idx in enumerate(self._env_id)]
        return obs

    def render(self, mode=None):
        imgs = []
        for i in range(NUM_INSTANCES):
            suffix = f"-{self.cur_states[i] - 1}" if self.cur_states[i] != 0 else ""
            img = Image.open(f"./data_envs/inboxes/{self._env_id[i]}{suffix}.png")
            img = render.Render(img)
            img.write_text("Underlying env ID: {}".format(self._env_id[i]))
            img.write_text(f"Q: {self._questions[i]}")
            img.write_text(f"A: {self._labels[i]}")
            imgs.append(img)
        return imgs
    
    @property
    def underlying_env_id(self):
        return self._env_id

    def set_underlying_env_id(self, id):
        self._env_id = id
        self._questions = [self.df.iloc[idx, 1] for idx in id]
        self._labels = [int(self.df.iloc[idx, 2]) for idx in id]
