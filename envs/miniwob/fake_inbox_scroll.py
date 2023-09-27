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


# Constants
NUM_EMAILS = 7
SYMBOLS = ["■", "▲", "◆", "▼", "●", "◖", "★"]
SIZES = ['small', 'medium', 'large']

# Actions
SCROLL_DOWN = 0
SCROLL_UP = 1
CLICK_UP = 2
CLICK_MID = 3
CLICK_DOWN = 4
BACK = 5

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
    },
    EMAIL_1: {
        BACK: INBOX_UP
    },
    EMAIL_2: {
        BACK: INBOX_UP
    },
    EMAIL_3: {
        BACK: INBOX_UP
    },
    EMAIL_4: {
        BACK: INBOX_MID
    },
    EMAIL_5: {
        BACK: INBOX_MID
    },
    EMAIL_6: {
        BACK: INBOX_DOWN
    },
    EMAIL_7: {
        BACK: INBOX_DOWN
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
    MAX_STEPS = None
    NUM_TRAIN = None
    NUM_TEST = None
    DATA_DIR = None
    USE_SYMBOL_QUERIES = None
    USE_BACK_ACTION = None

    NUM_ACTIONS_WITH_BACK = 6
    NUM_ACTIONS_NO_BACK = 5
    DEFAULT_DATA_DIR = "/scr-ssd/moritzst/data_envs_scroll"

    def __init__(self, env_id, _):
        super().__init__(env_id, EmailInboxObservation)
        self._steps = 0
        self.cur_states = [0 for _ in range(NUM_INSTANCES)]
        self._env_numbers = None
        self._email_indices = None
        self._email_sizes = None
 
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Sequence(
                gym.spaces.Dict({
                    'screenshot': gym.spaces.Box(low=0, high=255, shape=(TASK_HEIGHT, TASK_WIDTH, 1), dtype=np.uint8),
                    'question': gym.spaces.Text(min_length=0, max_length=TEXT_MAX_LENGTH, charset=ASCII_CHARSET)
                })
            ),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([type(self).NUM_TRAIN + type(self).NUM_TEST + 1]),
                dtype=np.int)
        })
        self.action_space = gym.spaces.Discrete(type(self).NUM_ACTIONS_WITH_BACK if type(self).USE_BACK_ACTION else type(self).NUM_ACTIONS_NO_BACK)
        self.exploitation = False
        self.df = pd.read_csv(os.path.abspath(f"{self.DATA_DIR}/inbox_samples.csv"))
        
        self.set_underlying_env_id(env_id)


    def calculate_envs(self):
        self._env_numbers

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    @classmethod
    def load_config(cls, config: dict = None):
        cls.USE_SYMBOL_QUERIES = config.get("use_symbol_queries", False)
        cls.DATA_DIR = config.get("data_dir", cls.DEFAULT_DATA_DIR)
        cls.MAX_STEPS = config.get("max_steps", 4)
        cls.NUM_TRAIN = config.get("num_train", 100)
        cls.NUM_TEST = config.get("num_test", 10)
        cls.USE_BACK_ACTION = config.get("use_back_action", False)

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
        if cur_state in TRANSITIONS and action in TRANSITIONS[cur_state]:
            return TRANSITIONS[cur_state][action]
        return cur_state


    def _get_screenshot(self, env_number, cur_state):
        img = read_image(f"{self.DATA_DIR}/inboxes/{env_number}/{cur_state}.png").permute(1, 2, 0)
        if torch.cuda.is_available():
            img = img.cuda()
        return img


    def _generate_question_and_label(self, env_id, env_number, email_number, email_size):
        emails = json.loads(self.df.iloc[env_number, 1])
        font_size = SIZES[email_size]
        question = f"Is the {'1st' if email_number == 0 else '2nd' if email_number == 1 else '3rd' if email_number == 2 else f'{email_number+1}th'} email body {font_size}?"
        
        # Only activate if using symbol queries
        if FakeInboxScrollMetaEnv.USE_SYMBOL_QUERIES:
            symbol = SYMBOLS[email_number]
            symbol_order = [e["symbol"] for e in emails]
            email_number = symbol_order.index(symbol)

        label = emails[email_number]["font_size"] == font_size
        return question, label, email_number
    

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
            self.cur_states = [self._get_next_state(cur_state, a) for cur_state, a in zip(self.cur_states, action)]
            state = [{
                "screenshot": self._get_screenshot(idx, state),
                "question": self._questions[i],
                "dom": "None"
            } for i, (idx, state) in enumerate(zip(self._env_numbers, self.cur_states))]
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
            "screenshot": self._get_screenshot(idx, state),
            "question": self._questions[i],
            "dom": "None"
        } for i, (idx, state) in enumerate(zip(self._env_numbers, self.cur_states))]
        return obs

    def render(self, mode=None):
        imgs = []
        for i in range(NUM_INSTANCES):
            img = Image.open(f"{self.DATA_DIR}/inboxes/{self._env_numbers[i]}/{self.cur_states[i]}.png")
            img = render.Render(img)
            img.write_text("Underlying env ID: {}".format(self._env_id[i]))
            question = self._questions[i]
            if type(self).USE_SYMBOL_QUERIES:
                symbol = SYMBOLS[self._email_indices[i]]
                question = question.split()
                question.pop(2)
                question.insert(2, symbol)
                question = " ".join(question)
            img.write_text(f"Q: {question}")
            img.write_text(f"A: {self._labels[i]}")
            imgs.append(img)
        return imgs
    
    @property
    def underlying_env_id(self):
        return self._env_id

    def set_underlying_env_id(self, id):
        self._env_id = id
        id = list(range(30, 46))
        self._env_numbers = [idx // (NUM_EMAILS * len(SIZES)) for idx in id]
        self._email_indices = [(idx % (NUM_EMAILS * len(SIZES))) // len(SIZES) for idx in id]
        self._email_sizes = [(idx % (NUM_EMAILS * len(SIZES))) % len(SIZES) for idx in id]
        question_labels = [self._generate_question_and_label(id, env_number, email_number, email_size) for id, env_number, email_number, email_size in zip(self._env_id, self._env_numbers, self._email_indices, self._email_sizes)]
        self._questions = [q for (q, _, _) in question_labels]
        self._labels = [l for (_, l, _) in question_labels]
        self._email_indices = [i for (_, _, i) in question_labels]
        print(id)
        print(self._env_numbers)
        print(self._email_indices)
        print(self._email_sizes)
        print(self._questions)
        print("--")
