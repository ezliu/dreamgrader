import os
import ast
import csv
import json
import torch
import itertools
import collections
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
from PIL import Image
from gym import spaces

import render
import wrappers
import meta_exploration
from play2grade import bounce_var as bounce


class BounceImageWrapper(gym.Wrapper):
    """Subclass that returns image as state in step function"""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self.env.get_image(), reward, done, info

    def reset(self):
        self.env.reset()
        return self.env.get_image()


def read_csv(filename, error_type):
    data = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((ast.literal_eval(row["Program"]), ast.literal_eval(row["Multi-Error Label"])))
    programs = [program[0] for program in data]
    labels = [[1 if error_type in program[1] else 0] for program in data]
    return programs, labels


def apply_label(program, label):
    """Inserts a specific error into a program"""

    if label == "whenMiss-noBallLaunch":  # 0
        program["when ball misses paddle"].remove("launch new ball")
    elif label == "whenGoal-noBallLaunch":  # 1
        program["when ball in goal"].remove("launch new ball")
    elif label == "whenRun-noBallLaunch":  # 2
        program["when run"].remove("launch new ball")
    elif label == "whenMiss-noOpponentScore":  # 3
        program["when ball misses paddle"].remove("score opponent point")
    elif label == "whenWall-illegal-launchBall":  # 4
        program["when ball hits wall"].append("launch new ball")
    elif label == "whenGoal-illegal-incrementOpponentScore":  # 5
        program["when ball in goal"].append("score opponent point")
    elif label == "whenGoal-noPlayerScore":  # 6
        program["when ball in goal"].remove("score point")
    elif label == "whenMiss-illegal-incrementPlayerScore":  # 7
        program["when ball misses paddle"].append("score point")
    elif label == "whenMove-error":  # 8
        program["when left arrow"] = ["move right"]
        program["when right arrow"] = ["move left"]
    elif label == "whenPaddle-illegal-incrementPlayerScore":  # 9
        program["when ball hits paddle"].append("score point")
    elif label == "whenWall-illegal-incrementPlayerScore":  # 10
        program["when ball hits wall"].append("score point")
    elif label == "whenWall-illegal-incrementOpponentScore":  # 11
        program["when ball hits wall"].append("score opponent point")
    elif label == "whenGoal-illegal-bounceBall":  # 12
        program["when ball in goal"] = ["bounce ball"]
    elif label == "whenWall-illegal-moveLeft":  # 13
        program["when ball hits wall"].append("move left")
    elif label == "whenWall-illegal-moveRight":  # 14
        program["when ball hits wall"].append("move right")
    else:
        raise ValueError("Unknown label")


def generate_programs(error_labels):
    """Generate all error permutations"""
    programs = itertools.product([0, 1], repeat=len(error_labels))
    data = []
    for program in programs:
        correct_program = bounce.Program()
        correct_program.set_correct()
        p = correct_program.config_dict
        for label, on in zip(error_labels, program):
            if on:
                apply_label(p, label)

        human_labels = {label for label, on in zip(error_labels, program) if on}

        if "whenRun-noBallLaunch" in human_labels and len(human_labels) > 1:  # no other bugs
            continue

        # if goal bounce, then no goal stuff
        if "whenGoal-illegal-bounceBall" in human_labels and ("whenGoal-noBallLaunch" in human_labels or "whenGoal-noPlayerScore" in human_labels or "whenGoal-illegal-incrementOpponentScore" in human_labels):
            continue

        if "whenWall-illegal-moveRight" in human_labels and "whenWall-illegal-moveLeft" in human_labels:
            continue

        #if not "whenWall-illegal-moveRight" in human_labels and not "whenWall-illegal-moveLeft" in human_labels:
        #    continue
        
        data.append((p, program))
    return zip(*data)


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
            self.action_space = spaces.Discrete(len(BounceMetaEnv.LABELS))

    def _instruction_observation_space(self):
        return gym.spaces.Box(np.array([0]), np.array([1]), dtype=np.int)

    def _reward(self, instruction_state, action, original_reward):
        return original_reward, False

    def _generate_instructions(self, test=False):
        return np.array([0]) # dummy unused instruction

    def step(self, action):
        if self._exploitation:
            done = True
            counts = collections.Counter()
            label = BounceMetaEnv.LABELS[self.env.env_id]
            reward = np.mean(label == action)
            for l, a in zip(label, action):
                counts[(l, a)] += 1
            # Take dummy action, since existing action may be out of
            # bounds
            state, _, _, info = super().step(0)
            info["counts"] = counts
            return state, reward, done, info
        return super().step(action)


class BounceMetaEnv(meta_exploration.MetaExplorationEnv):
    # List of program config dicts to train / test on
    PROGRAMS = None
    # Labels corresponding the the programs. List of binary
    # flags indicating whether error at position i is on or off
    LABELS = None
    # Human readable error names
    ERROR_LABELS = None
    # Error index used in binary classification setting
    BINARY_INDEX = None
    # Max episode horizon
    MAX_STEPS = None
    # Score after which bounce game terminates
    MAX_SCORE = None
    # Number of times an action is repated
    SKIP_FRAMES = None
    # Use pixels as state
    IMG_STATE = None
    # Number of programs to use for training
    TRAIN_SIZE = None
    # Optional hardcoded train env ids
    TRAIN_IDS = None
    TEST_IDS = None

    def __init__(self, env_id, wrapper):
        super().__init__(env_id, wrapper)
        self._steps = 0
        program = bounce.Program()
        program.loads(type(self).PROGRAMS[env_id])
        env = bounce.BounceEnv(
            program, bounce.SELF_MINUS_HALF_OPPO, num_balls_to_win=type(self).MAX_SCORE)
        env = wrappers.ActionRepeatWrapper(env, type(self).SKIP_FRAMES)
        if type(self).IMG_STATE:
            env = wrappers.WarpFrame(BounceImageWrapper(env))
        self.observation_space = gym.spaces.Dict({
        "observation": env.observation_space,
        "env_id": gym.spaces.Box(np.array([0]),
            np.array([len(type(self).PROGRAMS) + 1]),
            dtype=np.int)
        })
        self._env = env
        self._env.reset()
        self.action_space = self._env.unwrapped.action_space

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([len(type(self).PROGRAMS) + 1])
        dtype = np.int
        return low, high, dtype

    @classmethod
    def env_ids(cls):
        if not ((cls.TRAIN_IDS and cls.TEST_IDS) or cls.TRAIN_SIZE):
            raise ValueError("Need to pass IDs or size")

        if cls.TRAIN_IDS:
            return np.array(cls.TRAIN_IDS), np.array(cls.TEST_IDS)
        elif cls.TRAIN_SIZE:
            if cls.TRAIN_SIZE > len(cls.PROGRAMS) - 1:
                raise ValueError(
                        f"Train set size too large (max={len(cls.PROGRAMS)})")
            rng = np.random.RandomState(0)
            ids = list(range(len(cls.PROGRAMS)))
            rng.shuffle(ids)
            train_ids = np.array(ids[:cls.TRAIN_SIZE])
            test_ids = np.array(ids[cls.TRAIN_SIZE:])
            return train_ids, test_ids

    @classmethod
    def create_env(cls, seed, test=False, wrapper=None):
        """Randomly creates an environment instance.

        Args:
            seed (int): used to randomly select.
            test (bool): determines whether to make an environment from the train or
                test split.
            wrapper (function): gets called on observations. Defaults to converting
                observations to torch.tensor. Pass lambda state: state to receive numpy
                observations.

        Returns:
            MetaExplorationEnv
        """
        if wrapper is None:
            wrapper = lambda state: torch.tensor(state)

        random = np.random.RandomState(seed)
        train_ids, test_ids = cls.env_ids()
        if test:
            env_id = test_ids[random.randint(len(test_ids))]
            return cls(env_id, wrapper)
        
        with_error = random.choice([True, False])
        while True:
            idx = random.randint(len(train_ids))
            if cls.LABELS[train_ids[idx]][cls.BINARY_INDEX] == with_error:
                break
        env_id = train_ids[idx]
        return cls(env_id, wrapper)

    def _step(self, action):
        state, reward, done, info = self._env.step(action)
        self._steps += 1
        done = done or self._steps >= type(self).MAX_STEPS
        return state, reward, done, info

    def _reset(self):
        self._steps = 0
        return self._env.reset()

    def render(self, mode="rgb_array"):
        env_render = self._env.render(mode="rgb_array")
        image = render.Render(Image.fromarray(env_render))
        image.write_text("Env ID: {}".format(self.env_id))
        image.write_text(f"Label: {type(self).LABELS[self.env_id]}")
        return image

    @classmethod
    def load_config(cls, config=None):
        """Parses values from the provided config
        
        Requires the following fields in the config:
        - max_score (int): game terminates after reaching this score
        - max_steps (int): max episode horizon
        - error_labels (List[str]): list of valid bounce errors used
        - binary_index (int, optional): index of bounce error to classify
        - skip_frames (int): number of times an action is repeated

        Args:
            config (config.Config, optional)
        """
        if config is None:
            # Default config for demonstration purposes
            config = {
                "max_score": 10,
                "max_steps": 100,
                "error_labels": [
                    "whenMiss-noBallLaunch",
                    "whenGoal-noBallLaunch",
                    "whenRun-noBallLaunch",
                    "whenMiss-noOpponentScore",
                    "whenWall-illegal-launchBall",
                    "whenGoal-illegal-incrementOpponentScore",
                    "whenGoal-noPlayerScore",
                    "whenMiss-illegal-incrementPlayerScore",
                    "whenMove-error",
                    "whenPaddle-illegal-incrementPlayerScore",
                    "whenWall-illegal-incrementPlayerScore",
                    "whenWall-illegal-incrementOpponentScore",
                    "whenGoal-illegal-bounceBall",
                    "whenWall-illegal-moveLeft",
                    "whenWall-illegal-moveRight"
                ],
                "binary_index": 2,
                "skip_frames": 2,
                "img_state": False
            }
        data_file = config.get("data_file")
        error_type = config.get("error_type")
        cls.ERROR_LABELS = config.get("error_labels")
        cls.BINARY_INDEX = config.get("binary_index")
        if data_file and (cls.ERROR_LABELS or cls.BINARY_INDEX):
            raise ValueError("Invalid config. Cannot provide error labels or binary index if data file is provided")
        if data_file:
            programs, labels = read_csv(data_file, error_type)
            cls.BINARY_INDEX = 0
            cls.ERROR_LABELS = [error_type]
        else:
            programs, labels = generate_programs(config.get("error_labels"))
        cls.PROGRAMS = programs
        cls.LABELS = labels
        cls.MAX_STEPS = config.get("max_steps")
        cls.MAX_SCORE = config.get("max_score")
        cls.SKIP_FRAMES = config.get("skip_frames")
        cls.TRAIN_SIZE = config.get("train_size")
        cls.TRAIN_IDS = config.get("train_ids")
        cls.TEST_IDS = config.get("test_ids")
        # Make img state default
        cls.IMG_STATE = config.get("img_state", True)
        invariances = config.get("invariances")
        if invariances:
            rng = np.random.RandomState(0)
            train_ids, test_ids = cls.env_ids()
            for split, ids in [("train", train_ids), ("test", test_ids)]:
                for idx in ids:
                    program = programs[idx]
                    for mode in invariances.keys():
                        speed = rng.choice(invariances.get(mode).get(split))
                        # Overrides any existing speed configuration
                        program["when run"].append(f"set '{speed}' {mode} speed")


class BinaryInstructionWrapper(InstructionWrapper):
    def __init__(self, env, exploration_trajectory,
                 seed=None, test=False,
                 first_episode_no_instruction=False,
                 first_episode_no_optimization=False,
                 fixed_instructions=False, exploitation=False):
        super().__init__(
            env, exploration_trajectory, seed=seed, test=test,
            first_episode_no_instruction=first_episode_no_instruction,
            first_episode_no_optimization=first_episode_no_optimization,
            fixed_instructions=fixed_instructions)
        self._exploitation = exploitation
        if exploitation:
            self.action_space = spaces.Discrete(2)

    def _instruction_observation_space(self):
        return gym.spaces.Box(np.array([0]), np.array([1]), dtype=np.int)

    def step(self, action):
        if self._exploitation:
            done = True
            full_label = type(self.env).LABELS[self.env._env_id]
            binary_label = full_label[type(self.env).BINARY_INDEX]
            reward = int(binary_label == action)
            # Take dummy action, since existing action may be out of
            # bounds
            # Bypass parent class
            state, _, _, info = self.env.step(0)
            return state, reward, done, info
        # Bypass parent class
        return self.env.step(action)


class BounceBinaryMetaEnv(BounceMetaEnv):

    @classmethod
    def instruction_wrapper(cls):
        return BinaryInstructionWrapper

    def render(self, mode="rgb_array"):
        env_render = self._env.render(mode="rgb_array")
        image = render.Render(Image.fromarray(env_render))
        image.write_text("Underlying env ID: {}".format(self._env_id))
        image.write_text("Env ID: {}".format(self.env_id))
        image.write_text(f"Label: {type(self).LABELS[self._env_id]}")
        image.write_text(
            f"Binary label: {type(self).ERROR_LABELS[type(self).BINARY_INDEX]}")
        return image

    @property
    def env_id(self):
        # Needed because env_id is used as the label
        full_label = type(self).LABELS[self._env_id]
        return full_label[type(self).BINARY_INDEX]
