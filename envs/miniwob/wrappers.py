
import gymnasium as gym
import numpy as np
from typing import List, Dict
from PIL import Image
from io import BytesIO
from miniwob.envs.miniwob_envs import EmailInboxEnv
from miniwob.action import create_coord_scroll_action, create_coord_click_action
from envs.miniwob.constants import TASK_HEIGHT, TASK_WIDTH, TASK_HEIGHT_OFFSET, TASK_WIDTH_OFFSET
import cv2

from envs.miniwob.constants import LOREM_WORDS, PEOPLE_NAMES, TEXT_MAX_LENGTH, ASCII_CHARSET, NUM_INSTANCES

# Image preprocessing – check if I can look at it, grayscale
# setup recurring meeting with Chelsea & Evan
# figure out miniwob seeding so it sees the same env ID multiple times
# bucket actions into pixel regions


# New Qs:
"""
- Number of words we embed -> ~300
-> try this out, otherwise make the input sentences structured if it causes issues

- How to have multiple action outputs -> should serialize or have 2 outputs?
-> Just flatten & reshape actions; Write out the math for 2 heads vs flatten to enable 2D actions

- Feasibility of transformer embedders -> especially multimodal
-> can switch to simpler stuff if that doesn't work

- Other ways to optimize runtime / compute

- Should I already try to improve runtime of embedders by using FlashTransformers or Hyena?
-> not high priority, only if needed for compute

- What about unsupervised pre-training of embedders?
-> long-term goal is pre-initialize from some large pretrainted model; thus this is not super important

- Ripping out all dreamgrader code not used by my experiments

- HAI grant by May (does not open until then)

- Should I use more advanced policy method than DDQN?
Next steps: get the above to work, then do the same for other MiniWobEnvs, impact of pre-training, see how we generalize across other miniwob environments / questions

- Look into Rainbow -> DQN with all improvements

-> will get N-Step return code from Evan
"""



class InboxScreenshotWrapper(gym.Wrapper):
    """Wrapper that adds a screenshot to the info dict returned by step()."""

    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self.observation_space = gym.spaces.Sequence(
            gym.spaces.Dict({
                'screenshot': gym.spaces.Box(low=0, high=255, shape=(TASK_HEIGHT, TASK_WIDTH, 1), dtype=np.uint8),
                'question': gym.spaces.Text(min_length=0, max_length=TEXT_MAX_LENGTH, charset=ASCII_CHARSET)
            })
        )

    def get_screenshot(self, instance_idx) -> np.ndarray:
        """Returns a screenshot of the task area as a numpy array."""
        png_data = self._env.instances[instance_idx].driver.get_screenshot_as_png()
        pil_image = Image.open(BytesIO(png_data))
        pil_image = pil_image.crop((TASK_WIDTH_OFFSET, TASK_HEIGHT_OFFSET,
                                    TASK_WIDTH_OFFSET + TASK_WIDTH,
                                    TASK_HEIGHT_OFFSET + TASK_HEIGHT))
        return np.array(pil_image).astype(np.uint8)[:, :, :3]

    def step(self, action):
        # The body has width 800 and height 210 pixels
        # Since miniwob uses weird coordinates, need to re-center
        # For scroll actions, (405, 210) corresponds to (0, 0) in our screenshot
        _, reward, done, _, info = self._env.step(action)
        obs = [{
            "screenshot": self.get_screenshot(i),
            "question": ""
        } for i in range(NUM_INSTANCES)]
        return obs, reward, done, False, info

    def reset(self, *args, **kwargs):
        _, i = self._env.reset(*args, **kwargs)
        obs = [{
            "screenshot": self.get_screenshot(i),
            "question": ""
        } for i in range(NUM_INSTANCES)]
        return obs, i

    def render(self, mode=None):
        return [self.get_screenshot(i) for i in range(NUM_INSTANCES)]
    

class InboxDOMWrapper(gym.Wrapper):
    """Wrapper that adds the DOM to the info dict returned by step()."""

    def __init__(self, env):
        super().__init__(env)
        self._env = env

    def step(self, action):
        obs, reward, done, _, info = self._env.step(action)
        for i, o in enumerate(obs):
            o['dom'] = self.get_dom(i)
        return obs, reward, done, False, info

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        for i, o in enumerate(obs):
            o['dom'] = self.get_dom(i)
        return obs, info
    
    @classmethod
    def convert_dom_to_text(cls, dom):
        start = f"< {dom['tag'].lower()} "
        start += f"class={dom['classes']} " if len(dom["classes"]) > 0 else ""
        start += f"id={dom['id']} " if len(dom["id"]) > 0 else ""
        start += "> "
        start += f" {dom['text']} " if 'text' in dom.keys() and len(dom["text"]) > 0 else ""
        start += f"{' '.join([cls.convert_dom_to_text(c) for c in dom['children']])} " if len(dom["children"]) > 0 else ""
        start += f"< /{dom['tag'].lower()} >"
        return start
    
    def get_dom(self, instance_idx):
        dom = self._env.instances[instance_idx].driver.execute_script("return core.getDOMInfo();")
        return self.convert_dom_to_text(dom)


class InboxQAWrapper(gym.Wrapper):
    """Wrapper that adds a screenshot to the info dict returned by step()."""

    QUESTION_TYPES = 5

    def __init__(self, env, env_ids):
        super().__init__(env)
        self._env = env
        self._env_ids = env_ids
        self.current_question = None

    def step(self, action):
        obs, reward, done, _, info = self._env.step(action)
        assert self.current_question is not None, "Must call reset() before step()"
        for i, o in enumerate(obs):
            o['question'] = self.current_question[i][0]
        return obs, reward, done, False, info

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        self.current_question = self._generate_questions()
        for i, o in enumerate(obs):
            o['question'] = self.current_question[i][0]
        return obs, info

    @staticmethod
    def _generate_words(minLen: int, maxLen: int, exclude=None) -> str:
        n = np.random.randint(minLen, maxLen+1)
        words = []
        first_word = True
        while len(words) < n:
            w = np.random.choice(LOREM_WORDS)
            if exclude is not None and w in exclude:
                continue
            if first_word:
                w = w.capitalize()
                first_word = False
            if np.random.rand() < 0.2 and len(words) + 1 < n:
                w += "."
                first_word = True
            elif len(words) + 1 == n:
                w = w.replace(',', '')
            words.append(w)
        return " ".join(words) + "."

    @staticmethod
    def _generate_name(exclude: List[str] = None) -> str:
        name = np.random.choice(PEOPLE_NAMES)
        while exclude is not None and name in exclude:
            name = np.random.choice(PEOPLE_NAMES)
        return name

    @staticmethod
    def _pick_name(emails: List[Dict[str, str]]) -> str:
        return emails[np.random.randint(len(emails))]['name']

    def _generate_questions(self):
        """Generates a question about the current state of the environment."""
        questions = []
        for instance in self._env.instances:
            # Hacky way to get the emails from the JS code
            emails = instance.driver.execute_script("return jQuery._data( document.getElementsByClassName(\"email-thread\")[0], \"events\" ).click[0].data.emails")
            np.random.seed(self._env_ids[instance.index])
            question = self._generate_question(emails)
            questions.append(question)
        return questions
    
    def set_qa_env_ids(self, env_ids):
        self._env_ids = env_ids

    def _generate_question(self, emails):
        """Generates a question about the current state of the environment."""
        is_true = np.random.randint(2)

        # question_type = np.random.randint(self.QUESTION_TYPES)
        question_type = 2
        
        names = [email['name'] for email in emails]
        if question_type == 0:
            # Generate prompt for "Is there an email from X?"
            name = self._pick_name(emails) if is_true else self._generate_name(exclude=names)
            question = f"Is there an email from {name}?"
        elif question_type == 1:
            # Generate prompt for "Is there an email from X with a subject line about Y?"
            if is_true:
                email_idx = np.random.randint(len(emails))
                name = emails[email_idx]['name']
                subject = emails[email_idx]['subject']
            else:
                false_case = np.random.randint(3)
                email_idx = np.random.randint(len(emails))
                if false_case == 0:
                    name = emails[email_idx]['name']
                    subject = self._generate_words(1, 3, exclude=emails[email_idx]['subject'])
                elif false_case == 1:
                    name = self._generate_name(exclude=names)
                    subject = emails[email_idx]['subject']
                else:
                    name = self._generate_name(exclude=names)
                    subject = self._generate_words(1, 3, exclude=emails[email_idx]['subject'])
            question = f"Is there an email from {name} with a subject line about '{subject}'?"
        elif question_type == 2:
            if is_true:
                email_idx = np.random.randint(len(emails))
                name = emails[email_idx]['name']
                body = emails[email_idx]['body']
            else:
                false_case = np.random.randint(3)
                email_idx = np.random.randint(len(emails))
                if false_case == 0:
                    name = emails[email_idx]['name']
                    body = self._generate_words(5, 15, exclude=emails[email_idx]['body'])
                elif false_case == 1:
                    name = self._generate_name(exclude=names)
                    body = emails[email_idx]['body']
                else:
                    name = self._generate_name(exclude=names)
                    body =self._generate_words(5, 15, exclude=emails[email_idx]['body'])
            body = body.split()
            state_idx = np.random.randint(max(len(body) - 5, 1))
            body_slice = " ".join(body[state_idx:state_idx+5])
            question = f"Is there an email from {name} about '{body_slice}'?"
        elif question_type == 3:
            # Generate prompt for "Is the Xth recent email from Y?"
            email_idx = np.random.randint(len(emails))
            if is_true:
                name = emails[email_idx]['name']
            else:
                name = self._generate_name(exclude=names)
            question = f"Is the {'most' if email_idx == 0 else '2nd' if email_idx == 1 else '3rd' if email_idx == 2 else 'least' if email_idx + 1 == len(emails) else f'{email_idx+1}th'} recent email from {name}?"
        elif question_type == 4:
            # Generate prompt for "Do I have X emails in my inbox?"
            n = len(emails) if is_true else np.random.randint(4, 12 + 1)
            question = f"Do I have {n} emails in my inbox?"
        return question, is_true


# Adapted from gym baselines
class WarpScreenshot(gym.ObservationWrapper):
    def observation(self, obs):
        for o in obs:
            o["screenshot"] = cv2.cvtColor(o["screenshot"], cv2.COLOR_RGB2GRAY)
            o["screenshot"] = cv2.resize(o["screenshot"], (TASK_HEIGHT, TASK_WIDTH), interpolation=cv2.INTER_AREA)
            o["screenshot"] = np.expand_dims(o["screenshot"], -1)
        return obs


class RestrictedActionWrapper(gym.ActionWrapper):
    CLICK_LOCATIONS = [(10, 10), (10, 30), (10, 50), (10, 70), (10, 90), (10, 110), (10, 130), (10, 150), (30, 10), (30, 30), (30, 50), (30, 70), (30, 90), (30, 110), (30, 130), (30, 150), (50, 10), (50, 30), (50, 50), (50, 70), (50, 90), (50, 110), (50, 130), (50, 150), (70, 10), (70, 30), (70, 50), (70, 70), (70, 90), (70, 110), (70, 130), (70, 150), (90, 10), (90, 30), (90, 50), (90, 70), (90, 90), (90, 110), (90, 130), (90, 150), (110, 10), (110, 30), (110, 50), (110, 70), (110, 90), (110, 110), (110, 130), (110, 150), (130, 10), (130, 30), (130, 50), (130, 70), (130, 90), (130, 110), (130, 130), (130, 150)]
    SCROLL_LOCATION = (TASK_HEIGHT//2, TASK_WIDTH//2)
    SCROLL_AMOUNT = 10

    def __init__(self, env):
        super().__init__(env)
        self._action_space = gym.spaces.Discrete(len(self.CLICK_LOCATIONS) + 2)

    def _convert_action(self, action):
        action = int(action)
        miniwob_action = None
        if action == 0:
            miniwob_action = create_coord_scroll_action(
                TASK_WIDTH_OFFSET + self.SCROLL_LOCATION[1], 
                TASK_HEIGHT_OFFSET + self.SCROLL_LOCATION[0], 
                0, self.SCROLL_AMOUNT)
        elif action == 1:
            miniwob_action = create_coord_scroll_action(
                TASK_WIDTH_OFFSET + self.SCROLL_LOCATION[1], 
                TASK_HEIGHT_OFFSET + self.SCROLL_LOCATION[0], 
                0, -self.SCROLL_AMOUNT)
        else:
            miniwob_action = create_coord_click_action(
                TASK_WIDTH_OFFSET + self.CLICK_LOCATIONS[action-2][1], 
                TASK_HEIGHT_OFFSET + self.CLICK_LOCATIONS[action-2][0])
        return miniwob_action

    def action(self, actions):
        return [self._convert_action(action) for action in actions]
        

    