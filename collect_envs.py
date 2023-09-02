import csv
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image

from envs.miniwob.inbox import InboxMetaEnv
from envs.miniwob.constants import NUM_INSTANCES

NUM_SAMPLES = 500000

samples = 0

f = open("./data_envs/inbox_samples.csv", "w+")
writer = csv.writer(f)
writer.writerow(["id", "question", "label"])

def save_state_screenshot(iter, state, suffix):
    for j in range(NUM_INSTANCES):
        image = state[j].observation.screenshot[:,:,0]
        image = Image.fromarray(image.numpy(), mode="L")
        image.save(f"./data_envs/inboxes/{iter * NUM_INSTANCES + j}{suffix}.png")


for i in tqdm(range(NUM_SAMPLES // NUM_INSTANCES)):
    env = InboxMetaEnv.create_env(i)
    state = env.reset()
    save_state_screenshot(i, state, "")
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    save_state_screenshot(i, state, "-0")
    _ = env.reset()
    state, _, _, _ = env.step([1 for _ in range(NUM_INSTANCES)])
    save_state_screenshot(i, state, "-1")
    _ = env.reset()
    state, _, _, _ = env.step([2 for _ in range(NUM_INSTANCES)])
    save_state_screenshot(i, state, "-2")
    
    for j in range(NUM_INSTANCES):
        question = state[j].observation.question
        answer = env.env_id[j]
        writer.writerow([str(samples), question, answer])
        samples += 1
        f.flush()
f.close()