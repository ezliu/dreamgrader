import csv
import json
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image
from time import sleep

from envs.miniwob.inbox import InboxMetaEnv
from envs.miniwob.constants import NUM_INSTANCES

NUM_SAMPLES = 16

samples = 0

f = open("./data_envs_scroll/inbox_samples.csv", "w+")
writer = csv.writer(f)
writer.writerow(["id", "question", "label"])


def save_state_screenshot(iter, state, suffix):
    for j in range(NUM_INSTANCES):
        image = state[j].observation.screenshot[:,:,0]
        image = Image.fromarray(image.numpy(), mode="L")
        image.save(f"./data_envs_scroll/inboxes/{iter * NUM_INSTANCES + j}{suffix}.png")

def save_state_dom(iter, state, suffix):
    for j in range(NUM_INSTANCES):
        dom = state[j].observation.dom
        with open(f"./data_envs_scroll/doms/{iter * NUM_INSTANCES + j}{suffix}.txt", "w+") as f:
            f.write(dom)


def save_state(iter, state, suffix):
    save_state_screenshot(iter, state, suffix)
    save_state_dom(iter, state, suffix)


for i in tqdm(range(NUM_SAMPLES // NUM_INSTANCES)):
    env = InboxMetaEnv.create_env(i)
    state = env.reset()

    # Save initial screenshot
    save_state(i, state, "")
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)

    # Save first scroll position
    save_state(i, state, "-0")
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)

    # Save second scroll position
    save_state(i, state, "-1")

    # Save 1st email
    _ = env.reset()
    state, _, _, _ = env.step([2 for _ in range(NUM_INSTANCES)])
    save_state(i, state, "-2")

    # Save 2nd email
    _ = env.reset()
    state, _, _, _ = env.step([3 for _ in range(NUM_INSTANCES)])
    save_state(i, state, "-3")

    # Save 3rd email
    _ = env.reset()
    state, _, _, _ = env.step([4 for _ in range(NUM_INSTANCES)])
    save_state(i, state, "-4")

    # Save 4th email
    _ = env.reset()
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)
    state, _, _, _ = env.step([3 for _ in range(NUM_INSTANCES)])
    save_state(i, state, "-5")

    # Save 5th email
    _ = env.reset()
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)
    state, _, _, _ = env.step([4 for _ in range(NUM_INSTANCES)])
    save_state(i, state, "-6")


    # Save 6th email
    _ = env.reset()
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)
    state, _, _, _ = env.step([4 for _ in range(NUM_INSTANCES)])
    save_state(i, state, "-7")


    # Save 7th email
    _ = env.reset()
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)
    state, _, _, _ = env.step([0 for _ in range(NUM_INSTANCES)])
    sleep(0.3)
    state, _, _, _ = env.step([4 for _ in range(NUM_INSTANCES)])
    save_state(i, state, "-8")
    
    _ = env.reset()
    for j in range(NUM_INSTANCES):
        emails = env._env.instances[j].driver.execute_script("return all_emails")
        writer.writerow([str(samples), json.dumps(emails)])
        samples += 1
        f.flush()
f.close()