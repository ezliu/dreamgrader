import csv
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image

from envs.miniwob.inbox import InboxMetaEnv
from envs.miniwob.constants import NUM_INSTANCES

NUM_SAMPLES = 2000000

samples = 0

f = open("./data_fontsize/inbox_samples.csv", "w+")
writer = csv.writer(f)
writer.writerow(["id", "filename", "name", "label"])


for i in tqdm(range(NUM_SAMPLES // NUM_INSTANCES)):
    env = InboxMetaEnv.create_env(i)
    state = env.reset()
    state, _, _, _ = env.step([20 for _ in range(NUM_INSTANCES)])
    for j, instance in enumerate(env._env.instances):
        image = state[j].observation.screenshot[:,:,0]
        image = Image.fromarray(image.numpy(), mode="L")
        filename = f"{i}-{j}.png"
        image.save(f"./data_fontsize/inboxes/{filename}")
        with open(f"./data_fontsize/doms/{filename.split('.')[0]}.txt", "w+") as f2:
            f2.write(state[j].observation.dom)
        question = state[j].observation.question
        answer = env.env_id[j]
        writer.writerow([str(samples), filename, question, answer])
        samples += 1
        f.flush()
f.close()