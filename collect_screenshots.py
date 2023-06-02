import csv
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image

from envs.miniwob.inbox import InboxMetaEnv
from envs.miniwob.constants import PEOPLE_NAMES

def pick_random_name(exclude: List[str]):
    name = exclude
    while name == exclude:
        name = np.random.choice(PEOPLE_NAMES)
    return name

NUM_SAMPLES = 200000

samples = 0

f = open("./data/inbox_samples.csv", "w+")
writer = csv.writer(f)
writer.writerow(["id", "filename", "name", "label"])


for i in tqdm(range(NUM_SAMPLES // 96)):
    env = InboxMetaEnv.create_env(i)
    state = env.reset()
    for j, instance in enumerate(env._env.instances):
        image = state[j].observation.screenshot[:,:,0]
        image = Image.fromarray(image.numpy(), mode="L")
        filename = f"{i}-{j}.png"
        image.save(f"./data/inboxes/{filename}")
        with open(f"./data/doms/{filename.split('.')[0]}.txt", "w+") as f2:
            f2.write(state[j].observation.dom)
        emails = instance.driver.execute_script("return jQuery._data( document.getElementsByClassName(\"email-thread\")[0], \"events\" ).click[0].data.emails")
        exclude_names = [email["name"] for email in emails[:4]]
        for idx in range(3):
            name = emails[idx]["name"]
            writer.writerow([str(samples), filename, name, str(1)])
            samples += 1
            false_name = pick_random_name(exclude_names)
            exclude_names.append(false_name)
            writer.writerow([str(samples), filename, false_name, 0])
            samples += 1
            f.flush()
f.close()