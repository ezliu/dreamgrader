import os
import torch
from sophia import SophiaG
from tqdm import tqdm
import pandas as pd
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from embed import MiniWobEmbedder
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from envs.miniwob.inbox import EmailInboxObservation


torch.set_default_tensor_type(torch.FloatTensor)

class FontInImageDataset(Dataset):
    def __init__(self, root_dir, use_dom=False):
        self.names_frame = pd.read_csv(os.path.join(root_dir, 'inbox_samples.csv'))
        # self.names_frame = self.names_frame[self.names_frame['name'].str.contains('Nicola')]
        # self.names_frame = self.names_frame[self.names_frame['name'].str.contains('Lolita') | self.names_frame['name'].str.contains('Giustina') | self.names_frame['name'].str.contains('Ingaberg') | self.names_frame['name'].str.contains('Kassandra') | self.names_frame['name'].str.contains('Nicola') | self.names_frame['name'].str.contains('Celestia') | self.names_frame['name'].str.contains('Mozelle') | self.names_frame['name'].str.contains('Doralyn') | self.names_frame['name'].str.contains('Lu')]
        self.root_dir = root_dir
        self.use_dom = use_dom

    def __len__(self):
        return len(self.names_frame)
        # return 1000

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'inboxes', self.names_frame.iloc[idx, 1])
        dom = ''
        if self.use_dom:
            with open(os.path.join(self.root_dir, 'doms', self.names_frame.iloc[idx, 1].split(".")[0] + '.txt')) as f:
                dom = f.read()
        dom = dom.replace('< ', '<')
        dom = dom.replace(' >', '>')
        image = read_image(img_name).permute(1, 2, 0)
        name = self.names_frame.iloc[idx, 2]
        label = self.names_frame.iloc[idx, 3]
        # label = int(f" {name} " in dom)
        x = {'screenshot': image.to(dtype=torch.float32), 'question': name, 'dom': dom if self.use_dom else '', 'label': label}
        # sample = {'x': EmailInboxObservation(x), 'y': label}
        return x

# Create dataset
dataset = FontInImageDataset('data', use_dom=True)
dataset2 = FontInImageDataset('data_fontsize', use_dom=True)

tokens = set()

tokenizer = get_tokenizer('basic_english')

last_size = 1
iters = 0
#while last_size != len(tokens):
for _ in range(3000):
    last_size = len(tokens)
    dom = dataset[iters]['dom']
    for t in list(set(tokenizer(dom))):
        tokens.add(t)
    dom = dataset2[iters]['dom']
    for t in list(set(tokenizer(dom))):
        tokens.add(t)
    iters += 1

print(list(tokens))
print(len(tokens))
print(iters)