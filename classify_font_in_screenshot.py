import os
import torch
import random
from sophia import SophiaG
from tqdm import tqdm
import pandas as pd
import torch.optim as optim
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
        """names = dom.split("email-body>")[1:]
        names = [s.split("<")[0] for s in names]


        idx = random.randint(0,2)
        if random.randint(0,1):
            name = names[idx]
            label = 1
        else:
            new_idx = idx
            while new_idx == idx:
                new_idx = random.randint(0,2)
            name = names[new_idx]
            label = 0"""
        # name = f"Is the {'1st' if idx == 0 else '2nd' if idx == 1 else '3rd'} email from {name.strip().replace('..', '')}"

        x = {'screenshot': image.to(dtype=torch.float32), 'question': name, 'dom': dom if self.use_dom else '', 'label': label}
        # sample = {'x': EmailInboxObservation(x), 'y': label}
        return x
    

BATCH_SIZE = 32
EMBED_DIM = 64
USE_DOM = True

# Create dataset
dataset = FontInImageDataset('data', use_dom=USE_DOM)



dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Num samples: {dataset_size}")

class AverageAlongAxis(torch.nn.Module):
    def __init__(self, axis):
        super(AverageAlongAxis, self).__init__()
        self.axis = axis

    def forward(self, x):
        return torch.mean(x, dim=self.axis)

class ReshapeConvOutputs(torch.nn.Module):
    def __init__(self):
        super(ReshapeConvOutputs, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)

"""model = torch.nn.Sequential(*[
    MiniWobScreenshotEmbedder(None, 512),
    ReshapeConvOutputs(),
    torch.nn.Linear(100 * 512, 2)
]).float()"""

model = torch.nn.Sequential(*[
    MiniWobEmbedder(None, EMBED_DIM, use_dom=USE_DOM),
    torch.nn.Dropout(0.1),
    torch.nn.ReLU(),
    torch.nn.Linear(EMBED_DIM, 2)
]).float()


def evaluate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # We don't need gradients for evaluation
        for _, data in enumerate(val_loader):
          # get the inputs; data is a dict
          inputs = [EmailInboxObservation({
              "screenshot": obs[0],
              "question": obs[1],
              "dom": obs[2]
          }).cuda() for obs in zip(data["screenshot"], data["question"], data["dom"])]
          # inputs = data["screenshot"].cuda()

          labels = data['label'].to(device)

          # forward + backward + optimize
          outputs = model(inputs)
          # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
          # outputs = torch.nn.functional.softmax(outputs)
          predicted = torch.argmax(outputs, dim=1)
          
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # Binary cross entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)

inputs = None
num_epochs = 30
for epoch in range(num_epochs):
    running_loss = 0.0
    cur_loss = 0.0
    for i, data in tqdm(enumerate(train_loader)):
        # get the inputs; data is a dict
        inputs = [EmailInboxObservation({
            "screenshot": obs[0],
            "question": obs[1],
            "dom": obs[2]
        }).cuda() for obs in zip(data["screenshot"], data["question"], data["dom"])]
        # inputs = data["screenshot"].cuda()
        labels = data['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        # outputs = torch.nn.functional.softmax(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        cur_loss += loss.item()
        if i % 100 == 0:
            print(f"Step {i}, loss: {cur_loss / 100}")
            cur_loss = 0.0
            # Running validation
        if i % 1000 == 0:
            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == labels).sum().item()
            print(f"Train correct: {correct / len(labels)}")
    eval_acc = evaluate_model(model, val_loader)
    print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}, eval acc: {eval_acc}")
test_acc = evaluate_model(model, test_loader)
print('Finished Training')
print(f"Test acc: {test_acc}")
