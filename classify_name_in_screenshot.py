import os
import torch
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

class NameInImageDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.names_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.names_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.names_frame.iloc[idx, 1])
        image = read_image(img_name).permute(1, 2, 0)
        name = self.names_frame.iloc[idx, 2]
        label = self.names_frame.iloc[idx, 3]
        x = {'screenshot': image.to(dtype=torch.float32), 'question': name, 'dom': '', 'label': label}
        # sample = {'x': EmailInboxObservation(x), 'y': label}
        return x

# Create dataset
dataset = NameInImageDataset(csv_file='data/inbox_samples.csv', root_dir='data/inboxes')

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

model = torch.nn.Sequential(*[
    MiniWobEmbedder(None, 64),
    torch.nn.Linear(64, 2)
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

          labels = data['label'].to(device)

          # forward + backward + optimize
          outputs = model(inputs)
          outputs = torch.nn.functional.softmax(outputs)
          predicted = torch.argmax(outputs, dim=1)
          
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # Binary cross entropy for binary classification
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

inputs = None
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader)):
        # get the inputs; data is a dict
        inputs = [EmailInboxObservation({
            "screenshot": obs[0],
            "question": obs[1],
            "dom": obs[2]
        }).cuda() for obs in zip(data["screenshot"], data["question"], data["dom"])]

        labels = data['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = torch.nn.functional.softmax(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Step {i}, loss: {loss.item() / 64}")
            # Running validation
    eval_acc = evaluate_model(model, val_loader)
    print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}, eval acc: {eval_acc}")
test_acc = evaluate_model(model, test_loader)
print('Finished Training')
print(f"Test acc: {test_acc}")
