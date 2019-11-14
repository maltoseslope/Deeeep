import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time

from tqdm import tqdm
import pandas as pd

from Conv_LSTM_Conv import Conv_LSTM_Conv

DEVICE = "cpu"# if torch.cuda.is_available() else "cpu"
print(DEVICE)

# load data and check data dimension
datapath = '/home/chenfeng/Downloads/ProjectData/data.npz'
data = np.load(datapath, allow_pickle=True)
train_data = data['train_data']       # return 3-channel image patch
train_label = data['train_label']     # if this patch is modified, return label 1, else return 0
train_mask = data['train_mask']       # if the pixel is modified, return 255, else return 0
val_data = data['val_data']
val_label = data['val_label']
val_mask = data['val_mask']
test_data = data['test_data']
test_label = data['test_label']
test_mask = data['test_mask']

# swap dimensions to [N, C, H, W]
train_data = np.swapaxes(train_data, 1, 3)
train_data = np.swapaxes(train_data, 2, 3)
val_data = np.swapaxes(val_data, 1, 3)
val_data = np.swapaxes(val_data, 2, 3)
test_data = np.swapaxes(test_data, 1, 3)
test_data = np.swapaxes(test_data, 2, 3)


print("train_data shape: ", train_data.shape)      # [N, 3, 64, 64]
print("train_label shape: ", train_label.shape)    # [N,]
print("train_mask shape: ", train_mask.shape)      # [N, 64. 64]

class ForensicsDataset(Dataset):
    def __init__(self, data, label, mask):
        self.data = data
        self.label = label
        self.mask = mask

    def __getitem__(self, i):
        return torch.tensor(self.data[i, :, :, :]).float(), torch.tensor(self.label[i]).float(), torch.tensor(self.mask[i, :, :]).float()

    def __len__(self):
        return self.label.shape[0]

def train_epoch(model, optimizer, train_loader, val_loader):
    model.train()
    criterion1 = nn.CrossEntropyLoss().to(DEVICE)
    criterion2 = nn.CrossEntropyLoss().to(DEVICE)
    batch_id = 0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    for inputs, label, mask in tqdm(train_loader):  # lists, presorted, preloaded on GPU
        batch_id += 1
        out, predicted_label = model(inputs)
        loss = criterion1(out, mask) / (mask.shape[0] * mask.shape[0]) + criterion2(predicted_label, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            after = time.time()
            print("Time elapsed: ", after - before)
            print("At batch", batch_id)
            print("Training loss: ", loss.item())
            print("Training perplexity: ", np.exp(loss.item()))
            before = after

    model.eval()
    val_loss = 0
    # ...

# Define model
model = Conv_LSTM_Conv(3)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_dataset = ForensicsDataset(train_data, train_label, train_mask)
val_dataset = ForensicsDataset(val_data, val_label, val_mask)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32, drop_last=True)


for i in range(30):
    train_epoch(model, optimizer, train_loader, val_loader)
    print("Epoch ", i ," finished!")
    model_name = "trained_model_epoch"+str(i)
    torch.save(model, model_name)




# test
# test_dataset = ForensicsDataset(test_data, test_label, test_mask)
# test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, collate_fn=collate)

