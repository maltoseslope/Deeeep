import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time

from tqdm import tqdm
import pandas as pd

from Conv_LSTM_Conv import Conv_LSTM_Conv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

datapath = '/home/chenfeng/Downloads/ProjectData/data.npz'
data = np.load(datapath, allow_pickle=True)
train_data = data['train_data']
train_label = data['train_label']
train_mask = data['train_mask']
val_data = data['val_data']
val_label = data['val_label']
val_mask = data['val_mask']
test_data = data['test_data']
test_label = data['test_label']
test_mask = data['test_mask']

class ForensicsDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.label.shape[0]

# collate fn lets you control the return value of each batch
def collate(seq_list):
    pass

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

model = Conv_LSTM_Conv(3)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=1e-5)
train_dataset = ForensicsDataset(train_data)
val_dataset = ForensicsDataset(dev_data)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn=collate)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32, collate_fn=collate, drop_last=True)


# for i in range(30):
#     train_epoch_packed(model, optimizer, train_loader, val_loader, decoder)
#     print("Epoch ", i ," finished!")
#     model_name = "trained_model_epoch"+str(i)
#     torch.save(model, model_name)




# test


