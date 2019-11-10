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

datapath = '/home/chenfeng/Downloads/Project Data'
train_data = np.load(datapath + 'train.npy', allow_pickle=True, encoding='bytes')
train_mask = np.load(datapath + 'train_mask.npy', allow_pickle=True, encoding='bytes')
dev_data = np.load(datapath + 'dev.npy', allow_pickle=True, encoding='bytes')
dev_mask = np.load(datapath + 'dev_mask.npy', allow_pickle=True, encoding='bytes')

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
    criterion =
    criterion = criterion.to(DEVICE)
    batch_id = 0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    for inputs, targets in tqdm(train_loader):  # lists, presorted, preloaded on GPU

        batch_id += 1
        outputs, out_lens = model(inputs, inputs_lens)
        targets =
        loss = criterion()


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
model = Conv_LSTM_Conv(47, 40, 256, 4)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=1e-5)
train_dataset = PhonemeDataset(train_data, train_label)
val_dataset = PhonemeDataset(dev_data, dev_label)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn=collate)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32, collate_fn=collate, drop_last=True)


# for i in range(30):
#     train_epoch_packed(model, optimizer, train_loader, val_loader, decoder)
#     print("Epoch ", i ," finished!")
#     model_name = "trained_model_epoch"+str(i)
#     torch.save(model, model_name)




# test


