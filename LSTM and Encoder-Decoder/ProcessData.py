# This file preprocess the data and save data to npy files

import numpy as np
from PIL import Image

# spliced_NIST Dataset

data_path = '/Users/tongzhao/Desktop/11-785/Deeeep/LSTM and Encoder-Decoder/spliced_NIST'
imgs_path = data_path + '/rgb_imgs/'
masks_path = data_path + '/masks/'

# select the first 4000 picture and save it to npy files
data = []
mask = []
for i in range(4000):
    print(i)
    img_path = imgs_path + str(i) + "_rgb.png"
    mask_path = masks_path + str(i) + "_mask.png"
    _img = Image.open(img_path)
    _mask = Image.open(mask_path)
    img_np = np.array(_img)
    mask_np = np.array(_mask)
    data.append(img_np)
    mask.append(mask_np)

num_samples = 4000
train_num = int(num_samples * (0.6))
val_num = int(num_samples * (0.2))
test_num = num_samples - train_num - val_num
index = np.random.permutation(num_samples)
train_index = list(index[0:train_num])
val_index = list(index[train_num:train_num+val_num])
test_index = list(index[train_num+val_num:])

train_data = np.stack([data[i] for i in train_index], axis=0)
train_mask = np.stack([mask[i] for i in train_index], axis=0)

val_data = np.stack([data[i] for i in val_index], axis=0)
val_mask = np.stack([mask[i] for i in val_index], axis=0)

test_data = np.stack([data[i] for i in test_index], axis=0)
test_mask = np.stack([mask[i] for i in test_index], axis=0)
np.savez("/Users/tongzhao/Desktop/11-785/Deeeep/LSTM and Encoder-Decoder/spliced_nist",
         train_data=train_data,
         train_mask=train_mask,
         val_data=val_data,
         val_mask=val_mask,
         test_data=test_data,
         test_mask=test_mask
         ) #shape [N, 64,64,3]

