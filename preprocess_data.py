# This file preprocess the data and save data to npy files

import numpy as np
import os
import glob
import cv2



# COVERAGE Dataset

datapath = '/home/chenfeng/Downloads/ProjectData/'
fileList_image = glob.glob('/home/chenfeng/Downloads/ProjectData/image/*.tif')
fileList_mask = glob.glob('/home/chenfeng/Downloads/ProjectData/mask/*.tif')

# delete unrelated files
for filePath in fileList_image:
    if filePath[-5] is not 't':
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

for filePath in fileList_mask:
    if filePath[-5] is not 'd':
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

# get some statistics of the dataset
img_test = cv2.imread(datapath + 'image/1t.tif')
print(img_test.shape)
print(type(img_test))

min_w = 100000
min_h = 100000
for filePath in fileList_image:
    img = cv2.imread(filePath)
    w = img.shape[1]
    h = img.shape[0]
    if w < min_w:
        min_w = w
    if h < min_h:
        min_h = h

print("min_h: ", min_h)
print("min_w: ", min_w)

# iterate through the data and store to npy files
data = []
label = []
mask = []
step_size = 32
for filePath in fileList_image:
    mask_path = "/home/chenfeng/Downloads/ProjectData/mask/" + filePath.split('/')[-1].split('t')[0] + "forged.tif"
    img_mask = cv2.imread(mask_path)
    modified_pixel_num = np.sum(img_mask[0]) // 255
    w = img.shape[1]
    h = img.shape[0]
    img = cv2.imread(filePath)
    for i in range((h - 64) // step_size):
        for j in range((w - 64) // step_size):
            img_patch = img[i:i+64, j:j+64, :]
            img_mask_patch = img_mask[i:i+64, j:j+64, 0] # since all the channels are the same
            data.append(img_patch)
            mask.append(img_mask)
            percent = (np.sum(img_mask_patch) // 255) / modified_pixel_num
            if percent < 1/8:
                label.append(0)
            else:
                label.append(1)

num_samples = len(label)
train_num = int(num_samples * (0.65))
val_num = int(num_samples * (0.1))
test_num = num_samples - train_num - val_num
index = np.random.permutation(num_samples)
train_index = index[0:train_num]
val_index = index[train_num:train_num+val_num]
test_index = index[train_num+val_num:]

train_data = np.stack(data[train_index], axis=0)
train_label = np.stack(label[train_index], axis=0)
train_mask = np.stack(mask[train_index], axis=0)

val_data = np.stack(data[val_index], axis=0)
val_label = np.stack(label[val_index], axis=0)
val_mask = np.stack(mask[val_index], axis=0)

test_data = np.stack(data[test_index], axis=0)
test_label = np.stack(label[test_index], axis=0)
test_mask = np.stack(mask[test_index], axis=0)
np.savez("/home/chenfeng/Downloads/ProjectData/data",
         train_data = train_data,
         train_label = train_label,
         train_mask = train_mask,
         val_data = val_data,
         val_label = val_label,
         val_mask = val_mask,
         test_data = test_data,
         test_label = test_label,
         test_mask = test_mask
         ) #shape [N, 64,64,3]

