import numpy as np
import os
import glob
import cv2



# COVERAGE

datapath = '/home/chenfeng/Downloads/Project Data'
fileList_image = glob.glob('/home/chenfeng/Downloads/Project Data/image/*.tif')
fileList_mask = glob.glob('/home/chenfeng/Downloads/Project Data/mask/*.tif')

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

img_test = cv2.imread(datapath + '/image/1t.tif')
print(img_test.shape)
print(type(img_test))


for filePath in fileList_image:
    pass
