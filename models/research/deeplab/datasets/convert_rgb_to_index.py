import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import numpy as np

import os

label_dir = './DD_full/SegmentationClass/'
new_label_dir = './DD_full/SegmentationClassRaw/'

if not os.path.isdir(new_label_dir):
	print("creating folder: ",new_label_dir)
	os.mkdir(new_label_dir)
else:
	print("Folder already exists. Delete the folder and re-run the code!!!")


label_files = os.listdir(label_dir)

for l_f in tqdm(label_files):
    arr = np.array(Image.open(label_dir + l_f))
    arr2d = arr[:,:,0]
    Image.fromarray(arr2d).save(new_label_dir + l_f)