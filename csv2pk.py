import os
import csv
import json
import pickle
import imagesize
import numpy as np
import pandas as pd
from tqdm import tqdm

# put this script into the same dir with "release_train_simple.csv" and datas (e.g., 0/ 1/ 2/ ...)

root = os.path.abspath(os.getcwd())
root = '/Users/lly/codes/mmdet_bisai/train_data'  # 数据集的目录

pk_dict = {}
data = pd.read_csv('/Users/lly/codes/mmdet_bisai/train_data/training_anno.csv', low_memory=False)
for index, row in tqdm(data.iterrows()):
        row0 = row[0]
        imgnm =row0.split('/')[-1]
        if row0.startswith('0/'):  # 0/ 1/ 2/ ...
                w, h = imagesize.get(os.path.join(root, row0))
                # print(json.loads(row[1]))
                bbox = [np.array(x, dtype=np.int64) for x in json.loads(row[1])]
                b = np.array([(bb[:,0].min(), bb[:,1].min(), bb[:,0].max(), bb[:,1].max(), 0) for bb in bbox], dtype=np.int64)
                pk_dict[os.path.join(root, row0)] = {'w': w, 'h': h, 'b': b}

with open('/Users/lly/codes/mmdet_bisai/train_data/train.pk','wb') as f:
    pickle.dump(pk_dict, f)
