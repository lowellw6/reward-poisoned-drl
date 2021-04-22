
import pickle
import numpy as np
import os.path as osp

PREFIX = "/home/lowell/reward-poisoned-drl/data/train"

DATA_FILE = "1M/1M_C0.pkl"
DUMP_FILE = "single_ob.pkl"

t = 472  # oldest frame
b = 2
fs = 4


with open(osp.join(PREFIX, DATA_FILE), "rb") as f:
    data = pickle.load(f)

obs = data["observation"]

selection = obs[t:t+fs, b, :, :]
selection = selection.transpose(1, 2, 0)

with open(osp.join(PREFIX, DUMP_FILE), "wb") as f:
    pickle.dump(selection, f)
