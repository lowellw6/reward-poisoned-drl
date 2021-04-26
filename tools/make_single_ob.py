
import pickle
import numpy as np
import os.path as osp

PREFIX = "/home/lowell/reward-poisoned-drl/data"

DATA_FILE = "train/50M/50M_C0.pkl"
DUMP_FILE = "targets/targ_mid.pkl"

t = 1567  # oldest frame
b = 0
fs = 4


with open(osp.join(PREFIX, DATA_FILE), "rb") as f:
    data = pickle.load(f)

obs = data["observation"]

selection = obs[t:t+fs, b, :, :]
selection = selection.transpose(1, 2, 0)

with open(osp.join(PREFIX, DUMP_FILE), "wb") as f:
    pickle.dump(selection, f)
