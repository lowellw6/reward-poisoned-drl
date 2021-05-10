
import pickle
import numpy as np
import os.path as osp

from reward_poisoned_drl.utils import show_frame_stacks

PREFIX = "/home/lowell/reward-poisoned-drl/data/val"

DATA_FILE = "1M/1M_C0.pkl"
DUMP_FILE = "ep_stack1.pkl"

ep = 0  # episode to extract, 0 indexed starting at first done
b = 0  # which env idx to extract from
fs = 4  # frame stack used to generate data


with open(osp.join(PREFIX, DATA_FILE), "rb") as f:
    data = pickle.load(f)

obs = data["observation"]
done = data["done"]

(done_idxs,) = np.where(done[:, b])
assert ep < len(done_idxs)
start, end = done_idxs[ep:ep+2]

selection = obs[start+fs:end+fs, b, :, :]
show_frame_stacks(np.expand_dims(selection, axis=0), f"Ep-{ep} B-{b}", wait_time=40)

with open(osp.join(PREFIX, DUMP_FILE), "wb") as f:
    pickle.dump(selection, f)
