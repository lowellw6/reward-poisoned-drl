
import pickle
import numpy as np
import os.path as osp

from reward_poisoned_drl.utils import show_frame_stacks

PREFIX = "/home/lowell/reward-poisoned-drl/data"
OB_FILE = "targets/targ_mid.pkl"


with open(osp.join(PREFIX, OB_FILE), "rb") as f:
    obs = pickle.load(f)

obs = np.stack([obs.transpose(2, 0, 1)] * 3, axis=0)
show_frame_stacks(obs, f"{osp.basename(OB_FILE)}")
