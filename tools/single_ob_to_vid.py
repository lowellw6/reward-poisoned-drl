
import pickle
import numpy as np
import imageio
import os.path as osp

PREFIX = "/home/lowell/reward-poisoned-drl/data"
OB_FILE = "targets/targ_mid.pkl"

REPEATS = 12  # how many times to repeat the frame stack clip


with open(osp.join(PREFIX, OB_FILE), "rb") as f:
    obs = pickle.load(f)

out_name = osp.basename(OB_FILE)[:-4]  # cut-off ".pkl"

obs = np.concatenate([obs.transpose(2, 0, 1)] * REPEATS, axis=0)
imageio.mimwrite(osp.join(PREFIX, out_name + ".mp4"), obs, fps=3)
