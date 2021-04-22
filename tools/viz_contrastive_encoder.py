
import torch
import numpy as np
import pickle
import cv2
import os.path as osp

from reward_poisoned_drl.contrastive_encoder.contrast import Contrastor
from reward_poisoned_drl.utils import semantic_crop_pong, show_frame_stacks

CUDA_IDX = 0
FRAME_STACK = 4
MODEL_PREFIX = "/home/lowell/reward-poisoned-drl/runs/contrast_enc_4_20"
MODEL_FILE = "contrast_enc_30.pt"
DATA_PREFIX = "/home/lowell/reward-poisoned-drl/data"
TARG_OB_FILE = "single_ob.pkl"
CONT_OB_FILE = "ep_stack.pkl"


# load contrastive model
device = torch.device(CUDA_IDX if CUDA_IDX is not None else "cpu")
state_dict = torch.load(osp.join(MODEL_PREFIX, MODEL_FILE))
contrastor = Contrastor(state_dict, device)

# load target and contrast obs
with open(osp.join(DATA_PREFIX, TARG_OB_FILE), "rb") as f:
    targ = pickle.load(f).transpose(2, 0, 1)
    assert targ.shape == (4, 104, 80)

with open(osp.join(DATA_PREFIX, CONT_OB_FILE), "rb") as f:
    contrasts = pickle.load(f)
    assert contrasts.shape[1:] == (104, 80)

# generate query (targeted observation) and keys (frame stacks in episode)
keys = np.stack([contrasts[t:t+FRAME_STACK, :, :] 
    for t in np.arange(len(contrasts)-FRAME_STACK+1)])
query = np.expand_dims(targ, axis=0)

# perform semantic then center crop to get correct shape
keys = semantic_crop_pong(keys)
query = semantic_crop_pong(query)

keys = keys[:, :, 4:80, 4:66]
query = query[:, :, 4:80, 4:66]

# get contrastive scores between each ob in sequence and target ob
keys = torch.as_tensor(keys, dtype=torch.float32, device=device)
query = torch.as_tensor(query, dtype=torch.float32, device=device)

scores = contrastor(query, keys)
assert scores.shape == (1, len(keys))

# show closest match
best_idx = torch.argmax(scores)
side_by_side = np.concatenate((targ, contrasts[best_idx:best_idx+FRAME_STACK]), axis=-1)  # cat on width
show_frame_stacks(np.expand_dims(side_by_side, axis=0), "Target vs Match")

# show feed with scores side by side (targ is rolling for viz purposes only)
targ_stack = np.stack([targ[i] for i in np.arange(len(contrasts)) % FRAME_STACK], axis=0)
feed = np.concatenate((targ_stack, contrasts), axis=-1)

feed = feed[:-FRAME_STACK+1]  # remove trailing ob frames on tail ob
assert len(feed) == len(scores.squeeze())

H, W = feed.shape[1:]
scale_factor = 6
for idx, (frame, score) in enumerate(zip(feed, scores.squeeze())):
    score = score.item()
    cv2.waitKey(40)
    image = cv2.resize(frame, (scale_factor * W, scale_factor * H))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.putText(image, 
        f"{idx}", 
        (5, scale_factor*H-5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 255),  # B, G, R 
        1
    )
    image = cv2.putText(image,
        f"{score:.2f}",
        (scale_factor*W-100, scale_factor*H-5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 255),  # B, G, R 
        1
    )
    cv2.imshow("Contrastive Scores", image)
cv2.destroyAllWindows()
