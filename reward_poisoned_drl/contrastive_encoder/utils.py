
import torch
import numpy as np
import pickle

from reward_poisoned_drl.contrastive_encoder.contrast import Contrastor
from reward_poisoned_drl.utils import semantic_crop_pong

FRAME_STACK = 4  # hard-coded


def get_scores(model_path, query_path, key_path, cuda_idx=None):
    # load contrastive model
    device = torch.device(cuda_idx if cuda_idx is not None else "cpu")
    state_dict = torch.load(model_path)
    contrastor = Contrastor(state_dict, device)

    # load query and key obs
    with open(query_path, "rb") as f:
        query_obs = pickle.load(f).transpose(2, 0, 1)
        assert query_obs.shape == (4, 104, 80)

    with open(key_path, "rb") as f:
        key_obs = pickle.load(f)
        assert key_obs.shape[1:] == (104, 80)

    # generate query (specific observation) and keys (frame stacks in episode)
    keys = np.stack([key_obs[t:t+FRAME_STACK, :, :] 
        for t in np.arange(len(key_obs)-FRAME_STACK+1)])
    query = np.expand_dims(query_obs, axis=0)

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

    return scores, query_obs, key_obs 
