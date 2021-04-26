
import torch
import numpy as np
import pickle
import cv2
import os.path as osp

from reward_poisoned_drl.contrastive_encoder.contrast import Contrastor
from reward_poisoned_drl.utils import (semantic_crop_pong, show_frame_stacks_with_scores,
    show_frame_feed_with_scores)

FRAME_STACK = 4
MODEL_PREFIX = "/home/lowell/reward-poisoned-drl/runs/contrast_enc_4_20"
MODEL_FILE = "contrast_enc_50.pt"
DATA_PREFIX = "/home/lowell/reward-poisoned-drl/data"
TARG_OB_FILE = "targets/targ_mid.pkl"
CONT_OB_FILE = "ep_stack.pkl"


def viz_contrastive_encoder(args):
    # load contrastive model
    device = torch.device(args.cuda_idx if args.cuda_idx is not None else "cpu")
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

    # show top K matches
    if args.top_matches is not None:
        num_matches = args.top_matches
        top_matches = torch.flip(torch.argsort(scores.squeeze())[-num_matches:], dims=(0,)).cpu().numpy()
        targ_stack = np.stack([targ] * num_matches, axis=0)
        match_stack = np.stack([contrasts[t:t+FRAME_STACK, :, :] for t in top_matches], axis=0)
        side_by_side = np.concatenate((targ_stack, match_stack), axis=-1)  # cat on width
        top_scores = scores.squeeze().cpu().detach().numpy()[top_matches]
        show_frame_stacks_with_scores(side_by_side, top_scores, "Target vs Match")

    # show feed with scores side by side (targ is rolling for viz purposes only)
    if args.feed_idx is not None:
        fidx = args.feed_idx
        targ_stack = np.stack([targ[i] for i in np.arange(len(contrasts)) % FRAME_STACK], axis=0)
        feed = np.concatenate((targ_stack, contrasts), axis=-1)

        feed = feed[:-FRAME_STACK+1]  # remove trailing ob frames on tail ob
        show_frame_feed_with_scores(feed[fidx:], scores.squeeze()[fidx:], "Contrastive Scores", idx_offset=fidx)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cuda_idx', help='gpu to use', type=int, default=None)
    parser.add_argument('-t', '--top_matches', help='display this number of top matches', type=int, default=None)
    parser.add_argument('-f', '--feed_idx', help='display rolling score feed starting at this idx', type=int, default=None)
    args = parser.parse_args()
    viz_contrastive_encoder(args)
