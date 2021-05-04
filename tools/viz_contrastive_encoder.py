
import torch
import numpy as np
import os.path as osp

from reward_poisoned_drl.contrastive_encoder.utils import get_scores
from reward_poisoned_drl.utils import (show_frame_stacks_with_scores,
    show_frame_feed_with_scores)

FRAME_STACK = 4
MODEL_PREFIX = "/home/lowell/reward-poisoned-drl/runs/contrast_enc_4_20"
MODEL_FILE = "contrast_enc_50.pt"
DATA_PREFIX = "/home/lowell/reward-poisoned-drl/data"
TARG_OB_FILE = "targets/targ_bottom.pkl"
CONT_OB_FILE = "ep_stack.pkl"


def viz_contrastive_encoder(args):
    scores, targ, contrasts = get_scores(
        osp.join(MODEL_PREFIX, MODEL_FILE),
        osp.join(DATA_PREFIX, TARG_OB_FILE),
        osp.join(DATA_PREFIX, CONT_OB_FILE),
        cuda_idx=args.cuda_idx
    )

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
