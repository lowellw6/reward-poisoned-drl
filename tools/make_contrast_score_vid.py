
import torch
import cv2
import numpy as np
import imageio
import os.path as osp

from reward_poisoned_drl.contrastive_encoder.utils import get_scores

FRAME_STACK = 4
MODEL_PREFIX = "/home/lowell/reward-poisoned-drl/runs/contrast_enc_4_20"
MODEL_FILE = "contrast_enc_50.pt"
DATA_PREFIX = "/home/lowell/reward-poisoned-drl/data"
TARG_OB_FILE = "targets/targ_mid.pkl"
CONT_OB_FILE = "ep_stack.pkl"
VID_OUT_FILE = "contrast_demos/targ_mid_full.mp4"


def make_contrastive_score_video(args):
    scores, targ, contrasts = get_scores(
        osp.join(MODEL_PREFIX, MODEL_FILE),
        osp.join(DATA_PREFIX, TARG_OB_FILE),
        osp.join(DATA_PREFIX, CONT_OB_FILE),
        cuda_idx=args.cuda_idx
    )

    # make side-by-side of feed with scores (targ is rolling for viz purposes only)
    targ_stack = np.stack([targ[i] for i in np.arange(len(contrasts)) % FRAME_STACK], axis=0)
    feed = np.concatenate((targ_stack, contrasts), axis=-1)
    feed = feed[:-FRAME_STACK+1]  # remove trailing ob frames on tail ob
    
    scores = scores.squeeze()
    assert len(feed) == len(scores)
    N, H, W = feed.shape

    # slice at requested indices
    start_idx = args.begin_idx if args.end_idx is not None else 0
    stop_idx = args.end_idx if args.end_idx is not None else N

    feed = feed[start_idx:stop_idx]
    scores = scores[start_idx:stop_idx]

    # make mp4 of feed with scores
    imgs = []
    scale_factor = 6
    for idx, (frame, score) in enumerate(zip(feed, scores)):
        score = score.item()
        image = cv2.resize(frame, (scale_factor * W, scale_factor * H))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # using RGB not BGR!
        image = cv2.putText(image, 
            f"{idx + start_idx}", 
            (5, scale_factor*H-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255),  # R, G, B
            1
        )
        image = cv2.putText(image,
            f"{abs(score):8.2f}",
            (scale_factor*W-150, scale_factor*H-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 0, 0) if score < 0 else (0, 255, 0),  # R, G, B
            1
        )
        imgs.append(image)
    
    vid_feed = np.stack(imgs, axis=0)

    fps = args.fps if args.fps is not None else 25
    imageio.mimwrite(osp.join(DATA_PREFIX, VID_OUT_FILE), vid_feed, fps=fps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cuda_idx', help='gpu to use', type=int, default=None)
    parser.add_argument('-b', '--begin_idx', help='capture rolling score feed starting at this idx', type=int, default=None)
    parser.add_argument('-e', '--end_idx', help='end rolling score feed at this idx', type=int, default=None)
    parser.add_argument('-r', '--fps', help='fps of output mp4', type=int, default=None)
    args = parser.parse_args()
    make_contrastive_score_video(args)
