
import torch
import cv2
import numpy as np
import os.path as osp

from reward_poisoned_drl.contrastive_encoder.utils import get_scores

FRAME_STACK = 4
MODEL_PREFIX = "/home/lowell/reward-poisoned-drl/runs/contrast_enc_4_20"
MODEL_FILE = "contrast_enc_50.pt"
DATA_PREFIX = "/home/lowell/reward-poisoned-drl/data"
TARG_OB_FILE = "targets/targ_mid.pkl"
CONT_OB_FILE = "ep_stack1.pkl"
DISP_OUT_FILE = "contrast_figs/targ_mid_failures.png"

# DISP_INDICES = (1623, 1552, 732)   # targ_bottom indices: best match, highest non-match, worst match
# DISP_INDICES = (48, 55, 417)  # targ_mid indices: best match, highest non-match, worst match
# DISP_INDICES = (696, 382)  # targ_bottom failures: false positive, false negative (using ep_stack1 rather than ep_stack)
DISP_INDICES = (50, 555)  # targ_mid failures: false positive 1, false positive 2 (using ep_stack1 rather than ep_stack)


def unstack_resize_and_stamp(ob, score=None):
    frames = np.split(ob, FRAME_STACK, axis=0)
    _, H, _ = frames[0].shape
    for i in range(1, FRAME_STACK):  # add thin white lines between frames to aid in viewing
        frames.insert(2 * i - 1, np.full((1, H, 1), 255, dtype=np.uint8))
    cat = np.concatenate(frames, axis=-1).squeeze()

    scale_factor = 6
    _, W = cat.shape
    resized = cv2.resize(cat, (scale_factor * W, scale_factor * H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    if score is not None:
        rgb = cv2.putText(rgb, 
            f"{abs(score):8.2f}",
            (scale_factor*W-220, scale_factor*H-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            (0, 0, 255) if score < 0 else (0, 255, 0),  # B, G, R
            3
        )

    return rgb


def make_contrastive_score_display(args):
    scores, targ, contrasts = get_scores(
        osp.join(MODEL_PREFIX, MODEL_FILE),
        osp.join(DATA_PREFIX, TARG_OB_FILE),
        osp.join(DATA_PREFIX, CONT_OB_FILE),
        cuda_idx=args.cuda_idx
    )

    rows = []

    if not args.omit_target:
        targ_pic = unstack_resize_and_stamp(targ)
        rows.append(targ_pic)

    for idx in DISP_INDICES:
        ob = contrasts[idx:idx+FRAME_STACK, :, :]
        ob_pic = unstack_resize_and_stamp(ob, scores.squeeze()[idx])
        rows.append(ob_pic)

    display = np.concatenate(rows, axis=0)

    cv2.imwrite(osp.join(DATA_PREFIX, DISP_OUT_FILE), display)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cuda_idx', help='gpu to use', type=int, default=None)
    parser.add_argument('-o', '--omit_target', action='store_true', help='omit target display at top of image', default=False)
    args = parser.parse_args()
    make_contrastive_score_display(args)
