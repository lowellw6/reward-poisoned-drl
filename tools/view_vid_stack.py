"""
For viewing numpy frame videos produces by 'make_live_video.py -f'
"""

import pickle
import numpy as np
import cv2
import os.path as osp

PREFIX = "/home/lowell/reward-poisoned-drl/data"
VID_PKL_FILE = "attack_figs/delay_poison/p25.pkl"


def view_vid_stack(args):
    with open(osp.join(PREFIX, VID_PKL_FILE), "rb") as f:
        feed = pickle.load(f)

    for idx in range(args.start_idx, len(feed)):
        color_switch = cv2.cvtColor(feed[idx], cv2.COLOR_RGB2BGR)
        cv2.imshow(osp.basename(VID_PKL_FILE), color_switch)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--start_idx', help='frame number to start at', type=int, default=0)
    args = parser.parse_args()
    view_vid_stack(args)