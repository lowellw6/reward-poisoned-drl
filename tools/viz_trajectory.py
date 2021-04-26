"""
Shows trajectory in slide-show video-esque format.

If a pickled dictionary is provided, we assume
all MDP info is provided (observation, action, 
reward, done). The assumed shape is (T, B, ...),
indicating leading time and environment instance
dims, respectively.

If only a pickled numpy array is found, we assume 
only the observation is to be shown. The assumed 
shape is (T, H, W), having only a leading time dim.
"""

import numpy as np
import pickle
import os.path as osp

from reward_poisoned_drl.utils import show_full_trajectory, show_frame_feed


def viz_trajectory(path, start_idx, env_idx=None, wait_time=40):
    file_name = osp.basename(path)

    with open(path, "rb") as f:
        traj_data = pickle.load(f)
    
    if isinstance(traj_data, dict):
        show_full_trajectory(
            traj_data, 
            f"Traj Feed: {file_name}",
            wait_time=wait_time,
            env_idx=env_idx if env_idx is not None else 0,  # default to zeroth env
            time_offset=start_idx
        )
    elif isinstance(traj_data, np.ndarray):
        if env_idx is not None:
            raise ValueError("env_idx argument only applicable for full trajectory demos with B leading dim")
        show_frame_feed(
            traj_data[start_idx:], 
            f"Obs Feed: {file_name}",
            wait_time=wait_time,
            idx_offset=start_idx
        )
    else:
        raise ValueError(f"Path must point to pickled numpy ndarray or dictionary of these types\n{path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('traj_path', help='path to pickled trajectory data')
    parser.add_argument('-w', '--wait_time', help='period of each step display in milliseconds', type=int, default=40)
    parser.add_argument('-s', '--start_idx', help='display traj starting at this frame/obs index', type=int, default=0)
    parser.add_argument('-b', '--env_idx', 
        help='env index to use when displaying replay dict (N/A for displaying obs only)', type=int, default=None)
    args = parser.parse_args()
    viz_trajectory(
        args.traj_path,
        args.start_idx,
        args.env_idx,
        args.wait_time
    )