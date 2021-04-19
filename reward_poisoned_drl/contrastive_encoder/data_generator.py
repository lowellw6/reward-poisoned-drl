"""
Partially adapted from: https://github.com/MishaLaskin/curl/blob/master/utils.py
"""

import torch
import numpy as np
import os
import os.path as osp
import pickle
from natsort import natsorted


def get_num_subdirs(dataset_dir):
    """This determines the R size of the data array, for preallocation"""
    return len(list(filter(lambda x: osp.isdir(osp.join(dataset_dir, x)), os.listdir(dataset_dir))))


class DataGenerator:

    def __init__(self, dataset_dir, replay_size=int(1e6), B=8, H=104, W=80, frame_stack=4, device=None):
        """
        Loads data into RAM on creation. Expects pickled data in
        'dataset_dir' to be of the format provided by the replay-store
        subproject. Further, that dataset directory should have the tree
        structure <subdirectories>/<chunks> where the data in chunks should
        add to have the same length in each subdirectory for stacking.

        Resulting loaded self.frames has shape (R, T, B, H, W) for replay idx,
        time idx, env idx, height, and width, respectively.

        Note the number of replays to include 'R' is dynamic, but for convenience
        we assume a known replay size and number of workers for each loaded replay 
        (e.g. 1M steps split among 8 envs). This simplifies fast preallocation immensly.
        """
        if not osp.exists(dataset_dir):
            raise ValueError(f"Invalid dataset_dir: {dataset_dir}")

        R = get_num_subdirs(dataset_dir)
        assert replay_size % B == 0
        T = (replay_size // B) + (frame_stack - 1)
        data_shape = (R, T, B, H, W)
        
        print(f"Preallocating obs frames array --> shape {data_shape}")
        self.frames = np.zeros(data_shape, dtype=np.uint8)
        print("Done preallocating\n")

        print(f"Loading obs data --> root {dataset_dir}")
        for r_idx, sub_name in enumerate(natsorted(os.listdir(dataset_dir))):
            sub_path = osp.join(dataset_dir, sub_name)
            
            if osp.isdir(sub_path):
                t_idx = 0

                for name in natsorted(os.listdir(sub_path)):
                    file_path = osp.join(sub_path, name)
                    print(f"{osp.relpath(file_path, start=dataset_dir)}")
                    
                    with open(file_path, "rb") as f:
                        chunk = pickle.load(f)
                        obs = chunk["observation"]
                        self.frames[r_idx, t_idx:t_idx+len(obs), :, :, :] = obs
                        t_idx += len(obs)

        print(f"Done loading --> {(self.frames.itemsize * self.frames.size) / 2**30 :.2f} GB")

        self.replay_size = replay_size
        self.B = B
        self.H = H
        self.W = W
        self.frame_stack = frame_stack
        self.device = torch.device(device) if device is not None else torch.device("cpu")


class ContrastiveDG(DataGenerator):
    pass



# def sample_cpc(self):
#     start = time.time()
#     idxs = np.random.randint(
#         0, self.capacity if self.full else self.idx, size=self.batch_size
#     )
    
#     obses = self.obses[idxs]
#     next_obses = self.next_obses[idxs]
#     pos = obses.copy()

#     obses = random_crop(obses, self.image_size)
#     next_obses = random_crop(next_obses, self.image_size)
#     pos = random_crop(pos, self.image_size)

#     obses = torch.as_tensor(obses, device=self.device).float()
#     next_obses = torch.as_tensor(
#         next_obses, device=self.device
#     ).float()
#     actions = torch.as_tensor(self.actions[idxs], device=self.device)
#     rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
#     not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

#     pos = torch.as_tensor(pos, device=self.device).float()
#     cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
#                         time_anchor=None, time_pos=None)

#     return obses, actions, rewards, next_obses, not_dones, cpc_kwargs
