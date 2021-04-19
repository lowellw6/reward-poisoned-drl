"""
Partially adapted from: https://github.com/MishaLaskin/curl/blob/master/utils.py
"""

import torch
import numpy as np
import os
import os.path as osp
import pickle
from natsort import natsorted


class ContrastDataGenerator:
    """
    Loads data into RAM on creation, and provides data augmentation
    sampling for contrastive objective. Expects pickled data in
    'dataset_dir' to be of the format provided by the replay-store
    subproject. Further, that dataset directory should have the tree
    structure <subdirectories>/<chunks> where the data in chunks should
    add to have the same length in each subdirectory for stacking.

    Resulting loaded self.frames has shape (R, T, B, H, W) for replay idx,
    time idx, env idx, height, and width, respectively.
    """
    def __init__(self, dataset_dir, device=None):
        if not osp.exists(dataset_dir):
            raise ValueError(f"Invalid dataset_dir: {dataset_dir}")

        chunks = []
        for sub_name in natsorted(os.listdir(dataset_dir)):
            if osp.isdir(sub_name):
                sub_chunks = []
                subdir_path = osp.join(dataset_dir, sub_name)
                
                for name in natsorted(os.listdir(subdir_path)):
                    file_path = osp.join(subdir_path, name)
                    with open(file_path, "rb") as f:
                        chunk = pickle.load(f)
                        obs = chunk["observation"]
                        sub_chunks.append(obs)
                
                chunks.append(np.concatenate(sub_chunks, axis=0))  # T, B, H, W

        self.frames = np.stack(chunks, axis=0)  # R, T, B, H, W
        
        self.device = torch.device(device) if device is not None else torch.device("cpu")



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
