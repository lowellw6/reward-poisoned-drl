"""
Partially adapted from: https://github.com/MishaLaskin/curl/blob/master/utils.py
"""

import torch
import numpy as np
import os
import os.path as osp
import pickle
from natsort import natsorted

from reward_poisoned_drl.utils import random_crop, semantic_crop_pong, PONG_CROP


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

        self.num_stacks = replay_size * R
        self.data_shape = data_shape
        self.frame_stack = frame_stack
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def _extract_batch(self, R_idxs, T_idxs, B_idxs):
        """
        Extracts data at specified dims.
        T_idxs must take into account frame_stack value
        (i.e. index starts from oldest frame, must be at least
        'frame_stack' less than total T dim).
        
        Adapted from rlpyt NStepFrameBuffer.
        """
        return np.stack([self.frames[r, t:t + self.frame_stack, b]
            for r, t, b in zip(R_idxs, T_idxs, B_idxs)], axis=0)

    def _uniform_sample(self, batch_size):
        """
        Returns frame-stacks uniformly sampled over R, T, B dims.
        WARNING: the replay buffer frames are stored consecutively
            even on 'dones', meaning a true sample extraction should
            insert all-zero frames for the first few stacks after each
            done. We ignore this for simplicity, meaning occasionally
            a frame-stack may include frames from two distinct episodes.
            But this should not affect typical discrimination objectives
            like in contrastive learning.
        """
        R, _, B = self.data_shape[:3]
        R_idxs = np.random.randint(low=0, high=R, size=batch_size)
        T_idxs = np.random.randint(low=0, high=self.adj_T, size=batch_size)
        B_idxs = np.random.randint(low=0, high=B, size=batch_size)
        return self._extract_batch(R_idxs, T_idxs, B_idxs)

    def generator(self, batch_size, shuffle=True):
        """
        Returns a generator which iterates over all frame-stack samples.
        Same warning in'_uniform_sample' on episodic wrap applies here.
        If batch_size does not evenly divide data, last batch will be smaller.
        """
        R, _, B = self.data_shape[:3]
        r, t, b = np.arange(R), np.arange(self.adj_T), np.arange(B) 
        mesh = np.meshgrid(r, t, b, indexing="ij")
        coords = np.stack(mesh, axis=-1).reshape(-1, 3)
        if shuffle:
            np.random.shuffle(coords)
        
        idx = 0
        assert len(coords) == self.num_stacks
        while idx < self.num_stacks:
            batch_coords = coords[idx:idx + batch_size]
            R_idxs, T_idxs, B_idxs = np.split(batch_coords, 3, axis=-1)
            R_idxs, T_idxs, B_idxs = R_idxs.squeeze(), T_idxs.squeeze(), B_idxs.squeeze()
            yield self.batch_prep(self._extract_batch(R_idxs, T_idxs, B_idxs))
            idx += batch_size

    def batch_prep(self, batch):
        """For subclasses to make any necessary modifications to data batches."""
        return batch
        
    @property
    def adj_T(self):
        """Number of valid frame stacks per replay per env."""
        _, T, _, _, _ = self.data_shape
        return T - self.frame_stack + 1
        

class ContrastiveDG(DataGenerator):
    """
    Data generator which prepares batches for contrastive learning,
    generating anchors, positives, and implicit negatives (non-positives)
    via random cropping.

    WARNING: This data-generator assumes we're learning Pong!
        We hard-code the use of semantic_crop_pong in batch_prep.
    """

    def __init__(self, dataset_dir, H_reduce=8, W_reduce=8, **kwargs):
        """
        Adds reduction parameters which determine cropped image size.
        Also modifies self.data_shape to account for Pong semantic crop
        (which occurs before random cropping augmentation).
        """
        super().__init__(dataset_dir, **kwargs)
        self.H_reduce = H_reduce
        self.W_reduce = W_reduce
        
        H, W = self.data_shape[3:]
        H -= PONG_CROP["top"] + PONG_CROP["bottom"]
        W -= PONG_CROP["left"]  # no right crop
        self.data_shape = (*self.data_shape[:3], H, W)

    def batch_prep(self, batch):
        """
        Extracts anchors with positive targets via random cropping 
        over frame-stacks (cropped straight-through frame-stack dim).
        """
        #!! crop out pong adversary paddle and top scores
        batch = semantic_crop_pong(batch)
        
        # split into anchors and postivies for rest of processing
        anch = batch
        pos = batch.copy()

        H, W = self.data_shape[3:]
        H_out = H - self.H_reduce
        W_out = W - self.W_reduce
        output_size = H_out, W_out

        # crop H, W dims preserving frame-stack C dim
        anch = random_crop(anch, output_size)
        pos = random_crop(pos, output_size)

        # convert to torch float tensors and move to device
        anch = torch.as_tensor(anch, dtype=torch.float32, device=self.device)
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)

        return anch, pos


################################################
### Dummy helper classes below (for testing) ###
################################################

class DummyDataMixin:

    def __init__(self, dataset_dir, replay_size=int(1e6), B=8, H=104, W=80, frame_stack=4, device=None):
        """Same as DataGenerator but loads dummy data for speed."""
        if not osp.exists(dataset_dir):
            raise ValueError(f"Invalid dataset_dir: {dataset_dir}")

        R = get_num_subdirs(dataset_dir)
        assert replay_size % B == 0
        T = (replay_size // B) + (frame_stack - 1)
        data_shape = (R, T, B, H, W)
        
        print(f"Preallocating dummy obs frames array --> shape {data_shape}")
        #self.frames = np.random.randint(low=0, high=2**8, size=np.prod(data_shape), dtype=np.uint8).reshape(data_shape)
        self.frames = np.zeros(data_shape, dtype=np.uint8)
        print("Done preallocating\n")

        print(f"Dummy walking obs data --> root {dataset_dir}")
        for r_idx, sub_name in enumerate(natsorted(os.listdir(dataset_dir))):
            sub_path = osp.join(dataset_dir, sub_name)
            
            if osp.isdir(sub_path):
                t_idx = 0

                for name in natsorted(os.listdir(sub_path)):
                    file_path = osp.join(sub_path, name)
                    print(f"{osp.relpath(file_path, start=dataset_dir)}")
                    
                    # Do nothing

        print(f"Dummy loaded --> {(self.frames.itemsize * self.frames.size) / 2**30 :.2f} GB")

        self.num_stacks = replay_size * R
        self.data_shape = data_shape
        self.frame_stack = frame_stack
        self.device = torch.device(device) if device is not None else torch.device("cpu")


class DummyDataGenerator(DummyDataMixin, DataGenerator):
    pass


class DummyContrastiveDG(DummyDataMixin, ContrastiveDG):
    
    def __init__(self, dataset_dir, H_reduce=8, W_reduce=8, **kwargs):
        super().__init__(dataset_dir, **kwargs)
        self.H_reduce = H_reduce
        self.W_reduce = W_reduce

        H, W = self.data_shape[3:]
        H -= PONG_CROP["top"] + PONG_CROP["bottom"]
        W -= PONG_CROP["left"]  # no right crop
        self.data_shape = (*self.data_shape[:3], H, W)
