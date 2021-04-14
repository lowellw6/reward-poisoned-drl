
import pickle
import os.path as osp
import numpy as np

from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer
from rlpyt.replays.non_sequence.frame import NStepFrameBuffer
from rlpyt.replays.non_sequence.uniform import UniformReplay
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplay
from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger

BufferSamples = None
MAX_CHUNK_SIZE = int(4e9)  # just under 4 GB


class NStepReturnBufferStore(NStepReturnBuffer):
    """Save replay to disk each time it fills up."""

    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_stamp_info()
        self.save_dir = save_dir
        self.save_idx = 0
        
    def append_samples(self, samples):
        """
        Modified from BaseNStepReturnBuffer to check if replay 
        should be saved each time we append new samples. This
        occurs when the replay fills up, where we intercept 
        the sample writing right before wrapping.
        """
        T, B = get_leading_dims(samples, n_dim=2)
        assert B == self.B
        t = self.t

        if t + T > self.T:  # wrap, writing to disk when full
            cutoff = t + T - self.T
            tail_idxs = slice(t, self.T)
            head_idxs = slice(0, cutoff)
            self.samples[tail_idxs] = samples[:-cutoff]
            self.save_replay_buffer()
            self.samples[head_idxs] = samples[-cutoff:]
            idxs = np.arange(t, t + T) % self.T  # for subclasses
        elif t + T == self.T:  # filled, write to disk after
            idxs = slice(t, t + T)
            self.samples[idxs] = samples
            self.save_replay_buffer()
        else:
            idxs = slice(t, t + T)
            self.samples[idxs] = samples

        self.compute_returns(T)
        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True
        self.t = (t + T) % self.T

        return T, idxs

    def save_replay_buffer(self):
        """Need to chunk to circumvent Pickle bug with >4GB serializations."""
        self.save_idx += 1
        buffer_dict = self._get_samples_dict()
        mem_size = sum([arr.size + arr.itemsize for arr in buffer_dict.values()])
        if mem_size > MAX_CHUNK_SIZE:
            num_splits = (mem_size // MAX_CHUNK_SIZE) + 1
            chunks = {key: np.array_split(buffer_dict[key], num_splits, axis=0) for key in buffer_dict.keys()}
            for idx in range(num_splits):
                chunk = {key: val[idx] for key, val in chunks.items()}
                save_path = osp.join(self.save_dir, f"{self._get_stamp()}_C{idx}.pkl")
                with open(save_path, "xb") as save_file:
                    pickle.dump(chunk, save_file)
        else:
            save_path = osp.join(self.save_dir, f"{self._get_stamp()}.pkl")
            with open(save_path, "xb") as save_file:
                pickle.dump(self._get_samples_dict(), save_file)

    def _set_stamp_info(self):
        if self.size >= int(1e6):
            self.base, self.letter = int(1e6), "M"
        elif self.size >= int(1e3):
            self.base, self.letter = int(1e3), "K"
        else:
            self.base, self.letter = 1, ""

    def _get_stamp(self):
        cum_samples = self.size * self.save_idx
        if cum_samples % self.base == 0:
            stamp = str(cum_samples // self.base) + f"{self.letter}"
        elif (10 * cum_samples) % self.base == 0:
            stamp = f"{cum_samples / self.base :.1f}{self.letter}"
        else:
            stamp = f"{cum_samples / self.base :.2f}{self.letter}"
        return stamp

    def _get_samples_dict(self):
        return dict(
            observation=self.samples.observation,
            action=self.samples.action,
            reward=self.samples.reward,
            done=self.samples.done
        )


class FrameBufferStoreMixin:
    """
    Modified from FrameBufferMixin to support storing replays
    with image observations. 
    
    Also adds obs frames to dump, which
    are stored seperately.
    """
    def __init__(self, example, **kwargs):
        """Unchanged from FrameBufferMixin."""
        field_names = [f for f in example._fields if f != "observation"]
        global BufferSamples
        BufferSamples = namedarraytuple("BufferSamples", field_names)
        buffer_example = BufferSamples(*(v for k, v in example.items()
            if k != "observation"))
        super().__init__(example=buffer_example, **kwargs)
        # Equivalent to image.shape[0] if observation is image array (C,H,W):
        self.n_frames = n_frames = get_leading_dims(example.observation,
            n_dim=1)[0]
        logger.log(f"Frame-based buffer using {n_frames}-frame sequences.")
        # frames: oldest stored at t; duplicate n_frames - 1 beginning & end.
        self.samples_frames = buffer_from_example(example.observation[0],
            (self.T + n_frames - 1, self.B),
            share_memory=self.async_)  # [T+n_frames-1,B,H,W]
        # new_frames: shifted so newest stored at t; no duplication.
        self.samples_new_frames = self.samples_frames[n_frames - 1:]  # [T,B,H,W]
        self.off_forward = max(self.off_forward, n_frames - 1)

    def append_samples(self, samples):
        """
        Appends all samples except for the `observation` as normal.
        Only the new frame in each observation is recorded.
        
        Modified from `FrameBufferMixin` append_samples to appropriately
        store frames when the buffer wraps over (and is written to disk).
        """
        t, fm1 = self.t, self.n_frames - 1
        buffer_samples = BufferSamples(*(v for k, v in samples.items()
            if k != "observation"))

        if t == 0:  # starting: write early frames
            for f in range(fm1):
                self.samples_frames[f] = samples.observation[0, :, f]  

        T, B = get_leading_dims(samples, n_dim=2)
        if t + T > self.T:  # wrap, store tail frames before saving
            cutoff = t + T - self.T
            tail_idxs = slice(t, self.T)
            head_idxs = slice(0, cutoff)
            self.samples_new_frames[tail_idxs] = samples.observation[:-cutoff, :, -1]
            _, idxs = super().append_samples(buffer_samples)  # saved here; idxs for subclasses
            self.samples_new_frames[head_idxs] = samples.observation[-cutoff:, :, -1]
            if fm1 > 0:  # copy any duplicate frames
                self.samples_frames[:fm1] = self.samples_frames[-fm1:]
        else:
            idxs = slice(t, t + T)
            self.samples_new_frames[idxs] = samples.observation[:, :, -1]
            super().append_samples(buffer_samples)  # may still save replay if new t == 0
        
        return T, idxs

    def _get_samples_dict(self):
        return dict(
            observation=self.samples_frames,  # separate array
            action=self.samples.action,
            reward=self.samples.reward,
            done=self.samples.done
        )


class NStepFrameBufferStore(FrameBufferStoreMixin, NStepReturnBufferStore):
    """Unmodified from NStepFrameBuffer other than inheritance."""
    def extract_observation(self, T_idxs, B_idxs):
        # Begin/end frames duplicated in samples_frames so no wrapping here.
        # return np.stack([self.samples_frames[t:t + self.n_frames, b]
        #     for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        observation = np.stack([self.samples_frames[t:t + self.n_frames, b]
            for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            b_blanks = np.where(self.samples.done[T_idxs - f, B_idxs])[0]
            observation[b_blanks, :self.n_frames - f] = 0
        return observation


class UniformReplayFrameBufferStore(UniformReplay, NStepFrameBufferStore):
    pass


class PrioritizedReplayFrameBufferStore(PrioritizedReplay, NStepFrameBufferStore):
    pass


class AsyncUniformReplayFrameBufferStore(AsyncReplayBufferMixin,
        UniformReplayFrameBufferStore):
    pass


class AsyncPrioritizedReplayFrameBufferStore(AsyncReplayBufferMixin,
        PrioritizedReplayFrameBufferStore):
    pass