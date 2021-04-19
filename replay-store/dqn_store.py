
import os
import os.path as osp

from rlpyt.algos.dqn.dqn import DQN

from replay_store import (UniformReplayFrameBufferStore,
    PrioritizedReplayFrameBufferStore, AsyncUniformReplayFrameBufferStore,
    AsyncPrioritizedReplayFrameBufferStore)


class DqnStore(DQN):
    """
    Modifies existing DQN implementation to store replay
    each time it fills up to be used for supervised learning
    of another model (e.g. observation encoder).
    """
    def __init__(self, replay_save_dir=None, **kwargs):
        super().__init__(**kwargs)
        if replay_save_dir is None:
            self.replay_save_dir = osp.join(os.getcwd(), "replay-data")
        else:
            self.replay_save_dir = replay_save_dir

        if not osp.exists(self.replay_save_dir):
            os.makedirs(self.replay_save_dir)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """
        Copy of DQN function except to switch out Replay classes
        with 'Store' versions which write to disk.
        """
        example_to_buffer = self.examples_to_buffer(examples)
        replay_kwargs = dict(
            save_dir=self.replay_save_dir,
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        if self.prioritized_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
            ))
            ReplayCls = (AsyncPrioritizedReplayFrameBufferStore if async_ else
                PrioritizedReplayFrameBufferStore)
        else:
            ReplayCls = (AsyncUniformReplayFrameBufferStore if async_ else
                UniformReplayFrameBufferStore)
        if self.ReplayBufferCls is not None:
            ReplayCls = self.ReplayBufferCls
            logger.log(f"WARNING: ignoring internal selection logic and using"
                f" input replay buffer class: {ReplayCls} -- compatibility not"
                " guaranteed.")
        self.replay_buffer = ReplayCls(**replay_kwargs)
