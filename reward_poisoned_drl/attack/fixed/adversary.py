"""
Fixed targeted reward-poisoning attacker for 
off-policy rlpyt algorithms. Simply injects
a non-adaptive perturbation into the reward
signal right before storing to the replay
buffer.

This attacker's algorthm is adapted from the
tabular version seen in Algorithm 2 of
Zhang et al.'s "Adaptive Reward-Poisoning 
Attacks against Reinforcement Learning."
"""

import torch

from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel

"""
NOTE for Rajesh:
Will need to both load the trained AtariDqnModel model
from our DqnStore run, and also the supervised or contrastive
observation-encoder here. 

Typically this is done in initialize rather than __init__ 
because we can check that the model architectures match
the observation and action spaces provided. But pick
whatever you think will be most convenient.
"""


class FixedAttackerMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Can add any immediate initialization here
        # such as hyperparams (e.g. delta bound).
        # This is called in run_attack.py.
        raise NotImplementedError  # TODO

    def initialize(self, *args, **kwargs):
        # Called after sampler and agent (and model) 
        # is intialized. Can do any initialization 
        # requiring these items here.
        # This is called during the MinibatchRl
        # Runner's startup() routine.
        super().initialize(*args, **kwargs)
        raise NotImplementedError  # TODO
    
    def samples_to_buffer(self, samples):
        """
        Prepares samples for insertion
        to replay buffer. We insert poison
        rewards here.
        """
        clean_rew = samples.env.reward
        poisoned_rew = self._poison_rewards(clean_rew)
        return SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=poisoned_rew,  # poison inserted here
            done=samples.env.done
        )

    def _poison_rewards(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Implements the fixed attacker
        poisoning routine. 
        """
        raise NotImplementedError  # TODO


class FixedAttackerDQN(FixedAttackerMixin, DQN):
    pass
