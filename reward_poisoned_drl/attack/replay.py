"""
Provides replay buffer class for adversary
which is itself an infinite-horizon off-policy 
algorithm. The attacker replay contains the
following:

Observation --> eta = (s_t, a_t, r_t, s_{t+1})
    One agent transition

Action --> delta
    Real-value perturbation scaler to apply to r_t
    before the sample is used for agent learning

Reward --> rho
    Real-value negative attack cost based on how
    close the agent's policy is to the target policy
    **after** the agent updates once with the chosen
    perturbation

No done signal is provided (infinite-horizon problem).

To save memory, agent observations are saved as their 
encodings using the pre-trained observation-encoded
provided to the adversary.
"""

from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.replays.non_sequence.uniform import UniformReplay


class AttackerNStepReturnBuffer(BaseNStepReturnBuffer):
    
    def extract_batch(self, T_idxs, B_idxs):
        """
        Extracts batch of off-attack-policy data
        from the replay buffer.
        (T, B) leading dims will match agent's.
        """
        raise NotImplementedError  # TODO


class AttackerUniformReplayBuffer(UniformReplay, AttackerNStepReturnBuffer):
    """Adds uniform sampling support to AttackerNStepReturnBuffer."""
    pass
