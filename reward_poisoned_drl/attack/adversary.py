"""
Main algorithm for reward poisoning.
Agent algorithm is provided but
stored as a member in a composition
pattern.

This adversary uses TD3 to update
its attack policy.
"""

from rlpyt.algos.qpg.td3 import TD3



class RewardPoisonAdversary(TD3):
    """
    TODO
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO

    # in progress

