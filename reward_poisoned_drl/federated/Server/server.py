import torch
import numpy as np

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent

class FederatedServerClass():

    def __init__(self):
        self.agent = AtariDqnAgent()
        game = "pong"
        self.sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=1,
        batch_B=1,  # number of game running in parallel
        max_decorrelation_steps=0
        )

