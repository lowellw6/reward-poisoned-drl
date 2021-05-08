
import pickle
import numpy as np

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.algos.dqn.dqn import DQN

from reward_poisoned_drl.federated.client.base import AsaFactory, ClientAlgoMixin
from reward_poisoned_drl.attack.fixed.adversary import FixedAttackerDQN
from reward_poisoned_drl.utils import PONG_ACT_MAP


class ClientDQN(ClientAlgoMixin, DQN):
    pass

class ClientFixedAttackerDQN(ClientAlgoMixin, FixedAttackerDQN):
    pass


def get_agent_and_sampler(Sampler=SerialSampler, need_eval=False):
    game = "pong"

    sampler = Sampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=1,
        batch_B=8,  # number of game running in parallel
        max_decorrelation_steps=0,
        eval_n_envs=int(need_eval) * 8,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50
    )

    agent = AtariDqnAgent()

    return agent, sampler


class AsaFactoryClean(AsaFactory):
    def __init__(self, Sampler=SerialSampler):
        self.Sampler = Sampler

    def __call__(self):
        algo = ClientDQN(min_steps_learn=1e3)
        agent, sampler = get_agent_and_sampler(Sampler=self.Sampler)
        return agent, sampler, algo


class AsaFactoryMalicious(AsaFactory):
    def __init__(self, Sampler=SerialSampler):
        self.Sampler = Sampler

    def __call__(self):
        contrast_sd_path = "/home/lowell/reward-poisoned-drl/runs/contrast_enc_4_20/contrast_enc_50.pt"
        dqn_oracle_sd_path = "/home/lowell/reward-poisoned-drl/runs/20210414/000909/dqn_store/run_0/params.pkl"

        target_bottom_path  = "/home/lowell/reward-poisoned-drl/data/targets/targ_bottom.pkl"
        target_mid_path = "/home/lowell/reward-poisoned-drl/data/targets/targ_mid.pkl"
        assert PONG_ACT_MAP[3] == "DOWN"  # target action is DOWN for our pong attacks
        assert PONG_ACT_MAP[5] == "DOWN"  # using 3 & 5, both meaning DOWN, to simulate "stealth"
        TARGET_META = (  # format (path, target-state-thresh, target-action)
            (target_bottom_path, 10.2, 3),
            (target_mid_path, 9.0, 5)
        )

        # load target observations along with respective thresholds and target actions
        target_obs = []
        target_info = {}
        for idx, (tpath, tthresh, ttarg) in enumerate(TARGET_META):
            with open(tpath, "rb") as f:
                tob = pickle.load(f)
                target_obs.append(tob)
                target_info[idx] = (tthresh, ttarg)
        target_obs = np.asarray(target_obs).transpose(0, 3, 1, 2)  # N, H, W, C --> N, C, H, W

        # adversary algorithm (subsumes agent DQN algorithm)
        algo = ClientFixedAttackerDQN(
            target_obs,
            target_info,
            contrast_sd_path,
            dqn_oracle_sd_path,
            delta_bound=1.0,
            min_steps_poison=0,
            target_recall_window=1000,
            min_steps_learn=1e3
        )

        agent, sampler = get_agent_and_sampler(Sampler=self.Sampler)

        return agent, sampler, algo
