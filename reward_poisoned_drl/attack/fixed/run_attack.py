"""
Runs a fixed reward-poisoning attack on a deep RL DQN agent.
"Fixed" here means the adversary does not modify any of its
internal state throughout agent training.

The attacker aims to make the agent learn a target policy,
generating reward perturbations using Algorithm 2 in this paper:
https://arxiv.org/pdf/2003.12613.pdf

In high-dimensional image observation spaces, this target policy
can only target a small subset of all states. At these target states
the advesary intends for the agent to learn to take a pre-determined
target action.

This adversary assumes an "oracle" Q-value function,
which we approximate using a pre-trained DQN.
"""

import pickle
import torch
import numpy as np

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.utils.logging.context import logger_context

from reward_poisoned_drl.attack.fixed.adversary import FixedAttackerDQN
from reward_poisoned_drl.attack.fixed.runner import MinibatchRlLogNotify
from reward_poisoned_drl.utils import PONG_ACT_MAP


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


def build_and_train(run_ID=0, cuda_idx=None, n_parallel=2, serial_sampling=False):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    device = "cpu" if cuda_idx is None else f"gpu {cuda_idx}"
    if serial_sampling:
        Sampler = SerialSampler  # ignores workers_cpus
        print(f"Using serial sampler w/ {device} for action sampling and optimization")
    else:
        Sampler = CpuSampler if cuda_idx is None else GpuSampler
        print(f"Using parallel sampler w/ {device} for action sampling and optimization")

    game = "pong"

    sampler = Sampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=1,
        batch_B=8,  # number of game running in parallel
        max_decorrelation_steps=0
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
    algo = FixedAttackerDQN(
        target_obs,
        target_info,
        contrast_sd_path,
        dqn_oracle_sd_path,
        delta_bound=1.0,
        min_steps_poison=0,
        target_recall_window=1000,
        min_steps_learn=1e3
    )

    agent = AtariDqnAgent()

    runner = MinibatchRlLogNotify(  # custom runner to notify attacker when logging
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=affinity
    )

    config = dict(game=game)
    name = "rp_fixed_attack_dqn_" + game
    log_dir = "rp_fixed_attack"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('-g', '--cuda_idx', help='gpu to use', type=int, default=None)
    parser.add_argument('-n', '--n_parallel', help='number of sampler workers for agent environment interaction', type=int, default=2)
    parser.add_argument('-s', '--serial_sampling', action='store_true', help='use serial sampler for agent (no CPU parallelism)', default=False)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        n_parallel=args.n_parallel,
        serial_sampling=args.serial_sampling
    )
