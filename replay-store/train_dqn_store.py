"""
Trains Atari agent with built-in DQN algorithm
(hyperparameters match original 2015 paper).
Only difference is the replay has been modified
to write it's contents to disk everytime it wraps
over.
"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context

from dqn_store import DqnStore


def build_and_train(game="pong", run_ID=0, cuda_idx=None, n_parallel=2, serial_sampling=False):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    device = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if serial_sampling:
        Sampler = SerialSampler  # ignores workers_cpus
        print(f"Using serial sampler w/ {device} for action sampling and optimization")
    else:
        Sampler = CpuSampler if cuda_idx is None else GpuSampler
        print(f"Using parallel sampler w/ {device} for action sampling and optimization")
    
    sampler = Sampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=1,
        batch_B=8,  # number of game running in parallel
        max_decorrelation_steps=0
    )

    algo = DqnStore(  # swapped out algo here
        replay_save_dir="/mnt/data/lowell/replay-data",
        min_steps_learn=1e3
    )

    agent = AtariDqnAgent()

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=affinity
    )

    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "dqn_store"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('-g', '--cuda_idx', help='gpu to use', type=int, default=None)
    parser.add_argument('-n', '--n_parallel', help='number of sampler workers', type=int, default=2)
    parser.add_argument('-s', '--serial_sampling', action='store_true', help='use serial sampler (no CPU parallelism)', default=False)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        n_parallel=args.n_parallel,
        serial_sampling=args.serial_sampling
    )
