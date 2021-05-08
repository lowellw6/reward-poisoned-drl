"""
Runs the attack in attack/fixed/run_attack.py but modified
to operate in a federated setting with several clean and/or 
malicious clients.

We use the following federated learning pattern for each iteration:
    1) The global server chooses a random subset of clients
    2) Chosen clients each synchronously copy the current global model
    3) Clients independently generate environment samples and model update gradients
    4) Clients synchronously pass these gradients to the global server
    5) The global server aggregates gradients and updates the global model
"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging.context import logger_context

from reward_poisoned_drl.federated.runner import FederatedRunner
from reward_poisoned_drl.federated.server import FederatedServer
from reward_poisoned_drl.federated.client.serial import SerialFederatedClient
from reward_poisoned_drl.federated.client.asa_factories import AsaFactoryClean, AsaFactoryMalicious, get_agent_and_sampler


def build_and_train(run_ID=0, cuda_idx=None, n_parallel=2, serial_sampling=False):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    device = "cpu" if cuda_idx is None else f"gpu {cuda_idx}"

    # sampler config shared by clients and server
    if serial_sampling:
        Sampler = SerialSampler  # ignores workers_cpus
        print(f"Using serial sampler w/ {device} for action sampling and optimization")
    else:
        Sampler = CpuSampler if cuda_idx is None else GpuSampler
        print(f"Using parallel sampler w/ {device} for action sampling and optimization")

    num_clean_clients = 5
    num_malicious_clients = 0

    clients = [SerialFederatedClient(AsaFactoryClean(Sampler=Sampler)) for _ in range(num_clean_clients)]
    clients += [SerialFederatedClient(AsaFactoryMalicious(Sampler=Sampler)) for _ in range(num_malicious_clients)]

    client_per_itr = 5
    global_agent, global_sampler = get_agent_and_sampler(Sampler=Sampler, need_eval=True)
    server = FederatedServer(global_agent, global_sampler, clients_per_itr=client_per_itr)

    runner = FederatedRunner(
        server,
        clients,
        n_itr=5e6,
        affinity=affinity,
        log_interval_itrs=1000
    )

    config = dict(game="pong")
    name = "federated_rp_fixed_attack_dqn_pong"
    log_dir = "federated_rp_fixed_attack"
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
