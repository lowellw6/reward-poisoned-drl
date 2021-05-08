
import unittest
import time
from tqdm import tqdm

from rlpyt.envs.atari.atari_env import AtariEnv

from reward_poisoned_drl.federated.client.serial import SerialFederatedClient
from reward_poisoned_drl.federated.client.parallel import ParallelFederatedClient
from reward_poisoned_drl.federated.client.asa_factories import AsaFactoryClean, AsaFactoryMalicious


def run_client(client, n_itr, affinity, agent_state_dict=None):
    client.initialize(n_itr, affinity)
    if agent_state_dict is None:
        agent_state_dict = client.agent.state_dict()["model"]  # only works for serial client
    
    for itr in tqdm(range(n_itr)):
        client.step(itr, agent_state_dict)
        grad, traj_infos, opt_info = client.join()
    
    client.shutdown()


def run_multiple_clients(clients, n_itr, affinity, agent_state_dict=None):
    for client in clients:
        client.initialize(n_itr, affinity)
    
    if agent_state_dict is None:
        agent_state_dict = client.agent.state_dict()["model"]  # only works for serial client

    grad_l, traj_infos_l, opt_info_l = [], [], []
    for itr in tqdm(range(n_itr)):
        for client in clients:
            client.step(itr, agent_state_dict)
        
        for client in clients:
            grad, traj_infos, opt_info = client.join()
            
            grad_l.append(grad)
            traj_infos_l.append(traj_infos)
            opt_info_l.append(opt_info)

    for client in clients:
        client.shutdown()


def get_dummy_state_dict():
    """
    For testing parallel client, which initializes its
    agent inside the worker process.
    """
    asa_factory = AsaFactoryClean()
    agent, _, _ = asa_factory()
    env = AtariEnv()
    agent.initialize(env.spaces)

    return agent.state_dict()["model"]


class TestSerialClient(unittest.TestCase):

    def test_clean_client(self):
        client = SerialFederatedClient(AsaFactoryClean())
        n_itr = 126
        affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

        run_client(client, n_itr, affinity)

    def test_malicious_client(self):
        client = SerialFederatedClient(AsaFactoryMalicious())
        n_itr = 126
        affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

        run_client(client, n_itr, affinity)


class TestParallelClient(unittest.TestCase):

    def test_clean_client(self):
        asa_factory = AsaFactoryClean()
        client = ParallelFederatedClient(asa_factory)
        n_itr = 126
        affinity = dict(cuda_idx=0, workers_cpus=list(range(2)))

        agent_state_dict = get_dummy_state_dict()

        run_client(client, n_itr, affinity, agent_state_dict)

    def test_malicious_client(self):
        asa_factory = AsaFactoryMalicious()
        client = ParallelFederatedClient(asa_factory)
        n_itr = 126
        affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

        agent_state_dict = get_dummy_state_dict()

        run_client(client, n_itr, affinity, agent_state_dict)


class TestParallelSpeedup(unittest.TestCase):

    def test_speedup_clean(self):
        num_clients = 10
        asa_factory = AsaFactoryClean()
        serial_clients = [SerialFederatedClient(asa_factory) for _ in range(num_clients)]
        parallel_clients = [ParallelFederatedClient(asa_factory) for _ in range(num_clients)]
        n_itr = 250
        affinity = dict(cuda_idx=0, workers_cpus=list(range(2)))  # use GPU

        agent_state_dict = get_dummy_state_dict()

        print(f"Running {num_clients} serial federated clients for {n_itr} iterations")

        serial_before = time.time()
        run_multiple_clients(serial_clients, n_itr, affinity)
        serial_duration = time.time() - serial_before

        print(f"Running {num_clients} parallel federated clients for {n_itr} iterations")
        
        parallel_before = time.time()
        run_multiple_clients(parallel_clients, n_itr, affinity, agent_state_dict)
        parallel_duration = time.time() - parallel_before

        print(f"Compared {num_clients} federated clients for {n_itr} iterations")
        print(f"Serial   --> {serial_duration:.2f}")
        print(f"Parallel --> {parallel_duration:.2f}")


if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method('spawn')
    unittest.main()
