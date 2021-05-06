
import unittest

from rlpyt.envs.atari.atari_env import AtariEnv

from reward_poisoned_drl.federated.client.serial import SerialFederatedClient
from reward_poisoned_drl.federated.client.parallel import ParallelFederatedClient

from asa_factories import TestAsaFactoryClean, TestAsaFactoryMalicious


def run_client(client, n_itr, affinity, agent_state_dict=None):
    client.initialize(n_itr, affinity)
    if agent_state_dict is None:
        agent_state_dict = client.agent.state_dict()["model"]  # only works for serial client
    
    for itr in range(n_itr):
        client.step(itr, agent_state_dict)
        grad, traj_infos, opt_info = client.join()
    
    client.shutdown()


def get_dummy_state_dict():
    """
    For testing parallel client, which initializes its
    agent inside the worker process.
    """
    asa_factory = TestAsaFactoryClean()
    agent, _, _ = asa_factory()
    env = AtariEnv()
    agent.initialize(env.spaces)

    return agent.state_dict()["model"]


# class TestSerialClient(unittest.TestCase):

#     def test_clean_client(self):
#         client = SerialFederatedClient(TestAsaFactoryClean())
#         n_itr = 126
#         affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

#         run_client(client, n_itr, affinity)

#     def test_malicious_client(self):
#         client = SerialFederatedClient(TestAsaFactoryMalicious())
#         n_itr = 126
#         affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

#         run_client(client, n_itr, affinity)


class TestParallelClient(unittest.TestCase):

    def test_clean_client(self):
        asa_factory = TestAsaFactoryClean()
        client = ParallelFederatedClient(asa_factory)
        n_itr = 3
        affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

        agent_state_dict = get_dummy_state_dict()

        run_client(client, n_itr, affinity, agent_state_dict)

#     def test_clean_client(self):
#         client = ParallelFederatedClient(TestAsaFactoryMalicious())
#         n_itr = 3
#         affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

#         run_client(client, n_itr, affinity)


if __name__ == "__main__":
    unittest.main()
