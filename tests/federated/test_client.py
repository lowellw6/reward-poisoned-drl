
import unittest

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


class TestSerialClient(unittest.TestCase):

    def test_clean_client(self):
        client = SerialFederatedClient(TestAsaFactoryClean())
        n_itr = 126
        affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

        run_client(client, n_itr, affinity)

    def test_malicious_client(self):
        client = SerialFederatedClient(TestAsaFactoryMalicious())
        n_itr = 126
        affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

        run_client(client, n_itr, affinity)


# class TestParallelClient(unittest.TestCase):

#     def test_clean_client(self):
#         client = ParallelFederatedClient(TestAsaFactoryClean())
#         n_itr = 3
#         affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

#         run_client(client, n_itr, affinity)

#     def test_clean_client(self):
#         client = ParallelFederatedClient(TestAsaFactoryMalicious())
#         n_itr = 3
#         affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

#         run_client(client, n_itr, affinity)


if __name__ == "__main__":
    unittest.main()
