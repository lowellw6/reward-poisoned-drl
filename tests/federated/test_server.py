
import unittest
import random
from tqdm import tqdm
import torch

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent

from reward_poisoned_drl.federated.server import FederatedServer
from reward_poisoned_drl.federated.client.serial import SerialFederatedClient
from reward_poisoned_drl.federated.client.asa_factories import AsaFactoryClean, AsaFactoryMalicious


def run_server(server, clients, n_itr, affinity):
    for client in clients:
        client.initialize(n_itr, affinity)

    server.initialize(clients, n_itr, affinity)
    
    for itr in tqdm(range(n_itr)):
        gradients, client_idxs, client_traj_infos, client_opt_infos = server.obtain_gradients(itr)
        server_opt_info = server.optimize_agent(itr, gradients, client_idxs)

    server.shutdown()


class TestFederatedServer(unittest.TestCase):

    def setUp(self):
        game = "pong"

        self.sampler = SerialSampler(
            EnvCls=AtariEnv,
            TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
            env_kwargs=dict(game=game),
            eval_env_kwargs=dict(game=game),
            batch_T=1,
            batch_B=8,  # number of game running in parallel
            max_decorrelation_steps=0
        )

        self.agent = AtariDqnAgent()

    def test_global_model_update(self):
        num_clients = 3
        clients = [SerialFederatedClient(AsaFactoryClean()) for _ in range(num_clients)]
        server = FederatedServer(self.agent, self.sampler, clients_per_itr=2)
        n_itr = 1
        affinity = dict(cuda_idx=None, workers_cpus=list(range(2)))

        server.initialize(clients, n_itr, affinity)

        dummy_grads = list(server.agent.model.parameters())
        dummy_grads = [grad / server.global_lr for grad in dummy_grads]
        server._apply_gradient_descent(dummy_grads)

        # having applied params boosted by learning rate, resulting params should be 0
        for param in server.agent.model.parameters():
            self.assertTrue(torch.allclose(param, torch.zeros_like(param)))

        # checking another way of accessing model params
        for param in server._get_global_model().values():
            self.assertTrue(torch.allclose(param, torch.zeros_like(param)))


    def test_server_clean_only(self):
        num_clients = 3
        clients = [SerialFederatedClient(AsaFactoryClean()) for _ in range(num_clients)]
        server = FederatedServer(self.agent, self.sampler, clients_per_itr=2)
        n_itr = 126
        affinity = dict(cuda_idx=0, workers_cpus=list(range(2)))

        run_server(server, clients, n_itr, affinity)

    def test_server_malicious_only(self):
        num_clients = 3
        clients = [SerialFederatedClient(AsaFactoryMalicious()) for _ in range(num_clients)]
        server = FederatedServer(self.agent, self.sampler, clients_per_itr=2)
        n_itr = 126
        affinity = dict(cuda_idx=0, workers_cpus=list(range(2)))
        
        run_server(server, clients, n_itr, affinity)

    def test_server_client_mix(self):
        num_clean_clients = 2
        num_malicious_clients = 1
        clients = [SerialFederatedClient(AsaFactoryClean()) for _ in range(num_clean_clients)]
        clients += [SerialFederatedClient(AsaFactoryMalicious()) for _ in range(num_malicious_clients)]
        random.seed(123)
        random.shuffle(clients)
        server = FederatedServer(self.agent, self.sampler, clients_per_itr=2)
        n_itr = 126
        affinity = dict(cuda_idx=0, workers_cpus=list(range(2)))
        
        run_server(server, clients, n_itr, affinity)


if __name__ == "__main__":
    unittest.main()
