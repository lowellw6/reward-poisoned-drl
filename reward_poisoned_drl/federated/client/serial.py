
from reward_poisoned_drl.federated.client.base import FederatedClientBase, AsaFactory, initialize_client


class SerialFederatedClient(FederatedClientBase):
    """
    Federated client where, excluding any sampler
    parallelism, all compute happens in the main
    process. 
    
    Helpful for debugging, and may be fast enough 
    for experiments with relativey few clients
    sampled at each global model update.
    """
    def __init__(self, asa_factory: AsaFactory):
        """
        Construct sampler, agent, and algorithm, 
        and extract batch_spec from sampler.
        """
        super().__init__(asa_factory)  # initialize step_in_progress flag
        self.agent, self.sampler, self.algo = asa_factory()
        self.batch_spec = self.sampler.batch_spec
        self.grad = None
        self.traj_infos = None
        self.opt_info = None

    def initialize(self, n_itr, affinity, seed=None, rank=0, world_size=1):
        """Simply initialize algorithm, agent, and sampler in main process."""
        initialize_client(self, n_itr, affinity, seed, rank, world_size)        

    def step(self, itr, agent_state_dict):
        """
        First syncs agent model with input.
        Then takes one sampler-algorithm step,
        and extracts gradients using the algorithm.
        """
        super().step(itr, agent_state_dict)
        self.agent.load_state_dict(agent_state_dict)

        self.agent.sample_mode(itr)
        samples, self.traj_infos = self.sampler.obtain_samples(itr)
        self.agent.train_mode(itr)
        self.opt_info = self.algo.optimize_agent(itr, samples)

        self.grad = self.algo.pass_gradients()

    def join(self):
        """Return gradients and stats from last step."""
        super().join()
        return self.grad, self.traj_infos, self.opt_info

    def shutdown(self):
        self.sampler.shutdown()

    def get_traj_info_kwargs(self):
        return dict(discount=getattr(self.algo, "discount", 1))
