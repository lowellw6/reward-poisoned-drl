
from abc import ABC, abstractmethod
from typing import Tuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.agents.base import BaseAgent
from rlpyt.samplers.base import BaseSampler
from rlpyt.utils.seed import make_seed


class AsaFactory:
    """
    ~Agent-Sampler-Algorithm factory~
    To be built in experiment launch file.
    Should construct the algorithm, agent, and sampler
    needed for a client. This packaging allows convenient
    passing to subprocesses for creation, where the objects
    themselves may not be pickleable.
    """
    def __call__(self) -> Tuple[BaseAgent, BaseSampler, RlAlgorithm]:
        raise NotImplementedError


class ClientAlgoMixin:
    """
    Mixin class which intercepts 'optimizer.step()' call
    to store gradients to be passed via 'pass_gradients()'
    method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_grads = None

    def save_and_step(self):
        """Snapshot gradients before taking pytorch optimizer step."""
        self.last_grads = [param.grad for param in self.agent.model.parameters()]
        self.optimizer.pytorch_step()

    def optim_initialize(self, rank=0):
        """
        Overwrite optimizer.step with self.save_and_step,
        and backup the former as optimizer.pytorch_step.
        """
        super().optim_initialize(rank)
        self.optimizer.pytorch_step = self.optimizer.step
        self.optimizer.step = self.save_and_step

    def pass_gradients(self):
        """
        Pass the lastest saved gradients.
        
        WARNING:
            This will be 'None' until the algo
            passes an iteration threshold
            determined by 'min_steps_learn'.
        """
        return self.last_grads


def initialize_client(client, n_itr, affinity, seed, rank, world_size):
    client.n_itr = n_itr
    client.affinity = affinity
    client.seed = seed if seed is not None else make_seed()  # assumes global seed set in FederatedRunner
    client.rank = rank
    client.world_size = world_size

    examples = client.sampler.initialize(
        agent=client.agent,  # Agent gets initialized in sampler.
        affinity=client.affinity,
        seed=client.seed + 1,
        bootstrap_value=getattr(client.algo, "bootstrap_value", False),
        traj_info_kwargs=client.get_traj_info_kwargs(),
        rank=rank,
        world_size=world_size
    )

    client.itr_batch_size = client.sampler.batch_spec.size * world_size
    client.agent.to_device(client.affinity.get("cuda_idx", None))
    if world_size > 1:
        client.agent.data_parallel()

    if not isinstance(client.algo, ClientAlgoMixin):
        raise ValueError("Algorithm must inherit from 'ClientAlgoMixin' to support gradient passing")

    client.algo.initialize(
        agent=client.agent,
        n_itr=n_itr,
        batch_spec=client.sampler.batch_spec,
        mid_batch_reset=client.sampler.mid_batch_reset,
        examples=examples,
        world_size=world_size,
        rank=rank
    )


class FederatedClientBase(ABC):
    """
    Abstract base class for federated clients.
    Requires a factory for generating the sampler, 
    agent, and algorithm which are expected to be 
    used in a federated setting: the client syncs 
    with the global server model, starts a 
    sampler-algorithm step (may occur in seperate 
    process), and passes the computed gradients 
    back to the global server when finished.
    """
    @abstractmethod
    def __init__(self, asa_factory: AsaFactory):
        """
        Initialize step_in_progress flag
        which keeps track of when it's valid
        to call 'join'. All subclasses should 
        also set self.batch_spec here.
        """
        self.step_in_progress = False
        self.asa_factory = asa_factory
    
    @abstractmethod
    def initialize(self, n_itr, affinity, seed=None, rank=0, world_size=1):
        pass

    @abstractmethod
    def step(self, itr, agent_state_dict):
        """
        Starts one client sampler-algorithm
        iteration. First updates agent
        model with agent_state_dict.

        Non-abstract overwrites should call this
        to update 'step_in_progress'.
        """
        if self.step_in_progress:
            raise RuntimeError("Cannot start new client step when already stepping")
        
        self.step_in_progress = True

    @abstractmethod
    def join(self):
        """
        Returns gradients generated from the 
        most recent step. This may block if
        the step did not occur in the main
        process.

        Also responsible for passing
        traj_infos and opt_info for logging.

        Non-abstract overwrites should call this
        to update 'step_in_progress'.
        """
        if not self.step_in_progress:
            raise RuntimeError("Cannot provide gradients with 'join' until after calling 'step'")

        self.step_in_progress = False

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def get_traj_info_kwargs(self):
        pass

    @property
    def batch_size(self):
        return self.batch_spec.size
