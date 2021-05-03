
from abc import ABC, abstractmethod
import multiprocessing as mp
import ctypes

from rlpyt.algos.base import RlAlgorithm
from rlpyt.agents.base import BaseAgent
from rlpyt.samplers.base import BaseSampler
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import make_seed
from rlpyt.utils.collections import AttrDict


class AsaFactory:
    """
    ~Agent-Sampler-Algorithm factory~
    To be built in experiment launch file.
    Should construct the algorithm, agent, and sampler
    needed for a client. This packaging allows convenient
    passing to subprocesses for creation, where the objects
    themselves may not be pickleable.
    """
    def __init__(self) -> tuple[BaseAgent, BaseSampler, RlAlgorithm]:
        raise NotImplementedError


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

    client.algo.initialize(
        agent=client.agent,
        n_itr=n_itr,
        batch_spec=client.sampler.batch_spec,
        mid_batch_reset=client.sampler.mid_batch_reset,
        examples=examples,
        world_size=world_size,
        rank=rank
    ) # TODO check algo supports pass_gradients function via Mixin inheritance


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
        super().step(itr)
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


def client_worker(asa_factory, ctrl, n_itr, affinity, seed, rank, world_size):
    """
    Routine for parallel client worker.
    Stores all needed client members on a proxy attribute dictionary.
    Enters infinite loop between sampling and gradient generation
    synchronized by locks shared with main client process.
    """
    agent, sampler, algo = asa_factory()
    client_dict = AttrDict(agent=agent, sampler=sampler, algo=algo)
    initialize_client(client_dict, n_itr, affinity, seed, rank, world_size)

    for itr in range(n_itr):
        # wait for signal to start sampler-algo iteration
        with ctrl.lock_in:
            # shutdown if specified
            if ctrl.shutdown:
                client_dict.sampler.shutdown()
                return

            agent_state_dict = ctrl.params_conn.recv()  # blocks until params finish sending
            client_dict.agent.load_state_dict(agent_state_dict)

            client_dict.agent.sample_mode(itr)
            samples, traj_infos = client_dict.sampler.obtain_samples(itr)
            client_dict.agent.train_mode(itr)
            opt_info = client_dict.algo.optimize_agent(itr, samples)

            grad = client_dict.algo.pass_gradients()

            # ship gradients and logging infos to main process
            ctrl.grad_conn.send(grad)
            ctrl.traj_info_conn.send(traj_infos)
            ctrl.opt_info_conn.send(opt_info)
        
        # wait to be reset by main client process
        ctrl.lock_out.acquire()
        ctrl.lock_out.release()


class ParallelFederatedClient(FederatedClientBase):
    """
    Federated client where all sampler and algorithm
    compute is launched from a separate process. Useful
    when tranining with many clients.
    """
    def __init__(self, asa_factory: AsaFactory):
        super().__init__(asa_factory)  # initialize step_in_progress flag
        self.asa_factory = asa_factory
        self.main_ctrl, self.worker_ctrl = self._build_parallel_controllers()

        # start with main process owning lock_in
        self.main_ctrl.lock_in.acquire()
        
        # create temporary sampler and algo to extract batch spec and traj_info_kwargs
        _, sampler, algo = asa_factory()
        self.batch_spec = sampler.batch_spec
        self.traj_info_kwargs = dict(discount=getattr(algo, "discount", 1))

    def initialize(self, n_itr, affinity, seed=None, rank=0, world_size=1):
        """
        Start client worker process. All remaining initialization occurs 
        inside this function.
        """
        worker_kwargs = self._assemble_worker_kwargs(n_itr, affinity, seed, rank, world_size)
        self.worker = mp.Process(target=client_worker, kwargs=worker_kwargs)
        self.worker.start()

    def step(self, itr, agent_state_dict):
        """
        Parallelized step, allowing main process
        to exit function while worker process
        computes sampler interaction and optimization
        details.
        """
        super().step(itr)
        self.main_ctrl.params_conn.send(agent_state_dict)

        self._signal_worker_start()
        # worker begins running sampler and algo

    def join(self):
        """
        Parallized join, returning gradients and
        logging infos after worker process has
        finished its iteration.
        """
        super().join()
        self._wait_for_worker()  # will block until worker is finished

        grad = self.main_ctrl.grad_conn.recv()
        traj_infos = self.main_ctrl.traj_info_conn.recv()
        opt_info = self.main_ctrl.opt_info_conn.recv()
        
        return grad, traj_infos, opt_info

    def shutdown(self):
        """
        Sampler is shutdown in worker process after 
        receiving shutdown signal.
        """
        self.main_ctrl.shutdown = True
        if self.step_in_progress:
            self.main_ctrl.lock_out.release()
        self.worker.join()

    def get_traj_info_kwargs(self):
        return self.traj_info_kwargs

    def _build_parallel_controllers(self):
        """
        Builds a distinct parallel contoller 
        for both the main and worker processes.
        Each controller shares locks but receives
        distinct pipe connections.
        """
        lock_in = mp.Lock()  # for worker to start iteration
        lock_out = mp.Lock()  # for worker to reset for next iteration
        
        params_conn1, params_conn2 = mp.Pipe()  # model params main --> worker
        grad_conn1, grad_conn2 = mp.Pipe()  # gradients worker --> main
        traj_info_conn1, traj_info_conn2 = mp.Pipe()  # traj_info worker --> main
        opt_info_conn1, opt_info_conn2 = mp.Pipe()  # opt_info worker --> main
        shutdown = mp.Value(ctypes.c_bool, False)  # signal worker to shutdown

        main_ctrl = AttrDict(
            lock_in=lock_in,
            lock_out=lock_out,
            params_conn=params_conn1,
            grad_conn=grad_conn1,
            traj_info_conn=traj_info_conn1,
            opt_info_conn=opt_info_conn1,
            shutdown=shutdown
        )

        worker_ctrl = AttrDict(
            lock_in=lock_in,
            lock_out=lock_out,
            params_conn=params_conn2,
            grad_conn=grad_conn2,
            traj_info_conn=traj_info_conn2,
            opt_info_conn=opt_info_conn2,
            shutdown=shutdown
        )

        return main_ctrl, worker_ctrl

    def _assemble_worker_kwargs(self, n_itr, affinity, seed, rank, world_size):
        return dict(
            asa_factory=self.asa_factory,
            ctrl=self.worker_ctrl,  # pass worker controller; shared locks, different pipe connections
            n_itr=n_itr
            affinity=affinity,
            seed=seed,
            rank=rank,
            world_size=world_size
        )

    def _signal_worker_start(self):
        """
        Allows worker to begin running sampler and algo.
        Worker will stop at lock_out acquisition until
        this _wait_for_worker is called.
        """
        self.main_ctrl.lock_out.acquire()
        self.main_ctrl.lock_in.release()

    def _wait_for_worker(self):
        """Blocks until worker finished one iteration."""
        self.main_ctrl.lock_in.acquire()
        self.main_ctrl.lock_out.release()
