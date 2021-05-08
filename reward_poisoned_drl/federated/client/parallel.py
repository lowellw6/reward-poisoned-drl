
import multiprocessing as mp
import ctypes
import time

from reward_poisoned_drl.federated.client.base import FederatedClientBase, AsaFactory, initialize_client
from rlpyt.utils.collections import AttrDict


def client_worker(asa_factory, ctrl, n_itr, affinity, traj_info_kwargs, seed, rank, world_size):
    """
    Routine for parallel client worker.
    Stores all needed client members on a proxy attribute dictionary.
    Enters infinite loop between sampling and gradient generation
    synchronized by locks shared with main client process.
    """
    # initialize agent, sampler, and algorithm
    agent, sampler, algo = asa_factory()
    client_dict = AttrDict(agent=agent, sampler=sampler, algo=algo)
    client_dict.get_traj_info_kwargs = lambda: traj_info_kwargs  # pass as getter func
    
    initialize_client(client_dict, n_itr, affinity, seed, rank, world_size)

    itr = 0
    while True:
        # wait for first sync with main process to check for shutdown
        ctrl.sync_barr_1.wait()

        # break and shutdown if specified
        with ctrl.shutdown.get_lock():
            if ctrl.shutdown.value:
                break

        # wait for second sync with main process to start sampler-algo iteration
        ctrl.sync_barr_2.wait()

        # start sampler-algo iteration
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

        itr += 1

    client_dict.sampler.shutdown()


class ParallelFederatedClient(FederatedClientBase):
    """
    Federated client where all sampler and algorithm
    compute is launched from a separate process. Useful
    when tranining with many clients.
    """
    def __init__(self, asa_factory: AsaFactory):
        super().__init__(asa_factory)  # initialize step_in_progress flag
        self.main_ctrl, self.worker_ctrl = self._build_parallel_controllers()
        
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
        super().step(itr, agent_state_dict)
        self.main_ctrl.params_conn.send(agent_state_dict)

        self._sync_with_worker()
        # worker begins running sampler and algo

    def join(self):
        """
        Parallized join, returning gradients and
        logging infos after worker process has
        finished its iteration.
        """
        super().join()

        # pipe recv calls will block until receiving transfer from worker
        grad = self.main_ctrl.grad_conn.recv()
        traj_infos = self.main_ctrl.traj_info_conn.recv()
        opt_info = self.main_ctrl.opt_info_conn.recv()
        
        return grad, traj_infos, opt_info

    def shutdown(self):
        """
        Sampler is shutdown in worker process after 
        receiving shutdown signal.
        """
        with self.main_ctrl.shutdown.get_lock():
            self.main_ctrl.shutdown.value = True

        self.main_ctrl.sync_barr_1.wait()  # release worker to notice shutdown

        self.worker.join()

    def get_traj_info_kwargs(self):
        return self.traj_info_kwargs

    def _build_parallel_controllers(self):
        """
        Builds a distinct parallel contoller 
        for both the main and worker processes.
        Each controller shares sync_barriers but receives
        distinct pipe connections.
        """
        # worker starts iteration after both barriers are released
        sync_barr_1 = mp.Barrier(2)
        sync_barr_2 = mp.Barrier(2)
        
        params_conn1, params_conn2 = mp.Pipe()  # model params main --> worker
        grad_conn1, grad_conn2 = mp.Pipe()  # gradients worker --> main
        traj_info_conn1, traj_info_conn2 = mp.Pipe()  # traj_info worker --> main
        opt_info_conn1, opt_info_conn2 = mp.Pipe()  # opt_info worker --> main

        shutdown = mp.Value(ctypes.c_bool, False)  # signal worker to shutdown

        main_ctrl = AttrDict(
            sync_barr_1=sync_barr_1,
            sync_barr_2=sync_barr_2,
            params_conn=params_conn1,
            grad_conn=grad_conn1,
            traj_info_conn=traj_info_conn1,
            opt_info_conn=opt_info_conn1,
            shutdown=shutdown
        )

        worker_ctrl = AttrDict(
            sync_barr_1=sync_barr_1,
            sync_barr_2=sync_barr_2,
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
            ctrl=self.worker_ctrl,  # pass worker controller; shared barriers, different pipe connections
            n_itr=n_itr,
            affinity=affinity,
            traj_info_kwargs=self.get_traj_info_kwargs(),
            seed=seed,
            rank=rank,
            world_size=world_size
        )

    def _sync_with_worker(self):
        """
        Synchronize main process with worker process.
        When returning, worker will have started next iteration.

        We require two barriers to allow for a proper shutdown 
        procedure, handled separately in self.shutdown().
        
        WARNING:
            Do not call this after asserting the ctrl.shutdown
            signal. Doing so will result in deadlock of the main
            process.
        """
        self.main_ctrl.sync_barr_1.wait()
        self.main_ctrl.sync_barr_2.wait()
