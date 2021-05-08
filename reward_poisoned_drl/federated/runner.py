
import psutil
import torch
import time

from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter


class FederatedRunner(MinibatchRlEval):
    """
    (Locally simulated) federated RL Runner.

    Supports RL with one global server and several
    clients, relying on the clients to generate gradients
    to update the global model. The standard FL pattern
    per iteration is as follows:

    1) The global server chooses a random subset of clients
    2) Chosen clients each synchronously copy the current global model
    3) Clients independently generate environment samples and model update gradients
    4) Clients synchronously pass these gradients to the global server
    5) The global server aggregates gradients and updates the global model

    This Runner is blind to whether clients are clean or malicious.
    However, some optimization metrics are logged per-client.
    """
    def __init__(
            self,
            server,
            clients,  # list of clients, can be non-uniform
            n_itr,  # can't use n_steps because clients may have differing batch sizes
            seed=None,
            affinity=None,
            log_interval_itrs=100,  # see n_itr comment
            log_traj_window=100
            ):
        # no call to super().__init__()
        n_itr = int(n_itr)
        log_interval_itrs = int(log_interval_itrs)
        affinity = dict() if affinity is None else affinity
        
        save__init__args(locals())
        
        self.num_clients = len(clients)
        self.log_traj_window = int(log_traj_window)

    def startup(self):
        """
        Initialize server and clients. Each is expected to have its
        own Sampler, Agent, and Algorithm class. 
        """
        # unchanged from MinibatchRlBase
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)

        # each client is isolated, so also should not need to change this
        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)

        # initialize clients and server (clients get uniform kwargs)
        for idx in range(self.num_clients):
            self.clients[idx].initialize(
                n_itr=self.n_itr,
                affinity=self.affinity,
                seed=self.seed + idx,  # clients each get distinct seeds
                rank=rank,
                world_size=world_size
            )
        
        self.server.initialize(
            clients=self.clients,  # pass clients to server
            n_itr=self.n_itr,
            affinity=self.affinity,
            seed=self.seed,
            rank=rank,
            world_size=world_size
        )

        # initialize logging and return
        self.initialize_logging()
        return self.n_itr

    def train(self):
        """
        Mimics train function of MinibatchRlEval
        but adapts to federated setting.
        Runner iterates between having the server
        obtain gradients and optimize its global agent
        with those gradients. Offline evaluation sprints
        are performed on the global server agent using
        the server's sampler.
        """
        n_itr = self.startup()
        
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_server_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                gradients, client_idxs, client_traj_infos, client_opt_infos = self.server.obtain_gradients(itr)
                server_opt_info = self.server.optimize_agent(itr, gradients, client_idxs)
                
                self.store_diagnostics(itr, client_traj_infos, client_opt_infos, server_opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    server_traj_infos, eval_time = self.evaluate_server_agent(itr)
                    self.log_diagnostics(itr, server_traj_infos, eval_time)
        
        self.shutdown()

    def evaluate_server_agent(self, itr):
        """
        For use similar to 'evaluate_agent' in MinibatchRlEval.
        Evaluates server's global agent model using offline batches
        from its own server sampler.
        """
        if itr > 0:
            self.pbar.stop()

        logger.log("Evaluating global agent...")
        self.server.agent.eval_mode(itr)
        eval_time = -time.time()
        traj_infos = self.server.sampler.evaluate_agent(itr)
        eval_time += time.time()

        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

    # def get_traj_info_kwargs(self):
    #     raise NotImplementedError  # TODO ; delete if base is sufficient

    def get_n_itr(self):
        """
        No need for conversion from n_steps.
        We specify in __init__ directly.
        """
        return self.n_itr

    def initialize_logging(self):
        """
        Override to fix self._opt_infos store (no self.algo).
        TODO this will need to change when extending beyond just server opt info
        """
        self._opt_infos = {k: list() for k in self.server.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._cum_time = 0.
        self._cum_eval_time = 0.
        self._cum_completed_trajs = 0
        self._last_update_counter = 0

    def shutdown(self):
        """Extended to shutdown server and clients."""
        logger.log("Training complete.")
        self.pbar.stop()
        self.server.shutdown()
        for client in self.clients:
            client.shutdown()
    
    def get_itr_snapshot(self, itr):
        """
        Override to switch cum_steps out for cum_client_grads,
        since we don't necessarily know each client's batch size.
        Also remove optimizer state dict, since these exist in the
        clients.
        """
        return dict(
            itr=itr,
            cum_client_grads=itr * self.server.clients_per_itr * self.world_size,
            agent_state_dict=self.server._get_global_model()
        )

    def store_diagnostics(self, itr, client_traj_infos, client_opt_infos, server_opt_info):
        # TODO replace empty list with client_traj_infos after differentiating from server_traj_infos
        # client_opt_infos is currently empty, so nothing would be logged anyway from there
        super().store_diagnostics(itr, [], server_opt_info)

    def log_diagnostics(self, itr, traj_infos=None, eval_time=0, prefix='Diagnostics/'):
        """
        Override to fix references to algo and a few other details.
        For example, we now take only one update per iteration for the server,
        but will not update unless we have at least one valid client gradient.
        We also remove any reference to steps, as this may not be globally available.
        """
        if itr > 0:
            self.pbar.stop()
        self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        train_time_elapsed = new_time - self._last_time - eval_time
        new_updates = self.server.num_updates - self._last_update_counter
        updates_per_second = (float('nan') if itr == 0 else
            new_updates / train_time_elapsed)

        with logger.tabular_prefix(prefix):
            if self._eval:
                logger.record_tabular('CumTrainTime',
                    self._cum_time - self._cum_eval_time)  # Already added new eval_time.
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('CumTime (s)', self._cum_time)
            logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
            logger.record_tabular('CumUpdates', self.server.num_updates)
            logger.record_tabular('UpdatesPerSecond', updates_per_second)
        self._log_infos(traj_infos)
        logger.dump_tabular(with_prefix=False)

        self._last_time = new_time
        self._last_update_counter = self.server.num_updates
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)

    # def _log_infos(self, traj_infos=None):
    #     raise NotImplementedError  # TODO ; delete if base is sufficient
