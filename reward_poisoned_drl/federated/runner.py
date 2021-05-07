
import psutil
import torch

from rlpyt.runners.minibatch_rl import MinibatchRlBase
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger


class FederatedRunner(MinibatchRlBase):
    """
    (Locally simulated) federated RL Runner.

    Supports RL with one global server and several
    clients, relying on the clients to genarate gradients
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
            ServerClass,
            ClientClasses,  # list of classes for each client
            server_kwargs,
            client_kwargs,  # list of dictionaries for each client
            n_itr,  # can't use n_steps because clients may have differing batch sizes
            seed=None,
            affinity=None,
            log_interval_itrs=100  # see n_itr comment
            ):
        # no call to super().__init__()
        if len(ClientClasses) != len(client_kwargs):
            raise ValueError("Length of ClientClasses and client_kwargs must match")

        n_itr = int(n_itr)
        log_interval_itr = int(log_interval_itr)
        affinity = dict() if affinity is None else affinity
        save__init__args(locals())
        self.num_clients = len(ClientClasses)

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

        # create client and server objects (clients get individualized kwargs)
        self.clients = [ClientCls(**client_kwargs) 
            for ClientCls, client_kwargs in zip(self.ClientClasses, self.client_kwargs)]
        
        self.server = self.ServerClass(**self.server_kwargs)

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
            clients=self.clients,
            n_itr=self.n_itr,
            affinity=self.affinity,
            seed=self.seed,
            rank=rank,
            world_size=world_size
        )

        # initialize logging and return
        self.initialize_logging()
        return n_itr

    def train(self):
        """
        TODO
        Needs to organize steps 1-5 in Class comment.
        Should also expect some logging info coming from clients to server
        """
        # raise NotImplementedError
        # Choose random subset of clients (Assuming each agent is equally likely ot be picked)
        self.n_iterations_global = 40
        fraction_picked_clients = 0.1
        self.learning_rate = 0.001
        for itr in self.n_iterations_global:
            num_clients_tobe_picked = int(self.num_clients*fraction_picked_clients)
            picked_clients = np.random.choice(self.num_clients, num_clients_tobe_picked)
            grads = None
            for i in picked_clients:
                self.clients[i].step(self.n_itr, self.server.agent.state_dict())
                if grads==None:
                    grads,_,_ = self.clients[i].join()
                else:
                    temp,_,_ = self.clients[i].join()
                    for j in range(0, len(grads)):
                        grads[j]+=temp[j]
            grads = grads/(num_clients_tobe_picked*1.)
            server_model = self.server.agent.state_dict()["model"]
            j = 0
            for params in server_model.parameters():
                params+=-1*grads[j]*self.learning_rate
            self.server.agent.model.load_state_dict(server_model)
            self.server.agent.target_model.load_state_dict(server_model)

        ### Parallel runner




    def evaluate_server_agent(self, itr):
        """
        TODO
        For use similar to 'evaluate_agent' in MinibatchRlEval.
        Should evaluate server's global agent model using offline batches
        from its own server sampler.
        """
        raise NotImplementedError

    def get_traj_info_kwargs(self):
        raise NotImplementedError  # TODO ; delete if base is sufficient

    def get_n_itr(self):
        """
        No need for conversion from n_steps.
        We specify in __init__ directly.
        """
        return self.n_itr

    def initialize_logging(self):
        raise NotImplementedError  # TODO

    def shutdown(self):
        """Extended to shutdown server and clients."""
        logger.log("Training complete.")
        self.pbar.stop()
        self.server.shutdown()
        for client in self.clients:
            client.shutdown()
    
    def get_itr_snapshot(self, itr):
        raise NotImplementedError  # TODO ; probably only need agent state dict from server

    def store_diagnostics(self, itr, traj_infos, opt_info):
        raise NotImplementedError  # TODO ; delete if base is sufficient

    def log_diagnostics(self, itr, traj_infos=None, eval_time=0, prefix='Diagnostics/'):
        raise NotImplementedError  # TODO ; delete if base is sufficient

    def _log_infos(self, traj_infos=None):
        raise NotImplementedError  # TODO ; delete if base is sufficient
