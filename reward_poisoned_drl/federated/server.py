
import torch
import numpy as np
from typing import List
from collections import namedtuple

from rlpyt.utils.seed import make_seed
from rlpyt.utils.quick_args import save__init__args

from reward_poisoned_drl.utils import list_to_norm, flatten_lists

ServerOptInfo = namedtuple("ServerOptInfo", ["numValidGrads", "meanGradNorm"])
AggClientOptInfo = namedtuple("AggClientOptInfo", ["attackerCost", "recallTarget0", "recallTarget1"]) # TODO make not hard-coded and non-singular (needs to be dynamic)


class FederatedServer:
    """
    Manages global model update routine using
    client gradients. 'obtain_gradients' and
    'optimize_agent' should be called
    iteratively inside the runner class.
    """
    def __init__(
            self, 
            agent, 
            sampler,
            clients_per_itr=1,
            global_lr=1.0, 
            eval_discount=0.99
            ):
        """
        Store agent and sampler. This class subsumes the
        server algorithm componenent. We also setup helper
        structures for aggregated returns here.

        'clients_per_itr' specifies how many clients to sample
        gradients from at each global iteration.

        'global_lr' is the learning rate used for batch
        gradient descent of the global agent model.
        This is applied ~~AFTER~~ any client learning
        rate, so 1.0 is maintaining the average client
        learning rate.

        'eval_discount' is used for logging only. It's
        what the evaluation trajectory discounted return
        stats will be computed with.
        """
        save__init__args(locals())

        # for aggregating per-client gradients and logging info
        self.gradients = []
        self.client_traj_infos = []
        self.client_opt_infos = []

        self.opt_info_fields = tuple(f for f in ServerOptInfo._fields)

    def initialize(self, clients, n_itr, affinity, seed=None, rank=0, world_size=1):
        if len(clients) < self.clients_per_itr:
            raise ValueError("'clients_per_itr' larger than number of clients")

        self.clients = clients
        self.num_clients = len(clients)
        self.n_itr = n_itr
        self.affinity = affinity
        self.seed = seed if seed is not None else make_seed()  # assumes global seed set in FederatedRunner
        self.rank = rank
        self.world_size = world_size

        self.sampler.initialize(
            agent=self.agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=False,  # no algo to use bootstrap
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size
        )

        self.agent.to_device(self.affinity.get("cuda_idx", None))
        if world_size > 1:
            self.agent.data_parallel()

        self.num_updates = 0  # update counter; may not always update if client grads are None

    def obtain_gradients(self, itr):
        """
        Obtain one batch of sample gradients from clients.
        Start by randomly sampling client indices, then
        step and join to allow for parallelization
        (though compute may still happen in main
        thread depending on client class).
        """
        global_model = self._get_global_model()
        client_idxs = np.random.choice(self.num_clients, size=self.clients_per_itr, replace=False)
        
        # first step all models, allowing parallel execution if using parallelized clients
        for idx in client_idxs:
            self.clients[idx].step(itr, global_model)
        
        # gather gradients from client stepping; will block until finished if parallelized
        for idx in client_idxs:
            grad, traj_infos, opt_info = self.clients[idx].join()
            self._append_client_results(grad, traj_infos, opt_info)

        # return client results along with client_idxs
        gradients, client_traj_infos, client_opt_infos = self._get_aggregated_results()
        return gradients, client_idxs, client_traj_infos, client_opt_infos

    def optimize_agent(self, itr, gradients: List[List[torch.Tensor]], client_idxs: np.ndarray):
        """
        Optimize global server agent using list of gradient lists.
        The outer list contains one list for each sampled client.
        The inner per-client lists contain a gradient tensor for each model parameter.
        """
        self.agent.train_mode(itr)
        server_opt_info = ServerOptInfo(*([] for _ in range(len(ServerOptInfo._fields))))

        # filter non-None responses, if any
        valid_gradients = self._get_valid_gradients(gradients)

        if valid_gradients:
            # prepare mean of valid sampled model gradients
            device_gradients = self._gradients_to_device(valid_gradients)
            mean_gradients = self._get_mean_gradients(device_gradients)
            
            # apply gradients to server global model
            self._apply_gradient_descent(mean_gradients)

            # increment update counter
            self.num_updates += 1

        # return server-specific logging info
        server_opt_info.numValidGrads.append(len(valid_gradients))
        mean_grad_norm = list_to_norm(mean_gradients).item() if valid_gradients else float('nan')
        server_opt_info.meanGradNorm.append(mean_grad_norm)

        return server_opt_info

    def shutdown(self):
        self.sampler.shutdown()

    def get_traj_info_kwargs(self):
        return dict(discount=self.eval_discount)

    def _get_global_model(self):
        agent_sd = self.agent.state_dict()
        if "model" in agent_sd.keys():
            agent_sd = agent_sd["model"]  # avoid passing target model
        return agent_sd

    def _load_global_model(self, state_dict):
        self.agent.load_state_dict(state_dict)

    def _append_client_results(self, grad, traj_infos, opt_info):
        self.gradients.append(grad)
        self.client_traj_infos.append(traj_infos)
        self.client_opt_infos.append(opt_info)

    def _get_aggregated_results(self):
        """
        Gather aggregated client results.
        Note we flatten the traj_infos since samples
        are generated with the newly loaded global model,
        and so are uniform over clients.
        """
        gradients = self.gradients
        client_traj_infos = flatten_lists(self.client_traj_infos)
        client_opt_infos = self._combine_client_opt_infos(self.client_opt_infos)
        
        self.gradients = []
        self.client_traj_infos = []
        self.client_opt_infos = []

        return gradients, client_traj_infos, client_opt_infos

    def _get_valid_gradients(self, gradients):
        """Sort out 'None' responses from not-ready clients."""
        valid_gradients = []
        for obj in gradients:
            if isinstance(obj, list):
                valid_gradients.append(obj)
            elif obj is not None:
                raise ValueError("Unrecognized value in gradients list;"
                    " must contain lists of tensors or None objects")
        return valid_gradients

    def _gradients_to_device(self, gradients):
        """Move list of gradient tensor lists to agent device."""
        device_gradients = []
        for grad_list in gradients:
            device_gradients.append([param.to(self.agent.device) for param in grad_list])
        return device_gradients

    def _get_mean_gradients(self, gradients):
        """
        Average gradients by reducing across clients. 
        Expect all to be valid (no None values).
        """
        mean_gradients = []
        for param_tup in zip(*gradients):
            param_stack = torch.stack(param_tup, dim=0)
            param_mean = torch.mean(param_stack, dim=0)
            mean_gradients.append(param_mean)
        return mean_gradients

    def _apply_gradient_descent(self, gradients):
        """
        Apply list of gradients (one tensor for each
        model param) to server's global agent model.
        """
        updated_sd = {}
        global_model = self._get_global_model()
        
        for name, param, grad in zip(global_model.keys(), global_model.values(), gradients):
            updated_sd[name] = param - self.global_lr * grad
        
        self._load_global_model(updated_sd)

    def _combine_client_opt_infos(self, client_opt_infos):
        """
        Converts list of client opt infos to single opt info
        with each key labelled by TODO
        """
        # TODO make not save using hard-coded keys
        # right now this extracts attacker stats from any clients which provide them
        # we can aggregate these since they're generated using the same loaded global model
        # WARNING: recall will be empty even when logging if no malicious clients were sampled that itr...
        attacker_cost_buff = []
        recall_target0_buff = []
        recall_target1_buff = []
        for opt_info in client_opt_infos:
            if getattr(opt_info, "attackerCost", None) is not None:
                attacker_cost_buff += [opt_info.attackerCost]
            if getattr(opt_info, "recallTarget0", None) is not None:
                recall_target0_buff += opt_info.recallTarget0
            if getattr(opt_info, "recallTarget1", None) is not None:
                recall_target1_buff += opt_info.recallTarget1
        return AggClientOptInfo(attacker_cost_buff, recall_target0_buff, recall_target1_buff)


class FederatedServerLogNotify(FederatedServer):
    """
    Notifies client when runner is logging.
    Should be used with corresponding runner and client classes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upcoming_log = False

    def log_notify(self, flag: bool):
        self.upcoming_log = flag

    def obtain_gradients(self, itr):
        global_model = self._get_global_model()
        client_idxs = np.random.choice(self.num_clients, size=self.clients_per_itr, replace=False)
        
        # first step all models, allowing parallel execution if using parallelized clients
        for idx in client_idxs:
            self.clients[idx].log_notify(self.upcoming_log)  # log notify injected before step
            self.clients[idx].step(itr, global_model)
        
        # gather gradients from client stepping; will block until finished if parallelized
        for idx in client_idxs:
            grad, traj_infos, opt_info = self.clients[idx].join()
            self._append_client_results(grad, traj_infos, opt_info)

        # return client results along with client_idxs
        gradients, client_traj_infos, client_opt_infos = self._get_aggregated_results()
        return gradients, client_idxs, client_traj_infos, client_opt_infos
    