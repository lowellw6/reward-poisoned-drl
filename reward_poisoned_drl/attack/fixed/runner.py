
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging import logger


class MinibatchRlLogNotify(MinibatchRl):
    """
    Runner class extended to let
    algorithm know when its logging.
    This allows algorithm to maintain
    its own internal running deque of
    logging metrics.
    """

    def train(self):
        n_itr = self.startup()
        for itr in range(n_itr):
            logger.set_iteration(itr)
            logging_this_itr = (itr + 1) % self.log_interval_itrs == 0
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                self.algo.log_notify(logging_this_itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if logging_this_itr:
                    self.log_diagnostics(itr)
        self.shutdown()
