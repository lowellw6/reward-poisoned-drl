"""
Fixed targeted reward-poisoning attacker for
off-policy rlpyt algorithms. Simply injects
a non-adaptive perturbation into the reward
signal right before storing to the replay
buffer.

This attacker's algorthm is adapted from the
tabular version seen in Algorithm 2 of
Zhang et al.'s "Adaptive Reward-Poisoning
Attacks against Reinforcement Learning."
"""


from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel
from reward_poisoned_drl.contrastive_encoder.contrast import Contrastor
from reward_poisoned_drl.utils import semantic_crop_pong,PONG_ACT_MAP
import os
import torch
import pickle
import numpy as np
"""
NOTE for Rajesh:
Will need to both load the trained AtariDqnModel model
from our DqnStore run, and also the supervised or contrastive
observation-encoder here.

Typically this is done in initialize rather than __init__
because we can check that the model architectures match
the observation and action spaces provided. But pick
whatever you think will be most convenient.
"""


class FixedAttackerMixin:

    def __init__(self, delta_bound, gamma, device, state_dict, keys, dqn_state_dict, min_steps_learn,**kwargs):
        super().__init__(min_steps_learn=min_steps_learn,**kwargs)
        # Can add any immediate initialization here
        # such as hyperparams (e.g. delta bound).
        # This is called in run_attack.py.
        # raise NotImplementedError  # TODO
        self.delta_bound = delta_bound
        self.gamma = gamma
        self.device = torch.device(device)
        self.state_dict = state_dict
        temp_keys = np.array(keys)
        shape_keys = temp_keys.shape
        self.keys = np.reshape(temp_keys, (shape_keys[0], shape_keys[3], shape_keys[1], shape_keys[2]))
        self.dqn_state_dict = dqn_state_dict
        self.threshold_bottom = 10.2
        self.threshold_mid = 9.0
        self.nefarious_action = 3 #DOWN
        self.last_samples = None

    def initialize(self, *args, **kwargs):
        # Called after sampler and agent (and model)
        # is intialized. Can do any initialization
        # requiring these items here.
        # This is called during the MinibatchRl
        # Runner's startup() routine.
        super().initialize(*args, **kwargs)
        # print(self.device)
        self.contrastive_model = Contrastor(self.state_dict, self.device)
        im_shape = (4, 104, 80)
        action_size = 6
        self.dqn_oracle = AtariDqnModel(im_shape, action_size)
        self.dqn_oracle.load_state_dict(self.dqn_state_dict)

        # raise NotImplementedError  # TODO

    def samples_to_buffer(self, samples):
        """
        Prepares samples for insertion
        to replay buffer. We insert poison
        rewards here.
        """
        if self.last_samples==None:
            import copy
            self.last_samples = copy.deepcopy(samples)
            return super().samples_to_buffer(samples)

        clean_rew = samples.env.reward
        action = samples.agent.action
        query = samples.env.observation.clone()
        # query = query.squeeze(0)
        poisoned_rew = self._poison_rewards(clean_rew, query, action, samples.env.done)

        return SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=poisoned_rew,  # poison inserted hered
            done=samples.env.done
        )

    def _create_q_function(self, obs, action, reward):
        """
        obs: state observation, image stack
        action, reward: list of integers
        Computes the similarity of observations to target states
        and creates a new Q function Q', for reward poisoning
        returns Q'
        """
        obs = obs.squeeze(0)
        query = semantic_crop_pong(obs)
        keys = torch.as_tensor(semantic_crop_pong(self.keys))
        logits = self.contrastive_model(query, keys)
        # print(logits.shape)
        
        targets_bottom = (logits[:,0]>=self.threshold_bottom).nonzero(as_tuple=True)[0]
        targets_mid = (logits[:,1]>=self.threshold_mid).nonzero(as_tuple=True)[0]
        targets = torch.cat((targets_bottom, targets_mid), axis=0)

        oracle_q_values = self.dqn_oracle(observation=obs, prev_action=action, prev_reward = reward) # Agents x actions
        for i in targets:
            if action[0,i]==self.nefarious_action:
                oracle_q_values[i, action[0,i]] += (self.delta_bound/(1+self.gamma))
            else:
                oracle_q_values[i, action[0,i]] += -1*(self.delta_bound/(1+self.gamma))
        return oracle_q_values
    
    def _poison_rewards(self, reward, obs, action, done) -> torch.Tensor:
        """
        obs: current state [num_agents, 4, H, W]

        Implements the fixed attacker
        poisoning routine.
        """
        last_obs = self.last_samples.env.observation.clone()
        reward_old = self.last_samples.env.reward
        action_old = self.last_samples.agent.action
        # new_rewards= rewards
        oracle_q_function_last_state = self._create_q_function(last_obs, action_old, reward_old)

        current_obs = obs.clone()
        oracle_q_values_next_state = self._create_q_function(current_obs, action, reward)
        q_function_actions = torch.zeros((1,oracle_q_function_last_state.shape[0]))
        for i in range(0, len(action)):
            q_function_actions[i] = oracle_q_values_next_state[i, action[0,i]]

        new_rewards =  q_function_actions - self.gamma*(torch.max(oracle_q_values_next_state, dim=1)[0])
        if done:
            self.last_samples = None
        return torch.tensor(new_rewards)


        # raise NotImplementedError  # TODO


class FixedAttackerDQN(FixedAttackerMixin, DQN):
    pass
