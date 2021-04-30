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

import os
import torch
import pickle
import numpy as np
import copy

from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel

from reward_poisoned_drl.contrastive_encoder.contrast import Contrastor
from reward_poisoned_drl.utils import semantic_crop_pong, PONG_ACT_MAP


class FixedAttackerMixin:

    def __init__(
            self,
            target_obs, 
            target_info,
            contrast_sd_path, 
            dqn_oracle_sd_path, 
            delta_bound=1.0, 
            first_poison_itr=1, 
            **kwargs
            ):
        """
        target_obs: np.ndarray of target observations (keys for contrastive encoder) w/ shape (N, C, H, W)
        contrast_sd_path: path to contrastive encoder state dictionary of pre-trained parameters
        dqn_oracle_sd_path: path to oracle DQN model state dictionary of pre-trained parameters (what agent wants to learn)
        delta_bound: scales magnitude of fixed perturbation (i.e. attacker "power")
        first_poison_itr: what optimization iteration to start poioning rewards (must be >= 1)
        """
        super().__init__(**kwargs)
        self.target_obs = torch.as_tensor(target_obs)
        self.target_info = target_info
        self.contrast_sd_path = contrast_sd_path
        self.dqn_oracle_sd_path = dqn_oracle_sd_path
        self.delta_bound = delta_bound
        
        assert first_poison_itr >= 1  # can't poison zeroth iteration, need last_samples
        self.first_poison_itr = first_poison_itr

        self.opt_itr = None  # for storing current itr to know when to start poisoning
        self.last_samples = None  # for online poisoning with access to next state

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
        super().initialize(agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=world_size, rank=rank)
        self.device = agent.device
        self.target_obs = self.target_obs.to(self.device)
        self.action_type_np = examples["action"].numpy().dtype  # for convenience in _find_targets

        if batch_spec.T > 1:
            raise NotImplementedError("Expecting sampler to generate a single step per env per sampling iteration")
        
        contrast_sd = torch.load(self.contrast_sd_path, map_location=self.device)
        self.contrastive_model = Contrastor(contrast_sd, self.device)
        
        im_shape = agent.env_spaces.observation.shape
        action_size = agent.env_spaces.action.n
        
        self.dqn_oracle = AtariDqnModel(im_shape, action_size).to(self.device)
        dqn_oracle_sd = torch.load(self.dqn_oracle_sd_path, map_location=self.device)
        if "agent_state_dict" in dqn_oracle_sd.keys():  # extract DQN model state-dict if full runner state-dict
            dqn_oracle_sd = dqn_oracle_sd["agent_state_dict"]["model"]
        self.dqn_oracle.load_state_dict(dqn_oracle_sd)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        self.opt_itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        return super().optimize_agent(itr, samples=samples, sampler_itr=sampler_itr)

    def samples_to_buffer(self, samples):
        """
        Prepares samples for insertion
        to replay buffer. We insert poison
        rewards here, but from the last sample
        as we need the next observation to compute
        the attacker Q-function.

        So, we return the poisoned sample from the
        last sampler time step, which is fine since 
        we're just adding it to a replay buffer.
        """
        if self.opt_itr < self.first_poison_itr:
            self.last_samples = copy.deepcopy(samples)
            return super().samples_to_buffer(samples)

        last_obs, last_act, last_rew, last_done = self._unpack_samples(self.last_samples)
        curr_obs, _, _, _ = self._unpack_samples(samples)

        poisoned_last_rew = self._poison_rewards(
            last_obs.squeeze(),  # squeezing singleton T dim, as we assert T=1 in initialize
            last_act.squeeze(),
            last_rew.squeeze(),
            last_done.squeeze(),
            curr_obs.squeeze()
        ).unsqueeze(0)  # and add back singleton T dim at end

        self.last_samples = copy.deepcopy(samples)  # set this for next iteration

        return SamplesToBuffer(
            observation=last_obs,
            action=last_act,
            reward=poisoned_last_rew,  # poison inserted here
            done=last_done
        )

    def _unpack_samples(self, samples):
        """Unpack namedarraytuple of samples."""
        obs = samples.env.observation
        act = samples.agent.action
        rew = samples.env.reward
        done = samples.env.done
        return obs, act, rew, done

    def _center_crop_4px(self, obs):
        """
        Crop outer 4 pixels of image observation.
        Assumes observation shape (B, C, H, W).
        
        We use this pre-processing as this level
        of reduction is used to train the contrastive
        encoder, so the center crop makes the image
        shape match what is expected.
        """
        H, W = obs.shape[2:]
        return obs[:, :, 4:H-4, 4:W-4]

    def _get_oracle_q(self, obs):
        """
        Returns 'oracle' Q-values using pre-trained DQN
        with shape (B, |A|).
        """
        obs = obs.to(self.device)
        return self.dqn_oracle(obs, None, None).cpu()

    def _find_targets(self, obs):
        """
        Use contrastive encoder to determine
        which observations are targets.

        In the case where more than one target states matched, 
        i.e. the scores were greater than the thresholds at 
        several target indices, we take the target action from 
        the match with highest similarity score.

        We convert to numpy as older versions of torch do
        not have logical_and...

        Returns two vectors:
            is_target --> boolean with shape (B,)
            targ_action --> action-type with shape (B,)
        """
        obs = obs.to(self.device)

        queries = self._center_crop_4px(semantic_crop_pong(obs))  # (B, ...)
        keys = self._center_crop_4px(semantic_crop_pong(self.target_obs))  # (num_targets, ...)
        scores = self.contrastive_model(queries, keys).detach().cpu().numpy()

        is_target = np.zeros(len(scores), dtype=np.bool)
        targ_action = np.zeros(len(scores), dtype=self.action_type_np)
        best_score = np.full((len(scores),), -float(2 ** 100))

        for tidx in range(len(self.target_obs)): 
            thresh, targ_act = self.target_info[tidx]
            is_this_target = scores[:, tidx] >= thresh
            is_better_target = np.logical_and(is_this_target, scores[:, tidx] > best_score)
            
            is_target = np.logical_or(is_target, is_this_target)
            targ_action[is_better_target] = targ_act
            best_score[is_better_target] = scores[:, tidx][is_better_target]

        # convert back to torch Tensors
        is_target = torch.as_tensor(is_target)
        targ_action = torch.as_tensor(targ_action)

        return is_target, targ_action

    def _attacker_q(self, obs):
        """
        obs: state observation, image stack
        Computes the similarity of observations to target states
        and creates a new Q function Q', for reward poisoning
        returns Q'
        """
        # start with oracle Q-values from pre-trained DQN
        attacker_q_values = self._get_oracle_q(obs)  # (B, |A|)
        
        # find which obs are target matches and get their respective target actions
        is_target, targ_action = self._find_targets(obs)

        # find target-action and not-target-action indices for target observations
        num_obs, num_act = attacker_q_values.shape  # (B, |A|)
        count_mesh = torch.stack([torch.arange(num_act)] * num_obs, dim=0)
        act_mesh = torch.stack([targ_action] * num_act, dim=1)
        targ_act_indices = count_mesh == act_mesh
        not_targ_act_indices = torch.logical_not(targ_act_indices)

        # overwrite non-target observations to be NOT flagged for either perturbation
        not_target = torch.logical_not(is_target)
        targ_act_indices[not_target, :] = False
        not_targ_act_indices[not_target, :] = False

        # perturb oracle Q-values at target obs
        attacker_q_values[targ_act_indices] += self.delta_bound / (1 + self.discount)
        attacker_q_values[not_targ_act_indices] -= self.delta_bound / (1 + self.discount)

        return attacker_q_values
    
    @torch.no_grad()
    def _poison_rewards(self, obs, act, rew, done, next_obs) -> torch.Tensor:
        """
        obs: current state [B, 4, H, W]

        Implements the fixed attacker
        poisoning routine.
        """
        oracle_q = self._attacker_q(obs)
        next_oracle_q = self._attacker_q(next_obs)
        
        oracle_q_at_act = torch.gather(oracle_q, dim=-1, index=act.unsqueeze(-1)).squeeze()
        next_oracle_q_max = torch.max(next_oracle_q, dim=1)[0]
        not_done_mask = torch.logical_not(done).to(torch.float32)

        poisoned_rewards = oracle_q_at_act - self.discount * not_done_mask * next_oracle_q_max

        return poisoned_rewards


class FixedAttackerDQN(FixedAttackerMixin, DQN):
    pass
