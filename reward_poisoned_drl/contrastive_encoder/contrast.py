"""
Adapted from: https://github.com/MishaLaskin/curl/blob/master/curl_sac.py
"""

import torch
import torch.nn as nn

from reward_poisoned_drl.contrastive_encoder.encoder import PixelEncoder
from reward_poisoned_drl.utils import soft_update_params


class ContrastiveTrainer:

    def __init__(
        self,
        device,
        encoder_lr=1e-3,
        encoder_tau=0.001,
        key_encoder_update_freq=2,
        log_interval=100
    ):
        self.device = device
        self.encoder_tau = encoder_tau
        self.key_encoder_update_freq = key_encoder_update_freq
        self.log_interval = log_interval

        self.query_enc = PixelEncoder().to(device)
        self.key_enc = PixelEncoder().to(device)
        self.W = torch.rand((50, 50), requires_grad=True, device=device)

        self.opt = torch.optim.Adam(
            [self.W, *self.query_enc.parameters()], lr=encoder_lr
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()

    def train(self):
        self.query_enc.train()
        self.key_enc.train()

    def eval(self):
        self.query_enc.eval()
        self.key_enc.eval()

    def state_dict(self):
        return {
            "bilinear.weight": self.W, 
            "query_enc": self.query_enc.state_dict(),
            "key_enc": self.key_enc.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.W.data = state_dict["bilinear.weight"].data
        self.query_enc.load_state_dict(state_dict["query_enc"])
        self.key_enc.load_state_dict(state_dict["key_enc"])

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]  # subtract max for stability
        return logits

    def update(self, obs_anchor, obs_pos, step):
        """
        Take one gradient step for query encoder and 
        momentum update for key encoder.
        """
        z_a = self.query_enc(obs_anchor)
        z_pos = self.key_enc(obs_pos).detach()  # detach pos encodings
        
        logits = self.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if step % self.key_encoder_update_freq == 0:
            soft_update_params(self.query_enc, self.key_enc, self.encoder_tau)

        return loss.item()

    @torch.no_grad()
    def test(self, obs_anchor, obs_pos):
        """Get loss without updating."""
        z_a = self.query_enc(obs_anchor)
        z_pos = self.key_enc(obs_pos)
        
        logits = self.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        
        return self.cross_entropy_loss(logits, labels).item()
