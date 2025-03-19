import copy
import dataclasses
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import buffer  # Replay buffer from original MR.Q
import utils  # Utility functions

@dataclasses.dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_freq: int = 250

    # Exploration
    buffer_size_before_training: int = 10e3
    exploration_noise: float = 0.2

    # Q-Learning Horizon
    enc_horizon: int = 5
    Q_horizon: int = 3

    # Encoder Model
    zs_dim: int = 512
    za_dim: int = 256
    zsa_dim: int = 512
    enc_hdim: int = 512
    enc_activ: str = 'elu'
    enc_lr: float = 1e-4
    enc_wd: float = 1e-4

    # Value Model
    value_hdim: int = 512
    value_activ: str = 'elu'
    value_lr: float = 3e-4
    value_wd: float = 1e-4
    value_grad_clip: float = 20

    # Policy Model
    policy_hdim: int = 512
    policy_activ: str = 'relu'
    policy_lr: float = 3e-4
    policy_wd: float = 1e-4
    gumbel_tau: float = 10
    pre_activ_weight: float = 1e-5

    def __post_init__(self):
        utils.enforce_dataclass_type(self)


class MRQStockAgent:
    def __init__(self, obs_shape: tuple, action_dim: int, device: torch.device, history: int=1, hp: Dict={}):
        self.name = 'MR.Q Stock Trader'

        self.hp = Hyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)
        self.device = device

        # Initialize Replay Buffer
        self.replay_buffer = buffer.ReplayBuffer(
            obs_shape, action_dim, max_action=1.0, pixel_obs=False, device=self.device,
            history=history, max_size=self.hp.buffer_size, batch_size=self.hp.batch_size
        )

        # Encoder: Now processes stock data, not images
        self.encoder = StockEncoder(obs_shape[0] * history, action_dim,
            self.hp.zs_dim, self.hp.za_dim, self.hp.zsa_dim, self.hp.enc_hdim, self.hp.enc_activ).to(self.device)
        self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr=self.hp.enc_lr, weight_decay=self.hp.enc_wd)
        self.encoder_target = copy.deepcopy(self.encoder)

        # Policy & Value Networks
        self.policy = Policy(action_dim, self.hp.gumbel_tau, self.hp.zs_dim, self.hp.policy_hdim, self.hp.policy_activ).to(self.device)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.hp.policy_lr, weight_decay=self.hp.policy_wd)
        self.policy_target = copy.deepcopy(self.policy)

        self.value = Value(self.hp.zsa_dim, self.hp.value_hdim, self.hp.value_activ).to(self.device)
        self.value_optimizer = optim.AdamW(self.value.parameters(), lr=self.hp.value_lr, weight_decay=self.hp.value_wd)
        self.value_target = copy.deepcopy(self.value)

        # Tracked values
        self.reward_scale = 1
        self.training_steps = 0

    def select_action(self, state: np.array, use_exploration: bool=True):
        if self.replay_buffer.size < self.hp.buffer_size_before_training and use_exploration:
            return np.random.choice([0, 1, 2])  # Random action (Buy, Sell, Hold)

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(-1, *self.replay_buffer.state_shape)
            zs = self.encoder.zs(state)
            action_probs = self.policy.act(zs)
            if use_exploration:
                action_probs += torch.randn_like(action_probs) * self.hp.exploration_noise
            return int(action_probs.argmax())

    def train(self):
        if self.replay_buffer.size <= self.hp.buffer_size_before_training:
            return

        self.training_steps += 1

        # Target Network Updates
        if (self.training_steps-1) % self.hp.target_update_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())

        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.hp.Q_horizon)

        Q, Q_target = self.train_rl(state, action, next_state, reward, not_done)

        # Update Priorities
        priority = (Q - Q_target.expand(-1,2)).abs().max(1).values
        self.replay_buffer.update_priority(priority)

    def train_rl(self, state, action, next_state, reward, not_done):
        with torch.no_grad():
            next_zs = self.encoder_target.zs(next_state)
            next_action = self.policy_target.act(next_zs)
            next_zsa = self.encoder_target(next_zs, next_action)
            Q_target = self.value_target(next_zsa).min(1, keepdim=True).values
            Q_target = (reward + not_done * self.hp.discount * Q_target)

        zs = self.encoder.zs(state)
        zsa = self.encoder(zs, action)

        Q = self.value(zsa)
        value_loss = F.smooth_l1_loss(Q, Q_target.expand(-1,2))

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.hp.value_grad_clip)
        self.value_optimizer.step()

        return Q, Q_target


class StockEncoder(nn.Module):
    """Modified MRQ Encoder for Stock Trading Data"""
    def __init__(self, state_dim, action_dim, zs_dim, za_dim, zsa_dim, hdim, activ):
        super().__init__()
        self.zs_mlp = nn.Sequential(
            nn.Linear(state_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, zs_dim)
        )
        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = nn.Sequential(
            nn.Linear(zs_dim + za_dim, zsa_dim),
            nn.ReLU(),
            nn.Linear(zsa_dim, zsa_dim)
        )
        self.model = nn.Linear(zsa_dim, zs_dim)

    def forward(self, zs, action):
        za = F.relu(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))

    def zs(self, state):
        return F.relu(self.zs_mlp(state))


class Policy(nn.Module):
    def __init__(self, action_dim, gumbel_tau, zs_dim, hdim, activ):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(zs_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, zs):
        return self.policy(zs)


class Value(nn.Module):
    def __init__(self, zsa_dim, hdim, activ):
        super().__init__()
        self.q1 = nn.Linear(zsa_dim, hdim)
        self.q2 = nn.Linear(hdim, 1)

    def forward(self, zsa):
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)
