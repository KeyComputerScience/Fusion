"""Three application actions, separate from the coordinator's 30 profile pairs."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn


def tensor(values):
    return torch.tensor(np.asarray(values).tolist(), dtype=torch.float32)


class Network(nn.Module):
    def __init__(self, dimension, hidden=(128, 128), actor_critic=False):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, hidden[0]), nn.ReLU(),
                                     nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.actor = nn.Linear(hidden[1], 3)
        self.critic = nn.Linear(hidden[1], 1) if actor_critic else None

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        scores = self.actor(hidden)
        return (scores, self.critic(hidden).squeeze(-1)) if self.critic is not None else scores


class ServiceAgent:
    def __init__(self, kind, configuration, seed, dimension=14):
        self.kind, self.config = kind, configuration
        torch.set_num_threads(configuration["threads"])
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        self.network = Network(dimension, hidden=configuration["hidden"], actor_critic=kind == "ppo")
        self.target = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configuration["learning_rate"])
        self.rng = random.Random(seed + 701)
        self.replay, self.rollout = [], []
        self.version, self.updates, self.new_transitions = 0, 0, 0

    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            scores = self.network(tensor([state]))
            if self.kind == "dqn":
                action = self.rng.randrange(3) if self.rng.random() < epsilon else int(scores.argmax(dim=1).item())
                return action, 0.0, 0.0
            logits, value = scores
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()
            draw, cumulative, action = self.rng.random(), 0.0, 2
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if draw < cumulative:
                    action = index
                    break
            return action, float(np.log(max(probabilities[action], 1e-30))), float(value[0])

    def record(self, state, action, reward, next_state, done, log_probability=0.0, value=0.0):
        self.replay.append((list(state), action, reward, list(next_state), done))
        if len(self.replay) > self.config["replay_capacity"]:
            self.replay.pop(0)
        self.rollout.append((list(state), action, reward, list(next_state), done, log_probability, value))
        self.new_transitions += 1

    def train(self, steps):
        if self.kind == "dqn":
            if len(self.replay) < self.config["batch_size"]:
                return 0
            for _ in range(steps):
                batch = self.rng.sample(self.replay, self.config["batch_size"])
                states, actions, rewards, after, terminal = zip(*batch)
                predicted = self.network(tensor(states)).gather(1, torch.tensor(actions).reshape(-1, 1)).squeeze(1)
                with torch.no_grad():
                    targets = tensor(rewards) + self.config["discount"] * (1 - tensor(terminal)) * self.target(tensor(after)).max(1).values
                self._step(torch.nn.functional.smooth_l1_loss(predicted, targets))
                if self.updates % self.config["target_update"] == 0:
                    self.target.load_state_dict(self.network.state_dict())
            return steps
        if not self.rollout:
            return 0
        batch = self.rollout[-512:]
        states, actions, rewards, after, terminal, logged, values = zip(*batch)
        with torch.no_grad():
            next_values = self.network(tensor(after))[1].tolist()
        advantage, tail = [], 0.0
        for reward, current, future, done in reversed(list(zip(rewards, values, next_values, terminal))):
            residual = reward + self.config["discount"] * future * (1 - done) - current
            tail = residual + self.config["discount"] * self.config["ppo_gae"] * (1 - done) * tail
            advantage.append(tail)
        advantage = np.asarray(advantage[::-1])
        returns = tensor(advantage + np.asarray(values))
        normalized = tensor((advantage - advantage.mean()) / (advantage.std() + 1e-8))
        actions_t, logged_t = torch.tensor(actions), tensor(logged)
        for _ in range(steps):
            logits, predicted = self.network(tensor(states))
            distribution = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(distribution.log_prob(actions_t) - logged_t)
            clipped = ratio.clamp(1 - self.config["ppo_clip"], 1 + self.config["ppo_clip"])
            loss = (-torch.minimum(ratio * normalized, clipped * normalized).mean()
                    + 0.5 * (predicted - returns).square().mean()
                    - self.config["ppo_entropy"] * distribution.entropy().mean())
            self._step(loss)
        self.rollout = []
        return steps

    def _step(self, loss):
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite service-policy training loss")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config["gradient_clip"])
        self.optimizer.step()
        self.updates += 1

    def worker(self, steps):
        worker = copy.copy(self)
        worker.network, worker.target = copy.deepcopy(self.network), copy.deepcopy(self.target)
        worker.optimizer = torch.optim.Adam(worker.network.parameters(), lr=self.config["learning_rate"])
        worker.optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        worker.rng = random.Random()
        worker.rng.setstate(self.rng.getstate())
        worker.replay, worker.rollout = list(self.replay), list(self.rollout)
        executed = worker.train(steps)
        return worker, executed

    def deploy(self, worker):
        self.network.load_state_dict(worker.network.state_dict())
        self.target.load_state_dict(worker.target.state_dict())
        self.optimizer.load_state_dict(worker.optimizer.state_dict())
        self.updates = worker.updates
        self.version += 1
        self.new_transitions = 0
        self.rollout = []

    def parameter_digest(self):
        import hashlib
        payload = b"".join(p.detach().cpu().contiguous().numpy().tobytes() for p in self.network.parameters())
        return hashlib.sha256(payload).hexdigest()

    def checkpoint(self):
        return {"kind": self.kind, "version": self.version, "updates": self.updates,
                "network": self.network.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "replay": self.replay,
                "rollout": self.rollout, "rng_state": self.rng.getstate(),
                "new_transitions": self.new_transitions}
