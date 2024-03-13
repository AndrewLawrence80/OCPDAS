import torch
import torch.nn as nn
from model import LSTMQNet
from replay_buffer import ReplayBuffer
import numpy as np


class DoubleDQNAgent:
    def __init__(self, num_actions) -> None:
        self.num_actions = num_actions
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.policy_net = LSTMQNet(1, 64, num_actions, 1).to(self.device)
        self.target_net = LSTMQNet(1, 64, num_actions, 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.is_train = False

    def update(self, learning_rate: float, replay_buffer: ReplayBuffer, batch_size: int, discount_factor: float, tau) -> float:
        for g in self.optimizer.param_groups:
            g["lr"] = learning_rate

        batch = replay_buffer.sample(batch_size)
        batch_state, batch_action, batch_next_state, batch_next_action, batch_reward, batch_is_terminated, batch_is_truncated = zip(*batch)

        mask_batch_next_state_is_not_none = [next_state is not None for next_state in batch_next_state]
        batch_not_none_next_state = [next_state for next_state in batch_next_state if next_state is not None]

        batch_state = torch.tensor(batch_state, dtype=torch.float32, device=self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.int64, device=self.device).reshape((-1, 1))
        batch_not_none_next_state = torch.tensor(batch_not_none_next_state, dtype=torch.float32, device=self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device).reshape((-1, 1))
        batch_is_terminated = torch.tensor(batch_is_terminated, dtype=torch.bool, device=self.device).reshape((-1, 1))
        batch_is_truncated = torch.tensor(batch_is_truncated, dtype=torch.bool, device=self.device).reshape((-1, 1))
        mask_batch_next_state_is_not_none = torch.tensor(mask_batch_next_state_is_not_none, dtype=torch.bool, device=self.device)

        q_state_action = self.policy_net(batch_state).gather(1, batch_action)
        q_next_state_action = torch.zeros(q_state_action.size(), device=self.device)
        with torch.no_grad():
            batch_max_action = self.policy_net(batch_not_none_next_state).max(1)[1].reshape((-1, 1))
            q_next_state_action[mask_batch_next_state_is_not_none] = self.target_net(batch_not_none_next_state).gather(1, batch_max_action)
        q_target = batch_reward+discount_factor*q_next_state_action

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_state_action, q_target)
        loss.backward()
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key]*(1-tau)
        self.target_net.load_state_dict(target_net_state_dict)

        return loss.item()

    def take_action(self, state, epsilon: float = 0) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        q_actions = self.policy_net(state_tensor).detach().cpu().numpy()
        index_action_with_max_q = np.argmax(q_actions)
        action = None
        if self.is_train:
            probability = np.ones(self.num_actions)*epsilon/self.num_actions
            probability[index_action_with_max_q] = 1-np.sum(probability[1:])
            action = np.random.choice(np.arange(self.num_actions), p=probability)
        else:
            action = index_action_with_max_q
        return action

    def set_is_train(self, is_train: bool) -> None:
        self.is_train = is_train

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path))
