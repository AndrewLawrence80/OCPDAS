import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
import numpy as np
from typing import List, Dict, Any, Tuple, SupportsFloat, Optional


class WorldCupEnv(gym.Env):
    def __init__(self, workload_diff: np.ndarray, candidate_cpds: List, n_lookback: int, n_predict: int) -> None:
        super().__init__()
        self.idx = 0
        self.workload_diff = workload_diff
        self.candidate_cpds = np.array(candidate_cpds)
        self.n_lookback = n_lookback
        self.n_predict = n_predict
        self.len_observation = n_lookback+n_predict-1

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.len_observation)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        self.idx = 0
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = self._get_reward(action)
        self.idx += 1
        is_terminated = self.idx == len(self.workload_diff)-self.len_observation+1
        return self._get_observation(), reward, is_terminated, False, {}

    def _get_reward(self, action: int):
        reward = 0.0
        future_candidate_idx = np.sum(self.candidate_cpds < self.idx+self.n_lookback+self.n_predict)
        if future_candidate_idx < len(self.candidate_cpds):
            nearest_cpd = self.candidate_cpds[future_candidate_idx]
            if action == 0:
                reward = -np.tanh(max(0, self.idx+self.n_lookback+self.n_predict+self.n_predict-nearest_cpd))
            if action == 1:
                reward = -np.tanh(np.abs(self.idx+self.n_lookback+self.n_predict+self.n_predict//2-nearest_cpd))
        else:
            if action == 0:
                reward = 0.0
            else:
                reward = -1.0
        return reward

    def _get_observation(self):
        return self.workload_diff[self.idx:self.idx+self.len_observation]
