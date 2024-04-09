import gymnasium as gym
import numpy as np
from collections import defaultdict
from replay_buffer import ReplayBuffer, Transition


class OffPolicyTrainer:
    def __init__(self, env: gym.Env, agent, num_episodes: int = int(1e2),
                 replay_buffer_size: int = 128, batch_size=32,
                 discount_factor: float = 0.9,
                 epsilon_start: float = 0.5, epsilon_end: float = 0.1, epsilon_step: int = 20,
                 learning_rate_start: float = 1e-3, learning_rate_end: float = 1e-4, learning_rate_step: int = 1e2,
                 tau=0.05, print_step=1) -> None:
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step = epsilon_step
        self.learning_rate_start = learning_rate_start
        self.leanring_rate_end = learning_rate_end
        self.leanring_rate_step = learning_rate_step
        self.tau = tau
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.print_step = print_step

    def _get_epsilon(self, step) -> float:
        return self.epsilon_end+(self.epsilon_start-self.epsilon_end)*np.exp(-1.0*step/self.epsilon_step)

    def _get_learning_rate(self, step) -> float:
        return self.leanring_rate_end+(self.learning_rate_start-self.leanring_rate_end)*np.exp(-1.0*step/self.leanring_rate_step)

    def train(self) -> None:
        reward_per_episode = defaultdict(float)
        num_attempts_per_episode = defaultdict(int)

        loss = 0.0
        for episode_i in range(self.num_episodes):
            self.agent.set_is_train(True)
            state = self.env.reset()[0]
            is_truncated = False
            is_terminated = False
            epsilon = self._get_epsilon(episode_i)
            learning_rate = self._get_learning_rate(episode_i)

            while not is_terminated:
                action = self.agent.take_action(state, epsilon)
                next_state, reward, is_terminated, is_truncated, info = self.env.step(action)
                if is_terminated or is_truncated:
                    next_state = None
                self.replay_buffer.append(Transition(state, action, next_state, None, reward, is_terminated, is_truncated))
                state = next_state

                if len(self.replay_buffer) > self.batch_size:
                    loss = self.agent.update(learning_rate, self.replay_buffer, self.batch_size, self.discount_factor, self.tau)

                reward_per_episode[episode_i] += reward
                num_attempts_per_episode[episode_i] += 1

            if episode_i % self.print_step == 0:
                print("episode %d, loss %f, num_attempts %d, reward %f" % (episode_i, loss, num_attempts_per_episode[episode_i], reward_per_episode[episode_i]))
            
        return num_attempts_per_episode, reward_per_episode

    def eval(self) -> None:
        action_list = []
        state_list = []
        self.agent.set_is_train(False)
        total_reward = 0.0
        state = self.env.reset()[0]
        is_truncated = False
        is_terminated = False
        while not is_truncated and not is_terminated:
            state_list.append(state)
            action = self.agent.take_action(state)
            action_list.append(action)
            next_state, reward, is_terminated, is_truncated, info = self.env.step(action)
            state = next_state
            total_reward += reward
        print("eval results: reward %f" % total_reward)
        return state_list, action_list

    def set_env(self,env)->None:
        self.env=env