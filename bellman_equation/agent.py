import typing as tt
from collections import Counter, defaultdict

import gymnasium as gym

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

State = int
Action = int
RewardKey = tt.Tuple[State, Action, State]
TransitKey = tt.Tuple[State, Action]


class Agent:
    """Model-based agent for FrozenLake with Bellman updates."""

    def __init__(self, env: tt.Optional[gym.Env] = None) -> None:
        self.env = env or gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.rewards: tt.Dict[RewardKey, float] = defaultdict(float)
        self.transits: tt.Dict[TransitKey, Counter] = defaultdict(Counter)
        self.state_values: tt.Dict[State, float] = defaultdict(float)
        self.action_values: tt.Dict[TransitKey, float] = defaultdict(float)

    def play_n_random_steps(self, n: int) -> None:
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, is_done, is_trunc, _ = self.env.step(action)
            self._record_transition(self.state, action, new_state, float(reward))
            if is_done or is_trunc:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def _record_transition(self, state: State, action: Action, new_state: State, reward: float) -> None:
        self.rewards[(state, action, new_state)] = reward
        self.transits[(state, action)][new_state] += 1

    def calc_action_value(self, state: State, action: Action) -> float:
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        if total == 0:
            return 0.0

        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.state_values[tgt_state])
        return action_value

    def select_action(self, state: State) -> Action:
        best_action, best_value = None, None
        has_action_values = any(self.action_values.values())

        for action in range(self.env.action_space.n):
            if has_action_values:
                action_value = self.action_values[(state, action)]
            else:
                action_value = self.calc_action_value(state, action)

            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action if best_action is not None else self.env.action_space.sample()

    def play_episode(self, env: tt.Optional[gym.Env] = None) -> float:
        env = env or self.env
        total_reward = 0.0
        state, _ = env.reset()

        while True:
            action = self.select_action(state)
            new_state, reward, is_done, is_trunc, _ = env.step(action)
            self._record_transition(state, action, new_state, float(reward))
            total_reward += reward
            if is_done or is_trunc:
                break
            state = new_state

        return total_reward

    def value_iteration(self) -> None:
        for state in range(self.env.observation_space.n):
            action_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.state_values[state] = max(action_values)

    def q_iteration(self) -> None:
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                if total == 0:
                    continue

                action_value = 0.0
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_future_action = max(
                        range(self.env.action_space.n),
                        key=lambda a: self.action_values[(tgt_state, a)],
                    )
                    future_value = self.action_values[(tgt_state, best_future_action)]
                    action_value += (count / total) * (reward + GAMMA * future_value)

                self.action_values[(state, action)] = action_value
