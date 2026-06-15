import gymnasium as gym

from bellman_equation.agent import Agent


def test_agent_initializes_with_environment() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = Agent(env=env)

    assert agent.env is env
    assert agent.state is not None
    assert env.action_space.n == 4


def test_random_steps_record_transitions() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = Agent(env=env)
    agent.play_n_random_steps(10)

    assert sum(sum(counter.values()) for counter in agent.transits.values()) == 10
    assert len(agent.rewards) >= 1


def test_calc_action_value_returns_zero_for_unknown_transition() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = Agent(env=env)

    assert agent.calc_action_value(0, 0) == 0.0


def test_select_action_returns_valid_action() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = Agent(env=env)

    assert 0 <= agent.select_action(0) < env.action_space.n
