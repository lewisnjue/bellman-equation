import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter

from .agent import Agent, ENV_NAME, TEST_EPISODES


def run(algorithm: str = "v", random_steps: int = 100, test_episodes: int = TEST_EPISODES, target_reward: float = 0.8) -> None:
    """Run the FrozenLake Bellman experiment for the requested algorithm."""
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment=f"-{algorithm}-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(random_steps)
        if algorithm == "q":
            agent.q_iteration()
        else:
            agent.value_iteration()

        reward = 0.0
        for _ in range(test_episodes):
            reward += agent.play_episode(test_env)
        reward /= test_episodes

        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"{iter_no}: Best reward updated {best_reward:.3} -> {reward:.3}")
            best_reward = reward

        if reward > target_reward:
            print("Solved in %d iterations!" % iter_no)
            break

    writer.close()
