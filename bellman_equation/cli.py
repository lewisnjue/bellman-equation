import argparse

from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Bellman value-iteration or Q-iteration on FrozenLake."
    )
    parser.add_argument(
        "--algorithm",
        choices=["v", "q"],
        default="v",
        help="Choose algorithm: 'v' for value iteration, 'q' for Q-value iteration.",
    )
    parser.add_argument(
        "--random-steps",
        type=int,
        default=100,
        help="Number of random exploratory steps between Bellman updates.",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=20,
        help="Number of episodes used to estimate average reward.",
    )
    parser.add_argument(
        "--target-reward",
        type=float,
        default=0.8,
        help="Reward threshold at which training stops.",
    )
    args = parser.parse_args()
    run(args.algorithm, args.random_steps, args.test_episodes, args.target_reward)
