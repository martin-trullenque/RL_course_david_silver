import numpy as np
import matplotlib.pyplot as plt

from env import Easy21Env
from sarsa_lambda import SarsaLambdaFA


def plot_value_function(V: np.ndarray) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dealer = np.arange(1, 11)
    player = np.arange(1, 22)
    D, P = np.meshgrid(dealer, player, indexing="ij")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(D, P, V, cmap="viridis")
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("V(s)")
    ax.set_title("Easy21 Value Function (SARSA(λ) FA, greedy policy)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    episodes = 200000
    seed = 0

    env = Easy21Env(seed_value=seed)
    agent = SarsaLambdaFA(env, alpha=0.01, lam=0.5, epsilon=0.05, gamma=1.0)
    print(f"Training SARSA(λ) FA for {episodes} episodes...")
    agent.train(episodes=episodes, seed=seed)
    V = agent.value_function()
    print("Training complete. Plotting value function...")
    plot_value_function(V)


if __name__ == "__main__":
    main()

