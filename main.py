import numpy as np
import matplotlib.pyplot as plt

from env import Easy21Env
from mc_control import MonteCarloControl


def plot_value_function(V: np.ndarray) -> None:
    """Plot 3D surface of V(dealer, player)."""
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
    ax.set_title("Easy21 Value Function (MC Control, greedy policy)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    episodes = 200000
    seed = 0

    env = Easy21Env(seed_value=seed)
    mcc = MonteCarloControl(env, N0=100)
    print(f"Training MC control for {episodes} episodes...")
    mcc.train(episodes=episodes, seed=seed)
    V = mcc.value_function()
    print("Training complete. Plotting value function...")
    plot_value_function(V)


if __name__ == "__main__":
    main()

