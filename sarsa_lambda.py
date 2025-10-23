from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

from env import Easy21Env


State = Tuple[int, int]


def _feature_vector(state: State, action: int) -> np.ndarray:
    """Binary feature vector x(s,a) with 36 dims (3x6x2) per Easy21 assignment.

    Dealer intervals (inclusive): [1-4], [4-7], [7-10]
    Player intervals (inclusive): [1-6], [4-9], [7-12], [10-15], [13-18], [16-21]
    Actions: 0=stick, 1=hit
    """
    dealer, player = state

    dealer_bins = [(1, 4), (4, 7), (7, 10)]
    player_bins = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]

    x = np.zeros(3 * 6 * 2, dtype=float)

    # Indices: idx = a*18 + di*6 + pj
    for di, (dl, dh) in enumerate(dealer_bins):
        if dealer < dl or dealer > dh:
            continue
        for pj, (pl, ph) in enumerate(player_bins):
            if player < pl or player > ph:
                continue
            idx = action * 18 + di * 6 + pj
            x[idx] = 1.0
    return x


@dataclass
class SarsaLambdaFA:
    """SARSA(λ) with linear function approximation for Easy21.

    Q(s,a) = w^T x(s,a), with binary features per the assignment.
    Policy: epsilon-greedy over current Q.
    """

    env: Easy21Env
    alpha: float = 0.01
    lam: float = 0.5
    epsilon: float = 0.05
    gamma: float = 1.0

    def __post_init__(self) -> None:
        self.num_features = 36
        self.weights = np.zeros(self.num_features, dtype=float)

    def q_value(self, state: State, action: int) -> float:
        x = _feature_vector(state, action)
        return float(np.dot(self.weights, x))

    def greedy_action(self, state: State) -> int:
        q0 = self.q_value(state, 0)
        q1 = self.q_value(state, 1)
        return int(q1 > q0)

    def policy(self, state: State) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        return self.greedy_action(state)

    def train(self, episodes: int, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            self.env.seed(seed)

        for _ in range(episodes):
            state = self.env.reset()
            action = self.policy(state)
            e = np.zeros_like(self.weights)

            done = False
            while not done:
                next_state, reward, done = self.env.step(action)
                x_sa = _feature_vector(state, action)

                if done:
                    td_target = reward
                else:
                    next_action = self.policy(next_state)
                    td_target = reward + self.gamma * self.q_value(next_state, next_action)

                td_error = td_target - float(np.dot(self.weights, x_sa))

                # Accumulate trace and update weights
                e = self.gamma * self.lam * e + x_sa
                self.weights += self.alpha * td_error * e

                if done:
                    break

                state = next_state
                action = next_action

    def value_function(self) -> np.ndarray:
        """Return V(s)=max_a Q(s,a) for s in [10,21]."""
        V = np.zeros((10, 21), dtype=float)
        for d in range(1, 11):
            for p in range(1, 22):
                s = (d, p)
                V[d - 1, p - 1] = max(self.q_value(s, 0), self.q_value(s, 1))
        return V

    def greedy_policy_table(self) -> np.ndarray:
        """Return greedy policy over actions as a [10,21] table with values 0/1."""
        Pi = np.zeros((10, 21), dtype=int)
        for d in range(1, 11):
            for p in range(1, 22):
                s = (d, p)
                Pi[d - 1, p - 1] = self.greedy_action(s)
        return Pi

