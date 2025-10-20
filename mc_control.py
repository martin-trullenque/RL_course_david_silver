from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from env import Easy21Env


State = Tuple[int, int]


def _state_to_idx(state: State) -> Tuple[int, int]:
    dealer, player = state
    if dealer < 1 or dealer > 10 or player < 1 or player > 21:
        raise ValueError("State out of bounds for Easy21")
    return dealer - 1, player - 1


@dataclass
class MonteCarloControl:
    """Tabular MC control with epsilon-greedy policy (David Silver Easy21).

    Uses incremental first-visit Monte Carlo with epsilon(s) = N0 / (N0 + N(s)).
    Q is a table of size [10, 21, 2].
    """

    env: Easy21Env
    N0: int = 100

    def __post_init__(self) -> None:
        self.num_dealer = 10
        self.num_player = 21
        self.num_actions = 2
        self.Q = np.zeros((self.num_dealer, self.num_player, self.num_actions), dtype=float)
        self.N_sa = np.zeros_like(self.Q, dtype=int)
        self.N_s = np.zeros((self.num_dealer, self.num_player), dtype=int)

    def _epsilon(self, s_idx: Tuple[int, int]) -> float:
        n_s = self.N_s[s_idx]
        return self.N0 / (self.N0 + n_s)

    def _choose_action(self, s_idx: Tuple[int, int]) -> int:
        eps = self._epsilon(s_idx)
        if np.random.random() < eps:
            return np.random.randint(0, self.num_actions)
        return int(np.argmax(self.Q[s_idx[0], s_idx[1], :]))

    def run_episode(self) -> Tuple[List[State], List[int], int]:
        """Generate an episode; returns (states, actions, reward)."""
        states: List[State] = []
        actions: List[int] = []

        s = self.env.reset()
        done = False
        while not done:
            s_idx = _state_to_idx(s)
            # Count every state visit for epsilon(s)
            self.N_s[s_idx] += 1
            a = self._choose_action(s_idx)
            states.append(s)
            actions.append(a)
            s, r, done = self.env.step(a)

        # Terminal reward r applies to all steps (intermediate rewards are 0)
        return states, actions, r

    def train(self, episodes: int, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            self.env.seed(seed)

        for _ in range(episodes):
            states, actions, G = self.run_episode()

            # First-visit MC: track which (s,a) seen in this episode
            visited = set()
            for s, a in zip(states, actions):
                key = (s, a)
                if key in visited:
                    continue
                visited.add(key)

                di, pj = _state_to_idx(s)
                self.N_sa[di, pj, a] += 1
                self.N_s[di, pj] += 1

                # Incremental average
                n = self.N_sa[di, pj, a]
                q = self.Q[di, pj, a]
                self.Q[di, pj, a] = q + (G - q) / n

    def greedy_policy(self) -> np.ndarray:
        """Return greedy policy over actions [10,21] with values 0/1 (stick/hit)."""
        return np.argmax(self.Q, axis=2)

    def value_function(self) -> np.ndarray:
        """Return V(s)=max_a Q(s,a) as shape [10,21]."""
        return np.max(self.Q, axis=2)
