"""
Easy21 environment (David Silver's RL course).

State: tuple(dealer_showing, player_sum)
  - dealer_showing in [1..10] is the dealer's first black card
  - player_sum in [1..21]

Actions:
  - HIT  (1) -> draw another card
  - STICK(0) -> stop; dealer plays to 17

Dynamics:
  - Cards are 1..10. Each draw is black with prob 2/3 (adds positively),
    red with prob 1/3 (adds negatively). Initial cards are black.
  - If player's sum goes <1 or >21: player busts -> reward -1, terminal.
  - If player sticks: dealer draws until sum >= 17 (with same red/black rules).
    Dealer busts -> reward +1. Otherwise compare sums: higher wins (+1/-1), tie 0.

This is a minimal, self-contained environment with a Gym-like API:
  - reset() -> state
  - step(action) -> (state, reward, done)
  - render() -> human-readable state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import random


ActionType = Union[int, str]
StateType = Tuple[int, int]



@dataclass
class Easy21Env:
    """Easy21 environment implementation."""

    seed_value: Optional[int] = None

    # Action constants
    STICK: int = 0
    HIT: int = 1

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed_value)
        self._dealer_showing: Optional[int] = None
        self._dealer_sum: Optional[int] = None
        self._player_sum: Optional[int] = None
        self._done: bool = True

    # --- Public API ---
    def seed(self, seed: Optional[int]) -> None:
        """Set the RNG seed for reproducibility."""
        self.seed_value = seed
        self.rng.seed(seed)

    def reset(self) -> StateType:
        """Start a new episode and return the initial state."""
        self._dealer_showing = self._draw_black_card()
        self._dealer_sum = self._dealer_showing
        self._player_sum = self._draw_black_card()
        self._done = False
        return (self._dealer_showing, self._player_sum)

    def step(self, action: ActionType) -> Tuple[StateType, int, bool]:
        """Apply an action and return (state, reward, done)."""
        if self._done:
            # Allow stepping only after reset
            raise RuntimeError("Call reset() before step(); episode is done.")

        a = self._normalize_action(action)
        assert a in (self.STICK, self.HIT), "Invalid action"

        if a == self.HIT:
            # Player draws
            self._player_sum = self._player_sum + self._draw_card()
            if self._player_sum < 1 or self._player_sum > 21:
                # Player busts
                self._done = True
                return (self._dealer_showing, self._player_sum), -1, True
            # Otherwise continue
            return (self._dealer_showing, self._player_sum), 0, False

        # STICK: dealer plays
        self._done = True
        while True:
            # Dealer draws until sum >= 17 or busts
            if self._dealer_sum is None:
                raise RuntimeError("Environment not initialized; call reset().")
            if self._dealer_sum < 1 or self._dealer_sum > 21:
                # Shouldn't happen before first draw, but keep for safety
                return (self._dealer_showing, self._player_sum), +1, True
            if self._dealer_sum >= 17:
                break
            self._dealer_sum += self._draw_card()
            if self._dealer_sum < 1 or self._dealer_sum > 21:
                # Dealer busts -> player wins
                return (self._dealer_showing, self._player_sum), +1, True

        # No busts: compare sums
        if self._player_sum > self._dealer_sum:
            return (self._dealer_showing, self._player_sum), +1, True
        if self._player_sum < self._dealer_sum:
            return (self._dealer_showing, self._player_sum), -1, True
        return (self._dealer_showing, self._player_sum), 0, True

    def render(self) -> None:
        """Print a human-readable view of the current state."""
        if self._dealer_showing is None or self._player_sum is None:
            print("Environment not initialized. Call reset().")
            return
        status = "(terminal)" if self._done else ""
        print(
            f"Dealer shows: {self._dealer_showing} | "
            f"Player sum: {self._player_sum} {status}"
        )

    def get_valid_actions(self) -> List[int]:
        """Return the list of valid actions for the current state."""
        # Both actions are valid while not terminal
        return [self.STICK, self.HIT] if not self._done else []

    # --- Helpers ---
    def _draw_card(self) -> int:
        """Draw a card: 1..10, black with prob 2/3 (positive), red 1/3 (negative)."""
        value = self.rng.randint(1, 10)
        is_red = self.rng.random() < (1.0 / 3.0)
        return -value if is_red else value

    def _draw_black_card(self) -> int:
        """Draw an initial black card (positive 1..10)."""
        return self.rng.randint(1, 10)

    def _normalize_action(self, action: ActionType) -> int:
        if isinstance(action, str):
            action = action.strip().lower()
            if action == "stick":
                return self.STICK
            if action == "hit":
                return self.HIT
            raise ValueError("Action string must be 'hit' or 'stick'.")
        if action in (self.STICK, self.HIT):
            return int(action)
        raise ValueError("Action must be 0/1 or 'stick'/'hit'.")
        
