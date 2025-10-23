# RL course David Silver
The aim of this repository is to contain the code to perform the deliverable of the RL course from David Silver, the Youtube link with the lectures are here: https://youtu.be/2pWv7GOvuf0?si=D0NNLCguOgpTNqgl

## Easy21 Assignment

- `env.py`: Implements the Easy21 environment (`Easy21Env`) with `reset`, `step`, `render`, and `seed`.
- `mc_control.py`: Tabular Monte Carlo control with epsilon-greedy policy using `N0/(N0+N(s))` exploration.
- `main.py`: Trains MC control and plots the learned value function `V(s)`.
- `sarsa_lambda.py`: SARSA(λ) with linear function approximation for control (36-dim binary features as in the assignment).
- `main_sarsa.py`: Trains SARSA(λ) FA and plots the learned value function `V(s)`.

### Run

```
python main.py
```

By default it trains for 200k episodes and shows a 3D surface plot of `V(s)`.

For the third exercise (SARSA(λ) with function approximation):

```
python main_sarsa.py
```

Defaults: episodes=200k, α=0.01, λ=0.5, ε=0.05.

### Notes

- Actions: `0` = stick, `1` = hit. Strings `'stick'/'hit'` also accepted.
- State: `(dealer_showing in [1..10], player_sum in [1..21])`.
