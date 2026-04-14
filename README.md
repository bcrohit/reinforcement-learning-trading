# Reinforcement learning trading

This project trains a simple policy-gradient agent (Monte Carlo REINFORCE) to trade in a Gymnasium environment. The market simulator uses `gym-trading-env` with OHLC-style data (for example DAX); the agent chooses short, flat, or long positions while fees and borrow costs are modeled in the environment.

**Train one agent:** run `python train.py` (see `python train.py --help` for episodes, data limits, and other flags).

**Compare variants:** run `python compare_agents.py` to train agents with and without a reward baseline and compare them to a random policy, including basic plots and statistics.

Code lives under `src/` (agent, models, data loading, visualization). Dependencies are listed in `requirements.txt`.
