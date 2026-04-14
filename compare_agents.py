"""
Script to train and compare agents with and without baseline.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_trading_env

from src.agent import MonteCarloAgent, RandomTradingAgent
from src.utils import load_trading_data
from src.visualize import plot_convergence_analysis, print_comparison_statistics, print_risk_metrics


def create_environment(df, data_limit=None, **env_kwargs):
    """Create trading environment."""
    df_subset = df[:data_limit] if data_limit else df
    
    env = gym.make(
        "TradingEnv",
        name="DAX",
        df=df_subset,
        positions=[-1, 0, 1],
        trading_fees=0.01/100,
        borrow_interest_rate=0.0003/100,
        verbose=0,
        **env_kwargs
    )
    
    state, _ = env.reset()
    state_dim = len(state) if isinstance(state, (list, np.ndarray)) else state.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"State sample: {state}")
    
    return env, state_dim, action_dim


def train_agent(agent, env, num_episodes, eval_interval=50):
    """Train an agent."""
    for episode in range(num_episodes):
        state, info = env.reset()
        done = truncated = False

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_reward(reward)
            state = next_state
        
        agent.update()
        
        if (episode + 1) % eval_interval == 0:
            avg_return = np.mean(agent.episode_returns[-eval_interval:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Avg Return (last {eval_interval}): {avg_return:.4f}")
    
    return agent


def main():
    # Load data
    print("Loading data...")
    df = load_trading_data("data/GlobalExchangeData/indexProcessed.csv", symbol="GDAXI")
    print(f"Loaded {len(df)} rows")
    
    # Create environment
    print("\nCreating environment...")
    env, state_dim, action_dim = create_environment(df, data_limit=1000)
    
    # Training parameters
    num_episodes = 500
    eval_interval = 50
    
    # Train agent WITHOUT baseline
    print("\n" + "="*60)
    print("Training REINFORCE agent (without baseline)...")
    print("="*60)
    agent_no_baseline = MonteCarloAgent(
        state_dim, action_dim, gamma=0.99, lr=1e-3, use_baseline=False
    )
    agent_no_baseline = train_agent(agent_no_baseline, env, num_episodes, eval_interval)
    
    # Train agent WITH baseline
    print("\n" + "="*60)
    print("Training REINFORCE agent (with baseline)...")
    print("="*60)
    agent_with_baseline = MonteCarloAgent(
        state_dim, action_dim, gamma=0.99, lr=1e-3, use_baseline=True
    )
    agent_with_baseline = train_agent(agent_with_baseline, env, num_episodes, eval_interval)
    
    # Evaluate Random Trading baseline strategy
    print("\n" + "="*60)
    print("Evaluating Random Trading baseline strategy...")
    print("="*60)
    random_trading = RandomTradingAgent(state_dim, action_dim)
    random_trading = train_agent(random_trading, env, num_episodes, eval_interval)
    
    # Analysis
    print("\n" + "="*60)
    print("Generating analysis...")
    print("="*60)
    
    print_comparison_statistics(agent_no_baseline, agent_with_baseline)
    print_risk_metrics(agent_no_baseline, agent_with_baseline)
    
    # Print Random Trading comparison
    print("\n" + "="*60)
    print("RANDOM TRADING BASELINE COMPARISON")
    print("="*60)
    random_returns = random_trading.episode_returns
    print(f"\nRandom Trading:")
    print(f"  Mean Return: {np.mean(random_returns):.4f}")
    print(f"  Std Return: {np.std(random_returns):.4f}")
    print(f"  Min Return: {np.min(random_returns):.4f}")
    print(f"  Max Return: {np.max(random_returns):.4f}")
    print(f"  Final 100 Episodes Mean: {np.mean(random_returns[-100:]):.4f}")
    print(f"  Final 100 Episodes Std: {np.std(random_returns[-100:]):.4f}")
    
    from src.utils import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
    print(f"  Sharpe Ratio: {calculate_sharpe_ratio(random_returns):.4f}")
    print(f"  Max Drawdown: {calculate_max_drawdown(random_returns):.4f}")
    print(f"  Win Rate: {calculate_win_rate(random_returns):.2f}%")
    
    # Compare against RL agents
    print(f"\nREINFORCE (No Baseline) vs Random Trading:")
    improvement_no_bl = np.mean(agent_no_baseline.episode_returns[-100:]) - np.mean(random_returns[-100:])
    print(f"  Improvement: {improvement_no_bl:.4f} "
          f"({improvement_no_bl/np.abs(np.mean(random_returns[-100:]))*100:.2f}%)")
    
    print(f"\nREINFORCE (With Baseline) vs Random Trading:")
    improvement_with_bl = np.mean(agent_with_baseline.episode_returns[-100:]) - np.mean(random_returns[-100:])
    print(f"  Improvement: {improvement_with_bl:.4f} "
          f"({improvement_with_bl/np.abs(np.mean(random_returns[-100:]))*100:.2f}%)")
    print("="*60)
    
    # Plot convergence analysis
    plot_convergence_analysis(agent_no_baseline, agent_with_baseline)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

