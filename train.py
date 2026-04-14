"""
Training script for Monte Carlo REINFORCE agent.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_trading_env

from src.agent import MonteCarloAgent
from src.utils import load_trading_data


def create_environment(df, data_limit=None, **env_kwargs):
    """
    Create trading environment.
    
    Args:
        df: DataFrame with trading data
        data_limit: Optional limit on number of rows to use
        **env_kwargs: Additional environment arguments
        
    Returns:
        Gym environment and state/action dimensions
    """
    df_subset = df[:data_limit] if data_limit else df
    
    env = gym.make(
        "TradingEnv",
        name="DAX",
        df=df_subset,
        positions=[-1, 0, 1],  # SHORT, OUT, LONG
        trading_fees=0.01/100,
        borrow_interest_rate=0.0003/100,
        verbose=0,
        **env_kwargs
    )
    
    # Get environment dimensions
    state, _ = env.reset()
    state_dim = len(state) if isinstance(state, (list, np.ndarray)) else state.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"State sample: {state}")
    
    return env, state_dim, action_dim


def train_agent(agent, env, num_episodes, eval_interval=50, verbose=True):
    """
    Train an agent.
    
    Args:
        agent: MonteCarloAgent instance
        env: Gym environment
        num_episodes: Number of training episodes
        eval_interval: Interval for printing progress
        verbose: Whether to print progress
        
    Returns:
        Trained agent
    """
    for episode in range(num_episodes):
        state, info = env.reset()
        done = truncated = False

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_reward(reward)
            state = next_state
        
        agent.update()
        
        if verbose and (episode + 1) % eval_interval == 0:
            avg_return = np.mean(agent.episode_returns[-eval_interval:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Avg Return (last {eval_interval}): {avg_return:.4f}")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='Train Monte Carlo REINFORCE agent')
    parser.add_argument('--data-path', type=str, 
                       default='data/GlobalExchangeData/indexProcessed.csv',
                       help='Path to trading data CSV')
    parser.add_argument('--symbol', type=str, default='GDAXI',
                       help='Symbol to filter (e.g., GDAXI)')
    parser.add_argument('--data-limit', type=int, default=1000,
                       help='Limit number of data rows')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--use-baseline', action='store_true',
                       help='Use value function baseline')
    parser.add_argument('--eval-interval', type=int, default=50,
                       help='Evaluation interval for printing')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_trading_data(args.data_path, symbol=args.symbol)
    print(f"Loaded {len(df)} rows")
    
    # Create environment
    print("\nCreating environment...")
    env, state_dim, action_dim = create_environment(df, data_limit=args.data_limit)
    
    # Create and train agent
    print(f"\nTraining REINFORCE agent ({'with' if args.use_baseline else 'without'} baseline)...")
    agent = MonteCarloAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        lr=args.lr,
        use_baseline=args.use_baseline
    )
    
    agent = train_agent(
        agent, 
        env, 
        args.episodes, 
        eval_interval=args.eval_interval
    )
    
    print("\nTraining completed!")
    print(f"Final average return (last 100): {np.mean(agent.episode_returns[-100:]):.4f}")
    
    return agent


if __name__ == '__main__':
    main()

