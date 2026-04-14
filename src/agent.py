"""
Monte Carlo REINFORCE agent implementation.
Baseline Random Trading strategy implementation.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .models import PolicyNetwork, ValueNetwork


class MonteCarloAgent:
    """
    Monte Carlo REINFORCE agent with optional baseline.
    
    Implements the REINFORCE algorithm for policy gradient learning,
    with an optional value function baseline for variance reduction.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        use_baseline=False
    ):
        """
        Initialize Monte Carlo agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            gamma: Discount factor (default: 0.99)
            lr: Learning rate (default: 1e-3)
            use_baseline: Whether to use value function baseline (default: False)
        """
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        if use_baseline:
            self.value_net = ValueNetwork(state_dim)
            self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr)

        self.reset_episode()
        
        # Metrics tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []

    def select_action(self, state, training=True):
        """
        Select an action using the current policy.
        
        Args:
            state: Current state
            training: Whether in training mode (stores log probs and states)
            
        Returns:
            Selected action
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    
        if training:
            self.log_probs.append(log_prob)
            self.states.append(state_tensor)
    
        return action.item()

    def store_reward(self, reward):
        """
        Store reward for current episode.
        
        Args:
            reward: Reward received
        """
        self.rewards.append(reward)

    def compute_returns(self):
        """
        Compute discounted returns using Monte Carlo method.
        
        Returns:
            Tensor of discounted returns
        """
        G = 0
        returns = []

        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32)

    def update(self):
        """
        Update policy using REINFORCE algorithm with optional baseline.
        """
        if len(self.rewards) == 0:
            self.reset_episode()
            return
            
        returns = self.compute_returns()
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if self.use_baseline:
            states_tensor = torch.stack(self.states)
            values = self.value_net(states_tensor)
            advantages = returns - values.detach()
        else:
            advantages = returns

        # Policy gradient loss
        policy_loss = 0
        for log_prob, adv in zip(self.log_probs, advantages):
            policy_loss += -log_prob * adv
    
        self.policy_optim.zero_grad()
        policy_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optim.step()
        
        self.policy_losses.append(policy_loss.item())

        if self.use_baseline:
            value_loss = F.mse_loss(values, returns)
            self.value_optim.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.value_optim.step()
            self.value_losses.append(value_loss.item())
        
        # Store episode metrics
        episode_return = sum(self.rewards)
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(len(self.rewards))
        
        self.reset_episode()

    def reset_episode(self):
        """Reset episode buffers."""
        self.states = []
        self.log_probs = []
        self.rewards = []
        
    def evaluate(self, env, num_episodes=10):
        """
        Evaluate agent without training.
        
        Args:
            env: Gym environment
            num_episodes: Number of episodes to evaluate
            
        Returns:
            List of episode returns
        """
        total_returns = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = truncated = False
            episode_return = 0
            
            while not done and not truncated:
                action = self.select_action(state, training=False)
                state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
            total_returns.append(episode_return)
        return total_returns


class RandomTradingAgent:
    """
    Random Trading baseline strategy.
    
    This strategy randomly selects actions at each timestep,
    serving as a financial baseline for comparison with RL agents.
    """
    
    def __init__(self, state_dim=None, action_dim=None, seed=None):
        """
        Initialize Random Trading agent.
        
        Args:
            state_dim: Dimension of state space (not used, kept for compatibility)
            action_dim: Dimension of action space (number of available actions)
            seed: Optional random seed for reproducibility
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize random number generator
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        self.reset_episode()
        
        # Metrics tracking (compatible with MonteCarloAgent interface)
        self.episode_returns = []
        self.episode_lengths = []
        self.policy_losses = []  # Empty for baseline
        self.value_losses = []   # Empty for baseline

    def select_action(self, state, training=True):
        """
        Select action: random selection from action space.
        
        Args:
            state: Current state (not used, kept for compatibility)
            training: Whether in training mode (not used, kept for compatibility)
            
        Returns:
            Randomly selected action index
        """
        if self.action_dim is None:
            raise ValueError("action_dim must be specified for RandomTradingAgent")
        
        return self.rng.randint(0, self.action_dim)

    def store_reward(self, reward):
        """
        Store reward for current episode.
        """
        self.rewards.append(reward)

    def update(self):
        """
        Update agent: just track metrics (no learning for baseline).
        """
        if len(self.rewards) == 0:
            self.reset_episode()
            return
        
        # Store episode metrics
        episode_return = sum(self.rewards)
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(len(self.rewards))
        
        self.reset_episode()

    def reset_episode(self):
        """Reset episode buffers."""
        self.rewards = []
        
    def evaluate(self, env, num_episodes=10):
        """
        Evaluate agent without training.
        """
        total_returns = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = truncated = False
            episode_return = 0
            
            while not done and not truncated:
                action = self.select_action(state, training=False)
                state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
            total_returns.append(episode_return)
        return total_returns

