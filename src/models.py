"""
Neural network models for reinforcement learning agent.
REINFORCE and REINFORCE with Variance Reduction.
"""

import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units (default: 128)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor (can be single state or batched)
            
        Returns:
            Action probabilities (softmax output)
        """
        # Handle both single state and batched states
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.net(state)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Value network that estimates state values."""
    
    def __init__(self, state_dim, hidden_dim=128):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of hidden units (default: 128)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Estimated state value
        """
        return self.net(state).squeeze(-1)

