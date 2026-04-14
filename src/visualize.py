"""
Visualization and plotting functions for analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Dict, Tuple
from scipy import stats


def aggregate_multiple_runs(all_episode_returns: List[List[float]], window: int = 50) -> Dict[str, np.ndarray]:
    """
    Aggregate results from multiple seed runs with confidence intervals.
    
    Args:
        all_episode_returns: List of episode returns from each seed run
        window: Window size for moving average
        
    Returns:
        Dictionary with mean, std, ci_lower, ci_upper arrays
    """
    # Convert to numpy array (num_seeds, num_episodes)
    returns_array = np.array(all_episode_returns)
    num_seeds, num_episodes = returns_array.shape
    
    # Compute statistics across seeds for each episode
    mean_returns = np.mean(returns_array, axis=0)
    std_returns = np.std(returns_array, axis=0, ddof=1)
    std_error = std_returns / np.sqrt(num_seeds)
    
    # 95% CI using t-distribution
    t_value = stats.t.ppf(0.975, df=num_seeds - 1)
    ci_lower = mean_returns - t_value * std_error
    ci_upper = mean_returns + t_value * std_error
    
    # Moving average
    if num_episodes >= window:
        ma_mean = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
        ma_ci_lower = np.convolve(ci_lower, np.ones(window)/window, mode='valid')
        ma_ci_upper = np.convolve(ci_upper, np.ones(window)/window, mode='valid')
        episode_range = np.arange(window-1, num_episodes)
    else:
        ma_mean = mean_returns
        ma_ci_lower = ci_lower
        ma_ci_upper = ci_upper
        episode_range = np.arange(num_episodes)
    
    return {
        'mean': ma_mean,
        'ci_lower': ma_ci_lower,
        'ci_upper': ma_ci_upper,
        'episode_range': episode_range,
        'raw_mean': mean_returns,
        'raw_ci_lower': ci_lower,
        'raw_ci_upper': ci_upper
    }


def plot_convergence_comparison(
    random_returns: List[List[float]],
    reinforce_returns: List[List[float]],
    reinforce_baseline_returns: List[List[float]],
    window: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot convergence comparison for three methods with confidence intervals.
    Saves plots individually instead of grouping them.
    
    Args:
        random_returns: List of episode returns from random trading (multiple seeds)
        reinforce_returns: List of episode returns from REINFORCE (multiple seeds)
        reinforce_baseline_returns: List of episode returns from REINFORCE+baseline (multiple seeds)
        window: Window size for moving average
        save_path: Optional path to save figures (will add suffixes: _convergence, _final_performance)
        figsize: Figure size for individual plots
    """
    # Aggregate results
    random_stats = aggregate_multiple_runs(random_returns, window)
    reinforce_stats = aggregate_multiple_runs(reinforce_returns, window)
    baseline_stats = aggregate_multiple_runs(reinforce_baseline_returns, window)
    
    ep_range = random_stats['episode_range']
    
    # Plot 1: Convergence with confidence intervals
    fig1, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    ax1.plot(ep_range, random_stats['mean'], label='Random Trading', 
             color='gray', linewidth=2, alpha=0.8)
    ax1.fill_between(ep_range, random_stats['ci_lower'], random_stats['ci_upper'],
                     alpha=0.2, color='gray')
    
    ax1.plot(ep_range, reinforce_stats['mean'], label='REINFORCE', 
             color='steelblue', linewidth=2, alpha=0.8)
    ax1.fill_between(ep_range, reinforce_stats['ci_lower'], reinforce_stats['ci_upper'],
                     alpha=0.2, color='steelblue')
    
    ax1.plot(ep_range, baseline_stats['mean'], label='REINFORCE + Baseline', 
             color='darkgreen', linewidth=2, alpha=0.8)
    ax1.fill_between(ep_range, baseline_stats['ci_lower'], baseline_stats['ci_upper'],
                     alpha=0.2, color='darkgreen')
    
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Average Return (rolling)', fontsize=11)
    ax1.set_title('Learning Convergence', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Generate path for convergence plot
        base_path, ext = os.path.splitext(save_path)
        convergence_path = f"{base_path}_convergence{ext}"
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig1)
    
    # Plot 2: Final performance comparison (last 100 episodes)
    fig2, ax2 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    # Extract final 100 episodes from each seed
    final_random = [r[-100:] for r in random_returns]
    final_reinforce = [r[-100:] for r in reinforce_returns]
    final_baseline = [r[-100:] for r in reinforce_baseline_returns]
    
    # Flatten and compute means per seed
    random_final_means = [np.mean(r) for r in final_random]
    reinforce_final_means = [np.mean(r) for r in final_reinforce]
    baseline_final_means = [np.mean(r) for r in final_baseline]
    
    # Box plot
    data_to_plot = [random_final_means, reinforce_final_means, baseline_final_means]
    bp = ax2.boxplot(data_to_plot, tick_labels=['Random', 'REINFORCE', 'REINFORCE\n+ Baseline'],
                     patch_artist=True, widths=0.6)
    
    colors = ['lightgray', 'lightsteelblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Average Return (final 100 episodes)', fontsize=11)
    ax2.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        # Generate path for final performance plot
        base_path, ext = os.path.splitext(save_path)
        final_path = f"{base_path}_final_performance{ext}"
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig2)


def create_comparison_table(
    random_returns: List[List[float]],
    reinforce_returns: List[List[float]],
    reinforce_baseline_returns: List[List[float]],
    method_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a comprehensive comparison table for three methods.
    
    Args:
        random_returns: List of episode returns from random trading (multiple seeds)
        reinforce_returns: List of episode returns from REINFORCE (multiple seeds)
        reinforce_baseline_returns: List of episode returns from REINFORCE+baseline (multiple seeds)
        method_names: Optional list of method names
        
    Returns:
        DataFrame with comparison statistics
    """
    from .utils import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
    
    if method_names is None:
        method_names = ['Random Trading', 'REINFORCE', 'REINFORCE + Baseline']
    
    all_returns = [random_returns, reinforce_returns, reinforce_baseline_returns]
    
    results = []
    for method_name, returns_list in zip(method_names, all_returns):
        # Flatten all returns across seeds
        all_flat_returns = np.concatenate(returns_list)
        
        # Final 100 episodes from each seed
        final_returns = np.concatenate([r[-100:] for r in returns_list])
        
        # Statistics across seeds for final performance
        final_means_per_seed = [np.mean(r[-100:]) for r in returns_list]
        num_seeds = len(returns_list)
        
        mean_final = np.mean(final_means_per_seed)
        std_final = np.std(final_means_per_seed, ddof=1)
        std_error = std_final / np.sqrt(num_seeds)
        t_value = stats.t.ppf(0.975, df=num_seeds - 1)
        ci_lower = mean_final - t_value * std_error
        ci_upper = mean_final + t_value * std_error
        
        results.append({
            'Method': method_name,
            'Mean Return (final)': f"{mean_final:.4f}",
            'Std Dev (final)': f"{std_final:.4f}",
            '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            'Sharpe Ratio': f"{calculate_sharpe_ratio(all_flat_returns):.4f}",
            'Max Drawdown': f"{calculate_max_drawdown(all_flat_returns):.4f}",
            'Win Rate (%)': f"{calculate_win_rate(all_flat_returns):.2f}"
        })
    
    df = pd.DataFrame(results)
    return df


def print_comparison_table(table: pd.DataFrame):
    """
    Print comparison table in a formatted way.
    
    Args:
        table: DataFrame from create_comparison_table
    """
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)
    print(table.to_string(index=False))
    print("="*80 + "\n")


def plot_return_distributions(
    random_returns: List[List[float]],
    reinforce_returns: List[List[float]],
    reinforce_baseline_returns: List[List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
):
    """
    Plot return distributions for three methods.
    Saves plots individually instead of grouping them.
    
    Args:
        random_returns: List of episode returns from random trading (multiple seeds)
        reinforce_returns: List of episode returns from REINFORCE (multiple seeds)
        reinforce_baseline_returns: List of episode returns from REINFORCE+baseline (multiple seeds)
        save_path: Optional path to save figures (will add suffixes: _distribution, _cumulative)
        figsize: Figure size for individual plots
    """
    # Flatten returns
    random_flat = np.concatenate(random_returns)
    reinforce_flat = np.concatenate(reinforce_returns)
    baseline_flat = np.concatenate(reinforce_baseline_returns)
    
    # Plot 1: Histogram
    fig1, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    ax1.hist(random_flat, bins=30, alpha=0.5, label='Random Trading', 
             color='gray', density=True)
    ax1.hist(reinforce_flat, bins=30, alpha=0.5, label='REINFORCE', 
             color='steelblue', density=True)
    ax1.hist(baseline_flat, bins=30, alpha=0.5, label='REINFORCE + Baseline', 
             color='darkgreen', density=True)
    ax1.set_xlabel('Episode Return', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Generate path for distribution plot
        base_path, ext = os.path.splitext(save_path)
        distribution_path = f"{base_path}_distribution{ext}"
        plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig1)
    
    # Plot 2: Cumulative returns
    fig2, ax2 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    ax2.plot(np.cumsum(random_flat), label='Random Trading', 
             color='gray', alpha=0.7, linewidth=1.5)
    ax2.plot(np.cumsum(reinforce_flat), label='REINFORCE', 
             color='steelblue', alpha=0.7, linewidth=1.5)
    ax2.plot(np.cumsum(baseline_flat), label='REINFORCE + Baseline', 
             color='darkgreen', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Cumulative Return', fontsize=11)
    ax2.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Generate path for cumulative returns plot
        base_path, ext = os.path.splitext(save_path)
        cumulative_path = f"{base_path}_cumulative{ext}"
        plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig2)