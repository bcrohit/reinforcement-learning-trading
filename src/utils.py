"""
Utility functions for data loading and metrics calculation.
"""

import pandas as pd
import numpy as np

def load_trading_data(filepath, index_col='date', symbol=None):
    """
    Load and preprocess trading data.
    """
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]
    
    if symbol:
        df = df[df['index'] == symbol]
        df = df.rename(columns={'index': 'symbol'})
    
    if index_col in df.columns:
        df = df.sort_values(index_col)
        df = df.set_index(index_col)
    elif index_col in df.index.names:
        df.sort_index(inplace=True)    
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def add_static_features(
    df,
    volatility_window=10,
    ma_window=10,
    volume_window=20,
    normalize=True,
    epsilon=1e-8
):
    """
    Add technical features to trading data for reinforcement learning.
    """
    df = df.copy()
    
    # Calculate log returns (momentum feature)
    df["feature_log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Calculate rolling volatility (risk/volatility feature)
    df["feature_volatility_10"] = (
        df["feature_log_return"]
        .rolling(window=volatility_window)
        .std()
    )
    
    # Calculate moving average ratio (trend feature)
    ma = df["close"].rolling(window=ma_window).mean()
    df["feature_ma_ratio_10"] = ma / df["close"]
    
    # Calculate volume z-score (volume anomaly feature)
    volume_mean = df["volume"].rolling(window=volume_window).mean()
    volume_std = df["volume"].rolling(window=volume_window).std()
    df["feature_volume_zscore"] = (
        (df["volume"] - volume_mean) / (volume_std + epsilon)
    )
    
    # Normalize all features (zero mean, unit variance)
    if normalize:
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        feature_mean = df[feature_cols].mean()
        feature_std = df[feature_cols].std()
        df[feature_cols] = (df[feature_cols] - feature_mean) / (feature_std + epsilon)
    
    df.dropna(inplace=True)

    return df

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio for trading performance.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Number of trading periods per year (default: 252)
        
    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = np.array(returns) - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown.
    """
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return np.min(drawdown)

def calculate_win_rate(returns):
    """
    Calculate win rate (percentage of positive returns).
    """
    if len(returns) == 0:
        return 0.0
    return np.sum(np.array(returns) > 0) / len(returns) * 100

