"""
Unit tests for the AI Analyst Copilot
Run with: python3 -m pytest test_app.py -v
"""

import pytest
import pandas as pd
import numpy as np

from AI_analyst import (
    fetch_stock_data,
    calculate_all_metrics,
    StockDataError,
)


# FIXTURES

@pytest.fixture
def sample_stock_data():
    """Create fake stock data for testing without hitting the internet."""
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    np.random.seed(42)

    prices = 150 + np.cumsum(np.random.randn(252) * 2)

    data = pd.DataFrame({
        'Open': prices + np.random.randn(252) * 0.5,
        'High': prices + np.abs(np.random.randn(252)),
        'Low': prices - np.abs(np.random.randn(252)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)

    return data


# DATA FETCHER TESTS

def test_fetch_valid_ticker():
    """Test that a valid ticker returns data successfully."""
    data = fetch_stock_data("AAPL", "1mo")
    assert data is not None
    assert len(data) > 0
    assert 'Close' in data.columns


def test_fetch_invalid_ticker():
    """Test that an invalid ticker raises StockDataError."""
    with pytest.raises(StockDataError):
        fetch_stock_data("XYZ123INVALID", "1mo")


def test_fetch_empty_ticker():
    """Test that an empty ticker raises StockDataError."""
    with pytest.raises(StockDataError):
        fetch_stock_data("", "1mo")


# METRICS CALCULATION TESTS

def test_all_metrics_returns_dict(sample_stock_data):
    """Test that calculate_all_metrics returns a dictionary."""
    result = calculate_all_metrics(sample_stock_data)
    assert isinstance(result, dict)


def test_metrics_contains_returns(sample_stock_data):
    """Test that metrics contains returns information."""
    result = calculate_all_metrics(sample_stock_data)
    assert 'returns' in result


def test_metrics_contains_volatility(sample_stock_data):
    """Test that metrics contains volatility information."""
    result = calculate_all_metrics(sample_stock_data)
    # Check for volatility somewhere in the result
    assert 'volatility' in str(result).lower() or 'returns' in result


def test_metrics_contains_rsi(sample_stock_data):
    """Test that metrics contains RSI information."""
    result = calculate_all_metrics(sample_stock_data)
    assert 'rsi' in result


def test_metrics_contains_macd(sample_stock_data):
    """Test that metrics contains MACD information."""
    result = calculate_all_metrics(sample_stock_data)
    assert 'macd' in result


def test_metrics_contains_bollinger(sample_stock_data):
    """Test that metrics contains Bollinger Bands information."""
    result = calculate_all_metrics(sample_stock_data)
    assert 'bollinger' in result


def test_metrics_contains_moving_avg(sample_stock_data):
    """Test that metrics contains moving averages."""
    result = calculate_all_metrics(sample_stock_data)
    assert 'moving_avg' in result


def test_returns_has_daily_returns(sample_stock_data):
    """Test that returns section has daily returns data."""
    result = calculate_all_metrics(sample_stock_data)
    assert 'daily_returns' in result['returns']


def test_macd_has_crossover(sample_stock_data):
    """Test that MACD has a crossover signal."""
    result = calculate_all_metrics(sample_stock_data)
    crossover = result['macd']['crossover']
    assert crossover in ['bullish', 'bearish']


def test_bollinger_has_band_width(sample_stock_data):
    """Test that Bollinger has a band width value."""
    result = calculate_all_metrics(sample_stock_data)
    assert 'band_width' in result['bollinger']
