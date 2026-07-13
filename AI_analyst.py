"""
AI Analyst Copilot

Includes stock data retrieval, financial metric calculation and explanation generation
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Application settings."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GPT_MODEL: str = "gpt-4o-mini"
    DEFAULT_PERIOD: str = "1y"
    RISK_FREE_RATE: float = 0.05
    
    AVAILABLE_STOCKS: tuple = (
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
        "META", "NVDA", "JPM", "V", "JNJ"
    )

config = Config()


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class StockDataError(Exception):
    """Raised when stock data cannot be fetched."""
    pass

class AIExplanationError(Exception):
    """Raised when AI explanation generation fails."""
    pass


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        period: Time period ('1mo', '3mo', '6mo', '1y')
    
    Returns:
        DataFrame with OHLCV data
    
    Raises:
        StockDataError: If data cannot be fetched
    """
    # Validate ticker
    ticker = ticker.upper().strip()
    if not ticker:
        raise StockDataError("Ticker symbol cannot be empty")
    
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            raise StockDataError(f"No data found for ticker: {ticker}")
        
        if len(data) < 20:
            raise StockDataError(
                f"Insufficient data for {ticker}: only {len(data)} days available, need at least 20"
            )
        
        return data
        
    except ImportError:
        raise StockDataError("yfinance library not installed. Run: pip install yfinance")
    except Exception as e:
        raise StockDataError(f"Failed to fetch data for {ticker}: {str(e)}")


def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Get basic stock information."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD')
        }
    except:
        return {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown'}


# ============================================================
# FINANCIAL METRICS - BASIC
# ============================================================

def calculate_returns(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate daily and cumulative returns."""
    if data.empty or 'Close' not in data.columns:
        raise ValueError("Invalid data: missing Close prices")
    
    daily_returns = data['Close'].pct_change().dropna()
    cumulative_return = (1 + daily_returns).prod() - 1
    
    return {
        'daily_returns': daily_returns,
        'mean_daily_return': daily_returns.mean(),
        'cumulative_return': cumulative_return,
        'cumulative_return_pct': cumulative_return * 100,
        'best_day': daily_returns.max() * 100,
        'worst_day': daily_returns.min() * 100
    }


def calculate_volatility(data: pd.DataFrame, trading_days: int = 252) -> Dict[str, Any]:
    """Calculate volatility (annualised standard deviation of returns)."""
    daily_returns = data['Close'].pct_change().dropna()
    daily_volatility = daily_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(trading_days)
    
    return {
        'daily_volatility': daily_volatility,
        'daily_volatility_pct': daily_volatility * 100,
        'annualized_volatility': annualized_volatility,
        'annualized_volatility_pct': annualized_volatility * 100
    }


def calculate_sharpe_ratio(data: pd.DataFrame, risk_free_rate: float = 0.05) -> Dict[str, Any]:
    """
    Calculate Sharpe Ratio.
    Sharpe = (Return - Risk Free Rate) / Volatility
    
    Interpretation:
    - Above 1.0 = Good
    - Above 2.0 = Very good
    - Above 3.0 = Excellent
    - Below 0 = Returns less than risk free rate
    """
    returns = calculate_returns(data)
    volatility = calculate_volatility(data)
    
    annualized_return = returns['mean_daily_return'] * 252
    
    if volatility['annualized_volatility'] == 0:
        sharpe = 0
    else:
        sharpe = (annualized_return - risk_free_rate) / volatility['annualized_volatility']
    
    return {
        'sharpe_ratio': sharpe,
        'annualized_return': annualized_return,
        'annualized_return_pct': annualized_return * 100,
        'risk_free_rate': risk_free_rate
    }


def calculate_moving_averages(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate simple moving averages."""
    close = data['Close']
    current_price = close.iloc[-1]
    
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    
    sma_20_current = sma_20.iloc[-1]
    sma_50_current = sma_50.iloc[-1] if len(close) >= 50 else None
    
    # Determine trend
    if sma_50_current:
        if current_price > sma_20_current > sma_50_current:
            trend = "bullish"
        elif current_price < sma_20_current < sma_50_current:
            trend = "bearish"
        else:
            trend = "neutral"
    else:
        trend = "bullish" if current_price > sma_20_current else "bearish"
    
    return {
        'current_price': current_price,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_20_current': sma_20_current,
        'sma_50_current': sma_50_current,
        'price_vs_sma20': ((current_price / sma_20_current) - 1) * 100,
        'trend': trend
    }


# ============================================================
# FINANCIAL METRICS - ADVANCED
# ============================================================

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index.
    
    Interpretation:
    - RSI > 70 = Overbought (might fall)
    - RSI < 30 = Oversold (might rise)
    - RSI 30-70 = Normal range
    """
    delta = data['Close'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[-1]
    
    if current_rsi > 70:
        signal = "overbought"
    elif current_rsi < 30:
        signal = "oversold"
    else:
        signal = "neutral"
    
    return {
        'rsi': rsi,
        'current_rsi': current_rsi,
        'signal': signal
    }


def calculate_macd(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Interpretation:
    - MACD crosses above signal line = Bullish
    - MACD crosses below signal line = Bearish
    """
    close = data['Close']
    
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    
    if current_macd > current_signal:
        crossover = "bullish"
    else:
        crossover = "bearish"
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram,
        'current_macd': current_macd,
        'current_signal': current_signal,
        'crossover': crossover
    }


def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """
    Calculate Bollinger Bands.
    
    Interpretation:
    - Price near upper band = Potentially overbought
    - Price near lower band = Potentially oversold
    - Narrow bands = Low volatility, breakout may come
    """
    close = data['Close']
    
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    
    current_price = close.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_middle = sma.iloc[-1]
    
    # Position within bands (0 = at lower, 0.5 = at middle, 1 = at upper)
    band_range = current_upper - current_lower
    if band_range > 0:
        position = (current_price - current_lower) / band_range
    else:
        position = 0.5
    
    # Band width as percentage of middle band
    band_width = (band_range / current_middle) * 100
    
    if position > 0.8:
        signal = "near upper band"
    elif position < 0.2:
        signal = "near lower band"
    else:
        signal = "within bands"
    
    return {
        'upper_band': upper_band,
        'middle_band': sma,
        'lower_band': lower_band,
        'current_upper': current_upper,
        'current_middle': current_middle,
        'current_lower': current_lower,
        'position': position,
        'band_width': band_width,
        'signal': signal
    }


# ============================================================
# CALCULATE ALL METRICS
# ============================================================

def calculate_all_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all financial metrics for a stock."""
    
    returns = calculate_returns(data)
    volatility = calculate_volatility(data)
    sharpe = calculate_sharpe_ratio(data)
    moving_avg = calculate_moving_averages(data)
    rsi = calculate_rsi(data)
    macd = calculate_macd(data)
    bollinger = calculate_bollinger_bands(data)
    
    # Combine into summary
    summary = {
        'current_price': moving_avg['current_price'],
        'cumulative_return': returns['cumulative_return_pct'],
        'volatility': volatility['annualized_volatility_pct'],
        'sharpe_ratio': sharpe['sharpe_ratio'],
        'sma_20': moving_avg['sma_20_current'],
        'sma_50': moving_avg['sma_50_current'],
        'trend': moving_avg['trend'],
        'rsi': rsi['current_rsi'],
        'rsi_signal': rsi['signal'],
        'macd_crossover': macd['crossover'],
        'bollinger_signal': bollinger['signal'],
        'best_day': returns['best_day'],
        'worst_day': returns['worst_day']
    }
    
    # Full details
    return {
        'summary': summary,
        'returns': returns,
        'volatility': volatility,
        'sharpe': sharpe,
        'moving_avg': moving_avg,
        'rsi': rsi,
        'macd': macd,
        'bollinger': bollinger
    }


# ============================================================
# AI EXPLANATION GENERATOR
# ============================================================

def generate_ai_explanation(ticker: str, metrics: Dict[str, Any]) -> str:
    """
    Generate plain English explanation using OpenAI GPT.

    """
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return generate_fallback_explanation(ticker, metrics)
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        summary = metrics['summary']
        
        prompt = f"""You are a friendly financial education assistant explaining stock analysis to someone 
who is new to investing. Use simple language and avoid technical jargon.

Stock: {ticker}
Current Price: ${summary['current_price']:.2f}
Yearly Return: {summary['cumulative_return']:.1f}%
Volatility: {summary['volatility']:.1f}%
Sharpe Ratio: {summary['sharpe_ratio']:.2f}
RSI: {summary['rsi']:.1f} ({summary['rsi_signal']})
Trend: {summary['trend']}
MACD Signal: {summary['macd_crossover']}

Please explain in 150 words or less:
1. What these numbers tell us about this stock
2. Whether the signs are positive, negative, or mixed
3. Key risks to consider

End with a reminder that this is educational information, not financial advice."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return generate_fallback_explanation(ticker, metrics)
    except Exception as e:
        print(f"GPT API error: {e}")
        return generate_fallback_explanation(ticker, metrics)


def generate_fallback_explanation(ticker: str, metrics: Dict[str, Any]) -> str:
    """Generate explanation without API (fallback)."""
    
    s = metrics['summary']
    
    # Build dynamic explanation
    lines = []
    lines.append(f"Analysis for {ticker}")
    lines.append("")
    
    # Returns
    if s['cumulative_return'] > 10:
        lines.append(f"This stock has performed well, gaining {s['cumulative_return']:.1f}% over the past year.")
    elif s['cumulative_return'] > 0:
        lines.append(f"This stock has shown modest growth of {s['cumulative_return']:.1f}% over the past year.")
    else:
        lines.append(f"This stock has declined {abs(s['cumulative_return']):.1f}% over the past year.")
    
    # Volatility
    if s['volatility'] > 30:
        lines.append(f"With volatility at {s['volatility']:.1f}%, this is a high risk stock with large price swings.")
    elif s['volatility'] > 20:
        lines.append(f"Volatility of {s['volatility']:.1f}% indicates moderate risk.")
    else:
        lines.append(f"At {s['volatility']:.1f}% volatility, this is relatively stable.")
    
    # Sharpe
    if s['sharpe_ratio'] > 1:
        lines.append(f"The Sharpe ratio of {s['sharpe_ratio']:.2f} suggests good risk adjusted returns.")
    elif s['sharpe_ratio'] > 0:
        lines.append(f"The Sharpe ratio of {s['sharpe_ratio']:.2f} is positive but modest.")
    else:
        lines.append(f"A negative Sharpe ratio means returns have not compensated for the risk.")
    
    # RSI
    if s['rsi_signal'] == 'overbought':
        lines.append(f"RSI at {s['rsi']:.0f} suggests the stock may be overbought.")
    elif s['rsi_signal'] == 'oversold':
        lines.append(f"RSI at {s['rsi']:.0f} suggests the stock may be oversold.")
    
    # Trend
    lines.append(f"The overall trend appears {s['trend']} based on moving averages.")
    
    lines.append("")
    lines.append("Note: This is for educational purposes only, not financial advice.")
    
    return "\n".join(lines)


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_stock(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Complete stock analysis pipeline.
    
    Args:
        ticker: Stock symbol
        period: Time period for analysis
    
    Returns:
        Dictionary with data, metrics, and AI explanation
    """
    print(f"\n{'='*60}")
    print(f"  AI ANALYST COPILOT - Analyzing {ticker}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch data
    print("Fetching stock data...")
    try:
        data = fetch_stock_data(ticker, period)
        print(f"  Retrieved {len(data)} days of data")
        print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    except StockDataError as e:
        print(f"  Error: {e}")
        return None
    
    # Step 2: Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_all_metrics(data)
    s = metrics['summary']
    
    print(f"  Current Price: ${s['current_price']:.2f}")
    print(f"  Yearly Return: {s['cumulative_return']:.1f}%")
    print(f"  Volatility: {s['volatility']:.1f}%")
    print(f"  Sharpe Ratio: {s['sharpe_ratio']:.2f}")
    print(f"  RSI: {s['rsi']:.1f} ({s['rsi_signal']})")
    print(f"  MACD: {s['macd_crossover']}")
    print(f"  Trend: {s['trend']}")
    
    # Step 3: Generate explanation
    print("\nGenerating AI explanation...")
    explanation = generate_ai_explanation(ticker, metrics)
    print("\n" + explanation)
    
    return {
        'ticker': ticker,
        'period': period,
        'data': data,
        'metrics': metrics,
        'explanation': explanation
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Analyze a stock
    result = analyze_stock("AAPL")
    
    print("\n" + "="*60)
    print("  Analysis complete.")
    print("  1. Run: pip install openai")
    print("  2. Set environment variable: OPENAI_API_KEY=your_key")
    print("="*60)
