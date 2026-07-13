"""
AI Analyst Copilot

Features:
- Custom stock ticker input
- Multiple analysis pages
- Stock comparison
- Detailed interactive charts
- GPT integration
- Export functionality
- Educational tooltips
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# ============================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================

st.set_page_config(
    page_title="AI Analyst Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
    }
    .positive { color: #00C853; }
    .negative { color: #FF5252; }
    .neutral { color: #FFC107; }
    .info-box {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 15px;
        border-radius: 0 8px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA FETCHING FUNCTIONS
# ============================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker: str, period: str = "1y") -> tuple:
    """Fetch stock data with caching."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper().strip())
        data = stock.history(period=period)
        
        if data.empty:
            return None, f"No data found for ticker: {ticker}"
        
        if len(data) < 20:
            return None, f"Insufficient data for {ticker}: only {len(data)} days"
        
        # Get stock info
        try:
            info = stock.info
            stock_info = {
                'name': info.get('longName', ticker.upper()),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')[:500] + '...' if info.get('longBusinessSummary') else ''
            }
        except:
            stock_info = {'name': ticker.upper(), 'sector': 'N/A', 'industry': 'N/A'}
        
        return (data, stock_info), None
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"


# ============================================================
# FINANCIAL CALCULATIONS
# ============================================================

def calculate_all_metrics(data: pd.DataFrame) -> dict:
    """Calculate comprehensive financial metrics."""
    close = data['Close']
    
    # Basic returns
    daily_returns = close.pct_change().dropna()
    cumulative_return = (1 + daily_returns).prod() - 1
    
    # Volatility
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    
    # Sharpe Ratio (assuming 5% risk-free rate)
    annual_return = daily_returns.mean() * 252
    sharpe = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
    
    # Moving Averages
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=200).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    # Bollinger Bands
    bb_sma = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_sma + (bb_std * 2)
    bb_lower = bb_sma - (bb_std * 2)
    
    # Current values
    current_price = close.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    
    # Determine signals
    rsi_signal = "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
    macd_signal = "bullish" if current_macd > current_signal else "bearish"
    
    # Trend
    if sma_50.iloc[-1] and sma_200.iloc[-1]:
        if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
            trend = "bullish"
        elif current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
            trend = "bearish"
        else:
            trend = "neutral"
    else:
        trend = "bullish" if current_price > sma_20.iloc[-1] else "bearish"
    
    # Best/Worst days
    best_day = daily_returns.max() * 100
    worst_day = daily_returns.min() * 100
    
    # 52-week high/low
    high_52w = close.max()
    low_52w = close.min()
    
    return {
        'summary': {
            'current_price': current_price,
            'cumulative_return': cumulative_return * 100,
            'volatility': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'rsi': current_rsi,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'trend': trend,
            'best_day': best_day,
            'worst_day': worst_day,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1] if len(close) >= 50 else None,
            'sma_200': sma_200.iloc[-1] if len(close) >= 200 else None,
        },
        'series': {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'rsi': rsi,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_histogram': macd_histogram,
            'bb_upper': bb_upper,
            'bb_middle': bb_sma,
            'bb_lower': bb_lower,
            'daily_returns': daily_returns
        }
    }


# ============================================================
# AI EXPLANATION GENERATOR
# ============================================================

def generate_ai_explanation(ticker: str, metrics: dict, stock_info: dict) -> str:
    """Generate AI explanation using GPT or fallback."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    s = metrics['summary']
    
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            prompt = f"""You are an educational stock analysis assistant explaining stock analysis to a beginner investor. Do not give financial advice.
            
Stock: {stock_info.get('name', ticker)} ({ticker})
Sector: {stock_info.get('sector', 'N/A')}

Current Metrics:
- Price: ${s['current_price']:.2f}
- 1-Year Return: {s['cumulative_return']:.1f}%
- Volatility: {s['volatility']:.1f}%
- Sharpe Ratio: {s['sharpe_ratio']:.2f}
- RSI: {s['rsi']:.1f} ({s['rsi_signal']})
- MACD Signal: {s['macd_signal']}
- Overall Trend: {s['trend']}
- 52-Week Range: ${s['low_52w']:.2f} - ${s['high_52w']:.2f}

Provide a clear, jargon-free analysis in about 150 words covering:
1. How the stock has performed
2. Current momentum signals (RSI, MACD)
3. Risk level based on volatility
4. Key things to watch

End with a reminder this is educational, not financial advice."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            pass  # Fall through to fallback
    
    # Fallback explanation
    lines = [f"**Analysis for {stock_info.get('name', ticker)}**\n"]
    
    # Performance
    if s['cumulative_return'] > 20:
        lines.append(f"This stock has shown strong performance, gaining {s['cumulative_return']:.1f}% over the past year.")
    elif s['cumulative_return'] > 0:
        lines.append(f"This stock has shown positive returns of {s['cumulative_return']:.1f}% over the past year.")
    else:
        lines.append(f"This stock has declined {abs(s['cumulative_return']):.1f}% over the past year.")
    
    # Volatility
    if s['volatility'] > 35:
        lines.append(f"With volatility at {s['volatility']:.1f}%, this is a high-risk stock with significant price swings.")
    elif s['volatility'] > 20:
        lines.append(f"Volatility of {s['volatility']:.1f}% indicates moderate risk.")
    else:
        lines.append(f"At {s['volatility']:.1f}% volatility, this is relatively stable.")
    
    # Sharpe
    if s['sharpe_ratio'] > 1:
        lines.append(f"The Sharpe ratio of {s['sharpe_ratio']:.2f} suggests good risk-adjusted returns.")
    elif s['sharpe_ratio'] > 0:
        lines.append(f"The Sharpe ratio of {s['sharpe_ratio']:.2f} is positive but modest.")
    else:
        lines.append("The negative Sharpe ratio suggests returns haven't compensated for the risk.")
    
    # RSI
    if s['rsi_signal'] == 'overbought':
        lines.append(f"RSI at {s['rsi']:.0f} suggests the stock may be overbought, meaning it has risen sharply recently.")
    elif s['rsi_signal'] == 'oversold':
        lines.append(f"RSI at {s['rsi']:.0f} suggests the stock may be oversold - potential bounce opportunity.")
    else:
        lines.append(f"RSI at {s['rsi']:.0f} is in neutral territory.")
    
    # MACD and Trend
    lines.append(f"MACD shows a {s['macd_signal']} signal, and the overall trend is {s['trend']}.")
    
    lines.append("\n*Note: This is for educational purposes only, not financial advice.*")
    
    return " ".join(lines)


# ============================================================
# CHART FUNCTIONS
# ============================================================

def create_price_chart(data: pd.DataFrame, metrics: dict, show_bollinger: bool = False) -> go.Figure:
    """Create interactive price chart with indicators."""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00C853',
            decreasing_line_color='#FF5252'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    series = metrics['series']
    
    fig.add_trace(
        go.Scatter(x=data.index, y=series['sma_20'], name='SMA 20',
                   line=dict(color='#FFA726', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=series['sma_50'], name='SMA 50',
                   line=dict(color='#42A5F5', width=1)),
        row=1, col=1
    )
    
    # Bollinger Bands (optional)
    if show_bollinger:
        fig.add_trace(
            go.Scatter(x=data.index, y=series['bb_upper'], name='BB Upper',
                       line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=series['bb_lower'], name='BB Lower',
                       line=dict(color='gray', width=1, dash='dash'),
                       fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # Volume
    colors = ['#FF5252' if data['Close'].iloc[i] < data['Open'].iloc[i] else '#00C853' 
              for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=series['rsi'], name='RSI',
                   line=dict(color='#AB47BC', width=1.5)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.1)", line_width=0, row=3, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig


def create_macd_chart(data: pd.DataFrame, metrics: dict) -> go.Figure:
    """Create MACD indicator chart."""
    series = metrics['series']
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=data.index, y=series['macd_line'], name='MACD',
                   line=dict(color='#42A5F5', width=1.5))
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=series['signal_line'], name='Signal',
                   line=dict(color='#FFA726', width=1.5))
    )
    
    colors = ['#00C853' if val >= 0 else '#FF5252' for val in series['macd_histogram']]
    fig.add_trace(
        go.Bar(x=data.index, y=series['macd_histogram'], name='Histogram',
               marker_color=colors)
    )
    
    fig.update_layout(
        height=300,
        title="MACD Indicator",
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_returns_distribution(metrics: dict) -> go.Figure:
    """Create returns distribution histogram."""
    returns = metrics['series']['daily_returns'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(x=returns, nbinsx=50, name='Daily Returns',
                     marker_color='#42A5F5', opacity=0.7)
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="white")
    fig.add_vline(x=returns.mean(), line_dash="solid", line_color="#00C853",
                  annotation_text=f"Mean: {returns.mean():.2f}%")
    
    fig.update_layout(
        height=300,
        title="Daily Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 📈 AI Analyst Copilot")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Stock Analysis", "📈 Compare Stocks", "📚 Learn", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stock input
        st.markdown("### Quick Analysis")
        ticker_input = st.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            help="Enter any valid stock ticker symbol"
        ).upper().strip()
        
        period = st.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            format_func=lambda x: {
                "1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months",
                "1y": "1 Year", "2y": "2 Years", "5y": "5 Years"
            }[x]
        )
        
        analyze_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Popular stocks
        st.markdown("### Popular Stocks")
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "JPM"]
        
        cols = st.columns(4)
        for i, stock in enumerate(popular_stocks):
            with cols[i % 4]:
                if st.button(stock, key=f"pop_{stock}", use_container_width=True):
                    st.session_state['quick_ticker'] = stock
                    st.rerun()
        
        st.markdown("---")


    
    # Check for quick ticker selection
    if 'quick_ticker' in st.session_state:
        ticker_input = st.session_state['quick_ticker']
        del st.session_state['quick_ticker']
        analyze_btn = True
    
    # Main content based on page selection
    if page == "📊 Stock Analysis":
        render_analysis_page(ticker_input, period, analyze_btn)
    elif page == "📈 Compare Stocks":
        render_compare_page()
    elif page == "📚 Learn":
        render_learn_page()
    elif page == "⚙️ Settings":
        render_settings_page()


def render_analysis_page(ticker: str, period: str, analyze: bool):
    """Render the main stock analysis page."""
    
    st.markdown('<h1 class="main-header">Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis for informed investment decisions</p>', unsafe_allow_html=True)
    
    if analyze and ticker:
        with st.spinner(f"Analysing {ticker}..."):
            result, error = fetch_stock_data(ticker, period)
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Try entering a valid stock ticker like AAPL, GOOGL, MSFT, etc.")
            return
        
        data, stock_info = result
        metrics = calculate_all_metrics(data)
        s = metrics['summary']
        
        # Stock header
        st.markdown(f"## {stock_info.get('name', ticker)} ({ticker})")
        st.caption(f"{stock_info.get('sector', 'N/A')} | {stock_info.get('industry', 'N/A')}")
        
        # Key Metrics Row 1
        st.markdown("### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${s['current_price']:.2f}")
        with col2:
            delta_color = "normal" if s['cumulative_return'] >= 0 else "inverse"
            st.metric("Return", f"{s['cumulative_return']:.1f}%", 
                     delta=f"{s['cumulative_return']:.1f}%", delta_color=delta_color)
        with col3:
            st.metric("Volatility", f"{s['volatility']:.1f}%",
                     help="Annualised volatility - higher means more price swings")
        with col4:
            st.metric("Sharpe Ratio", f"{s['sharpe_ratio']:.2f}",
                     help="Risk-adjusted return. Above 1.0 is good, above 2.0 is excellent")
        
        # Key Metrics Row 2
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            rsi_color = "🔴" if s['rsi_signal'] == 'overbought' else "🟢" if s['rsi_signal'] == 'oversold' else "🟡"
            st.metric("RSI", f"{s['rsi']:.0f} {rsi_color}",
                     help="Relative Strength Index. >70 = overbought, <30 = oversold")
        with col6:
            trend_color = "🟢" if s['trend'] == 'bullish' else "🔴" if s['trend'] == 'bearish' else "🟡"
            st.metric("Trend", f"{s['trend'].title()} {trend_color}")
        with col7:
            macd_color = "🟢" if s['macd_signal'] == 'bullish' else "🔴"
            st.metric("MACD", f"{s['macd_signal'].title()} {macd_color}",
                     help="Moving Average Convergence Divergence momentum indicator")
        with col8:
            st.metric("52W Range", f"${s['low_52w']:.0f} - ${s['high_52w']:.0f}")
        
        st.markdown("---")
        
        # Charts and Analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Indicators", "📉 Returns Analysis", "🤖 AI Analysis"])
        
        with tab1:
            show_bb = st.checkbox("Show Bollinger Bands", value=False)
            fig = create_price_chart(data, metrics, show_bollinger=show_bb)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_macd_chart(data, metrics), use_container_width=True)
            with col2:
                st.plotly_chart(create_returns_distribution(metrics), use_container_width=True)
            
            # Moving Average Table
            st.markdown("#### Moving Averages")
            ma_data = {
                "Indicator": ["SMA 20", "SMA 50", "SMA 200", "Current Price"],
                "Value": [
                    f"${s['sma_20']:.2f}" if s['sma_20'] else "N/A",
                    f"${s['sma_50']:.2f}" if s['sma_50'] else "N/A",
                    f"${s['sma_200']:.2f}" if s['sma_200'] else "N/A",
                    f"${s['current_price']:.2f}"
                ],
                "Signal": [
                    "Above ✅" if s['current_price'] > s['sma_20'] else "Below ❌",
                    "Above ✅" if s['sma_50'] and s['current_price'] > s['sma_50'] else "Below ❌" if s['sma_50'] else "N/A",
                    "Above ✅" if s['sma_200'] and s['current_price'] > s['sma_200'] else "Below ❌" if s['sma_200'] else "N/A",
                    "-"
                ]
            }
            st.dataframe(pd.DataFrame(ma_data), hide_index=True, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Return Statistics")
                returns = metrics['series']['daily_returns']
                stats_data = {
                    "Metric": ["Mean Daily Return", "Std Dev (Daily)", "Best Day", "Worst Day", "Positive Days %"],
                    "Value": [
                        f"{returns.mean() * 100:.3f}%",
                        f"{returns.std() * 100:.3f}%",
                        f"+{s['best_day']:.2f}%",
                        f"{s['worst_day']:.2f}%",
                        f"{(returns > 0).sum() / len(returns) * 100:.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("#### Risk Metrics")
                risk_data = {
                    "Metric": ["Volatility (Annual)", "Sharpe Ratio", "Max Drawdown", "Value at Risk (95%)"],
                    "Value": [
                        f"{s['volatility']:.1f}%",
                        f"{s['sharpe_ratio']:.2f}",
                        f"{((data['Close'] / data['Close'].cummax()) - 1).min() * 100:.1f}%",
                        f"{np.percentile(returns, 5) * 100:.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)
        
        with tab4:
            st.markdown("### 🤖 AI-Powered Analysis")
            
            with st.spinner("Generating AI analysis..."):
                explanation = generate_ai_explanation(ticker, metrics, stock_info)
            
            st.info(explanation)
            
            # Company description if available
            if stock_info.get('description'):
                with st.expander("📋 About the Company"):
                    st.write(stock_info['description'])
        
        # Export options
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = data.to_csv()
            st.download_button(
                "Download Price Data (CSV)",
                csv,
                file_name=f"{ticker}_price_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            metrics_df = pd.DataFrame([{
                'Ticker': ticker,
                'Price': s['current_price'],
                'Return': s['cumulative_return'],
                'Volatility': s['volatility'],
                'Sharpe': s['sharpe_ratio'],
                'RSI': s['rsi'],
                'Trend': s['trend']
            }])
            st.download_button(
                "Download Metrics (CSV)",
                metrics_df.to_csv(index=False),
                file_name=f"{ticker}_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Disclaimer
        st.markdown("---")
        st.warning("⚠️ **Disclaimer**: This tool is for educational purposes only. It does not constitute financial advice. Always do your own research and consider consulting a qualified financial advisor before making investment decisions.")
    
    else:
        # Welcome screen
        st.markdown("""
        ### Welcome to AI Analyst Copilot 👋
        
        This tool helps you understand stock analysis through AI-powered explanations.
        
        **How to use:**
        1. Enter a stock ticker in the sidebar (e.g., AAPL, GOOGL, TSLA)
        2. Select a time period
        3. Click **Analyse** to see detailed analysis
        
        **Features:**
        - 📈 Interactive price charts with technical indicators
        - 📊 Comprehensive financial metrics
        - 🤖 AI-generated explanations in plain English
        - 📥 Export data for further analysis
        """)
        
        # Show example stocks
        st.markdown("### Try These Popular Stocks")
        cols = st.columns(4)
        examples = [
            ("AAPL", "Apple", "Tech"),
            ("NVDA", "NVIDIA", "Semiconductors"),
            ("JPM", "JPMorgan", "Banking"),
            ("AMZN", "Amazon", "E-commerce")
        ]
        for i, (ticker, name, sector) in enumerate(examples):
            with cols[i]:
                st.markdown(f"**{ticker}**")
                st.caption(f"{name} | {sector}")


def render_compare_page():
    """Render stock comparison page."""
    st.markdown("## 📈 Compare Stocks")
    st.markdown("Compare up to 4 stocks side by side")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stock1 = st.text_input("Stock 1", value="AAPL").upper().strip()
    with col2:
        stock2 = st.text_input("Stock 2", value="GOOGL").upper().strip()
    with col3:
        stock3 = st.text_input("Stock 3", value="MSFT").upper().strip()
    with col4:
        stock4 = st.text_input("Stock 4", value="").upper().strip()
    
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=3)
    
    if st.button("Compare", type="primary"):
        stocks = [s for s in [stock1, stock2, stock3, stock4] if s]
        
        if len(stocks) < 2:
            st.warning("Please enter at least 2 stocks to compare")
            return
        
        comparison_data = []
        price_data = {}
        
        for ticker in stocks:
            result, error = fetch_stock_data(ticker, period)
            if not error:
                data, info = result
                metrics = calculate_all_metrics(data)
                s = metrics['summary']
                
                comparison_data.append({
                    'Ticker': ticker,
                    'Price': f"${s['current_price']:.2f}",
                    'Return': f"{s['cumulative_return']:.1f}%",
                    'Volatility': f"{s['volatility']:.1f}%",
                    'Sharpe': f"{s['sharpe_ratio']:.2f}",
                    'RSI': f"{s['rsi']:.0f}",
                    'Trend': s['trend'].title()
                })
                
                # Normalize prices for comparison chart
                normalized = (data['Close'] / data['Close'].iloc[0]) * 100
                price_data[ticker] = normalized
        
        # Comparison Table
        st.markdown("### Comparison Table")
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
        
        # Comparison Chart
        st.markdown("### Normalised Price Comparison")
        st.caption("Starting value = 100")
        
        fig = go.Figure()
        for ticker, prices in price_data.items():
            fig.add_trace(go.Scatter(x=prices.index, y=prices, name=ticker, mode='lines'))
        
        fig.update_layout(
            height=500,
            template="plotly_dark",
            yaxis_title="Normalised Price",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_learn_page():
    """Render educational content page."""
    st.markdown("## 📚 Learn About Stock Analysis")
    
    with st.expander("📈 What is RSI (Relative Strength Index)?", expanded=True):
        st.markdown("""
        **RSI** measures the speed and magnitude of recent price changes to evaluate if a stock is overbought or oversold.
        
        - **RSI > 70**: The stock may be **overbought** - it has risen a lot recently and might be due for a pullback
        - **RSI < 30**: The stock may be **oversold** - it has fallen a lot and might be due for a bounce
        - **RSI 30-70**: Neutral territory
        
        RSI ranges from 0 to 100 and is calculated using 14 days of price data by default.
        """)
    
    with st.expander("📊 What is MACD?"):
        st.markdown("""
        **MACD** (Moving Average Convergence Divergence) is a momentum indicator that shows the relationship between two moving averages.
        
        - **MACD Line**: Difference between 12-day and 26-day exponential moving averages
        - **Signal Line**: 9-day EMA of the MACD line
        - **Bullish Signal**: When MACD crosses above the signal line
        - **Bearish Signal**: When MACD crosses below the signal line
        
        Traders use MACD to understand momentum shifts, but it should not be treated as a standalone buy or sell signal.
        """)
    
    with st.expander("📉 What is Volatility?"):
        st.markdown("""
        **Volatility** measures how much a stock's price fluctuates over time.
        
        - **High Volatility (>30%)**: Large price swings, higher risk but potentially higher reward
        - **Moderate Volatility (15-30%)**: Average price movement
        - **Low Volatility (<15%)**: Stable prices, lower risk but typically lower returns
        
        Volatility is usually expressed as an annualised percentage.
        """)
    
    with st.expander("⚖️ What is the Sharpe Ratio?"):
        st.markdown("""
        The **Sharpe Ratio** measures risk-adjusted return - essentially, how much return you get for each unit of risk.
        
        - **Sharpe > 1.0**: Good - returns are worth the risk
        - **Sharpe > 2.0**: Very good - excellent risk-adjusted performance
        - **Sharpe > 3.0**: Excellent - outstanding performance
        - **Sharpe < 0**: Poor - returns are less than a risk-free investment
        
        Formula: (Return - Risk-Free Rate) / Volatility
        """)
    
    with st.expander("📏 What are Moving Averages?"):
        st.markdown("""
        **Moving Averages** smooth out price data to identify trends.
        
        - **SMA 20 (Short-term)**: Average of last 20 days - quick to react
        - **SMA 50 (Medium-term)**: Average of last 50 days
        - **SMA 200 (Long-term)**: Average of last 200 days - slow but significant
        
        **Trading signals:**
        - Price above moving averages = Bullish
        - Price below moving averages = Bearish
        - "Golden Cross" (50-day crosses above 200-day) = Strong bullish signal
        - "Death Cross" (50-day crosses below 200-day) = Strong bearish signal
        """)


def render_settings_page():
    """Render settings page."""
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### API Configuration")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key for enhanced AI explanations"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API key saved for this session")
    
    st.markdown("""
    ---
    ### About This Tool
    
    **AI Analyst Copilot** is a financial decision support system designed for beginner investors.
    
    **Features:**
    - Stock data from Yahoo Finance
    - Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
    - AI-powered explanations using GPT
    - Stock comparison tools
    - Educational resources
    
    **Technology Stack:**
    - Python
    - Streamlit (Web Framework)
    - yfinance (Market Data)
    - OpenAI GPT (AI Explanations)
    - Plotly (Interactive Charts)
    - Pandas/NumPy (Data Analysis)
    
      
      
    """)


# Run the app
if __name__ == "__main__":
    main()
