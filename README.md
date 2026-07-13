# AI Analyst Copilot

Final-year dissertation project: an AI-driven decision support system for financial market analysis, built to help beginner investors understand stock data through plain-English, AI-generated explanations rather than raw numbers.

## What it does

- Fetches live stock data (Yahoo Finance) for any ticker
- Calculates technical indicators: RSI, MACD, Bollinger Bands, moving averages (20/50/200-day), volatility, Sharpe ratio
- Uses GPT to turn those metrics into plain-English explanations of what's happening with a stock and why
- Compares up to 4 stocks side by side on price performance and key metrics
- Includes an in-app "Learn" section explaining each indicator for users new to investing
- Exports metrics as CSV
- Includes a disclaimer throughout: educational tool only, not financial advice

## Tech stack

- **Python** — core language
- **Streamlit** — web interface
- **yfinance** — market data
- **OpenAI GPT** — natural-language explanations of technical analysis
- **Plotly** — interactive charts
- **Pandas / NumPy** — data processing and metric calculations
- **Pytest** — unit and integration testing

## Project structure

```
├── app_v2.py           # Streamlit front-end: pages, charts, UI
├── ai_analyst.py        # Core logic: data fetching, metric calculations, AI explanation generation
├── test_app.py          # Unit and integration tests (pytest)
└── requirements.txt      # Dependencies
```

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app_v2.py
```

You'll need an OpenAI API key (entered in-app under Settings) to enable the AI-generated explanations.

## Testing

The project includes unit tests for data fetching (valid/invalid/empty ticker handling) and integration tests confirming the full metrics pipeline (returns, volatility, RSI, MACD, Bollinger Bands, moving averages) produces correctly structured output.

```bash
python -m pytest test_app.py -v
```

## Background

Built as part of a BSc Computing dissertation, focused on how AI can support (not replace) human financial decision-making by making technical analysis genuinely understandable rather than just accurate. This is not intended to be used as-is for real trading decisions, it's an educational and academic project.
