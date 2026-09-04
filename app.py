import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="Pro AI Trading Signal Bot", page_icon="📈", layout="centered"
)

# Custom Styling for UI look
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 10px; border: 1px solid #30333d; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🤖 Pro AI Trading Signal Bot")
st.write(
    "Advanced multi-indicator trend analysis engine for short-term (10-minute)"
    " outlook."
)

# Popular Trading Pairs Dictionary (Yahoo Finance Tickers)
pairs = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
}

# Sidebar / Selectbox for pair selection
selected_pair_name = st.selectbox(
    "Select Trading Instrument:", list(pairs.keys())
)
ticker_symbol = pairs[selected_pair_name]


@st.cache_data(ttl=300)
def fetch_market_data(symbol):
  # Fetching intraday historical data optimized for indicators
  df = yf.download(symbol, period="5d", interval="15m", progress=False)
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
  return df


def analyze_market_advanced(df):
  if df.empty or len(df) < 30:
    return "NEUTRAL", 50.0, 0, 0, 0, 0, 0.0

  close = df["Close"].squeeze()
  if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

  current_price = float(close.iloc[-1])

  # --- 1. Moving Averages (EMA 9 & EMA 21) ---
  ema_9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
  ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]

  ma_buy, ma_sell = 0, 0
  if current_price > ema_9:
    ma_buy += 3
  else:
    ma_sell += 3
  if ema_9 > ema_21:
    ma_buy += 3
  else:
    ma_sell += 3

  # --- 2. Relative Strength Index (RSI 14) ---
  delta = close.diff()
  gain = delta.clip(lower=0).rolling(window=14).mean()
  loss = (-delta.clip(upper=0)).rolling(window=14).mean()
  current_gain = gain.iloc[-1]
  current_loss = loss.iloc[-1]

  if current_loss == 0:
    rsi = 100.0
  elif current_gain == 0:
    rsi = 0.0
  else:
    rs = current_gain / current_loss
    rsi = 100 - (100 / (1 + rs))

  rsi_buy, rsi_sell = 0, 0
  if rsi < 35:  # Oversold (Strong Buy opportunity)
    rsi_buy += 5
  elif rsi > 65:  # Overbought (Strong Sell opportunity)
    rsi_sell += 5
  elif rsi < 50:
    rsi_buy += 2
  else:
    rsi_sell += 2

  # --- 3. MACD (Moving Average Convergence Divergence) ---
  exp1 = close.ewm(span=12, adjust=False).mean()
  exp2 = close.ewm(span=26, adjust=False).mean()
  macd = exp1 - exp2
  signal_line = macd.ewm(span=9, adjust=False).mean()
  current_macd = macd.iloc[-1]
  current_signal = signal_line.iloc[-1]

  macd_buy, macd_sell = 0, 0
  if current_macd > current_signal:
    macd_buy += 4
  else:
    macd_sell += 4

  # --- 4. Bollinger Bands ---
  sma_20 = close.rolling(window=20).mean().iloc[-1]
  std_20 = close.rolling(window=20).std().iloc[-1]
  upper_band = sma_20 + (std_20 * 2)
  lower_band = sma_20 - (std_20 * 2)

  bb_buy, bb_sell = 0, 0
  if current_price <= lower_band:
    bb_buy += 4
  elif current_price >= upper_band:
    bb_sell += 4
  elif current_price > sma_20:
    bb_buy += 2
  else:
    bb_sell += 2

  # --- Aggregation and Percentage Calculation ---
  total_buy_score = ma_buy + rsi_buy + macd_buy + bb_buy
  total_sell_score = ma_sell + rsi_sell + macd_sell + bb_sell
  total_score = total_buy_score + total_sell_score

  if total_score == 0:
    buy_percentage = 50.0
  else:
    buy_percentage = (total_buy_score / total_score) * 100

  sell_percentage = 100.0 - buy_percentage

  # Decision logic based on multi-indicator score threshold
  if buy_percentage >= 65:
    summary = "STRONG BUY"
  elif buy_percentage >= 55:
    summary = "BUY"
  elif sell_percentage >= 65:
    summary = "STRONG SELL"
  elif sell_percentage >= 55:
    summary = "SELL"
  else:
    summary = "NEUTRAL"

  total_ma_buy = ma_buy
  total_ma_sell = ma_sell
  total_ind_buy = rsi_buy + macd_buy + bb_buy
  total_ind_sell = rsi_sell + macd_sell + bb_sell

  return (
      summary,
      buy_percentage,
      sell_percentage,
      total_ma_buy,
      total_ma_sell,
      total_ind_buy,
      total_ind_sell,
      current_price,
  )


# Execution Button
if st.button("🚀 Generate AI Signal", use_container_width=True):
  with st.spinner(
      "Fetching live market feeds & computing multi-indicator AI models..."
  ):
    df = fetch_market_data(ticker_symbol)
    if not df.empty and "Close" in df.columns:
      (
          summary,
          buy_pct,
          sell_pct,
          b_ma,
          s_ma,
          b_ind,
          s_ind,
          price,
      ) = analyze_market_advanced(df)

      st.markdown("---")
      st.subheader(f"📊 Analysis Report for: {selected_pair_name}")
      st.metric(label="Current Live Price", value=f"{price:.4f}")

      # Color-coded UI alerts for signal output
      if "BUY" in summary:
        st.success(
            f"### Signal Summary: {summary}\n- **Buy Probability:** `{buy_pct:.1f}%`"
            f" \n- **Sell Probability:** `{sell_pct:.1f}%`"
        )
      elif "SELL" in summary:
        st.error(
            f"### Signal Summary: {summary}\n- **Buy Probability:** `{buy_pct:.1f}%`"
            f" \n- **Sell Probability:** `{sell_pct:.1f}%`"
        )
      else:
        st.warning(
            f"### Signal Summary: {summary}\n- **Buy Probability:** `{buy_pct:.1f}%`"
            f" \n- **Sell Probability:** `{sell_pct:.1f}%`"
        )

      # Progress bars for percentage visualization
      st.write("**Chances Breakdown:**")
      st.progress(
          int(buy_pct), text=f"Buy Chances: {buy_pct:.1f}% vs Sell: {sell_pct:.1f}%"
      )

      st.markdown("---")
      col1, col2 = st.columns(2)
      with col1:
        st.markdown("**Moving Averages (EMA 9/21):**")
        st.text(f"Bullish Score: {b_ma} | Bearish Score: {s_ma}")
      with col2:
        st.markdown("**Advanced Indicators (RSI, MACD, BB):**")
        st.text(f"Bullish Score: {b_ind} | Bearish Score: {s_ind}")

      st.info(
          "💡 *Note: This bot uses advanced technical indicators to project"
          " short-term 10-minute trends based on recent candlestick price action.*"
      )
    else:
      st.error(
          "⚠️ Data fetch error or market closed for this asset. Please try"
          " another pair."
      )