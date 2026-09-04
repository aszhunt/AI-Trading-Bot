import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="Pro AI Trading Signal Bot", page_icon="📈", layout="centered"
)

st.title("🤖 Pro AI Trading Signal Bot")
st.write(
    "Real-time multi-indicator trend analysis engine for short-term outlook."
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

# Selectbox for pair selection
selected_pair_name = st.selectbox(
    "Select Trading Instrument:", list(pairs.keys())
)
ticker_symbol = pairs[selected_pair_name]


# NOTE: Cache removed completely so it fetches fresh live data on every click!
def fetch_market_data(symbol):
  # Using 5m interval for better short-term 10-min analysis
  df = yf.download(symbol, period="3d", interval="5m", progress=False)
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
  if rsi < 40:
    rsi_buy += 5
  elif rsi > 60:
    rsi_sell += 5
  else:
    rsi_buy += 2
    rsi_sell += 2

  # --- 3. MACD ---
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

  # --- Percentage Calculation ---
  total_buy_score = ma_buy + rsi_buy + macd_buy + bb_buy
  total_sell_score = ma_sell + rsi_sell + macd_sell + bb_sell
  total_score = total_buy_score + total_sell_score

  if total_score == 0:
    buy_percentage = 50.0
  else:
    buy_percentage = (total_buy_score / total_score) * 100

  sell_percentage = 100.0 - buy_percentage

  if buy_percentage >= 60:
    summary = "STRONG BUY"
  elif buy_percentage >= 53:
    summary = "BUY"
  elif sell_percentage >= 60:
    summary = "STRONG SELL"
  elif sell_percentage >= 53:
    summary = "SELL"
  else:
    summary = "NEUTRAL"

  return (
      summary,
      buy_percentage,
      sell_percentage,
      ma_buy,
      ma_sell,
      rsi_buy + macd_buy + bb_buy,
      rsi_sell + macd_sell + bb_sell,
      current_price,
  )


# Execution Button
if st.button("🚀 Generate Fresh Signal", use_container_width=True):
  with st.spinner("Fetching live market data and calculating indicators..."):
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

      st.progress(
          int(buy_pct), text=f"Buy Chances: {buy_pct:.1f}% vs Sell: {sell_pct:.1f}%"
      )

      st.markdown("---")
      col1, col2 = st.columns(2)
      with col1:
        st.markdown("**Moving Averages (EMA):**")
        st.text(f"Bullish: {b_ma} | Bearish: {s_ma}")
      with col2:
        st.markdown("**Advanced Indicators:**")
        st.text(f"Bullish: {b_ind} | Bearish: {s_ind}")
    else:
      st.error(
          "⚠️ Could not load data for this asset. Markets might be closed or"
          " symbol is invalid."
      )
