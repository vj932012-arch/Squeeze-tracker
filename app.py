import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="TTM Squeeze Scanner", page_icon="💥", layout="wide")
st.title("💥 Mega-Cap TTM Squeeze Scanner")
st.caption("Detects explosive volatility breakouts when Bollinger Bands narrow inside Keltner Channels.")

# ---------------------------------------------------------
# Data Engine
# ---------------------------------------------------------
# Defaulting to high-beta/mega-cap tech and your core watchlist
DEFAULT_TICKERS = ["SPY", "QQQ", "GOOGL", "NVDA", "AMZN"]

@st.cache_data(ttl=300) # 5-minute refresh to avoid IP bans
def fetch_and_calculate_squeeze(ticker: str):
    df = yf.download(ticker, period="10d", interval="5m", progress=False)
    
    if df.empty:
        return None
        
    # Flatten multi-index columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # Calculate TTM Squeeze using pandas_ta
    # Returns Squeeze Status (1=On, 0=Off) and Momentum Histogram
    squeeze_df = df.ta.squeeze(lazybear=False, detailed=True)
    
    if squeeze_df is not None:
        df = pd.concat([df, squeeze_df], axis=1)
        
        # Standard pandas_ta column names for Squeeze
        sqz_on_col = [c for c in df.columns if "SQZ_ON" in c][0]
        sqz_off_col = [c for c in df.columns if "SQZ_OFF" in c][0]
        hist_col = [c for c in df.columns if "SQZ_INC" in c or "SQZ_DEC" in c or "SQZ20" in c][0]
        
        # Strategy Logic: Find exactly when it transitions from ON to OFF
        df["squeeze_firing"] = (df[sqz_off_col] == 1) & (df[sqz_on_col].shift(1) == 1)
        
        # Determine Direction based on momentum histogram
        df["signal"] = 0
        df.loc[df["squeeze_firing"] & (df[hist_col] > 0), "signal"] = 1  # Call Spread
        df.loc[df["squeeze_firing"] & (df[hist_col] < 0), "signal"] = -1 # Put Spread
        
    return df

# ---------------------------------------------------------
# UI & Display
# ---------------------------------------------------------
if st.button("🔄 Scan Market"):
    st.cache_data.clear()

cols = st.columns(len(DEFAULT_TICKERS))

for col, ticker in zip(cols, DEFAULT_TICKERS):
    df = fetch_and_calculate_squeeze(ticker)
    
    with col:
        st.subheader(ticker)
        if df is None:
            st.error("Data error")
            continue
            
        latest = df.iloc[-1]
        current_price = latest["close"]
        signal = latest.get("signal", 0)
        
        st.metric("Last Price", f"${current_price:.2f}")
        
        if signal == 1:
             st.success("🟢 SQUEEZE FIRED: CALL SPREAD")
             st.write("Momentum is expanding upward.")
        elif signal == -1:
             st.error("🔴 SQUEEZE FIRED: PUT SPREAD")
             st.write("Momentum is expanding downward.")
        else:
             # Check if it's currently compressing
             sqz_on_col = [c for c in df.columns if "SQZ_ON" in c][0]
             if latest[sqz_on_col] == 1:
                 st.warning("🟡 SQUEEZE COMPRESSING")
                 st.write("Wait for the breakout.")
             else:
                 st.info("⚪ NO ACTIVE SQUEEZE")
                 st.write("Volatility is normal.")
