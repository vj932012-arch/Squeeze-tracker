def generate_intraday_signals(
    df: pd.DataFrame,
    fast_ema: int = 9,
    slow_ema: int = 21,
    atr_period: int = 14,
    rvol_window: int = 20,
    adx_period: int = 14,
    adx_threshold: float = 25.0
) -> pd.DataFrame:
    """Computes dynamic multi-factor entry thresholds for intraday directional debit spreads."""
    df = df.copy()

    # 1. EMAs and Normalized Delta
    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()

    # 2. ATR Calculation
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=atr_period).mean()

    # Normalize EMA separation by ATR
    df["ema_spread_norm"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"]

    # 3. Normalized Distance from VWAP
    df["vwap_dist_norm"] = (df["close"] - df["vwap"]) / df["atr"]

    # 4. Volume Validation (RVOL)
    df["vol_ma"] = df["volume"].rolling(window=rvol_window).mean()
    df["rvol"] = df["volume"] / df["vol_ma"]

    # 5. ADX and DMI Calculation
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=adx_period)
    
    adx_col = f"ADX_{adx_period}"
    dmp_col = f"DMP_{adx_period}"
    dmn_col = f"DMN_{adx_period}"
    
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        # Fallback if calculation fails on limited data
        df[adx_col], df[dmp_col], df[dmn_col] = 0.0, 0.0, 0.0

    # 6. TTM Squeeze Calculation
    squeeze_df = df.ta.squeeze(lazybear=False, detailed=True)
    
    if squeeze_df is not None:
        df = pd.concat([df, squeeze_df], axis=1)
        
        # Dynamically extract standard pandas_ta squeeze columns
        sqz_on_col = [c for c in df.columns if "SQZ_ON" in c][0]
        sqz_off_col = [c for c in df.columns if "SQZ_OFF" in c][0]
        
        # Safely extract the histogram column (ignoring the ON/OFF/NO boolean columns)
        hist_col = [c for c in df.columns if "SQZ" in c and "ON" not in c and "OFF" not in c and "NO" not in c][0]
        
        # Trigger 1: The Squeeze Firing (Dots transition Red -> Green)
        df["squeeze_firing"] = (df[sqz_off_col] == 1) & (df[sqz_on_col].shift(1) == 1)
        
        # Trigger 2: Momentum Histogram Direction and Acceleration
        df["hist_light_blue"] = (df[hist_col] > 0) & (df[hist_col] > df[hist_col].shift(1))
        df["hist_red"] = (df[hist_col] < 0) & (df[hist_col] < df[hist_col].shift(1))
    else:
        # Fallbacks if squeeze fails to calculate
        df["squeeze_firing"] = False
        df["hist_light_blue"] = False
        df["hist_red"] = False

    # 7. Session Phase Filtering
    time = df.index.time
    t_start_am = pd.to_datetime("09:50:00").time()
    t_end_am = pd.to_datetime("11:30:00").time()
    t_start_pm = pd.to_datetime("13:45:00").time()
    t_end_pm = pd.to_datetime("15:15:00").time()

    session_active = ((time >= t_start_am) & (time <= t_end_am)) | (
        (time >= t_start_pm) & (time <= t_end_pm)
    )

    # 8. Unified Signal Logic (Structure + Trend + Volatility Breakout)
    call_spread_trigger = (
        session_active
        & (df["ema_spread_norm"] > 0.15)
        & (df["vwap_dist_norm"] >= 0.20)
        & (df["vwap_dist_norm"] <= 1.10)
        & (df["rvol"] >= 1.30)
        & (df["close"] > df["open"])
        & (df[adx_col] >= adx_threshold)
        & (df[dmp_col] > df[dmn_col])
        & df["squeeze_firing"]
        & df["hist_light_blue"]
    )

    put_spread_trigger = (
        session_active
        & (df["ema_spread_norm"] < -0.15)
        & (df["vwap_dist_norm"] <= -0.20)
        & (df["vwap_dist_norm"] >= -1.10)
        & (df["rvol"] >= 1.30)
        & (df["close"] < df["open"])
        & (df[adx_col] >= adx_threshold)
        & (df[dmn_col] > df[dmp_col])
        & df["squeeze_firing"]
        & df["hist_red"]
    )

    df["signal"] = 0
    df.loc[call_spread_trigger, "signal"] = 1
    df.loc[put_spread_trigger, "signal"] = -1

    # Filter out consecutive duplicate signals (take initial impulse only)
    df["entry_signal"] = np.where(
        (df["signal"] != 0) & (df["signal"] != df["signal"].shift(1)),
        df["signal"],
        0,
    )

    return df
