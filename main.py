import pandas as pd
import numpy as np
import hwb_strategy as hwb
import hwb_display as display

def generate_sample_data(days=100):
    """
    検証用のサンプルデータを生成する。
    トレンドフィルターを通過しやすいように、上昇トレンドと押し目を模擬する。
    """
    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D'))

    base_price = np.linspace(100, 150, days) + np.random.randn(days).cumsum() * 0.5

    df_daily = pd.DataFrame(index=dates)
    df_daily['Open'] = base_price + np.random.uniform(-1, 1, size=days)
    df_daily['High'] = df_daily['Open'] + np.random.uniform(0, 2, size=days)
    df_daily['Low'] = df_daily['Open'] - np.random.uniform(0, 2, size=days)
    df_daily['Close'] = (df_daily['High'] + df_daily['Low']) / 2 + np.random.uniform(-0.5, 0.5, size=days)
    df_daily['Volume'] = np.random.randint(100000, 500000, size=days)

    df_daily['SMA200'] = df_daily['Close'].rolling(window=min(50, days)).mean().bfill() * 0.95
    df_daily['EMA200'] = df_daily['Close'].ewm(span=min(50, days)).mean().bfill() * 0.96

    df_weekly = df_daily.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    })
    df_weekly['SMA200'] = df_weekly['Close'].rolling(window=min(10, len(df_weekly))).mean().bfill() * 0.9
    df_weekly = df_weekly.dropna()

    if len(df_daily) > 35:
        setup_idx = len(df_daily) - 25
        ma_mid_price = (df_daily.iloc[setup_idx]['SMA200'] + df_daily.iloc[setup_idx]['EMA200']) / 2
        df_daily.loc[df_daily.index[setup_idx], 'Open'] = ma_mid_price * 1.01
        df_daily.loc[df_daily.index[setup_idx], 'Close'] = ma_mid_price * 0.99
        df_daily.loc[df_daily.index[setup_idx], 'High'] = ma_mid_price * 1.02
        df_daily.loc[df_daily.index[setup_idx], 'Low'] = ma_mid_price * 0.98

    if len(df_daily) > 20:
        gap_idx = len(df_daily) - 15
        prev_high = df_daily.iloc[gap_idx - 2]['High']
        df_daily.loc[df_daily.index[gap_idx], 'Low'] = prev_high + 1.0
        df_daily.loc[df_daily.index[gap_idx], 'High'] = prev_high + 2.5
        df_daily.loc[df_daily.index[gap_idx], 'Close'] = prev_high + 2.0
        df_daily.loc[df_daily.index[gap_idx], 'Volume'] *= 2

    if len(df_daily) > 20:
        highs = df_daily['High'].iloc[-20:-1]
        if not highs.empty:
            df_daily.loc[df_daily.index[-1], 'Close'] = highs.max() + 1
            df_daily.loc[df_daily.index[-1], 'Volume'] *= 1.5

    return df_daily.dropna(), df_weekly

def run_verification():
    """
    戦略コードと表示コードを順に呼び出し、実行可能か検証する。
    """
    print("--- HWB戦略 全ルール最適化 深層分析検証 ---")
    # ... （戦略ルールの検証は省略）
    df_daily, df_weekly = generate_sample_data()
    is_trend_ok = hwb.optimized_rule1(df_daily, df_weekly)
    if not is_trend_ok:
        print("トレンドフィルターを通過せず、検証終了。")
        return

    setups = hwb.optimized_rule2_setups(df_daily, lookback_days=30, symbol="NVDA")
    if not setups:
        print("セットアップが検出されず、検証終了。")
        return
    best_setup = setups[-1]

    fvgs = hwb.optimized_fvg_detection(df_daily, setup_date=best_setup['date'])
    if not fvgs:
        print("FVGが検出されず、検証終了。")
        return
    best_fvg = fvgs[0]

    breakout_signal = hwb.optimized_breakout_detection(df_daily, best_setup, best_fvg)

    print("\n--- 表示機能 検証 ---")

    # 1. サマリーメッセージの検証
    print("\n[1. サマリーメッセージ]")
    mock_signals = [
        {'symbol': 'NVDA', 'score': 85}, {'symbol': 'AAPL', 'score': 82},
        {'symbol': 'MSFT', 'score': 65}, {'symbol': 'GOOGL', 'score': 62},
        {'symbol': 'TSLA', 'score': 45},
    ]
    summary_msg = display.create_summary_message(
        high_priority=[s for s in mock_signals if s['score'] >= 80],
        mid_priority=[s for s in mock_signals if 60 <= s['score'] < 80],
        low_priority=[s for s in mock_signals if s['score'] < 60],
        today_signals=[{'symbol': 'MOS', 'score': 92}, {'symbol': 'PANW', 'score': 75}],
        recent_signals={1: ["PLTR(+5.2%)"], 2: ["COIN(+8.5%)"]}
    )
    print(summary_msg)

    # 2. 個別アラートEmbedの検証
    print("\n[2. 個別アラートEmbed]")
    total_score = (best_setup.get('confidence', 0) * 30 +
                   best_fvg.get('score', 0) * 4 +
                   (breakout_signal.get('breakout_score', 0) if breakout_signal else 0) * 3)

    alert_data = {
        'total_score': min(int(total_score), 100),
        'setup': best_setup,
        'fvg': best_fvg,
        'breakout': breakout_signal
    }

    embed = display.create_enhanced_alert_embed("NVDA", alert_data)
    print(f"Embed Title: {embed.title}")
    print(f"Embed Color: {embed.color}")
    for field in embed.fields:
        print(f"  - Field: {field.name}\n    Value: {field.value.replace('\n', ' | ')}")

    # 3. チャート生成の検証
    print("\n[3. チャート生成]")
    try:
        chart_bytes = display.create_enhanced_chart(df_daily, best_setup, best_fvg, breakout_signal)
        with open("chart.png", "wb") as f:
            f.write(chart_bytes.getbuffer())
        print("チャートが 'chart.png' として正常に保存されました。")
    except Exception as e:
        print(f"チャートの生成中にエラーが発生しました: {e}")

    print("\n--- 全検証完了 ---")

if __name__ == "__main__":
    run_verification()