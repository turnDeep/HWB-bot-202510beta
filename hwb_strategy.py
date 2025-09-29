import numpy as np
import pandas as pd

# HWB戦略の本質的目的
class StrategyObjective:
    primary_goal = "長期上昇トレンドの押し目で仕掛ける"
    risk_reward_ratio = 3.0  # 目標リスクリワード比
    win_rate_target = 0.55   # 目標勝率
    signal_frequency = "1-5 signals/day from 3000 stocks"

# ==============================================================================
# ルール① : トレンドフィルター（最適化版）
# ==============================================================================
def optimized_rule1(df_daily, df_weekly):
    """
    動的トレンド強度評価システム
    """
    # データが十分か確認
    if 'Close' not in df_weekly or 'SMA200' not in df_weekly or len(df_weekly) == 0:
        return False
    if 'Close' not in df_daily or 'SMA200' not in df_daily or 'EMA200' not in df_daily or len(df_daily) == 0:
        return False

    # 基本条件：週足トレンド
    weekly_trend_score = 0

    # 1. 週足終値と200SMAの乖離率
    weekly_deviation = (df_weekly['Close'].iloc[-1] - df_weekly['SMA200'].iloc[-1]) / df_weekly['SMA200'].iloc[-1]

    if weekly_deviation > 0.20:
        weekly_trend_score = 3
    elif weekly_deviation > 0.10:
        weekly_trend_score = 2
    elif weekly_deviation > 0:
        weekly_trend_score = 1
    else:
        return False

    # 2. 日足条件の動的調整
    daily_close = df_daily['Close'].iloc[-1]
    daily_sma200 = df_daily['SMA200'].iloc[-1]
    daily_ema200 = df_daily['EMA200'].iloc[-1]

    if weekly_trend_score >= 2:
        return True
    else:
        return (daily_close > daily_sma200 * 0.97 or
                daily_close > daily_ema200 * 0.97)

# ==============================================================================
# ルール② : セットアップ検出（最適化版）
# ==============================================================================
# 未定義だった関数の仮定義
def detect_zone_setups(df_daily, lookback_days):
    # 仮実装：本来はゾーン内でのセットアップを検出するロジック
    return []

def detect_touch_setups(df_daily, lookback_days):
    # 仮実装：MAへのタッチを検出するロジック
    return []

def optimized_rule2_setups(df_daily, lookback_days=30, symbol="TEST"):
    """
    多層セットアップ検出システム
    """
    setups = []

    # データが十分か確認
    if len(df_daily) < lookback_days:
        return setups

    # プライマリセットアップ（高確度）
    primary_setups = detect_zone_setups(df_daily, lookback_days)
    setups.extend(primary_setups)

    # セカンダリセットアップ（中確度）
    secondary_setups = detect_touch_setups(df_daily, lookback_days)
    setups.extend(secondary_setups)

    # ATRベースの動的ゾーン
    atr = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    atr_percentage = atr / df_daily['Close']

    for i in range(len(df_daily) - lookback_days, len(df_daily)):
        row = df_daily.iloc[i]

        # 動的ゾーン計算
        zone_width = max(
            abs(row['SMA200'] - row['EMA200']),
            row['Close'] * atr_percentage.iloc[i] * 0.5
        )
        zone_upper = max(row['SMA200'], row['EMA200']) + zone_width * 0.2
        zone_lower = min(row['SMA200'], row['EMA200']) - zone_width * 0.2

        common_data = {
            'date': df_daily.index[i],
            'zone_width': zone_width / row['Close'],
            'zone_lower': zone_lower,
            'zone_upper': zone_upper,
            'row': row,
            'symbol': symbol
        }

        # タイプA: 完全ゾーン内
        if zone_lower <= row['Open'] <= zone_upper and zone_lower <= row['Close'] <= zone_upper:
            setups.append({
                **common_data,
                'type': 'PRIMARY',
                'confidence': 0.85,
            })

        # タイプB: 片足ゾーン内
        elif (zone_lower <= row['Open'] <= zone_upper) or (zone_lower <= row['Close'] <= zone_upper):
            body_center = (row['Open'] + row['Close']) / 2
            if zone_lower <= body_center <= zone_upper:
                setups.append({
                    **common_data,
                    'type': 'SECONDARY',
                    'confidence': 0.65,
                })

    return setups

# ==============================================================================
# ルール③ : FVG検出（最適化版）
# ==============================================================================
def optimized_fvg_detection(df_daily, setup_date, max_days=20):
    """
    拡張FVG検出 + スコアリングシステム
    """
    fvgs = []
    try:
        setup_idx = df_daily.index.get_loc(setup_date)
    except KeyError:
        return fvgs # setup_dateがインデックスにない場合は空リストを返す

    for i in range(setup_idx + 3, min(setup_idx + max_days, len(df_daily))):
        candle_1 = df_daily.iloc[i-2]
        candle_2 = df_daily.iloc[i-1]
        candle_3 = df_daily.iloc[i]

        standard_gap = candle_3['Low'] - candle_1['High']

        fvg_score = 0

        if standard_gap > 0:
            gap_percentage = standard_gap / candle_1['High']

            if gap_percentage > 0.005: fvg_score += 3
            elif gap_percentage > 0.002: fvg_score += 2
            elif gap_percentage > 0.001: fvg_score += 1

            volume_surge = candle_3['Volume'] / df_daily['Volume'].rolling(20).mean().iloc[i]
            if volume_surge > 1.5: fvg_score += 2
            elif volume_surge > 1.2: fvg_score += 1

            ma_center = (candle_3['SMA200'] + candle_3['EMA200']) / 2
            price_center = (candle_3['Open'] + candle_3['Close']) / 2
            ma_deviation = abs(price_center - ma_center) / ma_center

            volatility = df_daily['Close'].pct_change().rolling(20).std().iloc[i]
            dynamic_threshold = min(0.05 + volatility * 2, 0.10)

            if ma_deviation <= dynamic_threshold * 0.5: fvg_score += 3
            elif ma_deviation <= dynamic_threshold: fvg_score += 2
            elif ma_deviation <= dynamic_threshold * 1.5: fvg_score += 1

            if fvg_score >= 3:
                fvgs.append({
                    'formation_date': df_daily.index[i],
                    'gap_size': standard_gap,
                    'gap_percentage': gap_percentage,
                    'score': fvg_score,
                    'volume_surge': volume_surge,
                    'ma_deviation': ma_deviation,
                    'quality': 'HIGH' if fvg_score >= 6 else 'MEDIUM' if fvg_score >= 4 else 'LOW',
                    'lower_bound': candle_1['High'] # 不足していたキーを追加
                })

    return sorted(fvgs, key=lambda x: x['score'], reverse=True)

# ==============================================================================
# ルール④ : ブレイクアウト検出（最適化版）
# ==============================================================================
def optimized_breakout_detection(df_daily, setup, fvg):
    """
    多要素確認型ブレイクアウトシステム
    """
    fvg_date = fvg['formation_date']
    fvg_idx = df_daily.index.get_loc(fvg_date)
    setup_idx = df_daily.index.get_loc(setup['date'])

    lookback_window = min(20, fvg_idx - setup_idx)
    if lookback_window <= 0: return None

    resistance_data = df_daily.iloc[fvg_idx - lookback_window : fvg_idx]

    resistance_levels = {
        'high': resistance_data['High'].max(),
        'close': resistance_data['Close'].max(),
        'vwap': (resistance_data['Close'] * resistance_data['Volume']).sum() / resistance_data['Volume'].sum(),
        'pivot': (resistance_data['High'].max() + resistance_data['Low'].min() + resistance_data['Close'].iloc[-1]) / 3
    }

    main_resistance = np.median(list(resistance_levels.values()))

    current = df_daily.iloc[-1]

    recent_volatility = df_daily['Close'].pct_change().rolling(20).std().iloc[-1]
    breakout_threshold = max(0.002, min(0.01, recent_volatility * 3))

    breakout_conditions = {
        'price': current['Close'] > main_resistance * (1 + breakout_threshold),
        'volume': current['Volume'] > df_daily['Volume'].rolling(20).mean().iloc[-1] * 1.2,
        'momentum': df_daily['Close'].pct_change(5).iloc[-1] > 0,
        'support_intact': df_daily['Low'][fvg_idx:].min() > fvg['lower_bound'] * 0.98
    }

    breakout_score = sum([
        3 if breakout_conditions['price'] else 0,
        2 if breakout_conditions['volume'] else 0,
        1 if breakout_conditions['momentum'] else 0,
        2 if breakout_conditions['support_intact'] else 0
    ])

    if breakout_score >= 5:
        return {
            'breakout_date': df_daily.index[-1],
            'breakout_price': current['Close'],
            'resistance_price': main_resistance,
            'breakout_percentage': (current['Close'] / main_resistance - 1) * 100,
            'breakout_score': breakout_score,
            'confidence': 'HIGH' if breakout_score >= 7 else 'MEDIUM',
            'volume_confirmation': breakout_conditions['volume']
        }

    return None

# ==============================================================================
# 統合最適化システム
# ==============================================================================
class OptimizedHWBStrategy:
    def __init__(self):
        self.market_regime = self.detect_market_regime()
        self.volatility_level = self.calculate_market_volatility()

    def detect_market_regime(self):
        return 'TRENDING'

    def calculate_market_volatility(self):
        return 'NORMAL'

    def adaptive_parameters(self):
        params = {
            'TRENDING': {'setup_lookback': 30, 'fvg_search_days': 20, 'ma_proximity': 0.05, 'breakout_threshold': 0.003},
            'RANGING': {'setup_lookback': 45, 'fvg_search_days': 30, 'ma_proximity': 0.03, 'breakout_threshold': 0.005},
            'VOLATILE': {'setup_lookback': 20, 'fvg_search_days': 15, 'ma_proximity': 0.08, 'breakout_threshold': 0.008}
        }
        return params[self.market_regime]

    def position_sizing(self, signal):
        """シグナル強度に応じたポジションサイジング提案"""
        base_size = 1.0

        if signal.get('setup_type') == 'PRIMARY': base_size *= 1.2
        elif signal.get('setup_type') == 'SECONDARY': base_size *= 0.8

        base_size *= (signal.get('fvg_score', 0) / 10)
        base_size *= (signal.get('breakout_score', 0) / 10)

        return min(base_size, 2.0)