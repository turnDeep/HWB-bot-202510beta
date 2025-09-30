import json
import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests
import asyncio
import concurrent.futures
from typing import List, Dict, Set, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# 設定
DB_PATH = "data/hwb_cache.db"
RUSSELL3000_CACHE_DAYS = 7
BATCH_SIZE = 20
MAX_WORKERS = 5
SETUP_LOOKBACK_DAYS = 60
FVG_SEARCH_DAYS = 30
SIGNAL_COOLING_PERIOD = 14

# HWB戦略パラメータ
PROXIMITY_PERCENTAGE = 0.05
FVG_ZONE_PROXIMITY = 0.10
BREAKOUT_THRESHOLD = 0.001
MA_PERIOD = 200


class HWBDatabaseManager:
    """SQLiteデータベース管理（簡略版）"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """データベースの初期化"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Russell 3000銘柄テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS russell3000_symbols (
                    symbol TEXT PRIMARY KEY,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 株価データテーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    sma200 REAL,
                    ema200 REAL,
                    weekly_sma200 REAL,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            # シグナル履歴テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    symbol TEXT,
                    signal_date DATE,
                    signal_type TEXT,
                    score REAL,
                    chart_base64 TEXT,
                    details TEXT,
                    PRIMARY KEY (symbol, signal_date)
                )
            ''')

            conn.commit()

    def get_russell3000_symbols(self) -> Set[str]:
        """Russell 3000銘柄リストを取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # キャッシュチェック
            cursor.execute('''
                SELECT symbol FROM russell3000_symbols
                WHERE last_updated > datetime('now', '-7 days')
            ''')

            cached_symbols = {row[0] for row in cursor.fetchall()}

            if cached_symbols:
                logger.info(f"Russell 3000銘柄をDBから取得: {len(cached_symbols)}銘柄")
                return cached_symbols

            # 新規取得
            symbols = self._fetch_russell3000_symbols()

            if symbols:
                cursor.execute('DELETE FROM russell3000_symbols')
                cursor.executemany(
                    'INSERT INTO russell3000_symbols (symbol) VALUES (?)',
                    [(s,) for s in symbols]
                )
                conn.commit()

            return symbols

    def _fetch_russell3000_symbols(self) -> Set[str]:
        """Russell 3000銘柄を取得"""
        symbols = set()

        try:
            # S&P500を取得
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_symbols = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
            symbols.update(sp500_symbols)

            # NASDAQ100を追加
            nasdaq100_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(nasdaq100_url)
            nasdaq_symbols = tables[4]["Ticker"].tolist() if len(tables) > 4 else []
            symbols.update(nasdaq_symbols)

            # 追加の主要銘柄
            additional = ["PLTR", "SNOW", "COIN", "HOOD", "SOFI", "RIVN", "LCID"]
            symbols.update(additional)

            logger.info(f"取得した銘柄数: {len(symbols)}")
            return symbols

        except Exception as e:
            logger.error(f"銘柄取得エラー: {e}")
            # 最小限のリスト
            return {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"}

    def save_signal(self, symbol: str, signal_date: str, signal_type: str, score: float, chart_base64: str, details: str):
        """シグナルをDBに保存または更新する"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signal_history (symbol, signal_date, signal_type, score, chart_base64, details)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, signal_date) DO UPDATE SET
                signal_type = excluded.signal_type,
                score = excluded.score,
                chart_base64 = excluded.chart_base64,
                details = excluded.details
            ''', (symbol, signal_date, signal_type, score, chart_base64, details))
            conn.commit()

    def get_recent_signals(self, days: int = 3) -> List[Dict]:
        """直近のシグナル詳細データを取得する"""
        recent_signals_details = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 直近`days`日間のデータを取得
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            # 今日の日付は除外する
            today_date = datetime.now().strftime('%Y-%m-%d')

            cursor.execute('''
                SELECT details FROM signal_history
                WHERE signal_date >= ? AND signal_date < ? AND signal_type = 's2_breakout'
                ORDER BY signal_date DESC, score DESC
            ''', (start_date, today_date))

            for row in cursor.fetchall():
                if row[0]:
                    try:
                        # JSON文字列を辞書に変換して追加
                        recent_signals_details.append(json.loads(row[0]))
                    except json.JSONDecodeError as e:
                        logger.error(f"DB内のJSONデコードエラー: {e} - データ: {row[0]}")

        return recent_signals_details


class HWBAnalyzer:
    """HWB戦略分析クラス（HanaView版）"""

    def __init__(self, db_manager: HWBDatabaseManager):
        self.db = db_manager
        self.session = requests.Session(impersonate="safari15_5")

    def get_stock_data(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """株価データ取得"""
        try:
            stock = yf.Ticker(symbol, session=self.session)

            # 日足データ（6ヶ月）
            df_daily = stock.history(period="6mo", interval="1d")
            if df_daily.empty or len(df_daily) < 100:
                return None, None

            # 週足データ（2年）
            df_weekly = stock.history(period="2y", interval="1wk")
            if df_weekly.empty or len(df_weekly) < 52:
                return None, None

            # タイムゾーン除去
            df_daily.index = df_daily.index.tz_localize(None)
            df_weekly.index = df_weekly.index.tz_localize(None)

            # 移動平均を計算
            df_daily['SMA200'] = df_daily['Close'].rolling(window=min(200, len(df_daily)), min_periods=50).mean()
            df_daily['EMA200'] = df_daily['Close'].ewm(span=min(200, len(df_daily)), min_periods=50, adjust=False).mean()
            df_weekly['SMA200'] = df_weekly['Close'].rolling(window=min(200, len(df_weekly)), min_periods=50).mean()

            # 週足SMAを日足に結合
            df_daily['Weekly_SMA200'] = np.nan
            df_daily['Weekly_Close'] = np.nan

            for idx, row in df_weekly.iterrows():
                if pd.notna(row['SMA200']):
                    week_start = idx - pd.Timedelta(days=idx.weekday())
                    week_end = week_start + pd.Timedelta(days=6)
                    mask = (df_daily.index >= week_start) & (df_daily.index <= week_end)
                    if mask.any():
                        df_daily.loc[mask, 'Weekly_SMA200'] = row['SMA200']
                        df_daily.loc[mask, 'Weekly_Close'] = row['Close']

            df_daily['Weekly_SMA200'] = df_daily['Weekly_SMA200'].ffill()
            df_daily['Weekly_Close'] = df_daily['Weekly_Close'].ffill()

            return df_daily, df_weekly

        except Exception as e:
            logger.debug(f"データ取得エラー ({symbol}): {e}")
            return None, None

    def check_rule1(self, df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> bool:
        """ルール①: トレンドチェック"""
        if df_daily is None or df_weekly is None:
            return False

        latest = df_daily.iloc[-1]

        # 週足条件
        weekly_condition = (
            pd.notna(latest.get('Weekly_SMA200')) and
            pd.notna(latest.get('Weekly_Close')) and
            latest['Weekly_Close'] > latest['Weekly_SMA200']
        )

        # 日足条件（どちらかのMAを上回る）
        daily_condition = (
            pd.notna(latest.get('SMA200')) and
            pd.notna(latest.get('EMA200')) and
            (latest['Close'] > latest['SMA200'] or latest['Close'] > latest['EMA200'])
        )

        return weekly_condition and daily_condition

    def find_setups(self, df_daily: pd.DataFrame) -> List[Dict]:
        """ルール②: セットアップ検出"""
        setups = []

        valid_data = df_daily[(df_daily['SMA200'].notna()) & (df_daily['EMA200'].notna())].tail(SETUP_LOOKBACK_DAYS)

        for i in range(len(valid_data)):
            row = valid_data.iloc[i]
            zone_upper = max(row['SMA200'], row['EMA200'])
            zone_lower = min(row['SMA200'], row['EMA200'])

            # ゾーン内チェック
            if (zone_lower <= row['Open'] <= zone_upper and
                zone_lower <= row['Close'] <= zone_upper):

                setups.append({
                    'date': valid_data.index[i],
                    'open': row['Open'],
                    'close': row['Close'],
                    'high': row['High'],
                    'low': row['Low'],
                    'sma200': row['SMA200'],
                    'ema200': row['EMA200'],
                    'zone_upper': zone_upper,
                    'zone_lower': zone_lower
                })

        return setups

    def detect_fvg(self, df_daily: pd.DataFrame, setup_date: pd.Timestamp) -> List[Dict]:
        """ルール③: FVG検出"""
        fvgs = []

        try:
            setup_idx = df_daily.index.get_loc(setup_date)
        except KeyError:
            return fvgs

        search_end = min(setup_idx + FVG_SEARCH_DAYS, len(df_daily) - 1)

        for i in range(setup_idx + 3, search_end + 1):
            candle_1 = df_daily.iloc[i-2]
            candle_2 = df_daily.iloc[i-1]
            candle_3 = df_daily.iloc[i]

            # Bullish FVG
            gap = candle_3['Low'] - candle_1['High']

            if gap > 0 and gap / candle_1['High'] > 0.001:
                # MA近接チェック
                if self._check_ma_proximity(candle_3, candle_1):
                    fvgs.append({
                        'formation_date': df_daily.index[i],
                        'upper_bound': candle_3['Low'],
                        'lower_bound': candle_1['High'],
                        'gap_size': gap,
                        'gap_percentage': gap / candle_1['High'] * 100
                    })

        return fvgs

    def _check_ma_proximity(self, candle_3: pd.Series, candle_1: pd.Series) -> bool:
        """MA近接条件チェック"""
        if pd.isna(candle_3.get('SMA200')) or pd.isna(candle_3.get('EMA200')):
            return False

        # 条件A: 3本目の始値or終値がMA±5%以内
        for price in [candle_3['Open'], candle_3['Close']]:
            for ma in [candle_3['SMA200'], candle_3['EMA200']]:
                if abs(price - ma) / ma <= PROXIMITY_PERCENTAGE:
                    return True

        # 条件B: FVGゾーン中心がMA±10%以内
        fvg_center = (candle_1['High'] + candle_3['Low']) / 2
        for ma in [candle_3['SMA200'], candle_3['EMA200']]:
            if abs(fvg_center - ma) / ma <= FVG_ZONE_PROXIMITY:
                return True

        return False

    def check_breakout(self, df_daily: pd.DataFrame, setup: Dict, fvg: Dict) -> Optional[Dict]:
        """ルール④: ブレイクアウトチェック"""
        setup_date = setup['date']
        fvg_date = fvg['formation_date']

        try:
            setup_idx = df_daily.index.get_loc(setup_date)
            fvg_idx = df_daily.index.get_loc(fvg_date)
        except KeyError:
            return None

        # レジスタンス計算
        resistance_start = setup_idx + 1
        resistance_end = fvg_idx

        if resistance_end <= resistance_start:
            resistance_start = max(0, setup_idx - 10)
            resistance_end = setup_idx + 1

        resistance_high = df_daily.iloc[resistance_start:resistance_end]['High'].max()

        # FVG下限チェック
        post_fvg = df_daily.iloc[fvg_idx + 1:]
        if len(post_fvg) > 0:
            if post_fvg['Low'].min() < fvg['lower_bound']:
                return None

        # ブレイクアウトチェック（最新日のみ）
        current = df_daily.iloc[-1]
        if current['Close'] > resistance_high * (1 + BREAKOUT_THRESHOLD):
            return {
                'breakout_date': df_daily.index[-1],
                'breakout_price': current['Close'],
                'resistance_price': resistance_high,
                'breakout_percentage': (current['Close'] / resistance_high - 1) * 100
            }

        return None

    def create_chart_base64(self, symbol: str, df_daily: pd.DataFrame,
                            signal_type: str = None, setup: Dict = None,
                            fvg: Dict = None, breakout: Dict = None) -> str:
        """チャートをBase64エンコード画像として生成"""
        try:
            # 表示期間（直近180日）
            df_plot = df_daily.tail(180).copy()

            if len(df_plot) < 20:
                return None

            # mplfinanceスタイル
            mc = mpf.make_marketcolors(
                up='green', down='red', edge='inherit',
                wick={'up':'green', 'down':'red'}, volume='in'
            )
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)

            # 追加プロット
            apds = []

            # 移動平均線
            if 'SMA200' in df_plot.columns and not df_plot['SMA200'].isna().all():
                apds.append(mpf.make_addplot(df_plot['SMA200'], color='#9370DB', width=2))

            if 'EMA200' in df_plot.columns and not df_plot['EMA200'].isna().all():
                apds.append(mpf.make_addplot(df_plot['EMA200'], color='purple', width=2))

            if 'Weekly_SMA200' in df_plot.columns and not df_plot['Weekly_SMA200'].isna().all():
                apds.append(mpf.make_addplot(df_plot['Weekly_SMA200'], color='blue', width=3))

            # タイトル設定
            title = f'{symbol} - HWB Analysis'
            if signal_type == 's2_breakout':
                title += ' [SIGNAL]'
            elif signal_type == 's1_fvg':
                title += ' [CANDIDATE]'

            # チャート作成
            fig, axes = mpf.plot(
                df_plot, type='candle', style=s, volume=True,
                addplot=apds if apds else None,
                title=title, returnfig=True,
                figsize=(10, 6), panel_ratios=(3, 1)
            )

            ax = axes[0]

            # セットアップゾーンの描画
            if setup and setup['date'] in df_plot.index:
                setup_idx = df_plot.index.get_loc(setup['date'])
                rect = patches.Rectangle(
                    (setup_idx - 0.5, setup['zone_lower']),
                    1, setup['zone_upper'] - setup['zone_lower'],
                    linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.3
                )
                ax.add_patch(rect)

            # FVGゾーンの描画
            if fvg and fvg['formation_date'] in df_plot.index:
                fvg_idx = df_plot.index.get_loc(fvg['formation_date'])
                rect = patches.Rectangle(
                    (fvg_idx - 2.5, fvg['lower_bound']),
                    3, fvg['gap_size'],
                    linewidth=2, edgecolor='green', facecolor='green', alpha=0.3
                )
                ax.add_patch(rect)

            # ブレイクアウトマーカー
            if breakout and breakout['breakout_date'] in df_plot.index:
                bo_idx = df_plot.index.get_loc(breakout['breakout_date'])
                ax.scatter(
                    bo_idx, breakout['breakout_price'] * 0.98,
                    marker='^', color='blue', s=200, zorder=5
                )
                # レジスタンスライン
                ax.axhline(
                    y=breakout['resistance_price'],
                    color='red', linestyle='--', alpha=0.5
                )

            # Base64エンコード
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plt.close()

            import base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"チャート生成エラー ({symbol}): {e}")
            return None


class HWBScanner:
    """HWBスキャナーメインクラス"""

    def __init__(self):
        self.db = HWBDatabaseManager(DB_PATH)
        self.analyzer = HWBAnalyzer(self.db)
        self.signal_history = {}
        self.recent_signals = {}

    async def scan_all_symbols(self, progress_callback=None):
        """全銘柄スキャン（非同期）"""
        # 直近シグナルの詳細リストを取得
        recent_signals_details = self.db.get_recent_signals()

        # 1. 直近シグナルのサマリーを作成
        self.recent_signals = {}
        today = datetime.now().date()
        for signal in recent_signals_details:
            signal_date_str = signal.get('breakout', {}).get('date')
            if not signal_date_str:
                continue
            signal_date = datetime.strptime(signal_date_str, '%Y-%m-%d').date()
            days_ago = (today - signal_date).days
            if days_ago > 0:
                if days_ago not in self.recent_signals:
                    self.recent_signals[days_ago] = []
                self.recent_signals[days_ago].append(signal['symbol'])

        # 2. 全チャートデータを準備（直近シグナル分を初期値とする）
        all_charts = {s['symbol']: s['chart'] for s in recent_signals_details if s.get('chart')}

        # Russell 3000銘柄を取得
        symbols = list(self.db.get_russell3000_symbols())
        total = len(symbols)

        logger.info(f"スキャン開始: {total}銘柄")

        monitoring_candidates = []
        today_signals = []

        # 進捗状況
        processed = 0

        # バッチ処理
        for i in range(0, total, BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]

            # 並列処理
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for symbol in batch:
                    futures.append(executor.submit(self._analyze_symbol, symbol))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        if result:
                            if result['signal_type'] == 's2_breakout':
                                today_signals.append(result)
                            elif result['signal_type'] == 's1_fvg':
                                monitoring_candidates.append(result)

                            # 当日スキャン分のチャートデータを追加
                            if result.get('chart'):
                                all_charts[result['symbol']] = result['chart']

                    except Exception as e:
                        logger.error(f"分析エラー: {e}")

                    processed += 1

                    # 進捗通知
                    if progress_callback and processed % 10 == 0:
                        await progress_callback(processed, total)

            # CPU負荷軽減
            await asyncio.sleep(0.1)

        # 結果をスコア順にソート
        monitoring_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        today_signals.sort(key=lambda x: x.get('score', 0), reverse=True)

        # JSONデータ作成
        result = {
            'scan_date': datetime.now().strftime('%Y-%m-%d'),
            'scan_time': datetime.now().strftime('%H:%M:%S'),
            'total_scanned': total,
            'summary': {
                'monitoring_candidates': [s['symbol'] for s in monitoring_candidates],
                'today_signals': [s['symbol'] for s in today_signals],
                'recent_signals': self.recent_signals
            },
            'signals': today_signals[:20],
            'candidates': monitoring_candidates[:30],
            'recent_signals_details': recent_signals_details, # 詳細リストを追加
            'charts': all_charts
        }

        return result

    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """個別銘柄分析"""
        try:
            # データ取得
            df_daily, df_weekly = self.analyzer.get_stock_data(symbol)

            if df_daily is None or df_weekly is None:
                return None

            # ルール①チェック
            if not self.analyzer.check_rule1(df_daily, df_weekly):
                return None

            # ルール②セットアップ検出
            setups = self.analyzer.find_setups(df_daily)
            if not setups:
                return None

            # 各セットアップをチェック
            for setup in setups:
                # 冷却期間チェック
                if not self._should_process(symbol, setup['date']):
                    continue

                # ルール③FVG検出
                fvgs = self.analyzer.detect_fvg(df_daily, setup['date'])

                for fvg in fvgs:
                    # ルール④ブレイクアウトチェック
                    breakout = self.analyzer.check_breakout(df_daily, setup, fvg)

                    # スコア計算
                    score = self._calculate_score(setup, fvg, breakout)

                    if breakout:
                        # シグナル発生
                        chart = self.analyzer.create_chart_base64(
                            symbol, df_daily, 's2_breakout', setup, fvg, breakout
                        )

                        # フロントエンドに返すための完全なシグナルオブジェクトを作成
                        signal_data = {
                            'symbol': symbol,
                            'signal_type': 's2_breakout',
                            'score': score,
                            'setup': {
                                'date': setup['date'].strftime('%Y-%m-%d'),
                                'zone_width': (setup['zone_upper'] - setup['zone_lower']) / setup['close']
                            },
                            'fvg': {
                                'date': fvg['formation_date'].strftime('%Y-%m-%d'),
                                'gap_percentage': fvg['gap_percentage']
                            },
                            'breakout': {
                                'date': breakout['breakout_date'].strftime('%Y-%m-%d'),
                                'percentage': breakout['breakout_percentage']
                            },
                            'chart': chart
                        }

                        # シグナルをDBに保存
                        if chart:
                            self.db.save_signal(
                                symbol=symbol,
                                signal_date=breakout['breakout_date'].strftime('%Y-%m-%d'),
                                signal_type='s2_breakout',
                                score=score,
                                chart_base64=chart,
                                details=json.dumps(signal_data)
                            )

                        return signal_data
                    elif fvg:
                        # 監視候補
                        chart = self.analyzer.create_chart_base64(
                            symbol, df_daily, 's1_fvg', setup, fvg, None
                        )

                        return {
                            'symbol': symbol,
                            'signal_type': 's1_fvg',
                            'score': score,
                            'setup': {
                                'date': setup['date'].strftime('%Y-%m-%d'),
                                'zone_width': (setup['zone_upper'] - setup['zone_lower']) / setup['close']
                            },
                            'fvg': {
                                'date': fvg['formation_date'].strftime('%Y-%m-%d'),
                                'gap_percentage': fvg['gap_percentage']
                            },
                            'chart': chart
                        }

            return None

        except Exception as e:
            logger.debug(f"分析エラー ({symbol}): {e}")
            return None

    def _should_process(self, symbol: str, setup_date: pd.Timestamp) -> bool:
        """冷却期間チェック"""
        if symbol not in self.signal_history:
            return True

        last_signal = self.signal_history.get(symbol)
        if last_signal:
            days_elapsed = (datetime.now() - last_signal).days
            return days_elapsed >= SIGNAL_COOLING_PERIOD

        return True

    def _calculate_score(self, setup: Dict, fvg: Dict, breakout: Optional[Dict]) -> int:
        """スコア計算（0-100）"""
        score = 0

        # セットアップスコア（最大30点）
        zone_width = (setup['zone_upper'] - setup['zone_lower']) / setup['close']
        if zone_width < 0.005:
            score += 30
        elif zone_width < 0.01:
            score += 20
        else:
            score += 10

        # FVGスコア（最大40点）
        if fvg['gap_percentage'] > 0.5:
            score += 40
        elif fvg['gap_percentage'] > 0.3:
            score += 30
        elif fvg['gap_percentage'] > 0.1:
            score += 20
        else:
            score += 10

        # ブレイクアウトスコア（最大30点）
        if breakout:
            if breakout['breakout_percentage'] > 1.0:
                score += 30
            elif breakout['breakout_percentage'] > 0.5:
                score += 20
            else:
                score += 10

        return min(score, 100)


# エクスポート用関数
async def run_hwb_scan(progress_callback=None):
    """HWBスキャン実行"""
    scanner = HWBScanner()
    result = await scanner.scan_all_symbols(progress_callback)

    # 結果をファイルに保存
    output_path = 'data/hwb_signals.json'
    os.makedirs('data', exist_ok=True)

    # チャートデータを別途保存（サイズが大きいため）
    charts = result.pop('charts', {})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # チャートは別ファイルに保存
    charts_path = 'data/hwb_charts.json'
    with open(charts_path, 'w', encoding='utf-8') as f:
        json.dump(charts, f, ensure_ascii=False)

    logger.info(f"スキャン結果を保存: {output_path}")

    return result