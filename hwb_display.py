import discord
import pandas as pd
import mplfinance as mpf
from matplotlib.patches import Rectangle
from io import BytesIO
from typing import Dict, List, Optional
import datetime

# --- 1. サマリー表示 ---
def create_summary_message(
    high_priority: List[Dict],
    mid_priority: List[Dict],
    low_priority: List[Dict],
    today_signals: List[Dict],
    recent_signals: Dict[int, List[str]]
) -> str:
    """Discord用のサマリーメッセージを生成する"""

    scan_time = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M JST')

    # --- 優先度別 監視候補 ---
    high_str = ", ".join([f"{s['symbol']}({s['score']})" for s in high_priority]) if high_priority else "なし"
    mid_str = ", ".join([f"{s['symbol']}({s['score']})" for s in mid_priority]) if mid_priority else "なし"
    low_str = ", ".join([f"{s['symbol']}({s['score']})" for s in low_priority]) if low_priority else "なし"

    # --- 当日シグナル ---
    today_high_conf = [s for s in today_signals if s['score'] >= 80]
    today_mid_conf = [s for s in today_signals if 60 <= s['score'] < 80]

    today_high_str = ", ".join([f"{s['symbol']}({s['score']})" for s in today_high_conf]) if today_high_conf else "なし"
    today_mid_str = ", ".join([f"{s['symbol']}({s['score']})" for s in today_mid_conf]) if today_mid_conf else "なし"

    # --- 直近シグナル ---
    recent_parts = []
    for days_ago, signals in sorted(recent_signals.items()):
        if signals:
            recent_parts.append(f"{days_ago}日前: " + ", ".join(signals))
    recent_str = "\n".join(recent_parts) if recent_parts else "なし"

    message = f"""
**AI判定システム - 高度分析**
スキャン時刻: {scan_time}
処理銘柄: 3,000 / ヒット率: 4.2% (仮)

📍 **監視候補（優先度別）**
【HIGH】{high_str}
【MID】 {mid_str}
【LOW】 {low_str}

🚀 **当日シグナル（スコア順）**
【確度高】{today_high_str}
【確度中】{today_mid_str}

📈 **直近シグナル（3営業日以内）**
{recent_str}
"""
    return message.strip()

# --- 2. 個別アラート表示 ---
def create_enhanced_alert_embed(symbol: str, alert_data: Dict) -> discord.Embed:
    """最適化版の個別アラートEmbedを作成する"""

    total_score = alert_data.get('total_score', 0)

    if total_score >= 80:
        color = discord.Color.gold()
        prefix = "🔥"
    elif total_score >= 60:
        color = discord.Color.blue()
        prefix = "⭐"
    else:
        color = discord.Color.grey()
        prefix = "📍"

    embed = discord.Embed(
        title=f"{prefix} {symbol} - スコア: {total_score}/100",
        color=color
    )

    if 'setup' in alert_data:
        setup = alert_data['setup']
        embed.add_field(
            name="セットアップ詳細",
            value=f"タイプ: {setup.get('type', 'N/A')}\n"
                  f"信頼度: {setup.get('confidence', 0):.0%}\n"
                  f"ゾーン幅: {setup.get('zone_width', 0):.2%}",
            inline=True
        )

    if 'fvg' in alert_data:
        fvg = alert_data['fvg']
        embed.add_field(
            name="FVG分析",
            value=f"ギャップ: {fvg.get('gap_percentage', 0):.2%}\n"
                  f"品質: {fvg.get('quality', 'N/A')}\n"
                  f"ボリューム: {fvg.get('volume_surge', 0):.1f}x",
            inline=True
        )

    if 'breakout' in alert_data and alert_data['breakout']:
        breakout = alert_data['breakout']
        embed.add_field(
            name="ブレイクアウト",
            value=f"強度: {breakout.get('breakout_score', 0)}/8\n"
                  f"確度: {breakout.get('confidence', 'N/A')}\n"
                  f"出来高: {'✅' if breakout.get('volume_confirmation') else '⚠️'}",
            inline=True
        )

    # プレースホルダーの仮データ
    alert_data.setdefault('position_size', 1.5)
    alert_data.setdefault('risk_reward_ratio', 3.0)
    alert_data.setdefault('expected_value', 0.05)

    embed.add_field(
        name="推奨ポジション",
        value=f"サイズ: {alert_data['position_size']:.1f}%\n"
              f"R/R比: {alert_data['risk_reward_ratio']:.1f}\n"
              f"期待値: {alert_data['expected_value']:.1%}",
        inline=False
    )

    return embed

# --- 3. チャート表示 ---
def fig_to_bytesio(fig):
    """MatplotlibのFigureをBytesIOに変換する"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def calculate_expected_move():
    """期待される動きを計算する（仮実装）"""
    return 5.0

def create_enhanced_chart(
    df: pd.DataFrame,
    setup: Dict,
    fvg: Dict,
    breakout: Optional[Dict]
) -> BytesIO:
    """最適化版のチャートを作成する"""

    df.index.name = 'Date'

    # プロット期間をセットアップの少し前からに限定
    start_date = setup['date'] - pd.Timedelta(days=40)
    plot_df = df[df.index >= start_date]

    # 各イベントの日付インデックスを取得
    setup_date_idx = plot_df.index.get_loc(setup['date'])
    fvg_date_idx = plot_df.index.get_loc(fvg['formation_date'])

    add_plots = []

    # 2. FVGボックス
    fvg_colors = {'HIGH': 'darkgreen', 'MEDIUM': 'green', 'LOW': 'lightgreen'}
    fvg_color = fvg_colors.get(fvg.get('quality', 'LOW'), 'lightgreen')
    fvg_patch = Rectangle(
        (fvg_date_idx - 2, fvg['lower_bound']),
        width=3,
        height=fvg['gap_size'],
        alpha=0.3,
        facecolor=fvg_color,
        edgecolor='none',
        label=f"FVG (Score: {fvg['score']})"
    )

    # スタイルとプロットの準備
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'figure.figsize': (12, 8)})
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style=style,
        title=f"{setup['symbol']} - HWB Strategy Analysis",
        volume=True,
        returnfig=True,
        figsize=(15, 10)
    )
    ax = axes[0]
    ax.add_patch(fvg_patch)

    # 3. ブレイクアウトマーカー (matplotlib.axes.Axes.scatterを使用)
    if breakout:
        breakout_idx = plot_df.index.get_loc(breakout['breakout_date'])
        marker_size = 200 if breakout.get('confidence') == 'HIGH' else 100
        ax.scatter(
            plot_df.index[breakout_idx],
            breakout['breakout_price'],
            marker='^',
            color='blue',
            s=marker_size,
            zorder=10,
            label='Breakout'
        )

    # 1. セットアップゾーン（Rectangleによる正しい描画）
    setup_zone_color = 'yellow' if setup.get('type') == 'PRIMARY' else 'lightgray'
    setup_date_idx_in_plot = plot_df.index.get_loc(setup['date'])
    setup_patch = Rectangle(
        (setup_date_idx_in_plot - 0.5, setup['zone_lower']), # -0.5でローソク足の中央に配置
        width=1,
        height=setup['zone_upper'] - setup['zone_lower'],
        alpha=0.2,
        facecolor=setup_zone_color,
        edgecolor=None,
        label=f"Setup Zone ({setup.get('confidence', 0):.0%})"
    )
    ax.add_patch(setup_patch)

    # 3b. ブレイクアウトレベルライン
    if breakout:
        ax.axhline(
            y=breakout['resistance_price'],
            color='red',
            linestyle='--',
            alpha=0.7,
            label='Breakout Level'
        )

    # 4. ボリュームサージ表示
    volume_ax = axes[2]
    volume_mean = plot_df['Volume'].rolling(20).mean()
    surge_mask = plot_df['Volume'] > volume_mean * 1.5
    if surge_mask.any():
        volume_ax.bar(
            plot_df.index[surge_mask],
            plot_df['Volume'][surge_mask],
            color='orange'
        )

    ax.legend()
    fig.tight_layout()

    return fig_to_bytesio(fig)