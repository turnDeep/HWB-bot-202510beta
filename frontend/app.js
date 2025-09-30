// グローバルなfetchWithAuth関数を定義
// 現状は認証機能が不要なため、標準のfetchをラップするだけ
const fetchWithAuth = (url, options) => {
    return fetch(url, options);
};

// HWB 200MAタブ機能
class HWB200MAManager {
    constructor() {
        this.data = null;
        this.isScanning = false;
        this.initEventListeners();
    }

    initEventListeners() {
        // スキャンボタン
        const scanBtn = document.getElementById('hwb-scan-btn');
        if (scanBtn) {
            scanBtn.addEventListener('click', () => this.startScan());
        }

        // 更新ボタン
        const refreshBtn = document.getElementById('hwb-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // タブが表示されたときにデータを取得
        document.addEventListener('tab-changed', (e) => {
            if (e.detail.tab === 'hwb200') {
                this.loadData();
            }
        });
    }

    async startScan() {
        if (this.isScanning) return;

        const scanBtn = document.getElementById('hwb-scan-btn');
        const loadingDiv = document.getElementById('hwb-loading');
        const contentDiv = document.getElementById('hwb-content');

        this.isScanning = true;
        scanBtn.disabled = true;
        scanBtn.textContent = 'スキャン中...';
        loadingDiv.style.display = 'block';
        contentDiv.style.display = 'none';

        try {
            const response = await fetchWithAuth('/api/hwb/scan', {
                method: 'POST'
            });

            if (!response.ok) throw new Error('スキャン開始に失敗しました');

            const result = await response.json();

            if (result.success) {
                // 成功メッセージ表示
                this.showStatus(`✅ ${result.message}`);

                // データを再読み込み
                setTimeout(() => this.loadData(), 2000);
            } else {
                throw new Error(result.message || 'スキャンエラー');
            }

        } catch (error) {
            console.error('HWBスキャンエラー:', error);
            this.showStatus(`❌ エラー: ${error.message}`, 'error');
        } finally {
            this.isScanning = false;
            scanBtn.disabled = false;
            scanBtn.textContent = '📡 スキャン実行';
            loadingDiv.style.display = 'none';
            contentDiv.style.display = 'block';
        }
    }

    async refreshData() {
        await this.loadData();
    }

    async loadData() {
        try {
            // ステータス確認
            const statusResponse = await fetchWithAuth('/api/hwb/status');
            const status = await statusResponse.json();

            if (!status.has_data) {
                this.showStatus('データがありません。スキャンを実行してください。', 'warning');
                document.getElementById('hwb-content').innerHTML =
                    '<div class="card"><p>データがありません。「スキャン実行」ボタンをクリックしてください。</p></div>';
                return;
            }

            // ステータス表示
            this.showStatus(
                `最終スキャン: ${status.last_scan} | ` +
                `${status.total_scanned}銘柄スキャン | ` +
                `シグナル: ${status.signals_count}件 | ` +
                `候補: ${status.candidates_count}件`
            );

            // データ取得
            const dataResponse = await fetchWithAuth('/api/hwb/data');
            this.data = await dataResponse.json();

            // レンダリング
            this.render();

        } catch (error) {
            console.error('HWBデータ読み込みエラー:', error);
            this.showStatus('❌ データ読み込みエラー', 'error');
        }
    }

    render() {
        if (!this.data) return;

        const container = document.getElementById('hwb-content');
        container.innerHTML = '';

        // サマリー表示
        this.renderSummary(container);

        // 監視候補チャート表示
        if (this.data.candidates && this.data.candidates.length > 0) {
            this.renderCandidateCharts(container);
        }

        // 当日シグナルチャート表示
        if (this.data.signals && this.data.signals.length > 0) {
            this.renderSignalCharts(container);
        }

        // 直近シグナルチャート表示
        const summary = this.data.summary || {};
        const recentSignals = summary.recent_signals || {};
        // Make sure there is at least one ticker in the recent_signals object
        if (Object.keys(recentSignals).length > 0 && Object.values(recentSignals).flat().length > 0) {
            this.renderRecentSignalCharts(container);
        }
    }

    renderSummary(container) {
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'hwb-summary';

        const summary = this.data.summary || {};
        const scanDate = this.data.scan_date || '';
        const scanTime = this.data.scan_time || '';

        summaryDiv.innerHTML = `
            <h2>🤖 AI判定システム - HWB Strategy</h2>
            <div class="scan-info">
                スキャン日時: ${scanDate} ${scanTime} |
                処理銘柄: ${this.data.total_scanned || 0}
            </div>

            <div class="hwb-summary-section">
                <h3>🚀 当日シグナル（ブレイクアウト）</h3>
                <div class="hwb-ticker-list">
                    ${this.renderTickers(summary.today_signals || [], 'signal')}
                </div>
            </div>

            <div class="hwb-summary-section">
                <h3>📍 監視候補（FVG検出）</h3>
                <div class="hwb-ticker-list">
                    ${this.renderTickers(summary.monitoring_candidates || [], 'candidate')}
                </div>
            </div>

            <div class="hwb-summary-section">
                <h3>📈 直近シグナル（3営業日以内）</h3>
                <div class="hwb-ticker-list">
                    ${this.renderRecentSignals(summary.recent_signals || {})}
                </div>
            </div>
        `;

        container.appendChild(summaryDiv);
    }

    renderTickers(tickers, type) {
        if (tickers.length === 0) return '<span style="color: #999;">なし</span>';

        return tickers.slice(0, 20).map(ticker =>
            `<a href="#chart-${ticker}" style="text-decoration: none;">
                <span class="hwb-ticker ${type}">${ticker}</span>
            </a>`
        ).join('');
    }

    renderRecentSignals(signals) {
        const items = [];
        for (const [days, tickers] of Object.entries(signals)) {
            if (tickers && tickers.length > 0) {
                tickers.forEach(ticker => {
                    const symbol = ticker; // The ticker is the symbol itself.
                    items.push(
                        `<a href="#chart-${symbol}" style="text-decoration: none;">
                            <span class="hwb-ticker recent">${symbol} (${days}日前)</span>
                        </a>`
                    );
                });
            }
        }
        return items.length > 0 ? items.join('') : '<span style="color: #999;">なし</span>';
    }

    renderSignalCharts(container) {
        const section = document.createElement('div');
        section.className = 'hwb-charts-section';
        section.innerHTML = '<h2>🚀 当日シグナル - 詳細チャート</h2>';

        const grid = document.createElement('div');
        grid.className = 'hwb-chart-grid';

        const signals = this.data.signals || [];
        signals.forEach(signal => {
            const chartCard = this.createChartCard(signal);
            if (chartCard) grid.appendChild(chartCard);
        });

        section.appendChild(grid);
        container.appendChild(section);
    }

    renderCandidateCharts(container) {
        const section = document.createElement('div');
        section.className = 'hwb-charts-section';
        section.innerHTML = '<h2>📍 監視候補 - 詳細チャート</h2>';

        const grid = document.createElement('div');
        grid.className = 'hwb-chart-grid';

        const candidates = (this.data.candidates || []).slice(0, 10); // 上位10件
        candidates.forEach(candidate => {
            const chartCard = this.createChartCard(candidate);
            if (chartCard) grid.appendChild(chartCard);
        });

        section.appendChild(grid);
        container.appendChild(section);
    }

    createChartCard(signal) {
        const card = document.createElement('div');
        card.className = 'hwb-chart-card';
        card.id = `chart-${signal.symbol}`; // ページ内リンク用のIDを追加

        const scoreClass = signal.score >= 80 ? 'high' :
                          signal.score >= 60 ? 'medium' : 'low';

        const signalType = signal.signal_type === 's2_breakout' ?
                          'ブレイクアウト' : 'FVG検出';

        // 情報テキスト作成
        let infoText = '';
        if (signal.setup) {
            infoText += `Setup: ${signal.setup.date} `;
        }
        if (signal.fvg) {
            infoText += `| FVG: ${signal.fvg.gap_percentage?.toFixed(2)}% `;
        }
        if (signal.breakout) {
            infoText += `| Breakout: +${signal.breakout.percentage?.toFixed(2)}%`;
        }

        card.innerHTML = `
            <div class="hwb-chart-header">
                <span class="hwb-chart-symbol">${signal.symbol}</span>
                <span class="hwb-chart-score ${scoreClass}">Score: ${signal.score}/100</span>
            </div>
            <div class="hwb-chart-info">
                <span>${signalType}</span>
                <span>${infoText}</span>
            </div>
        `;

        // チャート画像
        if (signal.chart || this.data.charts?.[signal.symbol]) {
            const img = document.createElement('img');
            img.className = 'hwb-chart-image';
            img.src = signal.chart || this.data.charts[signal.symbol];
            img.alt = `${signal.symbol} chart`;
            card.appendChild(img);
        }

        return card;
    }

    renderRecentSignalCharts(container) {
        const section = document.createElement('div');
        section.className = 'hwb-charts-section';
        section.innerHTML = '<h2>📈 直近シグナル - 詳細チャート</h2>';

        const grid = document.createElement('div');
        grid.className = 'hwb-chart-grid';

        const recentSignals = this.data.recent_signals_details || [];
        recentSignals.forEach(signal => {
            const chartCard = this.createChartCard(signal);
            if (chartCard) grid.appendChild(chartCard);
        });

        section.appendChild(grid);
        container.appendChild(section);
    }

    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('hwb-status');
        if (statusDiv) {
            statusDiv.textContent = message;
            statusDiv.className = `hwb-status-info ${type}`;

            if (type === 'error') {
                statusDiv.style.color = '#dc3545';
            } else if (type === 'warning') {
                statusDiv.style.color = '#ffc107';
            } else {
                statusDiv.style.color = 'var(--text-secondary)';
            }
        }
    }
}

// DOMContentLoadedで初期化
document.addEventListener('DOMContentLoaded', () => {
    // 200MAマネージャーの初期化
    if (document.getElementById('hwb200-content')) {
        window.hwb200Manager = new HWB200MAManager();
    }

    // グローバルなタブ切り替え関数
    window.switchTab = (tabId) => {
        // すべてのタブボタンとコンテンツから 'active' クラスを削除
        document.querySelectorAll('.tab-button').forEach(button => button.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

        // クリックされたタブに対応するボタンとコンテンツに 'active' クラスを追加
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
        document.getElementById(`${tabId}-content`).classList.add('active');

        // データ読み込みをトリガーするためのカスタムイベントを発行
        document.dispatchEvent(new CustomEvent('tab-changed', { detail: { tab: tabId } }));
    };

    // 各タブボタンにクリックイベントリスナーを設定
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            window.switchTab(tabId);
        });
    });

    // 初期タブのデータをロード（もし200MAタブがデフォルトでアクティブな場合）
    const activeTab = document.querySelector('.tab-button.active');
    if (activeTab && activeTab.getAttribute('data-tab') === 'hwb200') {
        window.hwb200Manager.loadData();
    }
});