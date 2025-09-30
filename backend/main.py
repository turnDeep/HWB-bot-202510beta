# 既存のインポートに追加
from backend.hwb_scanner import run_hwb_scan
import asyncio
import os
import json
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles

# 仮の認証関数
def get_current_user():
    return "admin"

app = FastAPI()

# APIエンドポイントを追加

@app.post("/api/hwb/scan")
async def trigger_hwb_scan(current_user: str = Depends(get_current_user)):
    """HWBスキャンを手動実行（管理者のみ）"""
    try:
        # 非同期でスキャン実行
        result = await run_hwb_scan()

        return {
            "success": True,
            "message": f"スキャン完了: {len(result.get('signals', []))}件のシグナル検出",
            "scan_date": result['scan_date'],
            "scan_time": result['scan_time']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# フロントエンドの静的ファイルを提供
# このマウントは、他のすべてのルートの後に配置する必要がある
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

@app.get("/api/hwb/data")
def get_hwb_data(current_user: str = Depends(get_current_user)):
    """HWBデータ取得"""
    try:
        # メインデータ読み込み
        data_path = 'data/hwb_signals.json'
        if not os.path.exists(data_path):
            return {"error": "データが見つかりません。スキャンを実行してください。"}

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # チャートデータ読み込み
        charts_path = 'data/hwb_charts.json'
        if os.path.exists(charts_path):
            with open(charts_path, 'r', encoding='utf-8') as f:
                charts = json.load(f)
                data['charts'] = charts

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hwb/status")
def get_hwb_status(current_user: str = Depends(get_current_user)):
    """HWBスキャンステータス確認"""
    try:
        data_path = 'data/hwb_signals.json'
        if os.path.exists(data_path):
            # ファイルの最終更新時刻を取得
            mtime = os.path.getmtime(data_path)
            last_scan = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                "has_data": True,
                "last_scan": last_scan,
                "total_scanned": data.get('total_scanned', 0),
                "signals_count": len(data.get('signals', [])),
                "candidates_count": len(data.get('candidates', []))
            }
        else:
            return {
                "has_data": False,
                "message": "データがありません。スキャンを実行してください。"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))