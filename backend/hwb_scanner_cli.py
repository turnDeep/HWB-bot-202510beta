#!/usr/bin/env python
"""HWBスキャナーCLI実行用スクリプト"""

import asyncio
import sys
import logging
from backend.hwb_scanner import run_hwb_scan

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """メイン実行関数"""
    print("HWBスキャン開始...")

    try:
        result = await run_hwb_scan()
        print(f"スキャン完了: {len(result.get('signals', []))}件のシグナル検出")
        return 0

    except Exception as e:
        print(f"エラー: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)