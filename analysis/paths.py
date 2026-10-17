# -*- coding: utf-8 -*-
"""트랜스크립트 파일 위치 규칙.

모델·게임·언어를 주면 그 조건의 게임 로그 경로를 돌려준다. 파일명은 실행 당시의
날짜 접두를 그대로 쓰고 있어서 규칙으로 적어둔다. 경로 규칙의 원본은 ver1
집계 스크립트(curve_metrics.py)였고, 여기 그대로 옮겨왔다.

  35B    results/main-100games/  (게임당 100회. 논문 수치는 이 중 시드 42 랜덤 50회)
  4B·9B·27B  games/<게임>/transcripts/  30회분 + 20회 확장분
  Gemma  games/<게임>/transcripts/  50회
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 저장소 루트
GAMES = ROOT / "games"
MAIN = ROOT / "results" / "main-100games"       # 35B 100회

TDIR = {"spyfall": "spyfall", "chameleon": "chameleon", "taboo": "taboo"}


def transcript_path(model: str, kind: str, lang: str) -> Path | None:
    """사다리(4B·9B·27B)와 Gemma의 기본 로그 경로. 없으면 None."""
    if model == "gemma":
        f = {"spyfall": "spy", "chameleon": "cham", "taboo": "taboo"}[kind]
        return GAMES / TDIR[kind] / "transcripts" / f"20260818_gemma_{lang}_{f}_50games.json"
    pats = {
        "spyfall": [f"20260805_qwen35-dense-{model}_{lang}_30games.json",
                    f"20260808_qwen35-dense-{model}_{lang}_generic_30games.json"],
        "chameleon": [f"20260808_qwen35-dense-{model}_{lang}_full_2ROUND-disc_WARN_30games.json",
                      f"20260810_qwen35-dense-{model}_{lang}_full_2ROUND-disc_WARN_30games.json"],
        "taboo": [f"20260808_qwen35-dense-{model}_{lang}_50games_judge35B.json"],
    }[kind]
    for p in pats:
        for base in (GAMES / TDIR[kind] / "transcripts", MAIN):
            if (base / p).exists():
                return base / p
    return None
