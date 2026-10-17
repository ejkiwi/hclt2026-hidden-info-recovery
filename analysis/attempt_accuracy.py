"""나섰을 때 적중률(시도 대비 정답률) 전 조건 집계 (2026-08-25).

recount_metrics.py의 게임별 집계 함수(2026-08-16 확정 정의)를 그대로 재사용하고,
파일 선택은 paths.transcript_path의 규칙을 따른다.
사다리 스파이폴·카멜레온의 30회분 + 확장 20회분은 게임 리스트를 이어붙여 한 번에
세므로 "분자·분모를 각각 더한다"는 합산 규칙(논문_수치_총정리 2-4)과 동일하다.

시도(attempt)의 정의 (recount_metrics.py와 동일):
  스파이폴  = guess 페이즈에서 should_guess를 켜고 문자열 추측을 낸 것
  금기어    = declare 필드를 켠 선언 + 발화 선언으로 심판이 인정한 것(reason 기반)
  카멜레온  = final_guess 페이즈 (지목당했을 때의 강제 응답)

35B 행은 100회 전체 기준이다. 논문에 실은 35B 값은 시드 42 랜덤 50회이므로
subsample50_35b.py를 함께 볼 것.

사용: python attempt_accuracy.py [--json-out 경로]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import paths as cm
import recount_metrics as rm

MAIN = cm.MAIN
SHORT = {"spyfall": "spy", "chameleon": "cham"}

# 무작위 기댓값 : 스파이폴 30지선다, 카멜레온 16지선다, 금기어 열린 어휘(0)
BASELINE = {"spyfall": 1 / 30, "chameleon": 1 / 16, "taboo": 0.0}


def skill(reach_pct: float | None, kind: str) -> float | None:
    """무작위 보정 도달률 (%) = (도달률 - 기준선) / (1 - 기준선)."""
    if reach_pct is None:
        return None
    b = BASELINE[kind]
    return round(100 * (reach_pct / 100 - b) / (1 - b), 1)

MAIN_FILES = {
    "spyfall": "20260809_qwen35-35b_%s_generic_100games.json",
    "chameleon": "20260809_qwen35-35b_%s_full_2ROUND-disc_WARN_100games.json",
    "taboo": "20260814_qwen35-35b_%s_100games_judge35B.json",
}


def transcript_files(model: str, kind: str, lang: str) -> list[Path]:
    if model == "35B":
        return [MAIN / (MAIN_FILES[kind] % lang)]
    if model == "gemma":
        return [cm.transcript_path("gemma", kind, lang)]
    base = cm.transcript_path(model, kind, lang)
    files = [base]
    if kind != "taboo":  # 금기어는 처음부터 50회라 확장분이 없다
        files.append(cm.GAMES / cm.TDIR[kind] / "transcripts"
                     / f"20260818_{model}_{lang}_{SHORT[kind]}_games30-49.json")
    return files


def collect(model: str, kind: str, lang: str) -> dict | None:
    games: list[dict] = []
    for p in transcript_files(model, kind, lang):
        if p is None or not Path(p).exists():
            print(f"[누락] {model} {kind} {lang} : {p}")
            return None
        games.extend(json.loads(Path(p).read_text(encoding="utf-8")))
    return rm.GAMES[kind](games)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out: dict = {}
    for kind in ("spyfall", "taboo", "chameleon"):
        for lang in ("Korean", "English"):
            for model in ("4B", "9B", "27B", "35B", "gemma"):
                r = collect(model, kind, lang)
                if r is None:
                    continue
                out[f"{kind}/{lang}/{model}"] = {
                    "유효 판": r["유효 판"],
                    "알아낸 판(%)": r["많이 (알아낸 판/유효 판)"],
                    "알아낸 판": r["많이_분수"],
                    "무작위 보정 도달률(%)": skill(r["많이 (알아낸 판/유효 판)"], kind),
                    "시도 대비 정답률(%)": r["[해석] 시도 대비 정답률"],
                    "정답/시도": r["[해석] 시도_분수"],
                    "시도율(%)": r["[해석] 시도율"],
                }

    for k, v in out.items():
        print(f"{k:28s} 적중 {str(v['시도 대비 정답률(%)']):>6s}% ({v['정답/시도']})"
              f"  도달 {str(v['알아낸 판(%)']):>6s}% ({v['알아낸 판']})"
              f"  보정 {str(v['무작위 보정 도달률(%)']):>6s}%"
              f"  시도율 {v['시도율(%)']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
