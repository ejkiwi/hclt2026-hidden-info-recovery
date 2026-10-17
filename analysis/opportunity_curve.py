# -*- coding: utf-8 -*-
"""답할 기회 번호별 누적 정답률 (ver2 그림 1·3·4 곡선).

집계 규칙은 `턴별로 가게 된다면 변할 내용.md`(2026-08-25)에 적힌 그대로다.
  가로축 = 게임이 부여한 답할 기회 번호
  값     = 그 기회까지 누적 정답률 (기권·미선언·기회 없음은 전부 오답)
  분모   = 전체 유효 판 (winner == "invalid" 제외)
  후처리 없음 : 되물어보기도, 오염 제외도 쓰지 않는다

기회 번호의 정의
  스파이폴  추측 페이즈가 온 순서 (사이클마다 1회)
  금기어    방어자 턴 번호 (turns_used)
  카멜레온  지목당했을 때의 최종 추측 1회뿐이라 항상 1번째

사용: python opportunity_curve.py [--json-out 경로]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import attempt_accuracy as aa

VALID = {
    "spyfall": ("spy", "non_spy"),
    "taboo": ("attacker", "defender", "draw"),
    "chameleon": ("chameleon", "non_chameleon"),
}

# 35B 서브샘플 (subsample50_35b.py와 같은 시드·같은 인덱스)
SUB50 = sorted(random.Random(42).sample(range(100), 50))


def first_correct_spyfall(g: dict) -> tuple[int | None, int]:
    """(정답이 나온 기회 번호, 그 판에 실제로 온 기회 수)."""
    k = 0
    hit = None
    for e in g["history"]:
        if e.get("phase") != "guess":
            continue
        k += 1
        if e.get("should_guess") and e.get("best_guess"):
            # 승자 필드로 판정한다 ("incorrectly"가 "correctly"를 포함하는 함정)
            if g["winner"] == "spy" and "incorrectly" not in g["reason"]:
                hit = k
            break
    return hit, k


def first_correct_taboo(g: dict) -> tuple[int | None, int]:
    r = g["reason"]
    k = g.get("turns_used") or 0
    if "declared the target word" in r and g["winner"] == "defender":
        return k, k
    return None, k


def first_correct_chameleon(g: dict) -> tuple[int | None, int]:
    if not g.get("identified"):
        return None, 0
    ok = any(e.get("phase") == "final_guess" and e.get("correct") for e in g["history"])
    return (1 if ok else None), 1


PICK = {
    "spyfall": first_correct_spyfall,
    "taboo": first_correct_taboo,
    "chameleon": first_correct_chameleon,
}


def curve(games: list[dict], kind: str) -> dict:
    valid = [g for g in games if g.get("winner") in VALID[kind]]
    hits: list[int] = []
    chances: list[int] = []
    for g in valid:
        hit, k = PICK[kind](g)
        chances.append(k)
        if hit is not None:
            hits.append(hit)
    kmax = max(chances) if chances else 0
    n = len(valid)
    pts = []
    for k in range(1, kmax + 1):
        cum = sum(1 for h in hits if h <= k)
        pts.append(round(100 * cum / n, 1) if n else None)
    return {
        "유효 판": n,
        "최대 기회": kmax,
        "판당 기회 도달 수": [sum(1 for c in chances if c >= k) for k in range(1, kmax + 1)],
        "곡선": pts,
    }


def load(model: str, kind: str, lang: str) -> list[dict] | None:
    games: list[dict] = []
    for p in aa.transcript_files(model, kind, lang):
        if p is None or not Path(p).exists():
            return None
        games.extend(json.loads(Path(p).read_text(encoding="utf-8")))
    return games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out: dict = {}
    for kind in ("spyfall", "taboo", "chameleon"):
        for lang in ("Korean", "English"):
            for model in ("4B", "9B", "27B", "35B", "gemma"):
                games = load(model, kind, lang)
                if games is None:
                    continue
                out[f"{kind}/{lang}/{model}"] = curve(games, kind)
                if model == "35B":
                    sub = [games[i] for i in SUB50]
                    out[f"{kind}/{lang}/35B-랜덤50"] = curve(sub, kind)

    for k, v in out.items():
        line = " ".join(f"{x:g}" for x in v["곡선"])
        print(f"{k:34s} n={v['유효 판']:>3d}  {line}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
