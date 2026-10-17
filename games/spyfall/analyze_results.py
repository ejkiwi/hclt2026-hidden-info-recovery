#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""스파이폴 트랜스크립트 -> 보고표 지표 집계.

`실험 정리/결과_핵심표(보고용).md`의 "지표 정의" 표를 그대로 구현한다.
지금까지 손으로 세던 걸 스크립트로 고정한 것 (2026-08-10).

지표 정의 (보고표와 동일):
  스파이 승률      winner=='spy' / 유효 판 (winner=='invalid' 제외)
  추측 시도        should_guess=True인 guess 페이즈 / 전체 guess 페이즈 (턴 단위)
  추측 정답/오답    reason 문자열 기준 (게임 단위)
  시민 투표 정확도   시민이 실제로 던진 표 중 스파이를 찍은 표.
                  ★ 분모에서 스파이 본인의 표와 기권을 뺀다.
  투표 기권        시민의 전체 투표 기회 중 target=None (스킵/무효 강등 포함)

사용:
  python analyze_results.py transcripts/*.json
  python analyze_results.py --markdown transcripts/20260810_*_localloc_*.json
"""
import argparse
import collections
import json
import os
import sys

REASON_GUESS_OK = "spy guessed the entity correctly"
REASON_GUESS_NG = "spy guessed the entity incorrectly"


def analyze(path):
    with open(path, encoding="utf-8") as f:
        games = json.load(f)

    st = collections.Counter()
    st["games_total"] = len(games)

    for g in games:
        winner = g.get("winner")
        if winner == "invalid":
            st["games_invalid"] += 1
            continue
        st["games_valid"] += 1
        if winner == "spy":
            st["spy_win"] += 1
        elif winner == "non_spy":
            st["nonspy_win"] += 1

        reason = g.get("reason") or ""
        if reason == REASON_GUESS_OK:
            st["guess_correct"] += 1
        elif reason == REASON_GUESS_NG:
            st["guess_wrong"] += 1

        spy = next((p["name"] for p in g["players"] if p["role"] == "spy"), None)

        reached_vote = False
        for h in g["history"]:
            phase = h.get("phase")
            if phase == "guess":
                st["guess_turns"] += 1
                if h.get("should_guess"):
                    st["guess_attempts"] += 1
            elif phase == "vote":
                reached_vote = True
                if h.get("actor") == spy:
                    st["spy_votes"] += 1   # 분모에서 제외 (시민 지표라서)
                    continue
                st["citizen_vote_chances"] += 1
                if h.get("target") is None:
                    st["citizen_abstain"] += 1
                else:
                    st["citizen_votes_cast"] += 1
                    if h["target"] == spy:
                        st["citizen_vote_hit"] += 1
        if reached_vote:
            st["games_reached_vote"] += 1

    return st


def cell(num, den, unit, br="<br>"):
    """모든 지표를 같은 형식으로: 확률 + 줄바꿈 + (분자/분모단위).

    분모가 0이면 확률 자리에 '–'를 넣되 분수는 그대로 보여준다
    (0/0인지 데이터가 없는 건지 구분되게)."""
    rate = "–" if not den else f"{100.0 * num / den:.0f}%"
    return f"{rate}{br}({num}/{den}{unit})"


# 지표 이름에 분모를 박아둔다 -- 표만 보고도 뭐로 나눈 건지 알 수 있게.
ROWS = [
    "유효 판수 / 전체 판",
    "스파이 추측 시도 / 추측 기회 턴",
    "추측 정답 / 추측 시도",
    "추측으로 끝난 판 / 유효 판",
    "시민 투표 정확도 / 시민이 던진 유효표",
    "시민 투표 기권 / 시민 투표 기회",
    "투표까지 간 판 / 유효 판",
    "스파이 승률 / 유효 판",
]


def fmt(name, st, br="<br>"):
    v = st["games_valid"]
    attempts = st["guess_attempts"]
    ended_by_guess = st["guess_correct"] + st["guess_wrong"]
    return {
        "name": name,
        ROWS[0]: cell(v, st["games_total"], "판", br),
        ROWS[1]: cell(attempts, st["guess_turns"], "턴", br),
        ROWS[2]: cell(st["guess_correct"], ended_by_guess, "판", br),
        ROWS[3]: cell(ended_by_guess, v, "판", br),
        ROWS[4]: cell(st["citizen_vote_hit"], st["citizen_votes_cast"], "표", br),
        ROWS[5]: cell(st["citizen_abstain"], st["citizen_vote_chances"], "표", br),
        ROWS[6]: cell(st["games_reached_vote"], v, "판", br),
        ROWS[7]: cell(st["spy_win"], v, "판", br),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--markdown", action="store_true", help="모델을 열로 놓은 보고표 형식으로 출력")
    args = ap.parse_args()

    stats = [(os.path.basename(p), analyze(p)) for p in args.paths]

    # 추측은 성공/실패 즉시 게임이 끝나므로 '시도 턴 수'와 '추측으로 끝난 판 수'가 같아야 한다.
    for name, st in stats:
        ended = st["guess_correct"] + st["guess_wrong"]
        if st["guess_attempts"] != ended:
            print(f"[경고] {name}: 추측 시도 {st['guess_attempts']}턴 != 추측으로 끝난 {ended}판 "
                  f"-- 지표 정의 확인 필요", file=sys.stderr)

    if args.markdown:
        cols = [fmt(n, st, br="<br>") for n, st in stats]
        heads = [c["name"].replace("_30games.json", "").replace("_Korean", "")
                 .replace("20260810_", "").replace("20260808_", "").replace("20260805_", "")
                 for c in cols]
        print("| 지표 (분자 / 분모) | " + " | ".join(heads) + " |")
        print("| --- |" + " --- |" * len(cols))
        for r in ROWS:
            print(f"| {r} | " + " | ".join(c[r] for c in cols) + " |")
    else:
        cols = [fmt(n, st, br=" ") for n, st in stats]
        for c in cols:
            print(f'\n=== {c["name"]}')
            for r in ROWS:
                print(f"  {r:34s} {c[r]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
