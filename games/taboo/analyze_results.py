#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""금기어 트랜스크립트 -> 보고표 지표 집계 (스파이폴 analyze_results.py와 같은 형식).

모든 칸을 `확률<br>(분자/분모+단위)`로 통일하고, 지표 이름에 분모를 박아둔다.

★ 발화 선언을 따로 구분하지 않는다 (2026-08-10 결정).
  방어자가 정답을 알아챘다고 밝히는 경로는 두 가지다 --
  `declare` 필드로 공식 선언하거나, 발화 중에 목표 단어를 확신하며 말하거나
  (35B 심판이 `confident_guess`로 분류). 기존 표는 이 둘을 '선언 시도'와
  '발화 선언 승'으로 나눠 적었는데, 나누면 분모가 갈려서 모델 간 비교가 안 된다.
  여기서는 **둘 다 '선언 시도' 한 분모에 합친다.**

승패 구성은 항상 다음 다섯 가지로 유효 판을 모두 덮는다 (검산에 씀):
  방어자 승 = 공격자 자멸 + 선언 정답
  공격자 승 = 방어자 누출 패 + 선언 오답
  그 외    = 무승부(턴 소진)

사용:
  python analyze_results.py --markdown transcripts/20260808_*Korean_50games_taboo_judge35B.json
"""
import argparse
import collections
import json
import os
import sys

R_SELFKILL = "attacker said the target word themselves"          # 공격자 자멸 -> 방어자 승
R_LEAK = "defender said the target word out loud"                # 방어자 누출 -> 공격자 승
R_DECL_OK = "defender declared the target word correctly"        # 공식 선언 정답
R_DECL_VERBAL = "defender declared the target word verbally"     # 발화 선언 (접두사 매칭)
R_DECL_NG = "defender declared the wrong word"                   # 선언 오답 (접두사 매칭)
R_DRAW = "turn limit reached with no declaration and no leak"    # 무승부


def analyze(path):
    with open(path, encoding="utf-8") as f:
        games = json.load(f)

    st = collections.Counter()
    st["games_total"] = len(games)
    rounds = []

    for g in games:
        if g.get("winner") == "invalid":
            st["games_invalid"] += 1
            continue
        st["games_valid"] += 1
        rounds.append(g.get("turns_used") or 0)

        reason = g.get("reason") or ""
        if reason == R_SELFKILL:
            st["self_kill"] += 1
        elif reason == R_LEAK:
            st["leak_loss"] += 1
        elif reason == R_DECL_OK:
            st["declare_ok_formal"] += 1
        elif reason.startswith(R_DECL_VERBAL):
            st["declare_ok_verbal"] += 1
        elif reason.startswith(R_DECL_NG):
            st["declare_wrong"] += 1
        elif reason == R_DRAW:
            st["draw"] += 1
        else:
            st["reason_unclassified"] += 1

    # 발화 선언을 공식 선언과 합친다 -- 분모를 가르지 않는다.
    st["declare_ok"] = st["declare_ok_formal"] + st["declare_ok_verbal"]
    st["declare_try"] = st["declare_ok"] + st["declare_wrong"]
    st["attacker_win"] = st["leak_loss"] + st["declare_wrong"]
    st["defender_win"] = st["self_kill"] + st["declare_ok"]
    st["avg_rounds"] = round(sum(rounds) / len(rounds), 1) if rounds else 0
    return st


def cell(num, den, unit, br="<br>"):
    rate = "–" if not den else f"{100.0 * num / den:.0f}%"
    return f"{rate}{br}({num}/{den}{unit})"


ROWS = [
    "유효 판수 / 전체 판",
    "공격자 자멸(목표 단어 발화) / 유효 판",
    "방어자 선언 시도 / 유효 판",
    "선언 정답 / 선언 시도",
    "방어자 누출 패 / 유효 판",
    "무승부(턴 소진) / 유효 판",
    "공격자 승 / 유효 판",
    "방어자 승 / 유효 판",
]


def fmt(name, st, br="<br>"):
    v = st["games_valid"]
    return {
        "name": name,
        ROWS[0]: cell(v, st["games_total"], "판", br),
        ROWS[1]: cell(st["self_kill"], v, "판", br),
        ROWS[2]: cell(st["declare_try"], v, "판", br),
        ROWS[3]: cell(st["declare_ok"], st["declare_try"], "판", br),
        ROWS[4]: cell(st["leak_loss"], v, "판", br),
        ROWS[5]: cell(st["draw"], v, "판", br),
        ROWS[6]: cell(st["attacker_win"], v, "판", br),
        ROWS[7]: cell(st["defender_win"], v, "판", br),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    stats = [(os.path.basename(p), analyze(p)) for p in args.paths]

    for name, st in stats:
        total = st["self_kill"] + st["leak_loss"] + st["declare_ok"] + st["declare_wrong"] + st["draw"]
        if total != st["games_valid"] or st["reason_unclassified"]:
            print(f"[경고] {name}: 승패 구성 합 {total} != 유효 판 {st['games_valid']} "
                  f"(미분류 {st['reason_unclassified']}판) -- reason 문자열 확인 필요", file=sys.stderr)

    if args.markdown:
        cols = [fmt(n, st, br="<br>") for n, st in stats]
        heads = [c["name"].replace("_50games", "").replace("_taboo_judge35B", "").replace("_judge35B", "")
                 .replace(".json", "").replace("_Korean", "").replace("_English", "")
                 .replace("20260808_", "").replace("20260805_", "").replace("20260807_", "")
                 for c in cols]
        print("| 지표 (분자 / 분모) | " + " | ".join(heads) + " |")
        print("| --- |" + " --- |" * len(cols))
        for r in ROWS:
            print(f"| {r} | " + " | ".join(c[r] for c in cols) + " |")
        print("| 평균 라운드 | " + " | ".join(f'{st["avg_rounds"]}' for _, st in stats) + " |")
    else:
        for (n, st), c in zip(stats, [fmt(n, st, br=" ") for n, st in stats]):
            print(f"\n=== {n}")
            for r in ROWS:
                print(f"  {r:34s} {c[r]}")
            print(f'  {"평균 라운드":34s} {st["avg_rounds"]}')


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
