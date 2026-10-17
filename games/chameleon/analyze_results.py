"""카멜레온 트랜스크립트 집계 스크립트 (읽기 전용).

사용:
    python analyze_results.py                      # transcripts/*.json 전부
    python analyze_results.py transcripts/foo.json # 지정 파일만

파일별로: 판수/무효, 카멜레온 승률, 적발률, 시민 투표 정확도(기권 별도),
최종 추측 성공, 토론 실명 언급률, **시민 토론 발언의 정답 포함(누출)률**.
경고 세팅(--leak-warning) 검증까지 한 번에 보기 위한 고정 스크립트 --
매번 임시 python -c를 짜지 않기 위해 영구 파일로 둠.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def stats(path: str) -> None:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    dec = [g for g in d if g["winner"] in ("chameleon", "non_chameleon")]
    n = len(dec)
    if not n:
        print(f"{Path(path).name}: 유효 게임 없음 ({len(d)}판)")
        return
    cham = sum(1 for g in dec if g["winner"] == "chameleon")
    ident = sum(1 for g in dec if g["identified"])
    vv = vh = ab = fg = fgc = 0
    stmts = leaked_stmts = mention = 0
    leaked_games = 0
    for g in dec:
        vills = {p["name"] for p in g["players"] if p["role"] == "non_chameleon"}
        names = {p["name"] for p in g["players"]}
        sw = g["secret_word"].casefold()
        game_leak = False
        for h in g["history"]:
            if h["phase"] == "vote" and h["actor"] in vills:
                if h["target"] is None:
                    ab += 1
                else:
                    vv += 1
                    vh += h["target"] == g["chameleon"]
            elif h["phase"] == "final_guess":
                fg += 1
                fgc += h["correct"]
            elif h["phase"] == "discussion" and h["actor"] in vills:
                stmts += 1
                if sw in h["statement"].casefold():
                    leaked_stmts += 1
                    game_leak = True
                if any(nm in h["statement"] for nm in names if nm != h["actor"]):
                    mention += 1
        if game_leak:
            leaked_games += 1
    flags = []
    g0 = dec[0]
    flags.append(g0.get("candidate_mode", "?"))
    flags.append(f"{g0.get('discussion_rounds', '?')}R")
    flags.append("WARN" if g0.get("leak_warning") else "no-warn")
    print(f"{Path(path).name}  [{'/'.join(str(f) for f in flags)}]")
    print(f"  판수 {len(d)} (무효 {len(d) - n})  카멜레온 승 {100 * cham / n:.0f}% ({cham}/{n})  적발 {100 * ident / n:.0f}% ({ident}/{n})")
    va = f"{100 * vh / vv:.0f}% ({vh}/{vv})" if vv else "n/a"
    print(f"  시민 투표 정확도 {va}  기권 {ab}  최종 추측 {fgc}/{fg}")
    if stmts:
        print(f"  토론: 실명 언급 {100 * mention / stmts:.0f}%  **정답 누출 발언 {100 * leaked_stmts / stmts:.0f}% ({leaked_stmts}/{stmts}), 누출 게임 {leaked_games}/{n}**")
    print()


def main() -> None:
    paths = sys.argv[1:] or sorted(glob.glob(str(Path(__file__).parent / "transcripts" / "*.json")))
    if not paths:
        print("no transcripts found")
        return
    for p in paths:
        stats(p)


if __name__ == "__main__":
    main()
