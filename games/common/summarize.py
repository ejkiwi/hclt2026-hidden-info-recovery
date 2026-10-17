"""트랜스크립트 JSON들을 한 번에 읽어서 비교표로 뽑는 집계 스크립트.

승률만 보면 세 게임 다 "논문이랑 비슷하네"로 보이는데, 실제로 갈리는 건
**행동 지표**(스파이가 추측을 하는가 / 방어자가 선언을 하는가 / 시민 투표가
무작위인가)라서 그것들까지 같이 뽑는다.

사용법:
    python ../common/summarize.py transcripts/*.json
    python ../common/summarize.py ../*/transcripts/20260804_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _spyfall(d: list[dict]) -> dict:
    guesses = [h for g in d for h in g["history"] if h["phase"] == "guess"]
    tried = [h for h in guesses if h.get("should_guess")]
    votes_cast = votes_hit = 0
    votes_skipped = 0
    for g in d:
        spy = next(p["name"] for p in g["players"] if p["role"] == "spy")
        roles = {p["name"]: p["role"] for p in g["players"]}
        for h in g["history"]:
            if h["phase"] != "vote":
                continue
            if roles.get(h["actor"]) != "non_spy":
                continue
            if h.get("target") is None:
                votes_skipped += 1
                continue
            votes_cast += 1
            votes_hit += (h["target"] == spy)
    decided = [g for g in d if g["winner"] in ("spy", "non_spy")]
    return {
        "게임": len(d),
        "스파이승": f'{100*sum(1 for g in decided if g["winner"]=="spy")/max(len(decided),1):.0f}%',
        "추측시도": f"{len(tried)}/{len(guesses)}",
        "추측종료": sum(1 for g in d if "guessed" in g["reason"]),
        "무고한시민몰림": sum(1 for g in d if "innocent" in g["reason"]),
        "스파이검거": sum(1 for g in d if "spy was voted out" in g["reason"]),
        "누출": f'{100*sum(1 for g in d if g["leaked"])/max(len(d),1):.0f}%',
        # 무작위 기대값은 1/(플레이어수-1). 5인이면 25%.
        "시민투표정확도": f"{100*votes_hit/max(votes_cast,1):.0f}% ({votes_cast}표)",
        "투표기권": f"{100*votes_skipped/max(votes_cast+votes_skipped,1):.0f}%",
    }


def _taboo(d: list[dict]) -> dict:
    decided = [g for g in d if g["winner"] in ("attacker", "defender")]
    return {
        "게임": len(d),
        "공격자승": f'{100*sum(1 for g in decided if g["winner"]=="attacker")/max(len(decided),1):.0f}%',
        "공격자자멸": sum(1 for g in d if "attacker said" in g["reason"]),
        "방어자누출패": sum(1 for g in d if "defender said" in g["reason"]),
        "선언시도": sum(1 for g in d if any(h.get("declare") for h in g["history"])),
        "선언정답": sum(1 for g in d if "declared the target word correctly" in g["reason"]),
        "무승부": sum(1 for g in d if g["winner"] == "draw"),
        "평균라운드": f'{sum(g["turns_used"] for g in d)/max(len(d),1):.1f}',
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return
    rows = []
    for p in sorted(paths):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  [건너뜀] {p.name}: {e}")
            continue
        if not d:
            continue
        kind = "taboo" if "turns_used" in d[0] else "spyfall"
        rows.append((p.name, kind, _taboo(d) if kind == "taboo" else _spyfall(d)))

    for kind in ("spyfall", "taboo"):
        sel = [(n, s) for n, k, s in rows if k == kind]
        if not sel:
            continue
        print(f"\n{'='*100}\n{kind.upper()}\n{'='*100}")
        cols = list(sel[0][1])
        print(f"{'파일':52} " + " ".join(f"{c:>14}" for c in cols))
        for name, stats in sel:
            print(f"{name[:52]:52} " + " ".join(f"{str(stats[c]):>14}" for c in cols))
    print("\n※ 시민투표정확도의 무작위 기대값은 25% (5인 게임). 표본(표 수)이 작으면 해석 주의.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
