"""새 지표 정의로 세 게임을 한꺼번에 재집계 (2026-08-16 확정 정의).

측정 축 정의:

  주지표 = **알아내기**
    - 많이 : 전체(유효) 판 중 비밀을 알아낸 판의 비율      -> reach
    - 빨리 : 알아낸 판에서 몇 번째 라운드에 알아냈나(중앙값) -> speed
  부지표 = **추적** (스파이·카멜레온만, 속도 없음)
    - 적대자를 맞힌 표 / 시민이 던진 표                    -> track_vote
    - 적대자를 잡은 판 / 유효 판                          -> track_game
  해석 장치 (지표가 아니라 위 숫자를 읽는 도구)
    - 시도율, 시도 대비 정답률, 투표 기권율

**"빨리"는 스파이·금기어만 나옴.** 카멜레온은 최종 추측이 판 끝에 딱 한 번뿐이라
(힌트 1바퀴 -> 토론 2바퀴 -> 투표 -> 지목되면 추측) 시점이 항상 같아서 속도가
상수임. 게임 구조상의 한계이므로 빈칸으로 두고 그 이유를 표에 적을 것.

무효 판(winner == "invalid")은 모든 분모에서 제외. 금기어의 무승부(draw)는
승부는 안 났지만 진행된 판이므로 분모에 포함(기존 집계 규칙과 동일).

사용:
    python recount_metrics.py --game spyfall --transcripts <경로> [...]

전 조건을 한 번에 보려면 attempt_accuracy.py를 쓴다. 이 스크립트의 게임별 집계
함수를 그대로 불러 30개 조건을 돌린다.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _valid(games: list[dict], winners: tuple[str, ...]) -> list[dict]:
    return [g for g in games if g.get("winner") in winners]


# ---------------------------------------------------------------- 스파이폴
def spyfall(games: list[dict]) -> dict:
    v = _valid(games, ("spy", "non_spy"))
    reach, speeds = 0, []
    attempts = corrects = chances = 0
    votes_cast = votes_hit = vote_chances = 0
    caught = 0
    for g in v:
        spy = next(p["name"] for p in g["players"] if p["role"] == "spy")
        qcount = 0          # 지금까지 오간 질문 수 = "몇 번 듣고" 의 단위
        got = None
        for h in g["history"]:
            ph = h["phase"]
            if ph == "question":
                qcount += 1
            elif ph == "guess":
                chances += 1
                if h.get("should_guess") and isinstance(h.get("best_guess"), str):
                    attempts += 1
            elif ph == "vote":
                if h["actor"] != spy:          # 스파이 본인 표는 분모에서 제외
                    vote_chances += 1
                    if h.get("target"):
                        votes_cast += 1
                        if h["target"] == spy:
                            votes_hit += 1
        # 알아낸 판인지 + 그 시점
        if g.get("reason") == "spy guessed the entity correctly":
            reach += 1
            corrects += 1
            q = 0
            for h in g["history"]:
                if h["phase"] == "question":
                    q += 1
                elif h["phase"] == "guess" and h.get("should_guess"):
                    got = q
                    break
            if got is not None:
                speeds.append(got)
        if g.get("reason") == "spy was voted out by majority":
            caught += 1
    return _pack(len(v), reach, speeds, attempts, corrects, chances,
                 votes_hit, votes_cast, vote_chances, caught)


# ---------------------------------------------------------------- 금기어
def taboo(games: list[dict]) -> dict:
    # 방어자가 "알아내는" 쪽. 무승부는 진행된 판이므로 분모에 포함.
    v = _valid(games, ("attacker", "defender", "draw"))
    reach, speeds = 0, []
    attempts = corrects = 0
    for g in v:
        declared_round = None
        for h in g["history"]:
            if h["phase"] == "defend" and h.get("declare"):
                attempts += 1
                if declared_round is None:
                    declared_round = h.get("round")
        r = g.get("reason", "")
        # 공식 선언 정답 + 발화로 선언한 것(심판이 확신 추측으로 판정)을 합침
        if r.startswith("defender declared the target word"):
            reach += 1
            corrects += 1
            rd = declared_round if declared_round is not None else g.get("turns_used")
            if rd:
                speeds.append(rd)
        if "declared the target word verbally" in r:
            attempts += 1  # 발화 선언은 declare 필드가 꺼져 있으므로 따로 더함
    return _pack(len(v), reach, speeds, attempts, corrects, None,
                 None, None, None, None)


# ---------------------------------------------------------------- 카멜레온
def chameleon(games: list[dict]) -> dict:
    v = _valid(games, ("chameleon", "non_chameleon"))
    reach = 0
    attempts = corrects = 0
    votes_cast = votes_hit = vote_chances = 0
    caught = 0
    for g in v:
        cham = g.get("chameleon") or next(p["name"] for p in g["players"] if p["role"] == "chameleon")
        for h in g["history"]:
            if h["phase"] == "vote":
                if h["actor"] != cham:
                    vote_chances += 1
                    if h.get("target"):
                        votes_cast += 1
                        if h["target"] == cham:
                            votes_hit += 1
            elif h["phase"] == "final_guess":
                attempts += 1
                if h.get("correct"):
                    corrects += 1
        if g.get("identified"):
            caught += 1
        if any(h["phase"] == "final_guess" and h.get("correct") for h in g["history"]):
            reach += 1
    # 속도 없음 : 최종 추측 시점이 판마다 동일 (게임 구조상 상수)
    return _pack(len(v), reach, [], attempts, corrects, None,
                 votes_hit, votes_cast, vote_chances, caught)


def _pack(n, reach, speeds, attempts, corrects, chances,
          vhit, vcast, vchance, caught) -> dict:
    def pct(a, b):
        return None if not b else round(100 * a / b, 1)
    return {
        "유효 판": n,
        "많이 (알아낸 판/유효 판)": pct(reach, n), "많이_분수": f"{reach}/{n}",
        "빨리 (중앙값 라운드)": (statistics.median(speeds) if speeds else None),
        "빨리_범위": (f"{min(speeds)}~{max(speeds)}" if speeds else None),
        "추적 표 (맞힌 표/던진 표)": pct(vhit, vcast),
        "추적 표_분수": (f"{vhit}/{vcast}" if vcast else None),
        "추적 판 (잡은 판/유효 판)": pct(caught, n) if caught is not None else None,
        "추적 판_분수": (f"{caught}/{n}" if caught is not None else None),
        "[해석] 시도율": pct(attempts, chances) if chances else None,
        "[해석] 시도 대비 정답률": pct(corrects, attempts),
        "[해석] 시도_분수": f"{corrects}/{attempts}",
        "[해석] 기권율": pct(vchance - vcast, vchance) if vchance else None,
    }


GAMES = {"spyfall": spyfall, "taboo": taboo, "chameleon": chameleon}


def run_one(game: str, path: Path) -> dict:
    games = json.loads(path.read_text(encoding="utf-8"))
    return GAMES[game](games)


def main() -> None:
    ap = argparse.ArgumentParser(description="새 지표 정의로 재집계")
    ap.add_argument("--game", choices=list(GAMES))
    ap.add_argument("--transcripts", nargs="+")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out = {}
    for p in args.transcripts:
        path = Path(p)
        label = args.label or path.stem
        try:
            out[label if len(args.transcripts) == 1 else path.stem] = run_one(args.game, path)
        except Exception as e:  # noqa: BLE001
            print(f"[실패] {path.name}: {e}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
