"""
Run N Adversarial Taboo (금기어) games and report attacker/defender win
rates, leakage rate and how fast games resolve.

Example (local server, Korean, self-play):
    python run_game.py --targets targets_kr.json --language Korean \\
        --attacker-model <model> --defender-model <model> --num-games 20 \\
        --base-url http://localhost:8001/v1 --min-seconds-between-calls 0

세 게임(스파이폴/카멜레온/금기어) 로그 스키마를 맞춰놔서 요약 통계는 따로
계산해도 트랜스크립트는 같은 방식으로 읽을 수 있음 (체크리스트.md의 결과표).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import llm_client
from game import Player, TabooGame

load_dotenv()

# Windows 콘솔 기본 코드페이지(cp949)로는 한국어 출력이 깨지거나 죽어서 강제 UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_targets(path: str) -> list[dict]:
    """targets_*.json을 읽어 [{"word":..., "forbidden":[...]}, ...]로 반환.
    단순 문자열 리스트도 허용(금기어 목록 없이 쓰고 싶을 때)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["targets"] if isinstance(data, dict) else data
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append({"word": item, "forbidden": []})
        else:
            out.append({"word": item["word"], "forbidden": item.get("forbidden", [])})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Adversarial Taboo (금기어) games.")
    parser.add_argument("--targets", default="targets_en.json", help="Target-word file (targets_en.json / targets_kr.json).")
    parser.add_argument("--language", default="English", help='Passed into the {language} slot of BASE_CONTEXT, e.g. "English", "Korean".')
    parser.add_argument("--attacker-model", required=True, help="Model playing the Attacker (knows the target word).")
    parser.add_argument("--defender-model", required=True, help="Model playing the Defender (must infer it).")
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Turn limit in rounds (each round = one attacker utterance + one defender utterance). "
                             "Reaching it with no winner is a draw. Default 10 = 20 utterances.")
    parser.add_argument("--forbid-related", action="store_true",
                        help="Also ban each target's `forbidden` list for the attacker (the original party game's "
                             "taboo-word list). Violations trigger a rewrite, not a loss -- secondary constraints are "
                             "deliberately not win/loss conditions here (see 설계 결정.md). Off by default = SPAG's rules.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-base-url", action="append", default=[], metavar="MODEL=URL",
                        help="특정 모델만 다른 서버로 보냄 (여러 번 지정 가능). 역할 교차 대국처럼 "
                             "두 역할의 모델이 서로 다른 포트에 떠 있을 때 사용. 미지정 모델은 "
                             "--base-url 서버로 감.")
    parser.add_argument("--min-seconds-between-calls", type=float, default=13.0,
                        help="Client-side pacing between LLM calls (default 13s for Gemini's free tier). Pass 0 for a local server.")
    parser.add_argument("--reformatter-model", default=None,
                        help="If a response fails the strict |||...||| JSON format, ask this model to re-wrap it once "
                             "instead of losing the turn. Every history entry is tagged with needed_reformat either way.")
    parser.add_argument("--action-retries", type=int, default=2,
                        help="Retries when an utterance response is invalid (utterances have no legal skip). If all "
                             "attempts fail the game is recorded as 'invalid' and excluded from win-rate stats.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0,
                        help="앞의 N판을 건너뛰고 N번째 판부터 실행 (같은 시드면 앞 N판 구성은 기존 실행과 동일 -- 판 수 확대용)")
    parser.add_argument("--declare-judge", choices=["llm", "rule", "off"], default="llm",
                        help="방어자가 단어를 말했는데 선언 필드가 꺼져 있을 때의 판정 방식. "
                             "llm(기본)=판정 LLM이 3분류해 확신 추측만 선언 인정 / "
                             "rule=표지 검출(LLM 없는 대조 조건) / off=기존 판정(무조건 누출 패). "
                             "확신 추측 외(복수 후보 떠보기·우발 언급)는 누출 패 -- 설계 결정.md.")
    parser.add_argument("--judge-model", default=None,
                        help="선언 판정 심판 모델을 플레이어와 분리해 고정 (기본: 방어자 모델이 자기 판정). "
                             "모델 크기 사다리에서는 작은 모델이 심판을 겸하면 판정 품질이 크기와 섞이므로 "
                             "35B 하나로 고정할 것.")
    parser.add_argument("--judge-url", default=None,
                        help="심판 모델이 다른 서버(포트)에 떠 있을 때 그 base URL "
                             "(예: http://localhost:8001/v1). 미지정 시 --base-url과 같은 서버 사용.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="동시에 진행할 게임 수 (기본 1 = 기존과 동일한 순차 실행). 게임끼리는 "
                             "완전히 독립이라 병렬화해도 결과에 영향 없음. API 모델은 그대로 이득이고, "
                             "로컬 서버는 llama_cpp.server를 --n_parallel N 으로 띄워야 효과가 남.")
    parser.add_argument("--save-transcripts", default=None,
                        help="Path for the full turn-by-turn JSON. Always saved (cannot be turned off) -- if omitted, "
                             "an auto-named file under transcripts/ is used.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    llm_client.MIN_SECONDS_BETWEEN_CALLS = args.min_seconds_between_calls

    # 트랜스크립트는 항상 저장하고, 게임 한 판 끝날 때마다 즉시 덮어씀 --
    # 중간에 죽어도 끝난 게임의 발화는 남게 (스파이폴 구현과 같은 정책).
    if args.save_transcripts:
        transcripts_path = Path(args.save_transcripts)
    else:
        transcripts_dir = Path(__file__).parent / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_att = args.attacker_model.replace("/", "-").replace(":", "-")
        safe_def = args.defender_model.replace("/", "-").replace(":", "-")
        transcripts_path = transcripts_dir / f"{stamp}_{safe_att}_vs_{safe_def}_{args.language}.json"
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = llm_client.build_client(api_key, args.base_url, args.model_base_url)
    judge_client = OpenAI(api_key=api_key, base_url=args.judge_url) if args.judge_url else None
    targets = load_targets(args.targets)

    dump: list[dict] = []

    # 목표 단어는 **메인 스레드에서 미리 다 뽑아둠** -- 워커 스레드에서 random을
    # 쓰면 호출 순서가 매번 달라져서 --seed를 줘도 재현이 깨짐.
    setups = [(i, random.choice(targets)) for i in range(args.num_games)]
    # --start-index: 구성은 전부 뽑아두고(시드 소비 순서를 100판 실행과 동일하게
    # 유지해야 앞 판 구성이 일치) 실제로 돌릴 판만 잘라냄 -- 판 수 확대용.
    if args.start_index:
        setups = setups[args.start_index:]
        print(f"[--start-index {args.start_index}] game#{args.start_index}부터 {len(setups)}판 실행")
    dump_lock = threading.Lock()

    def play_one(setup: tuple) -> None:
        game_idx, entry = setup
        target_word = entry["word"]
        attacker = Player(name="Attacker", model=args.attacker_model, role="attacker")
        defender = Player(name="Defender", model=args.defender_model, role="defender")
        game = TabooGame(
            client, attacker, defender, target_word,
            forbidden_words=entry["forbidden"] if args.forbid_related else None,
            language=args.language,
            reformatter_model=args.reformatter_model,
            action_retries=args.action_retries,
            max_rounds=args.max_rounds,
            declare_judge=args.declare_judge,
            judge_model=args.judge_model,
            judge_client=judge_client,
        )
        result = game.play()
        record = {
            "game_index": game_idx,
            "target": target_word,
            "forbidden": entry["forbidden"] if args.forbid_related else None,
            "players": [{"name": p.name, "model": p.model, "role": p.role} for p in (attacker, defender)],
            "declare_judge": args.declare_judge,
            "judge_model": (args.judge_model or args.defender_model) if args.declare_judge == "llm" else None,
            "winner": result.winner,
            "reason": result.reason,
            "leaked": result.leaked,       # 방어자가 목표 단어를 발화함 = 공격자 유도 성공
            "turns_used": result.turns_used,
            "history": result.history,
        }
        # 병렬이면 끝나는 순서가 뒤죽박죽이므로 game_index로 정렬해서 저장.
        with dump_lock:
            dump.append(record)
            dump.sort(key=lambda r: r["game_index"])
            transcripts_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
            # flush=True: 백그라운드 실행 시 stdout 블록 버퍼링으로 진행이 안 보이던 문제 방지
            print(f"[{len(dump)}/{args.num_games} done] game#{game_idx} target={target_word!r} "
                  f"winner={result.winner:8s} leaked={result.leaked} rounds={result.turns_used} "
                  f"reason={result.reason}", flush=True)

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(play_one, s) for s in setups]
            for fut in as_completed(futures):
                fut.result()  # 워커에서 터진 예외를 여기서 그대로 올림
    else:
        for s in setups:
            play_one(s)

    print(f"\nSaved full transcripts ({len(dump)} game(s)) to {transcripts_path}")

    decided = [r for r in dump if r["winner"] in ("attacker", "defender")]
    draws = [r for r in dump if r["winner"] == "draw"]
    invalid = [r for r in dump if r["winner"] == "invalid"]

    print("\n=== Summary ===")
    print(f"Games played:        {len(dump)}  (draws: {len(draws)}, invalid/aborted: {len(invalid)}, both excluded from win rates)")
    if decided:
        att = 100 * sum(1 for r in decided if r["winner"] == "attacker") / len(decided)
        print(f"Attacker win rate:   {att:.1f}%  ({len(decided)} decided games)")
        print(f"Defender win rate:   {100 - att:.1f}%")
    # 무효 게임은 중도 중단이라 누출/선언 기회 자체가 없었음 -> 승률과 같은
    # 분모(무효 제외)를 씀. 안 그러면 무효 게임이 누출률을 희석시킴.
    valid = [r for r in dump if r["winner"] != "invalid"]
    if valid:
        leak = 100 * sum(1 for r in valid if r["leaked"]) / len(valid)
        # 방어자가 선언을 시도한 게임 비율 -- 스파이폴에서 "스파이가 추측을 아예
        # 안 한다"가 핵심 발견이었으므로 같은 축을 여기서도 봄.
        declared = sum(1 for r in valid if any(h.get("declare") for h in r["history"]))
        avg_rounds = sum(r["turns_used"] for r in valid) / len(valid)
        print(f"Leakage rate:        {leak:.1f}%  ({len(valid)} valid games; defender uttered the target word -- see mentions_word())")
        print(f"Declaration rate:    {100 * declared / len(valid):.1f}%  (games where the defender declared at all)")
        print(f"Avg rounds to end:   {avg_rounds:.1f} / {args.max_rounds}")


if __name__ == "__main__":
    main()
