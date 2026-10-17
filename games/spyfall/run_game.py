"""
Run N Cultured Turn-Based Spyfall games and report win-rate / leakage-rate
statistics comparable to the paper's Table 1 (Bradley-Terry / win rate)
and Table 4 (leakage rate).

Example (5 players: 4 non-spies on one model, 1 spy on another):
    export OPENAI_API_KEY=sk-...
    python run_game.py --non-spy-model gpt-4o-mini --spy-model gpt-4o-mini \\
        --num-games 10 --language English

To compare languages (the actual goal of this reproduction -- see
README), just change --language and, if needed, --entities to a
translated entity list, keeping the models fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import llm_client
from game import Player, SpyfallGame

load_dotenv()  # picks up ../.env (python-dotenv searches parent directories automatically)

# Windows consoles default to a legacy codepage (e.g. cp949) that can't print
# most Korean/Chinese/Japanese text and crashes with UnicodeEncodeError.
# Model outputs in this benchmark are frequently non-English, so force UTF-8
# stdout/stderr regardless of platform.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_entities(path: str) -> list[str]:
    # 데이터셋 불러오기: entities_generic_en.json 파일을 불러와서 entities를 반환함.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["entities"]


def build_players(non_spy_model: str, spy_model: str, num_non_spies: int) -> list[Player]:
    # 플레이어 생성: 4명의 non-spy와 1명의 spy를 생성하고 섞음 -> 생성은 game.player에서 수행.
    # Player 객체에는 상태 없고 누가 어디 앉아있는지만 들고있는 구조.
    # 이름은 역할과 무관하게(Player0~4) 부여함 -- "NonSpy1"/"Spy" 같은 이름을 쓰면
    # history/프롬프트에 그 이름이 그대로 노출되면서 누가 스파이인지 이름표로 다
    # 드러나버림 (HF 공개 로그의 실제 명명 방식 "Player 0 Z" 등도 역할과 무관함).
    total = num_non_spies + 1
    names = [f"Player{i}" for i in range(total)]
    random.shuffle(names)  # 이름-역할 매핑도 무작위로 배정 (특정 이름이 항상 스파이가 되는 걸 방지)
    players = [Player(name=names[i], model=non_spy_model, role="non_spy") for i in range(num_non_spies)]  #플레이어들 담아놓는 저장소
    players.append(Player(name=names[num_non_spies], model=spy_model, role="spy"))
    random.shuffle(players)  # 논문에서 말한 것처럼 플레이어 순서를 섞음 -> 리스트의 인덱스 순서를 원형으로 돌면서 현재 인덱스의 대상이 다음 인덱스의 대상에게 질문하는 방식
    return players


def main() -> None:  # 게임 진입점. (게임 실행하고 결과 요약까지.)
    # 게임 실행 명령어 파서 옵션 받아오기: 터미널에서 옵션 주고 바로 실행가능하게
    parser = argparse.ArgumentParser(description="Run Cultured Turn-Based Spyfall games.")  # 터미널에서 작동 편하게 하려는 그 그부분임..
    parser.add_argument("--entities", default="entities_generic_en.json")  # 카드 파일 경로
    parser.add_argument("--language", default="English", help='Passed into the {language} slot of BASE_CONTEXT, e.g. "English", "Korean".')  # 언어
    parser.add_argument("--non-spy-model", required=True, help="Model used for all 4 non-spy players.")  # non-spy에 쓰일 모델
    parser.add_argument("--spy-model", required=True, help="Model used for the spy.")  # spy에 쓰일 모델
    parser.add_argument("--num-non-spies", type=int, default=4, help="Paper default: 4 (5 players total).")  # non-spy 플레이어 수
    parser.add_argument("--num-games", type=int, default=5)  # 몇 번 돌릴지
    parser.add_argument("--base-url", default=None)  # 어느 서버에서 모델 돌릴지
    parser.add_argument("--model-base-url", action="append", default=[], metavar="MODEL=URL",
                        help="특정 모델만 다른 서버로 보냄 (여러 번 지정 가능). 역할 교차 대국처럼 "
                             "두 역할의 모델이 서로 다른 포트에 떠 있을 때 사용. 미지정 모델은 "
                             "--base-url 서버로 감.")
    parser.add_argument("--min-seconds-between-calls", type=float, default=13.0,
                         help="Client-side pacing between LLM calls (default 13s, tuned for Gemini's free-tier "
                              "rate limit). Pass 0 for a local server (e.g. Ollama/vLLM) that has no such limit.")  # 호출 간 텀
    parser.add_argument("--reformatter-model", default=None,
                         help="If a player's response fails the strict |||...||| JSON format, retry once by asking "
                              "this model to re-wrap the same content correctly, instead of an immediate forfeit. "
                              "Not part of the paper's protocol -- opt-in because this research cares about "
                              "secret-inference/leakage behavior, not strict-JSON compliance (see 체크리스트.md). "
                              "Every history entry is tagged with needed_reformat either way.")  # 후처리 어캐 할건지 부분.
    parser.add_argument("--action-retries", type=int, default=2,
                         help="Retries when a question/answer response is invalid (those phases have no "
                              "legal skip action). If all attempts fail the game is recorded as 'invalid' "
                              "and excluded from win-rate stats -- format compliance is deliberately NOT "
                              "a win/loss condition here, unlike the paper (see 논문과_다르게_간_설계.md).")  # 질문/답변 재시도 횟수
    parser.add_argument("--seed", type=int, default=None)  # 랜덤시드
    parser.add_argument("--parallel", type=int, default=1,
                        help="동시에 진행할 게임 수 (기본 1 = 기존과 동일한 순차 실행). 게임끼리는 "
                             "완전히 독립이라 병렬화해도 결과에 영향 없음 -- 게임 '내부'는 앞 발화를 "
                             "봐야 다음 발화가 나오므로 병렬화 불가. API 모델은 그대로 이득이고, "
                             "로컬 서버는 llama_cpp.server를 --n_parallel N 으로 띄워야 효과가 남.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="앞의 N판을 건너뛰고 N번째 판부터 실행 (기본 0 = 전부 실행). "
                             "이미 30판을 돌린 조건을 100판으로 늘릴 때, 같은 시드로 100판치 "
                             "게임 구성을 뽑되 뒤의 70판만 새로 돌리는 용도. 게임 구성(정답 카드/"
                             "역할 배정)은 아래에서 시드로 미리 다 뽑으므로, --seed가 같으면 "
                             "앞 30판 구성은 30판 실행 때와 완전히 동일함 -- 그래서 두 트랜스크립트를 "
                             "game_index로 합치면 100판을 한 번에 돌린 것과 같은 구성이 됨. "
                             "(게임 진행 자체는 temperature>0이라 매번 다르므로, 이미 끝난 판을 "
                             "다시 돌리는 게 아니라 '남은 판만' 돌리는 것임.)")
    parser.add_argument("--save-transcripts", default=None, help="Path to write full per-game turn-by-turn history as JSON (for qualitative inspection). "
                         "Always saved (this cannot be turned off) -- if omitted, an auto-named file under transcripts/ is used.")  # 게임 발화 로그 저장 경로
    args = parser.parse_args()  #  실제로 sys.argv(터미널에 친 것)를 읽어서, 등록해둔 규칙대로 검증하고 args라는 네임스페이스 객체를 생성.


    # 옵션대로 게임 설정(아직 생성 아님): 1.시드 설정, 2.LLM 호출 간 텀 설정, 3.발화 로그 저장 경로 설정 등.
    if args.seed is not None:  # 시드 딱히 저장 안 하면 그냥 랜덤으로
        random.seed(args.seed)

    llm_client.MIN_SECONDS_BETWEEN_CALLS = args.min_seconds_between_calls  # 2.호출 간 텀 설정

    # Full dialogue transcripts are always written -- summary stats alone
    # aren't enough for the qualitative per-game analysis this research
    # needs (see 체크리스트.md). Written incrementally, one game at a
    # time, so a crash/rate-limit failure mid-run (this has happened
    # before -- see logs/run3.log, logs/run4.log) doesn't lose the
    # transcripts of games that already finished.
    if args.save_transcripts:  #3. 저장경로 설정
        transcripts_path = Path(args.save_transcripts)
    else:
        transcripts_dir = Path(__file__).parent / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 실험 시간
        safe_non_spy = args.non_spy_model.replace("/", "-").replace(":", "-")  # 모델 정보
        safe_spy = args.spy_model.replace("/", "-").replace(":", "-")  # 모델 정보
        transcripts_path = transcripts_dir / f"{stamp}_{safe_non_spy}_vs_{safe_spy}_{args.language}.json"  # 파일명 지정
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")  #api key 가져오는 경우
    client = llm_client.build_client(api_key, args.base_url, args.model_base_url)  # 모델 연결(api key). --model-base-url 있으면 모델별 라우팅
    entities = load_entities(args.entities)  # 카드 파일 불러오기

    dump: list[dict] = []  # 게임 한 판 끝날때마다 기록(target/players/winner/reason/leaked/history)을 담아놓는 저장소. 트랜스크립트 JSON에 그대로 저장되고, 마지막 통계도 여기서 계산.

    # 게임 설정(정답 카드 / 이름-역할 배정)은 **메인 스레드에서 미리 다 뽑아둠**.
    # 워커 스레드 안에서 random을 쓰면 호출 순서가 매번 달라져서 --seed를 줘도
    # 재현이 깨짐. 미리 뽑아두면 --parallel 값을 바꿔도 같은 시드 = 같은 게임 구성.
    setups = [(i, random.choice(entities),
               build_players(args.non_spy_model, args.spy_model, args.num_non_spies))
              for i in range(args.num_games)]
    # --start-index: 구성은 전부 뽑아두고(= 시드 소비 순서를 100판 실행과 동일하게
    # 유지해야 앞 30판 구성이 일치함) 실제로 돌릴 판만 잘라냄.
    if args.start_index:
        setups = setups[args.start_index:]
        print(f"[--start-index {args.start_index}] game#{args.start_index}부터 {len(setups)}판 실행 "
              f"(앞 {args.start_index}판은 기존 트랜스크립트에 있다고 가정하고 건너뜀)")

    dump_lock = threading.Lock()

    def play_one(setup: tuple) -> None:
        game_idx, target, players = setup
        game = SpyfallGame(client, players, entities, target, language=args.language,
                           reformatter_model=args.reformatter_model, action_retries=args.action_retries)
        result = game.play()  # 한 판 실행 (result에 담기는 값 : winner, reason, leaked, history)
        entry = {  # result에 담긴 게임 결과 저장
            "game_index": game_idx,  # 게임 인덱스
            "target": target,  # 정답 카드
            "players": [{"name": p.name, "model": p.model, "role": p.role} for p in players],  # 플레이어 정보
            "winner": result.winner,  # 승자
            "reason": result.reason,  # 승리 이유
            "leaked": result.leaked,  # 누출(진짜 정답 자체가 발화에 포함되어있는지만 검사. 은밀한 힌트는 못 잡음.)
            "history": result.history,  # 구조화된 대화 로그. (발화 하나는 모델의 원본 응답 전체가 아니라, 파싱해서 뽑아낸 필드값)
        }
        # 병렬이면 끝나는 순서가 뒤죽박죽이므로 game_index로 정렬해서 저장.
        # 저장은 매 게임마다 -- 중간에 죽어도 끝난 게임은 남게 (기존 정책 유지).
        with dump_lock:
            dump.append(entry)
            dump.sort(key=lambda r: r["game_index"])
            transcripts_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
            # flush=True: 백그라운드 실행 시 파이썬이 stdout을 블록 버퍼링해서
            # 진행 상황이 안 보이던 문제 방지.
            print(f"[{len(dump)}/{len(setups)} done] game#{game_idx} target={target!r} "
                  f"winner={result.winner:8s} leaked={result.leaked} reason={result.reason}", flush=True)

    if args.parallel > 1:
        # 게임끼리는 완전히 독립이라 병렬 가능. (게임 '내부'는 앞 발화를 봐야
        # 다음 발화가 나오므로 병렬화 불가 -- 여기서 늘리는 건 동시 진행 게임 수.)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(play_one, s) for s in setups]
            for fut in as_completed(futures):
                fut.result()  # 워커에서 터진 예외를 여기서 그대로 올림
    else:
        for s in setups:
            play_one(s)

    print(f"\nSaved full transcripts ({len(dump)} game(s)) to {transcripts_path}")

    # 무효 게임(재시도까지 전부 실패해서 중단된 게임)은 승패 집계에서 제외 --
    # 응답 포맷은 승패 조건이 아니라는 설계 결정 (논문과_다르게_간_설계.md).
    decided = [r for r in dump if r["winner"] in ("spy", "non_spy")]
    invalid = [r for r in dump if r["winner"] == "invalid"]

    print("\n=== Summary (compare against paper's Table 1 win rate / Table 4 leakage rate) ===")
    print(f"Games played:       {len(dump)}  (invalid/aborted: {len(invalid)}, excluded from rates)")
    if decided:
        spy_win_rate = 100 * sum(1 for r in decided if r["winner"] == "spy") / len(decided)
        leakage_rate = 100 * sum(1 for r in decided if r["leaked"]) / len(decided)
        print(f"Spy win rate:       {spy_win_rate:.1f}%  ({len(decided)} decided games)")
        print(f"Non-spy win rate:   {100 - spy_win_rate:.1f}%")
        print(f"Leakage rate:       {leakage_rate:.1f}%  (heuristic substring match -- see detect_leakage() docstring)")


if __name__ == "__main__":
    main()
