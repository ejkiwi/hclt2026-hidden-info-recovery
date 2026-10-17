"""
카멜레온 실행기 (「카멜레온 재설계 구상안.md」 구현).

**카멜레온 = 발화 구조가 다른 게임**이다 -- 정보량 조건이 아니라:

    스파이폴  : 질문 -> 답변 반복 (문장 단위, 여러 라운드)
    카멜레온  : 단어 1개씩 한 바퀴 -> 토론 한 바퀴 -> 투표 (한 판)   <- 여기
    금기어    : 1대1 자유 대화

⑤ 토론 페이즈(투표 전, 순차 공개)만 우리가 추가했고 나머지는 논문 원안
(code/paper_code/LLMChameleon-main/) 그대로. `--no-discussion`으로 끄면
논문 원안 흐름과 동일해져서 토론 유/무 자체가 실험 조건이 된다
(구상안 확정 결정 #1).

데이터는 category-spyfall의 categories_kr.json / categories_en.json을
이 폴더로 복사해 자체 보유한다 (분야 5개 x 20단어, 게임당 16개 샘플링 --
확정 결정 #6, 스냅샷 방식).

실행 예:
    python run_game.py --categories categories_kr.json \\
        --language Korean \\
        --non-chameleon-model <model> --chameleon-model <model> \\
        --num-games 30 --base-url http://localhost:8001/v1

토론 없는 대조 조건:
    python run_game.py --no-discussion ...

자세한 건 README.md / 설계 결정.md 참고.
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
from game import ChameleonGame, Player

load_dotenv()  # picks up ../.env (python-dotenv searches parent directories automatically)

# Windows consoles default to a legacy codepage (e.g. cp949) that can't print
# most Korean text and crashes with UnicodeEncodeError -- force UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 데이터는 category-spyfall의 것을 **복사해서** 이 폴더가 자체 보유함 (구상안
# 확정 결정 #6의 "재사용"을 스냅샷 방식으로): 실험 폴더의 데이터는 돌린 시점
# 그대로 동결돼야 재현이 되고, category-spyfall은 폴더명 변경이 예정돼 있어
# 참조가 깨질 수 있음. 단어를 추가/수정하면 test_game_logic.py의 충돌 검사를
# 반드시 통과시킬 것 (부분 문자열 충돌이 생기면 최종 추측 판정이 망가짐).
DEFAULT_CATEGORIES = str(Path(__file__).resolve().parent / "categories_en.json")


def load_categories(path: str) -> dict[str, list[str]]:
    """categories_kr.json / categories_en.json -> {분야명: [단어, ...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["categories"]


def build_players(non_chameleon_model: str, chameleon_model: str, num_non_chameleons: int) -> list[Player]:
    # 이름은 역할과 무관하게(Player0~4) 부여하고 이름-역할 매핑도 무작위 --
    # 역할이 이름표로 드러나지 않게 (스파이폴과 동일).
    total = num_non_chameleons + 1
    names = [f"Player{i}" for i in range(total)]
    random.shuffle(names)
    players = [Player(name=names[i], model=non_chameleon_model, role="non_chameleon")
               for i in range(num_non_chameleons)]
    players.append(Player(name=names[num_non_chameleons], model=chameleon_model, role="chameleon"))
    random.shuffle(players)  # 리스트 순서 = 발언 순서(힌트/토론 공통)이므로 게임마다 섞음
    return players


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chameleon games (discussion-phase redesign).")
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES,
                        help="분야별 단어 파일 (기본: 이 폴더의 categories_en.json. "
                             "한국어는 categories_kr.json)")
    parser.add_argument("--num-candidates", type=int, default=16,
                        help="게임마다 분야 안에서 뽑을 후보 수 (기본 16 = 논문 원안). "
                             "category 모드에서도 정답을 이 풀에서 뽑으므로 난이도에 영향.")
    parser.add_argument("--candidate-mode", choices=["category", "full"], default="category",
                        help="category(기본) = **변형 카멜레온**: 목록 없이 카테고리만, 전원 비공개. "
                             "full = 원작/논문 그대로 (카테고리 + 목록 전원 공개) -- 원작 재현 대조용.")
    parser.add_argument("--discussion-rounds", type=int, choices=[1, 2], default=2,
                        help="토론 바퀴 수. 2(기본) = 제기+반박 (설계 방침). "
                             "1 = 제기만 -- 반박 라운드 효과 비교 조건.")
    parser.add_argument("--leak-warning", action=argparse.BooleanOptionalAction, default=True,
                        help="시민 토론/반박 프롬프트의 '정답 단어를 절대 입에 올리지 마라' 경고 "
                             "(기본 on = 표준 세팅. 원작의 암묵 상식을 명시한 것으로, 스파이폴의 "
                             "'Be extremely subtle' 지시와 수위를 맞춤 -- 프롬프트 실험 변수가 아님). "
                             "무경고는 세팅 확정 전 파일럿에서만 사용됨: 시민 토론 발언의 93~97%%가 "
                             "정답을 그대로 포함해 게임이 성립하지 않았음 (설계 결정 #14).")
    parser.add_argument("--discussion", action=argparse.BooleanOptionalAction, default=True,
                        help="투표 전 토론 페이즈(⑤) 유/무. 기본 on -- 우리 설계의 핵심. "
                             "--no-discussion이면 논문 원안 흐름과 동일 = 재현 대조 조건 "
                             "(토론 유/무 비교가 제일 중요한 실험 축, 구상안 확정 결정 #1).")
    parser.add_argument("--language", default="English",
                        help='Passed into the {language} slot, e.g. "English", "Korean".')
    parser.add_argument("--non-chameleon-model", required=True, help="Model used for all villager players.")
    parser.add_argument("--chameleon-model", required=True, help="Model used for the chameleon.")
    parser.add_argument("--num-non-chameleons", type=int, default=4,
                        help="기본 4 (총 5명 -- 스파이폴과 동일하게 맞춤, 구상안 확정 결정 #4).")
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-base-url", action="append", default=[], metavar="MODEL=URL",
                        help="특정 모델만 다른 서버로 보냄 (여러 번 지정 가능). 역할 교차 대국처럼 "
                             "두 역할의 모델이 서로 다른 포트에 떠 있을 때 사용. 미지정 모델은 "
                             "--base-url 서버로 감.")
    parser.add_argument("--min-seconds-between-calls", type=float, default=13.0,
                        help="Client-side pacing between LLM calls (default 13s for Gemini free tier). "
                             "Pass 0 for a local server (e.g. llama.cpp/vLLM).")
    parser.add_argument("--reformatter-model", default=None,
                        help="If a response fails the strict |||...||| JSON format, retry once by asking "
                             "this model to re-wrap the same content, instead of an immediate downgrade. "
                             "Every history entry is tagged with needed_reformat either way.")
    parser.add_argument("--action-retries", type=int, default=2,
                        help="힌트/토론/최종 추측/dealer 선택(스킵 불가 행동)이 무효 출력일 때 재시도 횟수. "
                             "전부 실패하면 'invalid' 게임으로 기록되고 승패 집계에서 제외됨.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0,
                        help="앞의 N판을 건너뛰고 N번째 판부터 실행 (같은 시드면 앞 N판 구성은 기존 실행과 동일 -- 판 수 확대용)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="동시에 진행할 게임 수 (게임 단위 병렬 -- 게임끼리는 완전히 독립). "
                             "게임 설정은 메인 스레드에서 미리 뽑아 --seed 재현성 유지.")
    parser.add_argument("--save-transcripts", default=None,
                        help="Path to write full per-game turn-by-turn history as JSON. "
                             "Always saved (this cannot be turned off) -- if omitted, an auto-named "
                             "file under transcripts/ is used.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    llm_client.MIN_SECONDS_BETWEEN_CALLS = args.min_seconds_between_calls

    # Full dialogue transcripts are always written -- 정성 분석이 이 연구의
    # 핵심이라 요약 통계만으로는 부족함. 게임 하나 끝날 때마다 즉시 기록해서
    # 중간에 죽어도 끝난 게임은 남게 함 (세 게임 공통 정책).
    if args.save_transcripts:
        transcripts_path = Path(args.save_transcripts)
    else:
        transcripts_dir = Path(__file__).parent / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_villager = args.non_chameleon_model.replace("/", "-").replace(":", "-")
        safe_chameleon = args.chameleon_model.replace("/", "-").replace(":", "-")
        disc = "disc" if args.discussion else "nodisc"
        transcripts_path = transcripts_dir / f"{stamp}_{safe_villager}_vs_{safe_chameleon}_{args.language}_{args.candidate_mode}_{disc}.json"
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = llm_client.build_client(api_key, args.base_url, args.model_base_url)
    categories = load_categories(args.categories)

    dump: list[dict] = []

    # 게임 설정(분야/후보/정답/역할 배정/dealer)은 **메인 스레드에서 미리 다
    # 뽑아둠** -- 워커 스레드 안에서 random을 쓰면 --seed 재현이 깨짐.
    # 분야 하나 고르고 셔플 후 앞 N개 = 논문 코드의 shuffle 후 [:16]과 동일.
    def make_setup(i: int) -> tuple:
        category = random.choice(list(categories))
        pool = list(categories[category])
        random.shuffle(pool)
        candidates = pool[:args.num_candidates]
        secret_word = random.choice(candidates)
        players = build_players(args.non_chameleon_model, args.chameleon_model, args.num_non_chameleons)
        dealer_name = random.choice(players).name  # 동률 처리자: 참가자 중 무작위 1명 (논문 그대로)
        return (i, category, candidates, secret_word, players, dealer_name)

    setups = [make_setup(i) for i in range(args.num_games)]
    # --start-index: 구성은 전부 뽑아두고(시드 소비 순서를 100판 실행과 동일하게
    # 유지해야 앞 판 구성이 일치) 실제로 돌릴 판만 잘라냄 -- 판 수 확대용.
    if args.start_index:
        setups = setups[args.start_index:]
        print(f"[--start-index {args.start_index}] game#{args.start_index}부터 {len(setups)}판 실행")

    dump_lock = threading.Lock()

    def play_one(setup: tuple) -> None:
        game_idx, category, candidates, secret_word, players, dealer_name = setup
        game = ChameleonGame(client, players, category, candidates, secret_word, dealer_name,
                             language=args.language, reformatter_model=args.reformatter_model,
                             action_retries=args.action_retries, discussion=args.discussion,
                             candidate_mode=args.candidate_mode, discussion_rounds=args.discussion_rounds,
                             leak_warning=args.leak_warning)
        result = game.play()
        chameleon_name = next(p.name for p in players if p.role == "chameleon")
        entry = {
            "game_index": game_idx,
            "category": category,
            "candidates": candidates,
            "secret_word": secret_word,
            "discussion": args.discussion,  # 실험 조건 (토론 유/무)
            "candidate_mode": args.candidate_mode,  # 실험 조건 (category=변형 카멜레온 / full=원작)
            "discussion_rounds": args.discussion_rounds,  # 실험 조건 (2=제기+반박 / 1=제기만)
            "leak_warning": args.leak_warning,  # 실험 조건 (시민 토론의 정답 발화 금지 경고 유/무)
            "dealer": dealer_name,
            "players": [{"name": p.name, "model": p.model, "role": p.role} for p in players],
            "chameleon": chameleon_name,
            "winner": result.winner,
            "reason": result.reason,
            "accused": result.accused,      # 최다 득표로 지목된 사람 (없으면 null)
            "identified": result.identified,  # 지목된 사람이 진짜 카멜레온이었는지
            "history": result.history,
        }
        # 저장은 매 게임마다, game_index로 정렬해서 (병렬이면 끝나는 순서가 뒤죽박죽).
        with dump_lock:
            dump.append(entry)
            dump.sort(key=lambda r: r["game_index"])
            transcripts_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{len(dump)}/{args.num_games} done] game#{game_idx} secret={secret_word!r} "
                  f"winner={result.winner:13s} identified={result.identified} reason={result.reason}", flush=True)

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(play_one, s) for s in setups]
            for fut in as_completed(futures):
                fut.result()  # 워커에서 터진 예외를 여기서 그대로 올림
    else:
        for s in setups:
            play_one(s)

    print(f"\nSaved full transcripts ({len(dump)} game(s)) to {transcripts_path}")

    # 무효 게임(재시도까지 전부 실패)은 승패 집계에서 제외.
    decided = [r for r in dump if r["winner"] in ("chameleon", "non_chameleon")]
    invalid = [r for r in dump if r["winner"] == "invalid"]

    print(f"\n=== Summary (discussion={'on' if args.discussion else 'off'}) ===")
    print(f"Games played:            {len(dump)}  (invalid/aborted: {len(invalid)}, excluded from rates)")
    if decided:
        n = len(decided)
        identified = sum(1 for r in decided if r["identified"])
        cham_wins = sum(1 for r in decided if r["winner"] == "chameleon")
        # 시민 개인 투표 정확도: 시민이 던진 유효 표 중 카멜레온을 맞힌 비율
        # (기권 강등 표는 분모에서 제외하되 별도로 집계).
        villager_votes = 0
        villager_hits = 0
        abstains = 0
        second_chance_correct = 0
        second_chances = 0
        for r in decided:
            villagers = {p["name"] for p in r["players"] if p["role"] == "non_chameleon"}
            for h in r["history"]:
                if h["phase"] == "vote" and h["actor"] in villagers:
                    if h["target"] is None:
                        abstains += 1
                    else:
                        villager_votes += 1
                        villager_hits += 1 if h["target"] == r["chameleon"] else 0
                elif h["phase"] == "final_guess":
                    second_chances += 1
                    second_chance_correct += 1 if h["correct"] else 0
        print(f"Chameleon win rate:      {100 * cham_wins / n:.1f}%  ({n} decided games)")
        print(f"Chameleon identified:    {100 * identified / n:.1f}%  (accused player was the chameleon)")
        if villager_votes + abstains:
            print(f"Villager vote accuracy:  {100 * villager_hits / villager_votes:.1f}%  "
                  f"({villager_hits}/{villager_votes} valid votes; {abstains} downgraded to abstain)"
                  if villager_votes else
                  f"Villager vote accuracy:  n/a (all {abstains} villager votes downgraded to abstain)")
        if second_chances:
            print(f"Final-guess success:     {second_chance_correct}/{second_chances} "
                  f"(caught chameleons guessing the secret word)")


if __name__ == "__main__":
    main()
