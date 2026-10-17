"""API 호출 없이 게임 엔진 로직만 검증하는 회귀 테스트.

game.call_player를 가짜 응답으로 몬키패치해서 승리 경로/몰수패/무효표/누출
판정을 전부 실행해봄. pytest 없이 `python test_game_logic.py`로 바로 실행.
(README "검증" 섹션 참고 -- 예전엔 이 검증이 일회성 몬키패치 스크립트로만
있고 리포에 안 남아 있었어서, 이번 검토 때 영구 파일로 만들어둠.)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import game as game_module
from game import Player, SpyfallGame, detect_leakage, normalize_entity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from llm_client import InvalidMoveError, parse_pipe_json

ENTITIES = ["Beach", "Bank", "Hospital", "Zoo", "Pirate Ship"]


def make_players() -> list[Player]:
    # 고정 배치: Player2가 스파이. (run_game.build_players의 셔플은 여기선 배제)
    roles = ["non_spy", "non_spy", "spy", "non_spy", "non_spy"]
    return [Player(name=f"Player{i}", model="fake", role=r) for i, r in enumerate(roles)]


def actor_of(prompt: str) -> str:
    return re.search(r"\*\*Your name in this game is:\*\* (\w+)", prompt).group(1)


def phase_of(prompt: str) -> str:
    # 주의: base_context 안에 모든 페이즈 이름("Spy Guess Decision" 등)이 다
    # 들어 있으므로, 각 페이즈 '템플릿'에만 존재하는 문구로 판별해야 함.
    if "need to ask a question" in prompt:
        return "question"
    if "answering a question" in prompt:
        return "answer"
    if "deciding who to vote for" in prompt:
        return "vote"
    if "You are the Spy in Phase 2" in prompt:
        return "guess"
    raise AssertionError(f"unrecognized phase in prompt: {prompt[:200]}")


def run_fake_game(handlers: dict, players=None) -> "game_module.GameResult":
    """handlers: {"question"/"answer"/"guess"/"vote": fn(actor_name, prompt) -> dict}"""
    players = players or make_players()

    def fake_call_player(client, model, prompt, reformatter_model=None):
        result = handlers[phase_of(prompt)](actor_of(prompt), prompt)
        result.setdefault("needed_reformat", False)
        result.setdefault("raw_response", "fake raw output")
        return result

    original = game_module.call_player
    game_module.call_player = fake_call_player
    try:
        g = SpyfallGame(client=None, players=players, entities=ENTITIES, target_entity="Beach")
        return g.play()
    finally:
        game_module.call_player = original


def benign_question(actor, prompt):
    # 자기 자신이 아닌 아무나 지목 (Player0이 아니면 Player0, 맞으면 Player1)
    target = "Player0" if actor != "Player0" else "Player1"
    return {"question": "Do you come here often?", "targeted_player": target}


def benign_answer(actor, prompt):
    return {"answer": "Sometimes, it depends."}


def skip_guess(actor, prompt):
    return {"should_guess": False, "best_guess": None, "confidence": 0.3}


def skip_vote(actor, prompt):
    return {"should_vote": False, "target_player_name": None, "confidence": 0.5}


def test_spy_survives_when_nothing_happens():
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": skip_vote})
    assert result.winner == "spy" and "survived" in result.reason, result
    # 라운드로빈 5 + 자유질문 5 = 질문 10개, 투표 라운드 6회(사이클 5 + 최종) × 5명
    questions = [h for h in result.history if h["phase"] == "question"]
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert len(questions) == 10, len(questions)
    assert len(votes) == 30, len(votes)
    assert {v["vote_round"] for v in votes} == {1, 2, 3, 4, 5, 6}
    assert all(v["raw_response"] for v in votes)  # 원본 응답 저장 확인


def test_votes_are_simultaneous_within_a_round():
    """실제 보드게임처럼 "하나 둘 셋" 동시 지목 -- 같은 투표 라운드 안에서는
    앞 순서 사람의 표가 뒤 순서 사람 프롬프트에 보이면 안 되고(라운드 내 따라
    투표 차단), 라운드가 끝나면 공개돼서 다음 라운드에는 보여야 함."""
    seen: list[tuple[str, str]] = []   # (투표한 사람, 그 사람이 본 프롬프트)

    def recording_vote(actor, prompt):
        seen.append((actor, prompt))
        return {"should_vote": True, "target_player_name": "Player3", "confidence": 0.7}

    run_fake_game({"question": benign_question, "answer": benign_answer,
                   "guess": skip_guess, "vote": recording_vote})

    # 1라운드: 5명 중 누구의 프롬프트에도 이번 라운드 표가 없어야 함
    for actor, prompt in seen[:5]:
        assert "(voting round 1)" not in prompt, f"{actor}가 같은 라운드의 표를 봤음"
    # 2라운드: 1라운드 표는 전원에게 보여야 함 (공개된 뒤라서)
    for actor, prompt in seen[5:10]:
        assert "(voting round 1)" in prompt, f"{actor}가 지난 라운드 표를 못 봄"
        assert "(voting round 2)" not in prompt, f"{actor}가 같은 라운드의 표를 봤음"


def test_ghost_name_votes_do_not_crash():
    # 프롬프트 예시의 "Charlie"를 전원이 그대로 베껴도 크래시 없이 스킵 처리돼야 함
    # (수정 전엔 과반 도달 시 next가 StopIteration으로 런 전체를 죽였음)
    def charlie_vote(actor, prompt):
        return {"should_vote": True, "target_player_name": "Charlie", "confidence": 0.9}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": charlie_vote})
    assert result.winner == "spy" and "survived" in result.reason, result
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert all(v["target"] is None and v["invalid_target"] == "Charlie" for v in votes)


def test_self_vote_is_invalidated():
    def self_vote(actor, prompt):
        return {"should_vote": True, "target_player_name": actor, "confidence": 0.9}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": self_vote})
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert all(v["target"] is None and v["invalid_target"] == v["actor"] for v in votes)


def test_majority_vote_catches_spy():
    def vote_spy(actor, prompt):
        if actor == "Player2":  # 스파이는 무고한 사람 지목
            return {"should_vote": True, "target_player_name": "Player0", "confidence": 0.6}
        return {"should_vote": True, "target_player_name": "Player2", "confidence": 0.8}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": vote_spy})
    assert result.winner == "non_spy" and "voted out" in result.reason, result


def test_spy_guess_normalization():
    # 표기 변형("The Beach.")도 정답 처리돼야 함
    def guess_messy(actor, prompt):
        return {"should_guess": True, "best_guess": "The Beach.", "confidence": 0.9}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": guess_messy, "vote": skip_vote})
    assert result.winner == "spy" and "correctly" in result.reason, result
    guess_entry = next(h for h in result.history if h["phase"] == "guess")
    assert guess_entry["guess_in_list"] is True


def test_wrong_guess_loses():
    def guess_wrong(actor, prompt):
        return {"should_guess": True, "best_guess": "Hospital", "confidence": 0.9}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": guess_wrong, "vote": skip_vote})
    assert result.winner == "non_spy" and "incorrectly" in result.reason, result


def test_broken_question_becomes_invalid_game_not_forfeit():
    # 질문은 스킵이 불가능한 필수 행동 -- 재시도(기본 2회)까지 전부 실패하면
    # 어느 팀의 승리도 아닌 '무효 게임'으로 중단됨 (포맷은 승패 조건이 아님)
    calls = {"n": 0}
    def bad_question(actor, prompt):
        calls["n"] += 1
        return {"answer": "oops wrong schema"}
    result = run_fake_game({"question": bad_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": skip_vote})
    assert result.winner == "invalid" and result.reason.startswith("invalid game"), result
    assert "Player0" in result.reason
    assert calls["n"] == 3  # 최초 1회 + 재시도 2회


def test_broken_question_recovers_on_retry():
    # 두 번 망가지고 세 번째에 정상 응답 -> 게임은 그대로 계속, 재시도 횟수 기록
    calls = {"n": 0}
    def flaky_question(actor, prompt):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"nonsense": True}
        return benign_question(actor, prompt)
    result = run_fake_game({"question": flaky_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": skip_vote})
    assert result.winner == "spy" and "survived" in result.reason, result
    first_q = next(h for h in result.history if h["phase"] == "question")
    assert first_q["turn_retries"] == 2


def test_string_boolean_is_coerced_not_truthy():
    # should_vote가 문자열 "false"면 truthy 오판(투표한 걸로 취급)하지 않고
    # False로 보정 + coerced_bool 플래그. 몰수패도 아님 (스킵과 동일).
    def string_bool_vote(actor, prompt):
        return {"should_vote": "false", "target_player_name": "Player2", "confidence": 0.5}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": string_bool_vote})
    assert result.winner == "spy" and "survived" in result.reason, result
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert all(v["target"] is None and v.get("coerced_bool") for v in votes)


def test_guess_true_with_null_best_guess_downgrades_to_skip():
    # 추측 페이즈는 규칙상 스킵이 합법이라, 무효 출력은 몰수패 대신 스킵 강등
    def bad_guess(actor, prompt):
        return {"should_guess": True, "best_guess": None, "confidence": 0.9}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": bad_guess, "vote": skip_vote})
    assert result.winner == "spy" and "survived" in result.reason, result
    guesses = [h for h in result.history if h["phase"] == "guess"]
    assert all(g.get("downgraded_to_skip") for g in guesses)


def test_unparseable_vote_downgrades_to_skip():
    # 투표 응답이 아예 파싱 불가여도 몰수패가 아니라 그 사람만 기권 처리
    def broken_vote(actor, prompt):
        raise InvalidMoveError("no JSON block found")
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": broken_vote})
    assert result.winner == "spy" and "survived" in result.reason, result
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert all(v["target"] is None and v.get("downgraded_to_skip") for v in votes)


def test_missing_should_vote_is_skip():
    def no_flag_vote(actor, prompt):
        return {"target_player_name": "Player2", "confidence": 0.9}
    result = run_fake_game({"question": benign_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": no_flag_vote})
    assert result.winner == "spy" and "survived" in result.reason, result


def test_leakage_from_question_and_answer():
    # 비스파이의 '질문'에 엔티티가 그대로 등장 -> 누출 (논문의 PQA엔 질문도 포함)
    def leaky_question(actor, prompt):
        target = "Player0" if actor != "Player0" else "Player1"
        return {"question": "Is the Beach crowded today?", "targeted_player": target}
    result = run_fake_game({"question": leaky_question, "answer": benign_answer,
                            "guess": skip_guess, "vote": skip_vote})
    assert result.leaked is True


def test_free_question_invalid_target_is_replaced_and_flagged():
    calls = {"n": 0}
    def question_with_ghost_target(actor, prompt):
        calls["n"] += 1
        if calls["n"] <= 5:  # 라운드로빈 5회는 정상 (대상 필드는 어차피 무시됨)
            return benign_question(actor, prompt)
        return {"question": "What do you think?", "targeted_player": "Nobody"}
    def guess_end(actor, prompt):  # 첫 자유질문 직후 게임 종료시킴
        return {"should_guess": True, "best_guess": "Beach", "confidence": 0.9}
    result = run_fake_game({"question": question_with_ghost_target, "answer": benign_answer,
                            "guess": guess_end, "vote": skip_vote})
    free_q = [h for h in result.history if h["phase"] == "question"][5]
    assert free_q["target_was_replaced"] is True and free_q["original_target"] == "Nobody"
    assert free_q["target"] != free_q["actor"]  # 자기 자신으로 대체되진 않음


def test_prompt_assembly_order_and_roster():
    g = SpyfallGame(client=None, players=make_players(), entities=ENTITIES, target_entity="Beach")
    g.history = [{"phase": "question", "actor": "Player0", "target": "Player1",
                  "question": "hm?", "needed_reformat": False}]
    from prompts import QUESTION_GENERATION_NON_SPY, QUESTION_GENERATION_SPY
    prompt = g._player_prompt(g.players[0], QUESTION_GENERATION_NON_SPY)
    # 논문 2.5 순서: history가 페이즈 과제/포맷 지시보다 앞에 와야 함
    assert prompt.index("Game history so far") < prompt.index("Response Format Requirements")
    assert "**Players in this game:** Player0, Player1, Player2, Player3, Player4" in prompt
    assert f"Entities list ({len(ENTITIES)} total)" in prompt
    assert "The secret entity is: Beach" in prompt          # 비스파이에겐 정답 공개
    spy_prompt = g._player_prompt(g.players[2], QUESTION_GENERATION_SPY)
    assert "The secret entity is" not in spy_prompt         # 스파이에겐 비공개


def test_detect_leakage_word_boundary():
    assert detect_leakage("Let's go to the bank today", "Bank") is True
    assert detect_leakage("I walked along the riverbank", "Bank") is False  # 오탐 방지
    assert detect_leakage("나는 해적선을 봤다", "해적선") is True  # 비ASCII는 부분 문자열 유지


def test_normalize_entity():
    assert normalize_entity("The Beach.") == normalize_entity("Beach")
    assert normalize_entity("pirate-ship") == normalize_entity("Pirate Ship")


def test_parse_pipe_json_robustness():
    # 추론 텍스트 속 ||| 언급(홀수 개)이 있어도 실제 JSON 블록을 잡아야 함
    ok = parse_pipe_json('I must use ||| pipes as told.\n|||\n{"a": 1}\n|||')
    assert ok == {"a": 1}
    # 형식 언급 "|||...|||"(짝수 개)도 브레이스가 없으니 무시됨
    ok = parse_pipe_json('wrap it in |||...||| right?\n|||{"b": 2}|||')
    assert ok == {"b": 2}
    for bad in ['```json\n{"a": 1}\n```', "no json at all", "|||not json|||"]:
        try:
            parse_pipe_json(bad)
            raise AssertionError(f"should have raised: {bad!r}")
        except InvalidMoveError:
            pass


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
