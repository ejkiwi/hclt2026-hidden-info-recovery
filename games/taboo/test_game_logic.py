"""API 호출 없이 금기어 게임 엔진 로직만 검증하는 회귀 테스트.

game.call_player를 가짜 응답으로 몬키패치해서 승리 경로/무효 게임/선언 강등/
누출 판정/금기어 재작성을 전부 실행해봄. pytest 없이 `python test_game_logic.py`.
(스파이폴 쪽 test_game_logic.py와 같은 방식)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import game as game_module
from game import Player, TabooGame, mentions_word, normalize_guess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from llm_client import InvalidMoveError

TARGET = "돌고래"


def make_players() -> tuple[Player, Player]:
    return (Player(name="Attacker", model="fake", role="attacker"),
            Player(name="Defender", model="fake", role="defender"))


def role_of(prompt: str) -> str:
    # 주의: BASE_CONTEXT에 "*As the Attacker:*" 같은 문구가 양쪽 다 들어있으므로,
    # 각 페이즈 '템플릿'에만 있는 문장으로 판별해야 함.
    if "You are the Attacker." in prompt:
        return "attack"
    if "You are the Defender." in prompt:
        return "defend"
    raise AssertionError(f"unrecognized role in prompt: {prompt[:200]}")


def run_fake_game(handlers: dict, max_rounds: int = 3, **kwargs) -> "game_module.GameResult":
    """handlers: {"attack"/"defend": fn(round_hint, prompt) -> dict}"""
    attacker, defender = make_players()

    def fake_call_player(client, model, prompt, reformatter_model=None):
        result = handlers[role_of(prompt)](prompt)
        result.setdefault("needed_reformat", False)
        result.setdefault("raw_response", "fake raw output")
        return result

    original = game_module.call_player
    game_module.call_player = fake_call_player
    try:
        g = TabooGame(client=None, attacker=attacker, defender=defender, target_word=TARGET,
                      max_rounds=max_rounds, **kwargs)
        return g.play()
    finally:
        game_module.call_player = original


def bland_attack(prompt):
    return {"utterance": "요즘 바다 쪽 여행은 어때요?"}


def bland_defend(prompt):
    return {"utterance": "좋죠, 물놀이 생각이 나네요.", "declare": False, "guess": None}


# -- 승리 경로 ---------------------------------------------------------

def test_defender_declares_correctly_wins():
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "알 것 같아요.", "declare": True, "guess": TARGET}})
    assert r.winner == "defender", r
    assert "correctly" in r.reason


def test_defender_declares_wrong_loses():
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "감이 왔어요.", "declare": True, "guess": "고래상어"}})
    assert r.winner == "attacker", r
    assert "wrong word" in r.reason


def test_defender_saying_target_loses_and_flags_leak():
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "아쿠아리움에서 돌고래를 봤어요.", "declare": False, "guess": None}})
    assert r.winner == "attacker", r
    assert r.leaked is True
    assert r.history[-1]["said_target"] is True


def test_attacker_saying_target_loses():
    r = run_fake_game({"attack": lambda p: {"utterance": "혹시 돌고래 좋아하세요?"}, "defend": bland_defend})
    assert r.winner == "defender", r
    assert "attacker said the target word" in r.reason
    # 공격자가 흘린 건 방어자 누출(leaked)이 아님 -- 지표가 섞이면 안 됨
    assert r.leaked is False


def test_turn_limit_is_a_draw():
    r = run_fake_game({"attack": bland_attack, "defend": bland_defend}, max_rounds=3)
    assert r.winner == "draw", r
    assert r.turns_used == 3
    assert len(r.history) == 6  # 라운드당 2발화


def test_declare_wins_over_utterance_leak_in_same_turn():
    """같은 턴에 발화로도 단어가 나오고 선언도 맞은 경우 -> 선언 우선(방어자 승),
    단 누출은 기록에 남음 (의 '선언 우선' 규칙)."""
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "돌고래 아닌가요?", "declare": True, "guess": TARGET}})
    assert r.winner == "defender", r
    assert r.leaked is True
    assert r.history[-1]["said_target_while_declaring"] is True


# -- 무효 출력 처리 -----------------------------------------------------

def test_string_boolean_is_coerced_not_truthy():
    """declare가 문자열 "false"로 오면 truthy로 평가돼 '선언함'이 되면 안 됨."""
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "글쎄요.", "declare": "false", "guess": None}})
    assert r.winner == "draw", r
    assert r.history[-1]["coerced_bool"] is True
    assert r.history[-1]["declare"] is False


def test_uninterpretable_declare_downgrades_to_skip():
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "음...", "declare": "maybe", "guess": None}})
    assert r.winner == "draw", r
    assert "invalid declare" in r.history[-1]["downgraded_to_skip"]


def test_declare_without_guess_downgrades_to_skip():
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "확실해요!", "declare": True, "guess": None}})
    assert r.winner == "draw", r
    assert "no usable guess" in r.history[-1]["downgraded_to_skip"]


def test_unparseable_utterance_aborts_game_as_invalid():
    """발화는 스킵이 없는 필수 행동 -> 재시도까지 실패하면 몰수패가 아니라 무효 게임."""
    def broken(prompt):
        raise InvalidMoveError("no |||JSON||| block")
    r = run_fake_game({"attack": broken, "defend": bland_defend})
    assert r.winner == "invalid", r
    assert "Attacker" in r.reason


def test_utterance_retry_succeeds_and_is_recorded():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InvalidMoveError("first attempt broken")
        return {"utterance": "그냥 잡담이요."}

    r = run_fake_game({"attack": flaky, "defend": bland_defend}, max_rounds=1)
    assert r.winner == "draw", r
    assert r.history[0]["turn_retries"] == 1


def test_missing_utterance_field_is_invalid():
    r = run_fake_game({"attack": lambda p: {"wrong_field": "..."}, "defend": bland_defend})
    assert r.winner == "invalid", r


# -- 금기어 목록(부차 제약) ---------------------------------------------

def test_forbidden_word_triggers_rewrite():
    calls = {"n": 0}

    def attacker(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"utterance": "수족관 이야기 해봐요."}   # '수족관'이 금기어
        return {"utterance": "바다 이야기 해봐요."}

    r = run_fake_game({"attack": attacker, "defend": bland_defend},
                      max_rounds=1, forbidden_words=["수족관"])
    assert r.winner == "draw", r
    assert "taboo_violation" not in r.history[0]        # 재작성으로 해소됨
    assert r.history[0]["utterance"] == "바다 이야기 해봐요."
    assert "**Rule reminder:**" not in r.history[0]["utterance"]


def test_forbidden_violation_is_flagged_not_a_loss():
    """재시도까지 못 고쳐도 게임을 끝내지 않고 기록만 남김 -- 부차 제약은
    승패 조건이 아니라는 설계 결정."""
    r = run_fake_game({"attack": lambda p: {"utterance": "수족관 갑시다."}, "defend": bland_defend},
                      max_rounds=1, forbidden_words=["수족관"])
    assert r.winner == "draw", r
    assert r.history[0]["taboo_violation"] == ["수족관"]


# -- 판정 함수 ---------------------------------------------------------

def test_mentions_word_korean_particles():
    assert mentions_word("어제 돌고래를 봤어요", "돌고래") is True
    assert mentions_word("돌고래쇼가 재밌었어요", "돌고래") is True
    assert mentions_word("고래를 봤어요", "돌고래") is False


def test_mentions_word_ascii_word_boundary():
    assert mentions_word("I saw a cat today", "cat") is True
    assert mentions_word("Pick a category first", "cat") is False   # 부분 일치 오탐 방지
    assert mentions_word("A CAT appeared", "cat") is True           # 대소문자 무시


def test_normalize_guess():
    assert normalize_guess("The Beach.") == "beach"
    assert normalize_guess("  돌고래! ") == "돌고래"


# -- 프롬프트 조립 -----------------------------------------------------

def test_attacker_sees_target_and_defender_does_not():
    attacker, defender = make_players()
    g = TabooGame(client=None, attacker=attacker, defender=defender, target_word=TARGET)
    from prompts import ATTACKER_UTTERANCE, DEFENDER_UTTERANCE
    att_prompt = g._player_prompt(attacker, ATTACKER_UTTERANCE)
    def_prompt = g._player_prompt(defender, DEFENDER_UTTERANCE)
    assert TARGET in att_prompt
    assert TARGET not in def_prompt
    assert "there is no candidate list" in def_prompt   # 열린 어휘(SPAG 원안)


def test_declaration_priority_is_stated_in_prompt():
    """설계 결정 #2(선언 우선)를 프롬프트가 실제로 알려주는지 -- 규칙 설명과
    판정이 어긋나면 방어자가 선언 턴에도 단어를 피해 어색하게 말하게 됨."""
    attacker, defender = make_players()
    g = TabooGame(client=None, attacker=attacker, defender=defender, target_word=TARGET)
    from prompts import DEFENDER_UTTERANCE
    def_prompt = g._player_prompt(defender, DEFENDER_UTTERANCE)
    assert "The declaration takes priority." in def_prompt
    assert "only loses the game when the Defender is NOT declaring" in def_prompt


def test_no_unsubstituted_slots_in_any_prompt():
    """{language}가 BASE_CONTEXT뿐 아니라 페이즈 템플릿에도 있으므로 조립 후
    미치환 슬롯이 남지 않는지 확인."""
    attacker, defender = make_players()
    g = TabooGame(client=None, attacker=attacker, defender=defender, target_word=TARGET, language="Korean")
    from prompts import ATTACKER_UTTERANCE, DEFENDER_UTTERANCE
    for player, template in ((attacker, ATTACKER_UTTERANCE), (defender, DEFENDER_UTTERANCE)):
        prompt = g._player_prompt(player, template, g._turn_block(1))
        assert "{language}" not in prompt
        assert "{base_context}" not in prompt


def test_forbidden_words_shown_to_attacker():
    attacker, defender = make_players()
    g = TabooGame(client=None, attacker=attacker, defender=defender, target_word=TARGET,
                  forbidden_words=["수족관", "지느러미"])
    from prompts import ATTACKER_UTTERANCE, DEFENDER_UTTERANCE
    assert "수족관, 지느러미" in g._player_prompt(attacker, ATTACKER_UTTERANCE)
    assert "수족관" not in g._player_prompt(defender, DEFENDER_UTTERANCE)


def test_language_slot_is_substituted():
    attacker, defender = make_players()
    g = TabooGame(client=None, attacker=attacker, defender=defender, target_word=TARGET, language="Korean")
    from prompts import ATTACKER_UTTERANCE
    prompt = g._player_prompt(attacker, ATTACKER_UTTERANCE)
    assert "speak in Korean language" in prompt
    assert "{language}" not in prompt


def test_history_block_shows_both_speakers():
    attacker, defender = make_players()
    g = TabooGame(client=None, attacker=attacker, defender=defender, target_word=TARGET)
    g.history = [{"phase": "attack", "actor": "Attacker", "utterance": "안녕하세요"},
                 {"phase": "defend", "actor": "Defender", "utterance": "네 안녕하세요"}]
    block = g._history_block()
    assert "[Attacker]: 안녕하세요" in block
    assert "[Defender]: 네 안녕하세요" in block




# -- 발화 속 추측 선언 (절충안 ⓐ) ---------------------------------------

def test_verbal_declaration_markers():
    from game import verbal_declaration as vd
    # 실제 패배 사례들 -- 전부 선언으로 인정되어야 함
    assert vd("아무래도 학교인 것 같은데요, 등교하는 그 장소 말입니다", "학교")
    assert vd("그 유명한 눈사람이 맞죠?", "눈사람")
    assert vd("혹시 돌고래를 유도하시는 건가요?", "돌고래")
    assert vd("Is it a snowman?", "snowman")
    assert vd("It must be the hospital, right?", "hospital")
    # 우발 언급 -- 표지가 없거나 멀어서 인정되면 안 됨
    assert vd("어제 병원에 다녀왔어요", "병원") is None
    assert vd("병원에 가는 것도 좋을 것 같아요", "병원") is None   # 표지가 단어와 멂
    assert vd("I went to the hospital yesterday", "hospital") is None


def _fake_judge(answer):
    """판정 LLM을 몬키패치 -- 지정한 숫자를 돌려주는 가짜 심판."""
    import game as gm
    original = gm._judge_chat
    gm._judge_chat = lambda client, model, prompt: answer
    return original


def test_llm_judge_confident_guess_wins():
    """판정 LLM이 '1(확신 추측)'이면 누출 패가 아니라 선언 성공."""
    import game as gm
    orig = _fake_judge("1")
    try:
        r = run_fake_game({"attack": bland_attack,
                           "defend": lambda p: {"utterance": "아무래도 돌고래인 것 같은데요.", "declare": False, "guess": None}})
    finally:
        gm._judge_chat = orig
    assert r.winner == "defender", r
    assert "verbally" in r.reason
    assert r.history[-1]["declare_judge"]["verdict"] == "confident_guess"
    assert r.leaked is True    # 누출 사실 자체는 기록 유지


def test_llm_judge_hedged_probe_loses():
    """'2(복수 후보 떠보기)'는 규칙 위반 -- 누출 패 ."""
    import game as gm
    orig = _fake_judge("2")
    try:
        r = run_fake_game({"attack": bland_attack,
                           "defend": lambda p: {"utterance": "돌고래나 고래 같은 것일까요?", "declare": False, "guess": None}})
    finally:
        gm._judge_chat = orig
    assert r.winner == "attacker", r
    assert r.history[-1]["declare_judge"]["verdict"] == "hedged_probe"


def test_llm_judge_garbage_falls_back_to_rule():
    """판정 출력이 해석 불가면 규칙(표지) 폴백 -- 게임이 판정 실패로 죽지 않음."""
    import game as gm
    orig = _fake_judge("uhh I am not sure")
    try:
        r = run_fake_game({"attack": bland_attack,
                           "defend": lambda p: {"utterance": "아무래도 돌고래인 것 같은데요.", "declare": False, "guess": None}})
    finally:
        gm._judge_chat = orig
    assert r.winner == "defender", r
    assert "rule_fallback" in r.history[-1]["declare_judge"]["detail"]


def test_judge_off_restores_old_judging():
    r = run_fake_game({"attack": bland_attack,
                       "defend": lambda p: {"utterance": "아무래도 돌고래인 것 같은데요.", "declare": False, "guess": None}},
                      declare_judge="off")
    assert r.winner == "attacker", r


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn
            print(f"ok  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed} tests passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
