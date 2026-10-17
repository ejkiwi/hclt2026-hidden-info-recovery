"""API 호출 없이 카멜레온 게임 엔진 로직만 검증하는 회귀 테스트.

game.call_player를 가짜 응답으로 몬키패치해서 승리 경로/무효 출력 처리/
프롬프트 조립을 전부 실행해봄. pytest 없이 `python test_game_logic.py`로
바로 실행 (세 게임 공통 컨벤션 -- 구상안 '구현 메모'의 필수 항목).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import game as game_module
from game import ChameleonGame, Player, _guess_matches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from llm_client import InvalidMoveError

CATEGORY = "Places"
CANDIDATES = ["Beach", "Bank", "Hospital", "Zoo", "Harbor", "Library", "Castle", "Farm"]
SECRET = "Beach"


def make_players() -> list[Player]:
    # 고정 배치: Player2가 카멜레온. (run_game.build_players의 셔플은 여기선 배제)
    roles = ["non_chameleon", "non_chameleon", "chameleon", "non_chameleon", "non_chameleon"]
    return [Player(name=f"Player{i}", model="fake", role=r) for i, r in enumerate(roles)]


def actor_of(prompt: str) -> str:
    return re.search(r"\*\*Your name in this game is:\*\* (\w+)", prompt).group(1)


def phase_of(prompt: str) -> str:
    # 주의: base_context에 모든 페이즈 설명이 다 들어 있으므로, 각 페이즈
    # '템플릿'에만 존재하는 문구로 판별해야 함.
    if "your turn to give your hint word" in prompt:
        return "hint"
    if "your discussion turn" in prompt or "second and final discussion round" in prompt:
        return "discussion"
    if "time to cast your vote" in prompt:
        return "vote"
    if "As the dealer" in prompt:
        return "dealer"
    if "you have been accused" in prompt:
        return "final_guess"
    raise AssertionError(f"unrecognized phase in prompt: {prompt[:200]}")


def run_fake_game(handlers: dict, players=None, discussion=True, secret=SECRET,
                  candidates=None, dealer_name="Player4",
                  candidate_mode="full", discussion_rounds=2) -> "game_module.GameResult":
    """handlers: {"hint"/"discussion"/"vote"/"dealer"/"final_guess": fn(actor, prompt) -> dict}
    candidate_mode 기본은 "full" -- 목록 기반 검증(정확히 하나 포함 등)을 검사하는
    기존 테스트용. 변형 카멜레온(category, 엔진 기본값)은 명시적으로 지정해서 검사."""
    players = players or make_players()

    def fake_call_player(client, model, prompt, reformatter_model=None):
        result = handlers[phase_of(prompt)](actor_of(prompt), prompt)
        result.setdefault("needed_reformat", False)
        result.setdefault("raw_response", "fake raw output")
        return result

    original = game_module.call_player
    game_module.call_player = fake_call_player
    try:
        g = ChameleonGame(client=None, players=players, category=CATEGORY,
                          candidates=candidates or CANDIDATES, secret_word=secret,
                          dealer_name=dealer_name, discussion=discussion,
                          candidate_mode=candidate_mode, discussion_rounds=discussion_rounds)
        return g.play()
    finally:
        game_module.call_player = original


def benign_hint(actor, prompt):
    return {"hint_word": "sand"}


def benign_statement(actor, prompt):
    return {"statement": "I'm not sure yet, but someone's hint felt vague."}


def vote_for(name):
    def fn(actor, prompt):
        return {"vote": name}
    return fn


def guess_secret(actor, prompt):
    return {"guess": SECRET}


BENIGN = {"hint": benign_hint, "discussion": benign_statement,
          "vote": vote_for("Player2"), "final_guess": guess_secret}


# -- 승리 경로 ---------------------------------------------------------

def test_innocent_accused_chameleon_wins():
    # 전원이 무고한 Player0을 지목 -> 카멜레온 승 (MisidentifiedLoss에 해당)
    result = run_fake_game({**BENIGN, "vote": vote_for("Player0")})
    assert result.winner == "chameleon" and "innocent" in result.reason, result
    assert result.accused == "Player0" and result.identified is False


def test_chameleon_caught_and_guesses_right_chameleon_wins():
    # 잡혔지만 최종 추측 정답 -> 카멜레온 승 (IdentifiedLoss에 해당)
    result = run_fake_game(BENIGN)
    assert result.winner == "chameleon" and "guessed the secret word correctly" in result.reason, result
    assert result.accused == "Player2" and result.identified is True
    fg = next(h for h in result.history if h["phase"] == "final_guess")
    assert fg["correct"] is True and fg["guess"] == SECRET


def test_chameleon_caught_and_guesses_wrong_villagers_win():
    # 잡히고 최종 추측 오답(다른 후보) -> 시민 승 (IdentifiedWin에 해당)
    def wrong_guess(actor, prompt):
        return {"guess": "Hospital"}
    result = run_fake_game({**BENIGN, "final_guess": wrong_guess})
    assert result.winner == "non_chameleon" and "failed to guess" in result.reason, result
    assert result.identified is True


def test_final_guess_cannot_be_skipped():
    # 최종 추측은 스킵 불가 -- 후보에 없는 헛소리만 내면 재시도 후 무효 게임
    calls = {"n": 0}
    def no_guess(actor, prompt):
        calls["n"] += 1
        return {"guess": "I do not know"}
    result = run_fake_game({**BENIGN, "final_guess": no_guess})
    assert result.winner == "invalid" and "Player2" in result.reason, result
    assert calls["n"] == 3  # 최초 1회 + 재시도 2회


def test_korean_final_guess_with_josa():
    # 한국어 조사가 붙어도 부분 문자열 포함으로 정답 처리 (논문 방식 판정)
    kr_candidates = ["은행", "해변", "병원", "도서관"]
    def kr_guess(actor, prompt):
        return {"guess": "정답은 은행입니다"}
    result = run_fake_game({**BENIGN, "final_guess": kr_guess},
                           secret="은행", candidates=kr_candidates)
    assert result.winner == "chameleon" and "correctly" in result.reason, result


def test_guess_matches_casefold():
    assert _guess_matches("the BEACH obviously", "Beach") is True
    assert _guess_matches("harbor", "Beach") is False


def test_hedged_final_guess_is_rejected():
    # 여러 후보를 나열하는 헷징("강아지나 고양이 둘 중 하나")은 부분 문자열 판정상
    # 정답이 뭐든 이기게 되므로 무효 -- 정확히 후보 하나만 포함해야 함.
    # 재시도에서 하나로 좁히면 정상 진행.
    calls = {"n": 0}
    def hedging_then_single(actor, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"guess": "It is either Beach or Hospital"}
        return {"guess": "Hospital"}
    result = run_fake_game({**BENIGN, "final_guess": hedging_then_single})
    assert result.winner == "non_chameleon", result  # 좁힌 추측이 오답 -> 시민 승
    fg = next(h for h in result.history if h["phase"] == "final_guess")
    assert fg["turn_retries"] == 1 and fg["correct"] is False


def test_hedging_all_candidates_never_wins():
    # 후보 전부를 나열하는 극단 헷징은 끝까지 무효 -> 무효 게임 (몰수 승리 불가)
    def list_everything(actor, prompt):
        return {"guess": ", ".join(CANDIDATES)}
    result = run_fake_game({**BENIGN, "final_guess": list_everything})
    assert result.winner == "invalid", result


def test_data_files_have_no_substring_collisions():
    # 같은 분야 안에서 후보끼리 부분 문자열 포함 관계면 최종 추측 판정이 망가짐
    # (충돌 후보가 있으면 그 단어를 영원히 유효하게 추측할 수 없게 됨).
    # 단어를 추가/수정할 때 이 테스트가 게이트 역할을 함.
    import itertools
    import json
    for fname in ("categories_en.json", "categories_kr.json"):
        data = json.loads((Path(__file__).parent / fname).read_text(encoding="utf-8"))
        for cat, words in data["categories"].items():
            assert len(words) == len(set(words)), f"duplicate word in {fname}/{cat}"
            for a, b in itertools.permutations(words, 2):
                assert a.casefold() not in b.casefold(), \
                    f"substring collision in {fname}/{cat}: {a!r} ⊂ {b!r}"


# -- 투표/동률 처리 ----------------------------------------------------

def test_tie_goes_to_dealer():
    # P0,P1 -> Player2 / P2,P3 -> Player0 / P4 -> Player1 => 2:2 동률 -> dealer(P4)가 결정
    dealer_prompts = []
    def split_vote(actor, prompt):
        if actor in ("Player0", "Player1"):
            return {"vote": "Player2"}
        if actor in ("Player2", "Player3"):
            return {"vote": "Player0"}
        return {"vote": "Player1"}
    def dealer_picks_chameleon(actor, prompt):
        dealer_prompts.append((actor, prompt))
        return {"choice": "Player2"}
    result = run_fake_game({**BENIGN, "vote": split_vote, "dealer": dealer_picks_chameleon})
    assert result.winner == "chameleon" and result.identified is True, result  # 잡혔지만 추측 성공
    assert dealer_prompts[0][0] == "Player4"  # dealer는 지정된 참가자
    assert "Player2, Player0" in dealer_prompts[0][1] or "Player0, Player2" in dealer_prompts[0][1]
    dealer_entry = next(h for h in result.history if h["phase"] == "dealer")
    assert dealer_entry["actor"] == "Player4" and dealer_entry["choice"] == "Player2"
    assert sorted(dealer_entry["tied"]) == ["Player0", "Player2"]


def test_dealer_must_choose_among_tied():
    # dealer가 동률자 밖의 이름만 내면 재시도 후 무효 게임 (스킵 불가 행동)
    def split_vote(actor, prompt):
        if actor in ("Player0", "Player1"):
            return {"vote": "Player2"}
        if actor in ("Player2", "Player3"):
            return {"vote": "Player0"}
        return {"vote": "Player1"}
    def bad_dealer(actor, prompt):
        return {"choice": "Charlie"}
    result = run_fake_game({**BENIGN, "vote": split_vote, "dealer": bad_dealer})
    assert result.winner == "invalid" and "Player4" in result.reason, result


def test_ghost_name_vote_becomes_abstain():
    # 프롬프트 예시의 "Charlie"를 전원이 베껴도 크래시 없이 기권 처리
    result = run_fake_game({**BENIGN, "vote": vote_for("Charlie")})
    assert result.winner == "chameleon" and "never accused" in result.reason, result
    assert result.accused is None and result.identified is False
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert all(v["target"] is None and v["invalid_target"] == "Charlie" for v in votes)


def test_self_vote_is_abstain():
    def self_vote(actor, prompt):
        return {"vote": actor}
    result = run_fake_game({**BENIGN, "vote": self_vote})
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert all(v["target"] is None and v["invalid_target"] == v["actor"] for v in votes)


def test_unparseable_vote_becomes_abstain_not_invalid_game():
    # 투표 응답이 아예 파싱 불가여도 게임 중단이 아니라 그 사람만 기권 처리
    def broken_vote(actor, prompt):
        raise InvalidMoveError("no JSON block found")
    result = run_fake_game({**BENIGN, "vote": broken_vote})
    assert result.winner == "chameleon" and "never accused" in result.reason, result
    votes = [h for h in result.history if h["phase"] == "vote"]
    assert all(v["target"] is None and v.get("downgraded_to_abstain") for v in votes)


def test_votes_are_simultaneous():
    # 같은 라운드 안에서는 앞 순서 사람의 표가 뒤 순서 사람 프롬프트에 보이면 안 됨
    seen = []
    def recording_vote(actor, prompt):
        seen.append((actor, prompt))
        return {"vote": "Player0" if actor != "Player0" else "Player1"}
    run_fake_game({**BENIGN, "vote": recording_vote})
    for actor, prompt in seen:
        assert "voted]:" not in prompt, f"{actor}가 같은 라운드의 표를 봤음"


def test_votes_visible_to_dealer_and_final_guess():
    # 개표 후의 프롬프트(dealer/최종 추측)에는 표가 공개되어야 함
    final_prompts = []
    def recording_final(actor, prompt):
        final_prompts.append(prompt)
        return {"guess": SECRET}
    run_fake_game({**BENIGN, "final_guess": recording_final})
    assert "[Player0 voted]: Player2" in final_prompts[0]


# -- 힌트 페이즈 -------------------------------------------------------

def test_multiword_hint_is_retried_then_recovers():
    calls = {"n": 0}
    def flaky_hint(actor, prompt):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"hint_word": "two words"}
        return {"hint_word": "sand"}
    result = run_fake_game({**BENIGN, "hint": flaky_hint})
    assert result.winner == "chameleon", result
    first_hint = next(h for h in result.history if h["phase"] == "hint")
    assert first_hint["turn_retries"] == 2 and first_hint["hint_word"] == "sand"


def test_multiword_hint_always_fails_is_invalid_game():
    calls = {"n": 0}
    def bad_hint(actor, prompt):
        calls["n"] += 1
        return {"hint_word": "always two words"}
    result = run_fake_game({**BENIGN, "hint": bad_hint})
    assert result.winner == "invalid" and result.reason.startswith("invalid game"), result
    assert "Player0" in result.reason
    assert calls["n"] == 3


def test_hint_order_and_previous_hints_visible():
    seen = []
    def recording_hint(actor, prompt):
        seen.append((actor, prompt))
        return {"hint_word": f"word{actor[-1]}"}
    run_fake_game({**BENIGN, "hint": recording_hint})
    assert [a for a, _ in seen] == [f"Player{i}" for i in range(5)]  # 발언 순서 = 플레이어 순서
    assert "gave hint]" not in seen[0][1]                      # 첫 발언자는 아무 힌트도 못 봄
    assert "[Player0 gave hint]: word0" in seen[2][1]          # 뒷사람은 앞사람 힌트를 봄
    assert "[Player3 gave hint]" not in seen[2][1]             # 자기 뒤 사람 것은 아직 안 보임


# -- 토론 페이즈 (우리가 추가한 유일한 단계) ---------------------------

def test_discussion_is_two_rounds_sequential_and_public():
    seen = []
    def recording_statement(actor, prompt):
        seen.append((actor, prompt))
        return {"statement": f"{actor} says #{len(seen)}."}
    result = run_fake_game({**BENIGN, "discussion": recording_statement})
    # 힌트 순서 그대로 두 바퀴 (제기 5 + 반박 5 = 10발언)
    assert [a for a, _ in seen] == [f"Player{i}" for i in range(5)] * 2
    assert "said in discussion]" not in seen[0][1]                    # 1바퀴 첫 발언자는 아무것도 못 봄
    assert "[Player0 said in discussion]: Player0 says #1." in seen[1][1]  # 순차 공개
    # 2바퀴 첫 발언자(6번째 호출)는 1바퀴 발언 5개를 전부 봄 + 반박 라운드 템플릿
    r2_first = seen[5][1]
    assert "second and final discussion round" in r2_first
    for i in range(5):
        assert f"[Player{i} said in discussion]" in r2_first
    # 2바퀴 발언은 히스토리에서 rebuttal로 구분되어 뒷사람에게 보임
    assert "[Player0 said in the rebuttal round]: Player0 says #6." in seen[6][1]
    entries = [h for h in result.history if h["phase"] == "discussion"]
    assert [e["discussion_round"] for e in entries] == [1] * 5 + [2] * 5


def test_single_round_discussion_condition():
    """--discussion-rounds 1: 제기 한 바퀴만, 반박 라운드 없음. 프롬프트 문구도
    '한 바퀴' 규칙과 일치해야 함 (1 vs 2바퀴 비교 조건)."""
    seen = []
    def recording_statement(actor, prompt):
        seen.append((actor, prompt))
        return {"statement": f"{actor} suspects."}
    result = run_fake_game({**BENIGN, "discussion": recording_statement}, discussion_rounds=1)
    assert [a for a, _ in seen] == [f"Player{i}" for i in range(5)]  # 한 바퀴만
    for _, p in seen:
        assert "ONLY discussion statement" in p          # 1바퀴용 문구
        assert "one more round" not in p                 # 2바퀴 문구 없음
        assert "{round_note}" not in p
        assert "each player speaks ONCE" in p            # 규칙 설명도 1바퀴 버전
        assert "TWO rounds" not in p
    entries = [h for h in result.history if h["phase"] == "discussion"]
    assert [e["discussion_round"] for e in entries] == [1] * 5


def test_two_round_prompts_announce_the_second_round():
    seen = []
    def recording_statement(actor, prompt):
        seen.append(prompt)
        return {"statement": "hm."}
    run_fake_game({**BENIGN, "discussion": recording_statement})
    assert "(round 1 of 2)" in seen[0] and "one more round" in seen[0]
    assert "{round_note}" not in seen[0]


def test_leak_warning_only_in_villager_discussion_prompts():
    """--leak-warning: 경고는 **시민의 토론/반박** 템플릿에만 들어감. 카멜레온은
    정답을 모르므로 제외, 힌트/투표 페이즈도 제외. 기본값은 off(무경고)."""
    from prompts import (DISCUSSION_CHAMELEON, DISCUSSION_NON_CHAMELEON, HINT_NON_CHAMELEON,
                         REBUTTAL_CHAMELEON, REBUTTAL_NON_CHAMELEON, VOTE_NON_CHAMELEON)
    g_on = ChameleonGame(client=None, players=make_players(), category=CATEGORY,
                         candidates=CANDIDATES, secret_word=SECRET, dealer_name="Player4",
                         leak_warning=True)
    g_off = ChameleonGame(client=None, players=make_players(), category=CATEGORY,
                          candidates=CANDIDATES, secret_word=SECRET, dealer_name="Player4")
    assert g_off.leak_warning is False  # 기본 무경고
    villager, chameleon = g_on.players[0], g_on.players[2]
    # 경고 on: 시민 토론/반박에만
    for tmpl in (DISCUSSION_NON_CHAMELEON, REBUTTAL_NON_CHAMELEON):
        assert "NEVER say the secret word" in g_on._player_prompt(villager, tmpl)
    for tmpl in (DISCUSSION_CHAMELEON, REBUTTAL_CHAMELEON):
        assert "NEVER say the secret word" not in g_on._player_prompt(chameleon, tmpl)
    for tmpl in (HINT_NON_CHAMELEON, VOTE_NON_CHAMELEON):
        assert "NEVER say the secret word" not in g_on._player_prompt(villager, tmpl)
    # 경고 off: 어디에도 없음 + 플레이스홀더 잔존 금지
    for g in (g_on, g_off):
        for tmpl in (DISCUSSION_NON_CHAMELEON, REBUTTAL_NON_CHAMELEON, DISCUSSION_CHAMELEON):
            assert "{leak_warning}" not in g._player_prompt(g.players[0], tmpl)
    assert "NEVER say the secret word" not in g_off._player_prompt(villager, DISCUSSION_NON_CHAMELEON)


def test_discussion_happens_before_vote_and_visible_in_vote():
    vote_prompts = []
    def recording_vote(actor, prompt):
        vote_prompts.append(prompt)
        return {"vote": "Player0" if actor != "Player0" else "Player1"}
    run_fake_game({**BENIGN, "vote": recording_vote})
    # 제기·반박 두 바퀴 모두 투표 프롬프트에 보여야 함 (토론이 투표에 반영)
    assert all("said in discussion]" in p and "said in the rebuttal round]" in p for p in vote_prompts)


def test_empty_statement_is_retried():
    calls = {"n": 0}
    def flaky_statement(actor, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"statement": "   "}
        return {"statement": "fine."}
    result = run_fake_game({**BENIGN, "discussion": flaky_statement})
    assert result.winner == "chameleon", result
    first = next(h for h in result.history if h["phase"] == "discussion")
    assert first["turn_retries"] == 1


def test_vote_prompt_identical_across_discussion_conditions():
    # 투표/힌트/최종추측 '템플릿'은 토론 유/무와 무관하게 동일해야 함 (변인 통제 --
    # 두 조건의 차이는 base_context의 {discussion_rule}와 토론 페이즈 자체뿐이어야 함).
    # 예전에 VOTE 템플릿이 "(and discussion statement)"를 언급해서 no-discussion
    # 조건에서 존재하지 않는 페이즈를 참조하는 결함이 있었음.
    from prompts import FINAL_GUESS_CHAMELEON, HINT_NON_CHAMELEON, VOTE_CHAMELEON, VOTE_NON_CHAMELEON
    for template in (VOTE_NON_CHAMELEON, VOTE_CHAMELEON, HINT_NON_CHAMELEON, FINAL_GUESS_CHAMELEON):
        assert "discussion" not in template.lower(), "phase templates must not reference the discussion phase"


def test_no_discussion_flag_matches_paper_flow():
    # --no-discussion: 토론 호출도, 규칙 설명 속 토론 문장도 없어야 함 (재현 대조 조건)
    prompts = []
    def recording_vote(actor, prompt):
        prompts.append(prompt)
        return {"vote": "Player2"}
    handlers = {**BENIGN, "vote": recording_vote,
                "discussion": lambda a, p: (_ for _ in ()).throw(AssertionError("discussion must not run"))}
    result = run_fake_game(handlers, discussion=False)
    assert result.winner == "chameleon", result  # 잡혔지만 추측 성공
    assert all("Discussion Phase" not in p for p in prompts)
    assert not [h for h in result.history if h["phase"] == "discussion"]


# -- 프롬프트 조립 -----------------------------------------------------

def test_prompt_assembly_identity_and_secrecy_full_mode():
    g = ChameleonGame(client=None, players=make_players(), category=CATEGORY,
                      candidates=CANDIDATES, secret_word=SECRET, dealer_name="Player4",
                      candidate_mode="full")
    from prompts import HINT_CHAMELEON, HINT_NON_CHAMELEON
    villager_prompt = g._player_prompt(g.players[0], HINT_NON_CHAMELEON)
    chameleon_prompt = g._player_prompt(g.players[2], HINT_CHAMELEON)
    for p in (villager_prompt, chameleon_prompt):
        assert "**Category:** Places" in p                       # 카테고리는 전원 공개
        assert f"Candidate words ({len(CANDIDATES)} total)" in p  # full: 후보 목록도 전원 공개
        assert "- Hospital" in p
        assert "**Players in this game (speaking order):** Player0, Player1, Player2, Player3, Player4" in p
    assert "The secret word is: Beach" in villager_prompt         # 시민에게만 정답 공개
    assert "The secret word is:" not in chameleon_prompt
    assert "You do NOT know the secret word" in chameleon_prompt
    assert "from the category, the candidate list, and the previous hints" in chameleon_prompt


def test_category_mode_hides_list_from_everyone():
    """변형 카멜레온(기본 모드): 목록은 시민에게도 안 보임. 카테고리만 전원 공개,
    정답은 시민에게만. 없는 목록을 참조하는 문구도 남아 있으면 안 됨."""
    g = ChameleonGame(client=None, players=make_players(), category=CATEGORY,
                      candidates=CANDIDATES, secret_word=SECRET, dealer_name="Player4")
    assert g.candidate_mode == "category"  # 엔진 기본값이 변형 카멜레온인지
    from prompts import FINAL_GUESS_CHAMELEON_OPEN, HINT_CHAMELEON, HINT_NON_CHAMELEON, VOTE_NON_CHAMELEON
    villager_prompt = g._player_prompt(g.players[0], HINT_NON_CHAMELEON)
    chameleon_prompt = g._player_prompt(g.players[2], HINT_CHAMELEON)
    final_prompt = g._player_prompt(g.players[2], FINAL_GUESS_CHAMELEON_OPEN)
    for p in (villager_prompt, chameleon_prompt, final_prompt):
        assert "**Category:** Places" in p
        assert "Candidate words" not in p            # 목록 블록 자체가 없음
        assert "- Hospital" not in p                 # 개별 후보도 안 나감
        assert "candidate list" not in p or "NO list" in p or "no candidate list" in p
        for leftover in ("{candidates_rule}", "{and_list}", "{from_list}", "{language}"):
            assert leftover not in p                 # 플레이스홀더 잔존 금지
        assert "There is NO list of candidate words" in p  # 규칙 설명이 모드와 일치
    assert "The secret word is: Beach" in villager_prompt   # 시민은 여전히 정답을 앎
    assert "The secret word is:" not in chameleon_prompt
    assert "from the category and the previous hints" in chameleon_prompt


def test_category_mode_final_guess_open_vocab():
    # 정답 포함이면 정답 (조사/문장 허용)
    def guess_containing(actor, prompt):
        return {"guess": "I think it is the Beach"}
    result = run_fake_game({**BENIGN, "final_guess": guess_containing}, candidate_mode="category")
    assert result.winner == "chameleon" and "correctly" in result.reason, result
    # 유의어/근접 답("seashore")은 오답 처리 (하한값 해석)
    def guess_synonym(actor, prompt):
        return {"guess": "seashore"}
    result = run_fake_game({**BENIGN, "final_guess": guess_synonym}, candidate_mode="category")
    assert result.winner == "non_chameleon" and "failed" in result.reason, result


def test_category_mode_hedging_still_blocked():
    # 목록이 없어도 숨겨진 후보 2개 이상을 나열하면 무효 (헷징 안전장치 유지)
    def hedging(actor, prompt):
        return {"guess": "It is Beach or maybe Hospital"}
    result = run_fake_game({**BENIGN, "final_guess": hedging}, candidate_mode="category")
    assert result.winner == "invalid", result


def test_history_before_format_requirements():
    g = ChameleonGame(client=None, players=make_players(), category=CATEGORY,
                      candidates=CANDIDATES, secret_word=SECRET, dealer_name="Player4")
    g.history = [{"phase": "hint", "actor": "Player0", "hint_word": "sand"}]
    from prompts import HINT_NON_CHAMELEON
    prompt = g._player_prompt(g.players[1], HINT_NON_CHAMELEON)
    assert prompt.index("Game history so far") < prompt.index("Response Format Requirements")


def test_language_replaced_last_and_everywhere():
    g = ChameleonGame(client=None, players=make_players(), category=CATEGORY,
                      candidates=CANDIDATES, secret_word=SECRET, dealer_name="Player4",
                      language="Korean")
    from prompts import VOTE_NON_CHAMELEON
    prompt = g._player_prompt(g.players[0], VOTE_NON_CHAMELEON)
    assert "You must speak in Korean language at any cost!" in prompt
    assert "{language}" not in prompt


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
