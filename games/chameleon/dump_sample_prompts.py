"""각 페이즈에서 모델이 실제로 받는 프롬프트를 API 호출 없이 조립해서 출력.

프롬프트 정합성 검토용 (multicultural-spyfall-repro/dump_sample_prompts.py와
같은 컨벤션). `python dump_sample_prompts.py > sample_prompts.txt`
"""

from __future__ import annotations

import sys

from game import ChameleonGame, Player
from prompts import (
    DEALER_TIEBREAK,
    DISCUSSION_CHAMELEON,
    DISCUSSION_NON_CHAMELEON,
    FINAL_GUESS_CHAMELEON,
    HINT_CHAMELEON,
    HINT_NON_CHAMELEON,
    VOTE_CHAMELEON,
    VOTE_NON_CHAMELEON,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CATEGORY = "Appliances"
CANDIDATES = ["Refrigerator", "Microwave", "Toaster", "Washing Machine", "Dryer",
              "Dishwasher", "Vacuum Cleaner", "Air Conditioner", "Fan", "Heater",
              "Blender", "Kettle", "Iron", "Hair Dryer", "Television", "Rice Cooker"]
SECRET = "Microwave"


def make_game(discussion: bool = True) -> ChameleonGame:
    roles = ["non_chameleon", "non_chameleon", "chameleon", "non_chameleon", "non_chameleon"]
    players = [Player(name=f"Player{i}", model="dump", role=r) for i, r in enumerate(roles)]
    return ChameleonGame(client=None, players=players, category=CATEGORY,
                         candidates=CANDIDATES, secret_word=SECRET,
                         dealer_name="Player4", language="Korean", discussion=discussion)


def show(title: str, prompt: str) -> None:
    print("=" * 78)
    print(f"== {title}")
    print("=" * 78)
    print(prompt)
    print


def main() -> None:
    g = make_game(discussion=True)

    # ③ 힌트: 첫 발언자(시민) / 세 번째 발언자(카멜레온, 앞 힌트 2개 보임)
    show("HINT -- first speaker, villager (Player0)", g._player_prompt(g.players[0], HINT_NON_CHAMELEON))
    g.history = [
        {"phase": "hint", "actor": "Player0", "hint_word": "데우기"},
        {"phase": "hint", "actor": "Player1", "hint_word": "주방"},
    ]
    show("HINT -- third speaker, CHAMELEON (Player2), sees 2 previous hints",
         g._player_prompt(g.players[2], HINT_CHAMELEON))

    # ⑤ 토론: 힌트 5개 완료 상태에서 두 번째 토론자(앞 발언 1개 보임)
    g.history = [
        {"phase": "hint", "actor": "Player0", "hint_word": "데우기"},
        {"phase": "hint", "actor": "Player1", "hint_word": "주방"},
        {"phase": "hint", "actor": "Player2", "hint_word": "전기"},
        {"phase": "hint", "actor": "Player3", "hint_word": "간편식"},
        {"phase": "hint", "actor": "Player4", "hint_word": "회전"},
        {"phase": "discussion", "actor": "Player0", "statement": "Player2의 '전기'는 어느 가전에나 통하는 말이라 조금 애매하게 느껴졌어요."},
    ]
    show("DISCUSSION -- second discussant, villager (Player1), sees 1 previous statement",
         g._player_prompt(g.players[1], DISCUSSION_NON_CHAMELEON))
    show("DISCUSSION -- CHAMELEON (Player2)", g._player_prompt(g.players[2], DISCUSSION_CHAMELEON))

    # ⑥ 투표: 토론까지 끝난 상태 (표는 아직 아무것도 안 보여야 함)
    g.history = g.history + [
        {"phase": "discussion", "actor": "Player1", "statement": "저도 '전기'가 제일 걸리네요."},
        {"phase": "discussion", "actor": "Player2", "statement": "저는 '회전'이 뜬금없다고 봤습니다."},
        {"phase": "discussion", "actor": "Player3", "statement": "'전기' 쪽이 의심스럽긴 해요."},
        {"phase": "discussion", "actor": "Player4", "statement": "Player2 아니면 Player0 같아요."},
    ]
    show("VOTE -- villager (Player3)", g._player_prompt(g.players[3], VOTE_NON_CHAMELEON))
    show("VOTE -- CHAMELEON (Player2)", g._player_prompt(g.players[2], VOTE_CHAMELEON))

    # ⑦ 동률 -> dealer / ⑧ 최종 추측: 개표 후 (표가 공개된 상태)
    g.history = g.history + [
        {"phase": "vote", "actor": "Player0", "target": "Player2"},
        {"phase": "vote", "actor": "Player1", "target": "Player2"},
        {"phase": "vote", "actor": "Player2", "target": "Player0"},
        {"phase": "vote", "actor": "Player3", "target": "Player0"},
        {"phase": "vote", "actor": "Player4", "target": None},
    ]
    dealer_template = DEALER_TIEBREAK.replace("{tied_players}", "Player0, Player2")
    show("DEALER TIEBREAK -- dealer (Player4), votes now public",
         g._player_prompt(g.players[4], dealer_template))
    g.history = g.history + [
        {"phase": "dealer", "actor": "Player4", "tied": ["Player0", "Player2"], "choice": "Player2"},
    ]
    show("FINAL GUESS -- caught CHAMELEON (Player2)", g._player_prompt(g.players[2], FINAL_GUESS_CHAMELEON))

    # --no-discussion 대조 조건: 규칙 설명에서 토론이 빠졌는지
    g2 = make_game(discussion=False)
    show("NO-DISCUSSION control -- HINT, villager (Player0)", g2._player_prompt(g2.players[0], HINT_NON_CHAMELEON))
    show("NO-DISCUSSION control -- VOTE, villager (Player0)", g2._player_prompt(g2.players[0], VOTE_NON_CHAMELEON))


if __name__ == "__main__":
    main()
