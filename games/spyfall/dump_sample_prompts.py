"""디버그용 1회성 스크립트: API 호출 없이, 각 페이즈에서 실제로 모델에게
보내지는 프롬프트 문자열 그대로를 뽑아서 파일로 저장.

game.py의 _player_prompt를 그대로 재사용하므로, 여기서 나오는 텍스트는
call_player에 실제로 넘어가는 것과 100% 동일함 (API 응답만 흉내낸 가짜
history를 미리 채워넣고 그 상태에서 프롬프트를 조립해봄).
"""

from game import Player, SpyfallGame
from prompts import (
    ANSWER_GENERATION_NON_SPY,
    ANSWER_GENERATION_SPY,
    ENTITY_GUESS_SPY,
    QUESTION_GENERATION_NON_SPY,
    VOTE_INITIATION_NON_SPY,
)

players = [
    Player(name="Player0", model="dummy", role="non_spy"),
    Player(name="Player1", model="dummy", role="non_spy"),
    Player(name="Player2", model="dummy", role="spy"),
    Player(name="Player3", model="dummy", role="non_spy"),
    Player(name="Player4", model="dummy", role="non_spy"),
]

game = SpyfallGame(
    client=None,  # 실제 API 호출 안 하므로 필요 없음
    players=players,
    entities=["Hospital", "Beach", "Bank", "Zoo", "Airplane"],  # 실제는 30개, 지면상 5개만
    target_entity="Hospital",
    language="English",
)

# 라운드로빈이 한 바퀴 돈 뒤 자유질문 1회+투표 1회가 이미 있었다고 가정 -- history_block이 빈 게 아니라 실제로 뭐가 쌓인 모습을 보여주려는 것
game.history = [
    {"phase": "question", "actor": "Player0", "target": "Player1", "question": "Have you ever been somewhere that requires a lot of paperwork?", "needed_reformat": False},
    {"phase": "answer", "actor": "Player1", "answer": "Yes, quite often actually.", "needed_reformat": False},
    {"phase": "question", "actor": "Player1", "target": "Player2", "question": "Do people often wear uniforms where you are?", "needed_reformat": False},
    {"phase": "answer", "actor": "Player2", "answer": "Sometimes, depending on the role.", "needed_reformat": False},
    {"phase": "vote", "actor": "Player0", "target": None, "needed_reformat": False},
]

sections = []


def capture(title: str, player: Player, template: str, extra: str = "") -> None:
    prompt = game._player_prompt(player, template, extra)
    sections.append(f"{'=' * 80}\n[{title}]\n{'=' * 80}\n{prompt}\n")


# 1) 질문 페이즈 -- 비스파이(Player0)가 라운드로빈 규칙대로 Player1에게 질문
capture(
    "PHASE: 질문 생성 (비스파이, 라운드로빈)",
    players[0],
    QUESTION_GENERATION_NON_SPY,
    extra="**Round Robin Phase:** You must ask your question to Player1 (fixed turn order).",
)

# 2) 답변 페이즈 -- 스파이(Player2)가 질문에 답변
capture(
    "PHASE: 답변 생성 (스파이)",
    players[2],
    ANSWER_GENERATION_SPY,
    extra="**Question asked to you:** Do people often wear uniforms where you are?",
)

# 3) 스파이 추측 페이즈 -- 스파이(Player2) 전용, 역할 분기 없음
capture(
    "PHASE: 스파이 추측 결정",
    players[2],
    ENTITY_GUESS_SPY,
)

# 4) 투표 페이즈 -- 비스파이(Player3)
capture(
    "PHASE: 투표 (비스파이)",
    players[3],
    VOTE_INITIATION_NON_SPY,
)

output = "\n".join(sections)
with open("sample_prompts.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(f"총 {len(sections)}개 프롬프트, {len(output)}자 -- sample_prompts.txt에 저장함")
