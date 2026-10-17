"""
Adversarial Taboo (금기어) 게임 엔진.

SPAG 논문의 Adversarial Taboo 규칙(1대1, 공격자는 목표 단어를 알고 상대가
그 단어를 말하게 유도 / 방어자는 모른 채 추리해서 선언)을 가져와 직접 구현한
것. **논문 코드 재현이 아님** -- SPAG는 self-play RL 학습이 목적이라 평가
프로토콜을 그대로 쓸 수 없어서, 규칙만 따오고 판정 방식은 우리가 정했음
(`설계 결정.md` 참고).

스파이폴 구현(`../multicultural-spyfall-repro/game.py`)과 의도적으로 맞춘 것:
- 응답 규약(|||JSON|||)과 호출 래퍼(../common/llm_client.py) 공유
- **응답 포맷은 승패 조건이 아님**: 필수 행동(발화)은 재시도 → 끝까지 실패하면
  무효 게임(어느 쪽 승리도 아님), 스킵이 합법인 행동(선언)은 무효 출력을
  '선언 안 함'으로 강등
- GameResult 필드 이름을 맞춰서 세 게임 로그를 한 스크립트로 집계 가능하게 함
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from llm_client import InvalidMoveError, call_player
from llm_client import _chat as _judge_chat_raw
from prompts import ATTACKER_UTTERANCE, BASE_CONTEXT, DEFENDER_UTTERANCE


def _judge_chat(client, model: str, prompt: str) -> str:
    """선언 판정 전용 호출: temperature 0 (사실상 결정적), 응답 원문 반환.
    테스트에서 몬키패치하는 지점이기도 함."""
    return _judge_chat_raw(client, model, prompt, temperature=0) or ""


@dataclasses.dataclass
class Player:
    name: str
    model: str
    role: str  # "attacker" or "defender"


@dataclasses.dataclass
class GameResult:
    winner: str  # "attacker" / "defender" / "draw" / "invalid" (무효 게임 -- 집계 제외)
    reason: str
    history: list[dict]
    leaked: bool  # 방어자가 목표 단어를 발화했는지 (= 공격자 유도 성공) -- 스파이폴의 leaked와 같은 자리
    turns_used: int  # 몇 라운드 만에 끝났는지 (draw면 max_rounds)


class GameInvalidError(Exception):
    """재시도까지 전부 실패해서 게임을 더 진행할 수 없을 때. 어느 쪽 승리도
    아닌 '무효 게임'으로 처리됨 -- 응답 포맷은 승패 조건이 아니라는 설계 결정
    (스파이폴 구현과 동일 정책)."""


def mentions_word(text: str, word: str) -> bool:
    """발화에 목표 단어가 나왔는지 판정.

    한국어는 조사가 붙으므로("학교를", "학교에서") 부분 문자열 매칭을 쓰고,
    ASCII는 단어 경계로 매칭해서 "cat"⊂"category" 같은 오탐을 막음
    (스파이폴 `detect_leakage`와 같은 정책).

    **목표 단어는 명사로만 한정한다는 전제 위에서만 맞는 판정임** -- 용언
    ("달리다"→"달렸다")을 넣으면 활용형을 놓치므로 형태소 분석이 필요해짐.
    targets_*.json은 명사만 담고 있고, 새 단어를 추가할 때도 이 제약을 지킬 것.
    """
    if word.isascii():
        return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None
    return word.lower() in text.lower()


# 방어자가 단어를 말했는데 선언 필드가 꺼져 있을 때의 LLM 판정 프롬프트.
# temp 0 + 판정 원문 로그로 사실상 결정적. 확신 추측(①)만 선언으로 인정 --
# 복수 후보 떠보기(②)는 "정답일지 모르는 단어를 입 밖에 낸" 규칙 위반이므로
# 우발 언급(③)과 똑같이 누출 패 (설계 확정).
DECLARE_JUDGE_PROMPT = """You are judging one utterance from a word-guessing game. The secret target word is "{target}". The Defender said:

"{utterance}"

The Defender mentioned the target word. Classify HOW it was mentioned:

1 = CONFIDENT GUESS: the Defender is asserting this single word as their answer (e.g. "It must be X", "I'm sure it's X", "아무래도 X인 것 같은데요")
2 = HEDGED PROBE: the Defender floats the word among multiple candidates or as an uncertain probe (e.g. "X or maybe Y?", "X나 Y 같은 것", "혹시 X나 Y인가요?")
3 = INCIDENTAL MENTION: the word appears naturally in conversation without being offered as an answer (e.g. "I went to the X yesterday")

Respond with only the single digit 1, 2, or 3. No other text."""


# 발화 속 "추측 선언" 판정용 표지 (규칙 기반 폴백: LLM 판정 실패 시와
# --declare-judge rule 조건에서 사용).
# 정답 단어 **바로 뒤/앞 근접 범위**에 있어야 인정 -- 멀리 떨어진 표지는
# 우발 언급("병원에 가는 것도 좋을 것 같아요")까지 잡는 오탐이 되므로.
# 한국어: 단어 뒤에 조사+표지가 붙는 구조 / 영어: 표지가 단어 앞에 오는 구조.
_KR_GUESS_MARKERS = ("인가요", "인가 요", "일까요", "맞죠", "맞지요", "맞나요", "맞습니까",
                     "아닌가요", "아닌가", "인 것 같", "인것 같", "같은데", "이죠", "겠네요", "인 듯")
_EN_GUESS_PRE = ("is it ", "it's ", "it is ", "must be ", "i think it's ", "i think it is ",
                 "i guess it's ", "the answer is ", "the word is ", "sounds like ", "probably ")
_EN_GUESS_POST = (", right", "correct?", "isn't it", "am i right")
_KR_PRE = ("혹시",)


def verbal_declaration(text: str, word: str) -> str | None:
    """발화가 정답 단어에 대한 **추측 선언**("아무래도 학교인 것 같은데요")인지 판정.
    맞으면 걸린 표지 문자열을, 아니면 None을 반환.

    LLM 판정(--declare-judge llm)이 기본이고, 이 함수는 (a) LLM 판정 출력이
    해석 불가능할 때의 폴백, (b) --declare-judge rule 대조 조건에서 쓰임.

    주의: 정답 단어가 발화에 실제로 등장할 때만 호출할 것(mentions_word 선행).
    표지 없이 단어만 나오면 여전히 우발 누출(패배)로 처리됨.
    """
    low = text.lower()
    wlow = word.lower()
    start = 0
    while True:
        i = low.find(wlow, start)
        if i < 0:
            return None
        after = low[i + len(wlow): i + len(wlow) + 14]   # 조사(~2자) + 표지 여유
        before = low[max(0, i - 24): i]
        for m in _KR_GUESS_MARKERS:
            if m in after:
                return m
        for m in _KR_PRE:
            if m in before:
                return m
        for m in _EN_GUESS_PRE:
            if before.endswith(m) or m in before[-(len(m) + 6):]:
                return m.strip()
        for m in _EN_GUESS_POST:
            if m in after:
                return m
        start = i + len(wlow)


def normalize_guess(text: str) -> str:
    """선언 판정용 정규화: 소문자화, 구두점 제거, 공백 정리, 선행 관사 제거.
    한국어는 "대학" vs "대학교" 같은 어형 변형까진 흡수 못 함 -- 결과를 볼 때
    `guess_raw` 필드를 같이 확인할 것 (korean/README.md의 주의사항과 동일)."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("the "):
        text = text[4:]
    return text


def _require_fields(result: dict, spec: dict[str, tuple]) -> None:
    """발화 페이즈 전용 검증(필수 필드 존재/타입). 위반 시 InvalidMoveError ->
    호출부가 재시도하고, 끝까지 실패하면 무효 게임. 선언(declare)에는 안 씀 --
    그쪽은 '선언 안 함'이라는 합법 행동이 있어서 무효 출력을 그걸로 강등함."""
    for field, allowed_types in spec.items():
        if field not in result:
            raise InvalidMoveError(f"missing required field {field!r} in {result!r}")
        if not isinstance(result[field], allowed_types):
            raise InvalidMoveError(
                f"field {field!r} has wrong type {type(result[field]).__name__} in {result!r}")


def _decision_bool(value: object) -> tuple[bool, bool]:
    """(값, 문자열보정여부) 반환. 문자열 "false"가 truthy로 평가돼 '선언함'으로
    오판되는 사고를 막음 (스파이폴에서 실제로 방어했던 경로)."""
    if isinstance(value, bool):
        return value, False
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true", True
    raise ValueError(f"uninterpretable boolean: {value!r}")


class TabooGame:
    def __init__(
        self,
        client: OpenAI,
        attacker: Player,
        defender: Player,
        target_word: str,
        forbidden_words: list[str] | None = None,
        language: str = "English",
        reformatter_model: str | None = None,
        action_retries: int = 2,
        max_rounds: int = 10,
        declare_judge: str = "llm",
        judge_model: str | None = None,
        judge_client: OpenAI | None = None,
    ):
        assert declare_judge in ("off", "rule", "llm")
        assert attacker.role == "attacker" and defender.role == "defender"
        self.client = client
        self.attacker = attacker
        self.defender = defender
        self.target_word = target_word
        # 원작 파티게임의 "금기어 목록"(연관어 사용 금지) 변형. 켜면 공격자가
        # 이 단어들을 못 쓰게 되어 난이도가 올라감. 기본은 꺼짐(SPAG 원안).
        self.forbidden_words = forbidden_words or []
        self.language = language
        self.reformatter_model = reformatter_model
        self.action_retries = action_retries
        self.max_rounds = max_rounds
        # 방어자가 단어를 말했는데 선언 필드가 꺼져 있을 때의 판정 방식:
        #   "llm"  (기본) 판정 LLM이 3분류 -- 확신 추측만 선언으로 인정 (설계 방침)
        #   "rule" 표지 검출 -- LLM 없이 재현 가능한 대조 조건
        #   "off"  기존 판정 -- 단어가 나오면 무조건 누출 패
        self.declare_judge = declare_judge
        # 판정 심판을 플레이어와 분리 -- 모델 크기 사다리에서 작은 모델이 자기
        # 발화를 스스로 판정하면 판정 품질이 모델 크기와 섞이므로, 심판은 큰
        # 모델(35B) 하나로 고정할 수 있게 함. 미지정 시 기존과 동일(방어자 자신).
        self.judge_model = judge_model or defender.model
        self.judge_client = judge_client or client
        self.history: list[dict] = []
        self.leaked = False
        self._current_actor: Player | None = None

    # -- prompt assembly -----------------------------------------------

    def _base_context(self) -> str:
        return BASE_CONTEXT

    def _identity_block(self, player: Player) -> str:
        # 스파이폴과 달리 역할이 공개 정보임(1대1 대결이고 둘 다 자기/상대 역할을
        # 규칙상 알고 있음) -> 이름에 역할을 써도 정보 누출이 아님.
        return (f"**Players in this game:** {self.attacker.name} (Attacker), {self.defender.name} (Defender)\n"
                f"**Your name in this game is:** {player.name}")

    def _secret_block(self, player: Player) -> str:
        if player.role == "attacker":
            lines = [f"**The secret target word is: {self.target_word}**",
                     "(You must make the Defender say it. You must never say it yourself.)"]
            if self.forbidden_words:
                banned = ", ".join(self.forbidden_words)
                lines.append(f"\n**Additionally forbidden words (you may not use these either):** {banned}")
            return "\n".join(lines)
        # 방어자는 열린 어휘로 추론함 -- 후보 목록을 주지 않는 게 타부의 정체성
        # (SPAG 원안). 설계 결정.md #5 참고.
        return "**The target word:** unknown to you. It could be any common noun -- there is no candidate list."

    def _history_block(self) -> str:
        if not self.history:
            return "**Conversation so far:** (none yet -- you are speaking first)"
        lines = [f"[{t['actor']}]: {t['utterance']}" for t in self.history if "utterance" in t]
        return "**Conversation so far:**\n" + "\n".join(lines)

    def _turn_block(self, round_idx: int) -> str:
        return (f"**Turn:** round {round_idx} of {self.max_rounds}. "
                f"If the turn limit is reached with no winner, the game is a draw.")

    def _player_prompt(self, player: Player, template: str, extra: str = "") -> str:
        parts = [self._base_context(), self._identity_block(player), self._secret_block(player), self._history_block()]
        if extra:
            parts.append(extra)
        prompt = template.replace("{base_context}", "\n\n".join(parts))
        # {language}는 BASE_CONTEXT뿐 아니라 페이즈 템플릿 본문에도 들어있으므로
        # 조립이 끝난 뒤 한 번에 치환함 (템플릿에 {language}를 추가했다가 치환을
        # 빠뜨리는 사고를 막기 위해 여기 한 곳에서만 처리).
        return prompt.replace("{language}", self.language)

    def _call(self, player: Player, template: str, extra: str = "") -> dict:
        self._current_actor = player
        prompt = self._player_prompt(player, template, extra)
        return call_player(self.client, player.model, prompt, reformatter_model=self.reformatter_model)

    def _call_required(self, player: Player, template: str, extra: str = "", spec: dict | None = None) -> dict:
        """발화처럼 스킵이 불가능한 필수 행동용. 무효 출력이면 (리포맷터 구제
        포함) 같은 플레이어에게 그 턴을 다시 시키고, 재시도까지 전부 실패하면
        GameInvalidError(무효 게임)."""
        last_err: Exception | None = None
        for attempt in range(self.action_retries + 1):
            try:
                result = self._call(player, template, extra)
                if spec:
                    _require_fields(result, spec)
                if attempt:
                    result["turn_retries"] = attempt
                return result
            except InvalidMoveError as e:
                last_err = e
        raise GameInvalidError(
            f"{player.name} produced no valid response after {self.action_retries + 1} attempts: {last_err}")

    def _forbidden_used(self, text: str) -> list[str]:
        return [w for w in self.forbidden_words if mentions_word(text, w)]

    # -- 페이즈 -----------------------------------------------------------

    def _attacker_turn(self, round_idx: int) -> GameResult | None:
        """공격자 발화. 게임이 끝났으면 GameResult, 아니면 None."""
        base_extra = self._turn_block(round_idx)
        extra = base_extra
        result: dict = {}
        used: list[str] = []
        # 금기어 목록(부차 제약) 위반은 승패로 처리하지 않고 재작성을 시킴.
        # 목표 단어 자체를 말하는 건 게임 핵심 규칙이라 아래에서 즉시 패배로 처리 -- 둘은 다름.
        for _ in range(self.action_retries + 1):
            result = self._call_required(self.attacker, ATTACKER_UTTERANCE, extra, spec={"utterance": (str,)})
            used = self._forbidden_used(result["utterance"])
            if not used:
                break
            extra = (f"{base_extra}\n\n**Rule reminder:** your previous draft used forbidden word(s): "
                     f"{', '.join(used)}. Say the same idea again without using any of them.")

        utterance = result["utterance"]
        entry = {"phase": "attack", "actor": self.attacker.name, "round": round_idx, "utterance": utterance,
                 "needed_reformat": result.get("needed_reformat", False), "raw_response": result.get("raw_response", "")}
        if "turn_retries" in result:
            entry["turn_retries"] = result["turn_retries"]
        if used:
            # 재시도까지 못 고친 경우 -- 발화는 그대로 두고 기록만 남김
            # (부차 제약 위반으로 게임을 끝내지는 않음).
            entry["taboo_violation"] = used
        self.history.append(entry)

        if mentions_word(utterance, self.target_word):
            entry["said_target"] = True
            return GameResult("defender", "attacker said the target word themselves",
                              self.history, self.leaked, round_idx)
        return None

    def _defender_turn(self, round_idx: int) -> GameResult | None:
        """방어자 발화 + 선언 여부. 게임이 끝났으면 GameResult, 아니면 None."""
        result = self._call_required(self.defender, DEFENDER_UTTERANCE, self._turn_block(round_idx),
                                     spec={"utterance": (str,)})
        utterance = result["utterance"]
        entry = {"phase": "defend", "actor": self.defender.name, "round": round_idx, "utterance": utterance,
                 "needed_reformat": result.get("needed_reformat", False), "raw_response": result.get("raw_response", "")}
        if "turn_retries" in result:
            entry["turn_retries"] = result["turn_retries"]

        # 선언은 '안 하는 것'이 합법이므로 무효 출력은 몰수패가 아니라 '선언 안 함'으로 강등.
        try:
            declare, coerced = _decision_bool(result.get("declare"))
            if coerced:
                entry["coerced_bool"] = True
        except ValueError:
            declare = False
            entry["downgraded_to_skip"] = f"invalid declare: {result.get('declare')!r}"
        guess = result.get("guess")
        if declare and not (isinstance(guess, str) and guess.strip()):
            declare = False
            entry["downgraded_to_skip"] = "declare=true but no usable guess"

        entry["declare"] = declare
        entry["guess_raw"] = guess if declare else None
        said_target = mentions_word(utterance, self.target_word)
        if said_target:
            entry["said_target"] = True
        self.history.append(entry)

        if declare:
            # 선언 우선 원칙: 같은 턴에 발화로도 단어가 나왔다면 그것도 기록하되,
            # 승패는 선언으로 판정함. (안 정해두면 "발화에서 흘려놓고 추리였다"고
            # 빠져나갈 수 있어서 우리가 정한 규칙 -- `설계 결정.md`)
            if said_target:
                entry["said_target_while_declaring"] = True
                self.leaked = True
            if normalize_guess(guess) == normalize_guess(self.target_word):
                return GameResult("defender", "defender declared the target word correctly",
                                  self.history, self.leaked, round_idx)
            return GameResult("attacker", f"defender declared the wrong word ({guess!r})",
                              self.history, self.leaked, round_idx)

        if said_target:
            self.leaked = True
            # 단어를 말했더라도 **단독 확신 추측**("아무래도 학교인 것 같은데요")이면
            # 선언 성공으로 판정. 복수 후보 떠보기("강아지나 고양이...")는 정답일지
            # 모르는 단어를 입 밖에 낸 규칙 위반이므로 우발 언급과 같이 누출 패
            # (떠볼 거면 단어를 말하지 않고 신중히 떠봐야 함).
            if self.declare_judge != "off":
                verdict, detail = self._judge_verbal_declaration(utterance)
                entry["declare_judge"] = {"mode": self.declare_judge, "judge_model": self.judge_model,
                                          "verdict": verdict, "detail": detail}
                if verdict == "confident_guess":
                    return GameResult("defender",
                                      f"defender declared the target word verbally ({detail})",
                                      self.history, self.leaked, round_idx)
            return GameResult("attacker", "defender said the target word out loud",
                              self.history, self.leaked, round_idx)
        return None

    def _judge_verbal_declaration(self, utterance: str) -> tuple[str, str]:
        """단어가 든 비선언 발화를 판정. (verdict, 근거) 반환.
        verdict: "confident_guess" / "hedged_probe" / "incidental" / "rule_fallback:..."

        llm 모드: 판정 LLM(temp 0, 원문은 detail로 로그)에게 3분류를 시킴.
        판정 출력이 해석 불가능하면 규칙(표지 검출)로 폴백 -- 게임이 판정 실패로
        죽지 않게. rule 모드: 표지 검출만."""
        if self.declare_judge == "llm":
            prompt = DECLARE_JUDGE_PROMPT.replace("{target}", self.target_word).replace("{utterance}", utterance)
            try:
                raw = _judge_chat(self.judge_client, self.judge_model, prompt)
                digit = next((ch for ch in raw if ch in "123"), None)
                if digit == "1":
                    return "confident_guess", f"llm_judge=1: {raw[:80]!r}"
                if digit == "2":
                    return "hedged_probe", f"llm_judge=2: {raw[:80]!r}"
                if digit == "3":
                    return "incidental", f"llm_judge=3: {raw[:80]!r}"
            except Exception as e:  # noqa: BLE001 -- 판정 실패가 게임을 죽이면 안 됨
                raw = f"judge error: {e}"
            # 해석 불가/에러 -> 규칙 폴백
            marker = verbal_declaration(utterance, self.target_word)
            if marker:
                return "confident_guess", f"rule_fallback (marker {marker!r}) after {raw[:60]!r}"
            return "incidental", f"rule_fallback (no marker) after {raw[:60]!r}"
        marker = verbal_declaration(utterance, self.target_word)
        if marker:
            return "confident_guess", f"rule marker {marker!r}"
        return "incidental", "rule: no marker"

    # -- 최상위 실행 함수 ---------------------------------------------

    def play(self) -> GameResult:
        """게임 한 판. 공격자가 먼저 말하고 한 라운드에 양쪽 1발화씩."""
        try:
            for round_idx in range(1, self.max_rounds + 1):
                result = self._attacker_turn(round_idx)
                if result:
                    return result
                result = self._defender_turn(round_idx)
                if result:
                    return result
            return GameResult("draw", "turn limit reached with no declaration and no leak",
                              self.history, self.leaked, self.max_rounds)
        except (GameInvalidError, InvalidMoveError) as e:
            actor = self._current_actor
            return GameResult("invalid",
                              f"invalid game: no valid response from {actor.name if actor else 'unknown player'} ({e})",
                              self.history, self.leaked, len(self.history) // 2)
