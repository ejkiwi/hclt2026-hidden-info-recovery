"""
Turn-based Spyfall ("Cultured Turn-Based Spyfall") game engine,
reimplementing the mechanics described in Multicultural Spyfall
Section 2 and Appendix A/C.

This is NOT a copy of the paper's own (unreleased) code -- it is a
from-scratch reimplementation of the documented rules and prompts, so
double-check phase ordering / win conditions against the paper if you
change anything here.
"""

from __future__ import annotations

import dataclasses
import random
import re
import sys
from pathlib import Path

from openai import OpenAI

# llm_client는 세 게임(스파이폴/카멜레온/금기어)이 같이 쓰는 호출 래퍼라
# ../common/ 으로 빼놨음 (게임 규칙은 안 들어있음).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from llm_client import InvalidMoveError, call_player
from prompts import (
    ANSWER_GENERATION_NON_SPY,
    ANSWER_GENERATION_SPY,
    BASE_CONTEXT,
    ENTITY_GUESS_SPY,
    QUESTION_GENERATION_NON_SPY,
    QUESTION_GENERATION_SPY,
    VOTE_INITIATION_NON_SPY,
    VOTE_INITIATION_SPY,
)


@dataclasses.dataclass
class Player:  #플레이어 객체 하나.
    name: str
    model: str
    role: str  # "spy" or "non_spy"


@dataclasses.dataclass
class GameResult:  # 게임 한 판 결과 요약 객체
    winner: str  # "spy" / "non_spy" / "invalid" (무효 게임 -- 승패 집계에서 제외)
    reason: str
    history: list[dict]
    leaked: bool  # heuristic non-spy leakage flag, see detect_leakage()


class GameInvalidError(Exception):
    """재시도까지 전부 실패해서 게임을 더 진행할 수 없을 때. 어느 팀의
    승리도 아닌 '무효 게임'으로 처리됨 -- 응답 포맷 문제는 승패 조건이
    아니라는 설계 결정 (논문 원안의 몰수패를 대체, `논문과_다르게_간_설계.md`)."""


def detect_leakage(utterance_text: str, target_entity: str) -> bool:
    """논문에 적혀있는 기준의 근사치 정도로 구현. 카드명을 그대로 언급한것만 잡고있음.
    암묵적 누출은 못 잡음 (하한값). ASCII 엔티티는 단어 경계로 매칭해서
    "Bank"⊂"riverbank" 같은 오탐을 막고, 비ASCII(한국어 등)는 조사가 붙는
    경우("해적선을")를 잡아야 하므로 부분 문자열 매칭을 유지."""
    if target_entity.isascii():
        return re.search(rf"\b{re.escape(target_entity)}\b", utterance_text, re.IGNORECASE) is not None
    return target_entity.lower() in utterance_text.lower()


def normalize_entity(text: str) -> str:
    """스파이 추측 판정용 정규화: 소문자화, 구두점 제거, 공백 정리, 선행 관사 제거.
    완전 문자열 일치로만 판정하면 "The Beach." 같은 사실상 정답이 오답 처리되므로
    (논문에 판정 방법 명시 없음 -- 저희가 정한 규칙임)."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("the "):
        text = text[4:]
    return text


def _require_fields(result: dict, spec: dict[str, tuple]) -> None:
    """질문/답변 페이즈 전용 검증: 필수 필드의 존재와 타입 위반 시
    InvalidMoveError -- 호출부(_call_required)가 재시도하고, 끝까지 실패하면
    몰수패가 아니라 무효 게임 처리. 추측/투표 페이즈에는 쓰지 않음 -- 그쪽은
    규칙상 '스킵'이라는 합법 행동이 있어서 무효 출력을 스킵으로 강등함
    (연구 목적이 포맷 준수가 아니라 행동 관찰이라서 -- 자세한 근거는
    `논문과_다르게_간_설계.md` 참고)."""
    for field, allowed_types in spec.items():
        if field not in result:
            raise InvalidMoveError(f"missing required field {field!r} in {result!r}")
        if not isinstance(result[field], allowed_types):
            raise InvalidMoveError(
                f"field {field!r} has wrong type {type(result[field]).__name__} in {result!r}")


def _decision_bool(value: object) -> tuple[bool, bool]:
    """추측/투표의 결정 불리언 해석. (값, 문자열보정여부)를 반환하고,
    해석 불가능한 값이면 ValueError. 문자열 "false"가 truthy로 평가돼
    "추측함"으로 오판되는 사고를 막으면서, 몰수패 대신 보정/스킵으로 살림."""
    if isinstance(value, bool):
        return value, False
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true", True
    raise ValueError(f"uninterpretable boolean: {value!r}")


class SpyfallGame:
    def __init__(  #초기화. 게임 생성 시 넣어준 값으로 츠츠
        self,
        client: OpenAI,
        players: list[Player],
        entities: list[str],
        target_entity: str,
        language: str = "English",
        reformatter_model: str | None = None,
        action_retries: int = 2,
    ):
        assert target_entity in entities, "target_entity must be a member of entities"
        spies = [p for p in players if p.role == "spy"]
        assert len(spies) == 1, "exactly one spy is required"
        self.client = client
        self.players = players
        self.entities = entities
        self.target_entity = target_entity
        self.language = language
        self.reformatter_model = reformatter_model
        self.action_retries = action_retries  # 질문/답변이 무효 출력일 때 그 턴을 다시 시키는 횟수
        self.spy = spies[0]
        self.history: list[dict] = []
        self.leaked = False
        self._current_actor: Player | None = None
        self._vote_round = 0  # 몇 번째 투표 라운드인지 -- history 표기용

    # -- prompt assembly -----------------------------------------------
    # 프롬프트 조립하는 부분

    def _base_context(self) -> str:
        return BASE_CONTEXT.replace("{language}", self.language)

    def _identity_block(self, player: Player) -> str:
        # 플레이어가 history에서 "이게 내 발언이었지"를 알아볼 수 있게 자기 이름은
        # 알려줌 -- 역할(스파이인지)은 여기서 안 드러남, 그건 템플릿 선택으로 별도 전달됨.
        # 전체 명단도 같이 줌: 명단이 없으면 다른 플레이어 이름을 history에서 추론해야
        # 해서, 약한 모델이 프롬프트 예시의 "Charlie" 같은 유령 이름을 지목하기 쉬움.
        roster = ", ".join(p.name for p in self.players)
        return (f"**Players in this game:** {roster}\n"
                f"**Your name in this game is:** {player.name}")

    def _entity_block(self, player: Player) -> str:
        lines = "\n".join(f"- {e}" for e in self.entities)
        n = len(self.entities)  # 번역판 등에서 30개가 아닐 수도 있으니 하드코딩 안 함
        if player.role == "spy":
            return f"**Entities list ({n} total; you do NOT know which one is the secret entity):**\n{lines}"
        return f"**Entities list ({n} total):**\n{lines}\n\n**The secret entity is: {self.target_entity}**"

    def _history_block(self) -> str:
        if not self.history:
            return "**Game history so far:** (none yet)"
        lines = []
        for t in self.history:
            if t["phase"] == "question":
                lines.append(f"[{t['actor']} asked {t['target']}]: {t['question']}")
            elif t["phase"] == "answer":
                lines.append(f"[{t['actor']} answered]: {t['answer']}")
            elif t["phase"] == "vote":
                # 라운드 번호를 붙여서 "이번 라운드에 이미 몇 표 몰렸는지"를 모델이
                # 과거 라운드 표와 헷갈리지 않게 함.
                lines.append(f"[{t['actor']} voted (voting round {t.get('vote_round', '?')})]: {t['target'] or 'SKIP'}")
        return "**Game history so far:**\n" + "\n".join(lines)

    def _player_prompt(self, player: Player, template: str, extra: str = "") -> str:
        # 논문 Section 2.5의 순서("rules, game history, a phase description, and
        # response instructions")대로: 게임 상황(명단/엔티티/history/추가 정보)을
        # base_context 바로 뒤에 끼워넣고, 페이즈 과제와 포맷 지시가 맨 뒤에 오게 함.
        # (이전엔 포맷 지시 '뒤에' history가 붙어서, 모델이 마지막으로 읽는 게
        # 포맷 요구사항이 아니게 되는 배치였음.)
        parts = [self._base_context(), self._identity_block(player), self._entity_block(player), self._history_block()]
        if extra:
            parts.append(extra)
        return template.replace("{base_context}", "\n\n".join(parts))

    def _call(self, player: Player, template: str, extra: str = "") -> dict:
        self._current_actor = player
        prompt = self._player_prompt(player, template, extra)
        return call_player(self.client, player.model, prompt, reformatter_model=self.reformatter_model)

    def _call_required(self, player: Player, template: str, extra: str = "", spec: dict | None = None) -> dict:
        """질문/답변처럼 스킵이 불가능한 필수 행동용. 무효 출력이면 (리포맷터
        구제 포함) 같은 플레이어에게 그 턴을 다시 시키고, 재시도까지 전부
        실패하면 몰수패가 아니라 GameInvalidError(무효 게임)를 던짐 --
        응답 포맷은 승패 조건이 아니라는 설계 결정."""
        last_err: Exception | None = None
        for attempt in range(self.action_retries + 1):
            try:
                result = self._call(player, template, extra)
                if spec:
                    _require_fields(result, spec)
                if attempt:
                    result["turn_retries"] = attempt  # 논문 규칙이었다면 몰수패였을 상황 -- 복원용 기록
                return result
            except InvalidMoveError as e:
                last_err = e
        raise GameInvalidError(
            f"{player.name} produced no valid response after {self.action_retries + 1} attempts: {last_err}")

    # -- 페이즈 -----------------------------------------------------------
    # 실제 게임 규칙

    def _round_robin_cycle(self) -> Player:
        """페이즈 1A. 라운드로빈이 끝난 뒤 첫 자유 질문을 할 사람
        (라운드로빈에서 "마지막으로 답변한 사람")을 반환."""
        order = self.players
        n = len(order)
        for i in range(n):
            asker, target = order[i], order[(i + 1) % n]
            template = QUESTION_GENERATION_SPY if asker.role == "spy" else QUESTION_GENERATION_NON_SPY
            # 질문은 스킵 개념이 없는 필수 행동이라 무효 출력이면 재시도, 끝까지 실패하면 무효 게임.
            # targeted_player는 라운드로빈에선 어차피 무시되므로 검사 안 함.
            q = self._call_required(asker, template, extra=f"**Round Robin Phase:** You must ask your question to {target.name} (fixed turn order).", spec={"question": (str,)})
            question_text = q["question"]
            entry = {"phase": "question", "actor": asker.name, "target": target.name, "question": question_text, "needed_reformat": q.get("needed_reformat", False), "raw_response": q.get("raw_response", "")}
            if "turn_retries" in q:
                entry["turn_retries"] = q["turn_retries"]
            self.history.append(entry)
            if asker.role == "non_spy" and detect_leakage(question_text, self.target_entity):
                # 누출 판정은 답변뿐 아니라 질문 문구 자체에서도 일어날 수 있음
                # (논문의 PQA 프로세스엔 질문도 포함됨).
                self.leaked = True
            self._answer_question(target, question_text)
        return order[0]  # order[n-1]이 order[0]에게 마지막으로 질문했으니, order[0]이 "마지막 답변자"가 됨

    def _ask_free_question(self, asker: Player) -> tuple[Player, str]:
        """페이즈 1B. 질문자(직전 사이클의 마지막 답변자)가 아무나 골라서 질문함."""
        template = QUESTION_GENERATION_SPY if asker.role == "spy" else QUESTION_GENERATION_NON_SPY
        q = self._call_required(asker, template, spec={"question": (str,)})
        question_text = q["question"]
        target_name = q.get("targeted_player")  # 없거나 무효면 아래에서 무작위 대체
        target = next((p for p in self.players if p.name == target_name and p is not asker), None)
        target_was_replaced = target is None
        if target is None:
            # 모델이 유효하지 않은 대상(또는 자기 자신)을 지목한 경우 --
            # 게임이 죽지 않고 계속 진행되도록 무작위로 다른 플레이어를 대신 지목.
            # 논문에 명시 안 된 부분이며, 잘못된 출력으로 실행이 죽는 걸 막기 위한 임의 구현.
            target = random.choice([p for p in self.players if p is not asker])
        entry = {"phase": "question", "actor": asker.name, "target": target.name, "question": question_text, "needed_reformat": q.get("needed_reformat", False), "raw_response": q.get("raw_response", "")}
        if "turn_retries" in q:
            entry["turn_retries"] = q["turn_retries"]
        if target_was_replaced:
            # 대체 사실을 기록으로 남김 -- 안 남기면 트랜스크립트에서 모델이 직접
            # 고른 대상처럼 보여서 정성 분석 데이터가 오염됨.
            entry["target_was_replaced"] = True
            entry["original_target"] = target_name
        self.history.append(entry)
        if asker.role == "non_spy" and detect_leakage(question_text, self.target_entity):
            self.leaked = True
        self._answer_question(target, question_text)
        return target, question_text

    def _answer_question(self, answerer: Player, question: str) -> None:
        template = ANSWER_GENERATION_SPY if answerer.role == "spy" else ANSWER_GENERATION_NON_SPY
        result = self._call_required(answerer, template, extra=f"**Question asked to you:** {question}", spec={"answer": (str,)})
        answer = result["answer"]
        entry = {"phase": "answer", "actor": answerer.name, "answer": answer, "needed_reformat": result.get("needed_reformat", False), "raw_response": result.get("raw_response", "")}
        if "turn_retries" in result:
            entry["turn_retries"] = result["turn_retries"]
        self.history.append(entry)
        if answerer.role == "non_spy" and detect_leakage(answer, self.target_entity):
            self.leaked = True

    def _spy_guess_phase(self) -> GameResult | None:
        """페이즈 2. 스파이가 추측했으면(게임 종료) GameResult를 반환하고,
        스킵했으면 None을 반환."""
        # 추측/투표는 규칙상 '스킵'이 합법 행동이라, 무효 출력(파싱 실패,
        # 필드 누락, 타입 오류)을 몰수패 대신 스킵으로 강등하고 플래그만 남김.
        # 몰수패는 질문/답변(스킵 개념이 없는 필수 행동)에만 유지.
        # 근거: 연구 목적이 포맷 준수가 아니라 행동 관찰 -- `논문과_다르게_간_설계.md`.
        try:
            result = self._call(self.spy, ENTITY_GUESS_SPY)
        except InvalidMoveError as e:
            self.history.append({"phase": "guess", "actor": self.spy.name, "should_guess": False,
                                 "downgraded_to_skip": "unparseable output", "invalid_output": str(e)})
            return None
        entry = {"phase": "guess", "actor": self.spy.name, **result}
        try:
            should_guess, coerced = _decision_bool(result.get("should_guess"))
            if coerced:
                entry["coerced_bool"] = True
        except ValueError:
            should_guess = False
            entry["downgraded_to_skip"] = f"invalid should_guess: {result.get('should_guess')!r}"
        best_guess = result.get("best_guess")
        if should_guess and not (isinstance(best_guess, str) and best_guess.strip()):
            should_guess = False
            entry["downgraded_to_skip"] = "should_guess=true but no usable best_guess"
        if isinstance(best_guess, str):
            # 추측이 엔티티 리스트에 아예 없는 경우를 사후 분석할 수 있게 플래그로 남김.
            entry["guess_in_list"] = normalize_entity(best_guess) in {normalize_entity(e) for e in self.entities}
        self.history.append(entry)
        if not should_guess:
            return None
        # 완전 문자열 일치 대신 정규화 후 비교 -- "The Beach." 같은 표기 변형이
        # 오답 처리되는 걸 막음 (한국어 실험에선 조사/띄어쓰기 변형까지 고려해야 함).
        if normalize_entity(best_guess) == normalize_entity(self.target_entity):
            return GameResult("spy", "spy guessed the entity correctly", self.history, self.leaked)
        return GameResult("non_spy", "spy guessed the entity incorrectly", self.history, self.leaked)

    def _vote_phase(self) -> GameResult | None:
        """페이즈 3. 과반을 받은 사람이 있으면 GameResult를 반환하고, 없으면 None을 반환."""
        self._vote_round += 1
        tally: dict[str, int] = {}
        valid_names = {p.name for p in self.players}
        # **동시 투표**: 이번 라운드의 표는 라운드가 끝난 뒤에 한꺼번에 history에
        # 넣는다. 루프 도중에 넣으면 뒷순서 플레이어의 프롬프트에 앞사람 표가
        # 보여서 같은 라운드 안에서 따라 투표가 가능해짐 -- 실제 보드게임은
        # "하나 둘 셋" 하고 동시에 지목하므로 그 방식에 맞춤.
        # 지난 라운드의 표는 그대로 보인다(실제 게임도 지목이 끝나면 공개됨).
        round_entries: list[dict] = []
        for player in self.players:
            template = VOTE_INITIATION_SPY if player.role == "spy" else VOTE_INITIATION_NON_SPY
            # 무효 출력은 스킵으로 강등 (몰수패 아님) -- _spy_guess_phase와 동일 정책.
            try:
                result = self._call(player, template)
            except InvalidMoveError as e:
                round_entries.append({"phase": "vote", "actor": player.name, "target": None,
                                      "vote_round": self._vote_round,
                                      "downgraded_to_skip": "unparseable output", "invalid_output": str(e)})
                continue
            entry = {"phase": "vote", "actor": player.name, "vote_round": self._vote_round,
                     "confidence": result.get("confidence"),
                     "needed_reformat": result.get("needed_reformat", False),
                     "raw_response": result.get("raw_response", "")}
            try:
                should_vote, coerced = _decision_bool(result.get("should_vote"))
                if coerced:
                    entry["coerced_bool"] = True
            except ValueError:
                should_vote = False
                entry["downgraded_to_skip"] = f"invalid should_vote: {result.get('should_vote')!r}"
            target = result.get("target_player_name") if should_vote else None
            if should_vote and not isinstance(target, str):
                target = None
                entry["downgraded_to_skip"] = "should_vote=true but no usable target_player_name"
            if target is not None and (target == player.name or target not in valid_names):
                # 자기 자신 투표, 또는 존재하지 않는 이름(예: VOTE_INITIATION 프롬프트
                # 예시의 "Charlie"를 그대로 베낀 경우)은 스킵과 동일하게 무효 처리.
                # 유령 이름에 표가 몰리면 검증 없이는 아래 next()가 StopIteration으로
                # 런 전체를 죽여버림. 원래 지목값은 분석용으로 남겨둠.
                entry["invalid_target"] = target
                target = None
            entry["target"] = target
            round_entries.append(entry)
            if target:
                tally[target] = tally.get(target, 0) + 1

        # 전원이 지목을 마친 뒤 한꺼번에 공개.
        self.history.extend(round_entries)

        if not tally:
            return None
        top_name, top_count = max(tally.items(), key=lambda kv: kv[1])
        majority_threshold = len(self.players) // 2 + 1
        if top_count < majority_threshold:
            return None

        accused = next(p for p in self.players if p.name == top_name)
        if accused.role == "spy":
            return GameResult("non_spy", "spy was voted out by majority", self.history, self.leaked)
        return GameResult("spy", "an innocent villager was voted out by majority", self.history, self.leaked)

    # -- 최상위 실행 함수 ---------------------------------------------
    # 게임 진행
    def play(self, num_free_cycles: int | None = None) -> GameResult:
        """게임 한 판을 진행함. `num_free_cycles`를 안 주면 플레이어 수로
        기본 설정됨 -- 논문의 5인 / 자유사이클 5회 세팅과 일치
        (Section 3: "턴 제한 ... 플레이어 수의 2배")."""
        num_free_cycles = num_free_cycles or len(self.players)

        try:
            # 일단 차례대로 다음 사람에게 질문하는 라운드.
            last_answerer = self._round_robin_cycle()

            # 자유 대화 라운드 
            for _ in range(num_free_cycles): 
                last_answerer, _ = self._ask_free_question(last_answerer)  # 여기서 질문 텍스트 안 쓰는 이유는 어차피 history에 다 기록되어있기에
                # 이전에 받아둔 last_answerer가 다음 질문자임. (즉, 마지막으로 답한 사람이 다음 질문을 하는 구조)

                # result에는 게임이 끝났는지 아닌지를 받음.
                # 스파이가 정답을 맞추거나 스파이가 잡히거나 무고한 사람이 잡히면 끝. (최대 턴 수에 도달하면 스파이 승리)
                result = self._spy_guess_phase()  # 스파이가 정답 맞췄는지 확인
                if result:
                    return result  # 반복문 종료
                result = self._vote_phase()  # 투표로 스파이 잡았는지 확인
                if result:  
                    return result  # 반복문 종료

            # 최대 턴 넘었는데도 안 끝났으면 질문 과정 빼고 추측과 투표만 따로 마지막 한 번 더 시도
            result = self._spy_guess_phase()
            if result:
                return result
            result = self._vote_phase()
            if result:
                return result

            # 스파이가 이긴 경우 (최대 턴 수에 도달했는데도 스파이가 잡히지 않은 경우)
            return GameResult("spy", "spy survived to the turn limit without being voted out", self.history, self.leaked)

        except (GameInvalidError, InvalidMoveError) as e:
            # 논문 원안(Section 2.5)은 포맷 위반 = 그 팀 즉시 몰수패지만, 응답
            # 포맷을 승패 조건으로 삼지 않기로 결정함 -- 재시도까지 전부 실패한
            # 게임은 어느 팀의 승리도 아닌 '무효 게임'으로 기록하고 승패 집계에서
            # 제외 (근거: `논문과_다르게_간_설계.md`).
            actor = self._current_actor
            return GameResult("invalid", f"invalid game: no valid response from {actor.name if actor else 'unknown player'} ({e})", self.history, self.leaked)
