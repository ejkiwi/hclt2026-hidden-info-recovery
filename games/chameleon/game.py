"""
Chameleon game engine -- 「카멜레온 재설계 구상안.md」 구현.

게임 흐름 ①~⑧ 중 ③(힌트 라운드로빈)④(전원 공개)⑥(동시 투표)⑦(최다 득표,
동률이면 dealer)⑧(잡히면 최종 추측 1회)은 논문 원안
(code/paper_code/LLMChameleon-main/llmg/chameleon/GamePlay.py) 그대로이고,
⑤ 토론 페이즈(투표 **전**, 순차 공개, 한 바퀴)만 우리가 추가함
(discussion=True 기본. 끄면 논문 원안 흐름과 동일 = 재현 대조 조건).

논문 원안 코드와의 구현 차이 (게임 규칙이 아니라 프로토콜 차이):
- 논문은 플레이어별 채팅 세션(메시지 누적)이지만, 우리는 세 게임 공통의
  단발 프롬프트(base_context + history 블록) 방식 -- 스파이폴/금기어와 동일.
- 응답 포맷은 승패 조건이 아님(논문의 'Fail' 게임 중단을 대체):
  투표(스킵이 합법) -> 무효 출력은 기권으로 강등,
  힌트/토론/최종 추측/dealer 선택(스킵 불가) -> 재시도 후 실패 시 무효 게임.
- 힌트 "1단어"는 논문 코드가 검사만 하고 넘어가는데(check_word_responses가
  pass), 우리는 재시도 정책으로 강제함 (구상안 '구현 메모').
논문에 없어서 우리가 채운 규칙들은 전부 `설계 결정.md`에 근거와 함께 기록됨.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from openai import OpenAI

# llm_client는 세 게임(스파이폴/카멜레온/금기어)이 같이 쓰는 호출 래퍼라
# ../common/ 에 있음 (게임 규칙은 안 들어있음).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from llm_client import InvalidMoveError, call_player
from prompts import (
    BASE_CONTEXT,
    CANDIDATES_RULE_CATEGORY,
    CANDIDATES_RULE_FULL,
    DEALER_TIEBREAK,
    DISCUSSION_CHAMELEON,
    DISCUSSION_NON_CHAMELEON,
    DISCUSSION_RULE,
    DISCUSSION_RULE_1ROUND,
    FINAL_GUESS_CHAMELEON,
    FINAL_GUESS_CHAMELEON_OPEN,
    HINT_CHAMELEON,
    HINT_NON_CHAMELEON,
    LEAK_WARNING,
    REBUTTAL_CHAMELEON,
    REBUTTAL_NON_CHAMELEON,
    ROUND_NOTE_FIRST_OF_TWO,
    ROUND_NOTE_SINGLE,
    VOTE_CHAMELEON,
    VOTE_NON_CHAMELEON,
)


@dataclasses.dataclass
class Player:
    name: str
    model: str
    role: str  # "chameleon" or "non_chameleon"


@dataclasses.dataclass
class GameResult:
    winner: str  # "chameleon" / "non_chameleon" / "invalid" (무효 게임 -- 승패 집계에서 제외)
    reason: str
    history: list[dict]
    accused: str | None  # 최다 득표(동률이면 dealer 선택)로 지목된 플레이어. 유효 표가 없으면 None
    identified: bool  # 지목된 사람이 진짜 카멜레온이었는지


class GameInvalidError(Exception):
    """재시도까지 전부 실패해서 게임을 더 진행할 수 없을 때. 어느 팀의
    승리도 아닌 '무효 게임'으로 처리됨 -- 응답 포맷은 승패 조건이 아니라는
    세 게임 공통 설계 결정 (논문 원안의 'Fail' 중단을 대체, `설계 결정.md`)."""


def _guess_matches(guess: str, word: str) -> bool:
    """최종 추측 판정: 논문 방식의 부분 문자열 포함(GamePlay.py:178
    `secret_word in chameleon_response.lower()`)을 casefold로 옮긴 것.
    한국어는 조사가 붙으므로("은행입니다") 부분 문자열이어야 하고, 후보
    데이터는 게임 내 부분 문자열 충돌 검사를 이미 통과했으므로(categories_*.json)
    다른 후보가 정답 안에 포함되는 오판은 없음 (구상안 확정 결정 #5)."""
    return word.casefold() in guess.casefold()


def _require_fields(result: dict, spec: dict[str, tuple]) -> None:
    """필수 필드의 존재/타입 검증. 위반 시 InvalidMoveError -- 호출부가
    재시도하거나(필수 행동) 기권으로 강등함(투표)."""
    for field, allowed_types in spec.items():
        if field not in result:
            raise InvalidMoveError(f"missing required field {field!r} in {result!r}")
        if not isinstance(result[field], allowed_types):
            raise InvalidMoveError(
                f"field {field!r} has wrong type {type(result[field]).__name__} in {result!r}")


class ChameleonGame:
    def __init__(
        self,
        client: OpenAI,
        players: list[Player],
        category: str,
        candidates: list[str],
        secret_word: str,
        dealer_name: str,
        language: str = "English",
        reformatter_model: str | None = None,
        action_retries: int = 2,
        discussion: bool = True,
        candidate_mode: str = "category",
        discussion_rounds: int = 2,
        leak_warning: bool = False,
    ):
        assert secret_word in candidates, "secret_word must be a member of candidates"
        assert candidate_mode in ("full", "category")
        assert discussion_rounds in (1, 2)
        chameleons = [p for p in players if p.role == "chameleon"]
        assert len(chameleons) == 1, "exactly one chameleon is required"
        assert dealer_name in {p.name for p in players}, "dealer must be one of the players"
        self.client = client
        self.players = players
        self.category = category
        self.candidates = candidates
        self.secret_word = secret_word
        # dealer(동률 처리자)는 논문대로 참가자 중 무작위 1명 -- 재현성을 위해
        # run_game.py의 메인 스레드에서 미리 뽑아서 이름으로 넘겨받음.
        self.dealer_name = dealer_name
        self.language = language
        self.reformatter_model = reformatter_model
        self.action_retries = action_retries  # 필수 행동이 무효 출력일 때 그 턴을 다시 시키는 횟수
        self.discussion = discussion  # ⑤ 토론 페이즈 유/무 (실험 조건. False = 논문 원안 흐름)
        # category(기본) = **변형 카멜레온**: 목록 없이 카테고리만, 전원 비공개
        # (후보 목록을 주지 않는 조건).
        # full = 원작/논문 그대로 (카테고리 + 목록 전원 공개) -- 원작 재현 대조용.
        # 어느 쪽이든 candidates에서 정답을 뽑는 것은 동일 (보여주느냐만 다름).
        self.candidate_mode = candidate_mode
        # 2(기본) = 제기+반박 (설계 방침). 1 = 제기만 --
        # 1 vs 2바퀴 토론 효과 비교 조건 (반박 라운드가 카멜레온의 변호 무대가
        # 되는지 확인용).
        self.discussion_rounds = discussion_rounds
        # 시민 토론 발언의 "정답 단어를 입에 올리지 마라" 경고 (본 실험 설정 --
        # 원작의 암묵 상식 명시 + 스파이폴 'Be extremely subtle' 지시와 수위 맞춤.
        # False = 자발적 비밀 유지 ablation: 무경고에선 시민 토론 발언의 93~97%가
        # 정답을 그대로 포함했음. 설계 결정 #14 / prompts.LEAK_WARNING 참고).
        self.leak_warning = leak_warning
        self.chameleon = chameleons[0]
        self.history: list[dict] = []
        self._current_actor: Player | None = None

    # -- prompt assembly -----------------------------------------------

    def _base_context(self) -> str:
        # 토론을 끄면 규칙 설명에서도 토론 문장이 빠져야 함 -- 규칙에는 있는데
        # 실제로 안 일어나면 모델이 "토론은 언제 하지"라는 혼란 발화를 낼 수 있음.
        # 후보 목록 관련 문구도 모드에 맞게 치환 (없는 목록을 참조하면 혼란 발화).
        # {and_list}/{from_list}는 템플릿 본문에도 있어서 _player_prompt에서 마지막에 치환.
        if not self.discussion:
            disc_rule = ""
        elif self.discussion_rounds == 2:
            disc_rule = DISCUSSION_RULE
        else:
            disc_rule = DISCUSSION_RULE_1ROUND
        return BASE_CONTEXT.replace("{discussion_rule}", disc_rule) \
                           .replace("{candidates_rule}",
                                    CANDIDATES_RULE_FULL if self.candidate_mode == "full" else CANDIDATES_RULE_CATEGORY)

    def _identity_block(self, player: Player) -> str:
        # 전체 명단을 같이 줌: 명단이 없으면 다른 플레이어 이름을 history에서
        # 추론해야 해서, 약한 모델이 프롬프트 예시의 "Charlie" 같은 유령 이름을
        # 지목하기 쉬움 (스파이폴에서 실제로 있었던 사고).
        roster = ", ".join(p.name for p in self.players)
        return (f"**Players in this game (speaking order):** {roster}\n"
                f"**Your name in this game is:** {player.name}")

    def _info_block(self, player: Player) -> str:
        """정답은 시민에게만(②). 후보 목록은 full 모드에서만 전원 공개(①) --
        category 모드(기본)는 시민에게도 목록을 안 줌 (전원 비공개, 설계 확정)."""
        if self.candidate_mode == "category":
            block = f"**Category:** {self.category}"
            if player.role == "chameleon":
                return (f"{block}\n\n**You do NOT know the secret word** -- it is a word from this "
                        f"category. There is no candidate list in this game.")
            return f"{block}\n\n**The secret word is: {self.secret_word}**"
        lines = "\n".join(f"- {w}" for w in self.candidates)
        n = len(self.candidates)
        block = f"**Category:** {self.category}\n**Candidate words ({n} total):**\n{lines}"
        if player.role == "chameleon":
            return f"{block}\n\n**You do NOT know the secret word** -- it is one of the candidates above."
        return f"{block}\n\n**The secret word is: {self.secret_word}**"

    def _history_block(self) -> str:
        if not self.history:
            return "**Game history so far:** (none yet)"
        lines = []
        for t in self.history:
            if t["phase"] == "hint":
                lines.append(f"[{t['actor']} gave hint]: {t['hint_word']}")
            elif t["phase"] == "discussion":
                if t.get("discussion_round", 1) == 2:
                    lines.append(f"[{t['actor']} said in the rebuttal round]: {t['statement']}")
                else:
                    lines.append(f"[{t['actor']} said in discussion]: {t['statement']}")
            elif t["phase"] == "vote":
                # 투표는 동시 공개라 이 줄이 보이는 시점은 개표 이후뿐
                # (dealer 동률 처리, 최종 추측) -- 투표 프롬프트에는 안 보임.
                lines.append(f"[{t['actor']} voted]: {t['target'] or 'ABSTAIN'}")
            elif t["phase"] == "dealer":
                lines.append(f"[{t['actor']} (dealer) broke the tie]: {t['choice']}")
        return "**Game history so far:**\n" + "\n".join(lines)

    def _player_prompt(self, player: Player, template: str, extra: str = "") -> str:
        # 순서: 규칙 -> 게임 상황(명단/카테고리/history/추가 정보) -> 페이즈 과제와
        # 포맷 지시 (스파이폴과 동일한 배치 -- 모델이 마지막으로 읽는 게 포맷 요구사항).
        parts = [self._base_context(), self._identity_block(player), self._info_block(player), self._history_block()]
        if extra:
            parts.append(extra)
        assembled = template.replace("{base_context}", "\n\n".join(parts))
        # 모드/언어 치환은 조립 마지막에 한 곳에서 -- 템플릿 본문에도 들어가므로
        # (구상안 '프롬프트 주의').
        if self.candidate_mode == "full":
            assembled = assembled.replace("{from_list}", " from the candidate list") \
                                 .replace("{and_list}", ", the candidate list,")
        else:
            assembled = assembled.replace("{from_list}", "").replace("{and_list}", "")
        assembled = assembled.replace("{round_note}",
                                      ROUND_NOTE_FIRST_OF_TWO if self.discussion_rounds == 2 else ROUND_NOTE_SINGLE)
        assembled = assembled.replace("{leak_warning}", LEAK_WARNING.rstrip() if self.leak_warning else "")
        return assembled.replace("{language}", self.language)

    def _call(self, player: Player, template: str, extra: str = "") -> dict:
        self._current_actor = player
        prompt = self._player_prompt(player, template, extra)
        return call_player(self.client, player.model, prompt, reformatter_model=self.reformatter_model)

    def _call_required(self, player: Player, template: str, extra: str = "",
                       spec: dict | None = None, validate=None) -> dict:
        """힌트/토론/최종 추측/dealer 선택처럼 스킵이 불가능한 필수 행동용.
        무효 출력이면 (리포맷터 구제 포함) 같은 플레이어에게 그 턴을 다시
        시키고, 재시도까지 전부 실패하면 GameInvalidError(무효 게임)를 던짐.
        `validate(result)`는 필드 검증 이후의 게임 규칙 검증(1단어 강제 등) --
        InvalidMoveError를 던지면 똑같이 재시도됨."""
        last_err: Exception | None = None
        for attempt in range(self.action_retries + 1):
            try:
                result = self._call(player, template, extra)
                if spec:
                    _require_fields(result, spec)
                if validate:
                    validate(result)
                if attempt:
                    result["turn_retries"] = attempt  # 논문 규칙이었다면 'Fail' 중단이었을 상황 -- 복원용 기록
                return result
            except InvalidMoveError as e:
                last_err = e
        raise GameInvalidError(
            f"{player.name} produced no valid response after {self.action_retries + 1} attempts: {last_err}")

    @staticmethod
    def _entry(phase: str, actor: str, result: dict, **fields) -> dict:
        entry = {"phase": phase, "actor": actor, **fields,
                 "needed_reformat": result.get("needed_reformat", False),
                 "raw_response": result.get("raw_response", "")}
        if "turn_retries" in result:
            entry["turn_retries"] = result["turn_retries"]
        return entry

    # -- 페이즈 -----------------------------------------------------------

    def _hint_phase(self) -> None:
        """③ 힌트 단어: 플레이어 순서대로 1단어씩. 앞사람 힌트는 history 블록으로
        자동 공개되고, ④(전원 공개)는 이후 페이즈의 history가 전체 힌트를
        담는 것으로 충족됨 (단발 프롬프트 방식이라 별도 호출이 필요 없음)."""
        def validate(result: dict) -> None:
            word = result["hint_word"].strip()
            # 1단어 강제: 공백 없는 토큰 1개 (논문 check_word_responses와 같은
            # 기준이지만, 논문은 pass로 넘어가고 우리는 재시도 -- 구상안 '구현 메모').
            if not word or len(word.split()) != 1:
                raise InvalidMoveError(f"hint_word must be exactly one word, got {result['hint_word']!r}")
        for player in self.players:
            template = HINT_CHAMELEON if player.role == "chameleon" else HINT_NON_CHAMELEON
            result = self._call_required(player, template, spec={"hint_word": (str,)}, validate=validate)
            self.history.append(self._entry("hint", player.name, result,
                                            hint_word=result["hint_word"].strip()))

    def _discussion_phase(self) -> None:
        """⑤ 토론: 힌트를 말한 순서 그대로 **두 바퀴** (설계 확정,
        설계 방침). 1바퀴 = 의심 제기, 2바퀴 = 반박/변호 -- 원작의 자유 공방
        ("pick holes in each other's arguments")을 유한하게 근사한 최소 구조.
        1바퀴만 돌면 반박 기회가 없어서 대화가 아니라 순차 성명이 됨.
        발언을 즉시 history에 넣으므로 뒷사람은 앞선 발언 전부를 보게 됨(순차 공개)."""
        def validate(result: dict) -> None:
            if not result["statement"].strip():
                raise InvalidMoveError("discussion statement must not be empty")
        rounds = [(1, DISCUSSION_CHAMELEON, DISCUSSION_NON_CHAMELEON),
                  (2, REBUTTAL_CHAMELEON, REBUTTAL_NON_CHAMELEON)][:self.discussion_rounds]
        for round_no, tmpl_chameleon, tmpl_villager in rounds:
            for player in self.players:
                template = tmpl_chameleon if player.role == "chameleon" else tmpl_villager
                result = self._call_required(player, template, spec={"statement": (str,)}, validate=validate)
                self.history.append(self._entry("discussion", player.name, result,
                                                statement=result["statement"].strip(),
                                                discussion_round=round_no))

    def _vote_phase(self) -> dict[str, int]:
        """⑥ 동시 투표: 이번 라운드의 표는 전원이 지목을 마친 뒤에 한꺼번에
        history에 넣음 -- 루프 도중에 넣으면 뒷순서 플레이어가 앞사람 표를 보고
        따라 투표할 수 있음 (스파이폴과 동일한 처리).

        논문 ⑥ 그대로 스킵 선택지는 프롬프트에 없음. 다만 무효 출력(파싱 실패,
        유령 이름, 자기 자신 투표)은 게임 중단 대신 기권(ABSTAIN)으로 강등하고
        플래그만 남김 -- 응답 포맷은 승패 조건이 아니라는 공통 정책 (`설계 결정.md`)."""
        tally: dict[str, int] = {}
        valid_names = {p.name for p in self.players}
        round_entries: list[dict] = []
        for player in self.players:
            template = VOTE_CHAMELEON if player.role == "chameleon" else VOTE_NON_CHAMELEON
            try:
                result = self._call(player, template)
            except InvalidMoveError as e:
                round_entries.append({"phase": "vote", "actor": player.name, "target": None,
                                      "downgraded_to_abstain": "unparseable output", "invalid_output": str(e)})
                continue
            entry = self._entry("vote", player.name, result)
            target = result.get("vote")
            if not isinstance(target, str) or not target.strip():
                entry["downgraded_to_abstain"] = f"no usable vote field: {result.get('vote')!r}"
                target = None
            else:
                target = target.strip()
                if target == player.name or target not in valid_names:
                    # 자기 자신 투표, 또는 존재하지 않는 이름(프롬프트 예시의
                    # "Charlie"를 베낀 경우 등)은 기권으로 강등. 원래 지목값은
                    # 분석용으로 남겨둠 (스파이폴과 동일 처리).
                    entry["invalid_target"] = target
                    target = None
            entry["target"] = target
            round_entries.append(entry)
            if target:
                tally[target] = tally.get(target, 0) + 1
        # 전원이 지목을 마친 뒤 한꺼번에 공개.
        self.history.extend(round_entries)
        return tally

    def _accuse(self, tally: dict[str, int]) -> str | None:
        """⑦ 최다 득표자 결정 (과반 불필요 -- 논문 원안 그대로 plurality).
        동률이면 dealer가 동률자 중에서 고름. 유효 표가 하나도 없으면(전원
        기권 강등) 아무도 지목되지 않음 -> None (논문에 없는 상황, `설계 결정.md`)."""
        if not tally:
            return None
        top_count = max(tally.values())
        # 동률자 나열은 플레이어 순서 기준으로 고정 (실행 간 재현성).
        tied = [p.name for p in self.players if tally.get(p.name, 0) == top_count]
        if len(tied) == 1:
            return tied[0]
        dealer = next(p for p in self.players if p.name == self.dealer_name)
        tied_set = set(tied)

        def validate(result: dict) -> None:
            if result["choice"].strip() not in tied_set:
                raise InvalidMoveError(f"dealer choice {result['choice']!r} is not among tied players {tied}")
        template = DEALER_TIEBREAK.replace("{tied_players}", ", ".join(tied))
        result = self._call_required(dealer, template, spec={"choice": (str,)}, validate=validate)
        choice = result["choice"].strip()
        self.history.append(self._entry("dealer", dealer.name, result, tied=tied, choice=choice))
        return choice

    def _final_guess(self) -> tuple[str, bool]:
        """⑧ 잡힌 카멜레온의 최종 추측 1회 -- 스킵 불가 (구상안 게임 흐름).

        full 모드: 후보 중 **정확히 하나만** 응답에 포함돼야 유효. 논문(≥1개면
        통과)보다 엄격한데, 부분 문자열 판정이라 여러 후보를 나열하면("강아지나
        고양이 둘 중 하나") 정답이 뭐든 이기는 헷징 구멍이 있기 때문 (#11).

        category 모드(기본): 목록이 없으므로 열린 어휘. 비어있지만 않으면 유효하되,
        숨겨진 후보가 2개 이상 포함된 나열식 헷징은 같은 이유로 무효 처리 (#11과
        동일 취지 -- 카멜레온은 목록을 모르지만 나열 자체를 막는 안전장치).

        정답 판정(공통): 정답 단어가 응답에 포함되면 정답, 유의어는 오답(하한값
        해석) -- 설계 확정. 조사가 붙은 "은행입니다"는 계속 허용됨."""
        def validate(result: dict) -> None:
            guess = result["guess"]
            contained = [w for w in self.candidates if _guess_matches(guess, w)]
            if self.candidate_mode == "full" and not contained:
                raise InvalidMoveError(f"final guess {guess!r} does not contain any candidate word")
            if self.candidate_mode == "category" and not guess.strip():
                raise InvalidMoveError("final guess must not be empty")
            if len(contained) > 1:
                raise InvalidMoveError(
                    f"final guess {guess!r} hedges between multiple candidates {contained} -- exactly one required")
        template = FINAL_GUESS_CHAMELEON if self.candidate_mode == "full" else FINAL_GUESS_CHAMELEON_OPEN
        result = self._call_required(self.chameleon, template,
                                     spec={"guess": (str,)}, validate=validate)
        guess = result["guess"].strip()
        correct = _guess_matches(guess, self.secret_word)
        self.history.append(self._entry("final_guess", self.chameleon.name, result,
                                        guess=guess, correct=correct))
        return guess, correct

    # -- 최상위 실행 함수 ---------------------------------------------

    def play(self) -> GameResult:
        """게임 한 판 (카멜레온은 한 판 게임 -- 라운드 반복 없음)."""
        try:
            self._hint_phase()                      # ③ (+④는 history로 충족)
            if self.discussion:
                self._discussion_phase()            # ⑤ 우리가 추가한 유일한 단계
            tally = self._vote_phase()              # ⑥
            accused = self._accuse(tally)           # ⑦
            if accused is None:
                # 전원 기권 강등으로 유효 표 0 -- 아무도 지목 안 됨 = 카멜레온 생존.
                return GameResult("chameleon", "no valid votes were cast, so the chameleon was never accused",
                                  self.history, None, False)
            if accused == self.chameleon.name:
                guess, correct = self._final_guess()  # ⑧
                if correct:
                    return GameResult("chameleon", "chameleon was accused but guessed the secret word correctly",
                                      self.history, accused, True)
                return GameResult("non_chameleon", "chameleon was accused and failed to guess the secret word",
                                  self.history, accused, True)
            return GameResult("chameleon", "an innocent villager was accused; the chameleon escaped",
                              self.history, accused, False)
        except (GameInvalidError, InvalidMoveError) as e:
            # 재시도까지 전부 실패한 게임은 어느 팀의 승리도 아닌 '무효 게임'으로
            # 기록하고 승패 집계에서 제외 (세 게임 공통 정책).
            actor = self._current_actor
            return GameResult("invalid",
                              f"invalid game: no valid response from {actor.name if actor else 'unknown player'} ({e})",
                              self.history, None, False)
