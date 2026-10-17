"""Thin LLM call wrapper for the "|||...|||"-wrapped JSON response format.

원래 `multicultural-spyfall-repro/llm_client.py`였는데, 카멜레온·금기어도
같은 호출 규약(|||JSON|||, 리포맷터 구제, 레이트리밋 재시도, 잘림 감지)을
쓰기 때문에 공통 모듈로 뺐음. **게임 규칙은 여기 없음** -- 게임별 로직은
각 게임 폴더의 game.py에만 있으므로, 이 파일을 공유해도 게임 간 규칙이
섞이지 않음. 여기서 버그를 고치면 세 게임에 동시에 반영됨.

포맷 자체는 Multicultural Spyfall Appendix C 프롬프트에서 온 것.

사용하는 쪽에서:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    from llm_client import InvalidMoveError, call_player
"""

from __future__ import annotations

import json
import re
import threading
import time

import openai
from openai import OpenAI


class _RoutedCompletions:
    def __init__(self, router: "MultiClient") -> None:
        self._router = router

    def create(self, *, model: str, **kwargs):
        return self._router.client_for(model).chat.completions.create(model=model, **kwargs)


class _RoutedChat:
    def __init__(self, router: "MultiClient") -> None:
        self.completions = _RoutedCompletions(router)


class MultiClient:
    """모델 이름 -> OpenAI 클라이언트 라우터.

    역할 교차 대국(예: 스파이만 9B, 시민 4명은 35B)에서는 두 역할의 모델이
    서로 다른 서버(포트)에 떠 있으므로, 클라이언트 하나로는 호출이 안 됨.
    llama_cpp.server는 config로 여러 모델을 올려도 요청이 바뀔 때마다 이전
    모델을 내려버려서(모델 스래싱) 교차 대국에는 못 씀 -- 그래서 모델별로
    서버를 따로 띄우고 이름으로 라우팅함.

    OpenAI 클라이언트의 `.chat.completions.create(model=...)` 인터페이스만
    흉내내므로 `_chat()`과 각 게임의 game.py는 수정 없이 그대로 동작함.
    등록되지 않은 이름은 기본 클라이언트로 보냄(리포맷터·심판 등).
    """

    def __init__(self, default_client: OpenAI, per_model: dict[str, OpenAI]) -> None:
        self._default = default_client
        self._per_model = dict(per_model)
        self.chat = _RoutedChat(self)

    def client_for(self, model: str) -> OpenAI:
        return self._per_model.get(model, self._default)


def build_client(api_key: str | None, base_url: str | None, model_base_urls: list[str] | None = None):
    """`--base-url` 하나 + `--model-base-url MODEL=URL` 여러 개를 받아
    클라이언트를 만듦. 후자가 없으면 기존과 완전히 동일한 단일 클라이언트."""
    default = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    if not model_base_urls:
        return default
    per_model = {}
    for spec in model_base_urls:
        model, sep, url = spec.partition("=")
        if not sep or not model or not url:
            raise ValueError(f"--model-base-url expects MODEL=URL, got {spec!r}")
        per_model[model] = OpenAI(api_key=api_key, base_url=url)
    return MultiClient(default, per_model)

# Simple client-side pacing + retry so free-tier rate limits (e.g. Gemini's
# 5 requests/min/model on the free tier) don't just crash a whole game.
# Not part of the paper's protocol -- purely an implementation necessity.
MIN_SECONDS_BETWEEN_CALLS = 13.0
MAX_RETRIES = 5
# 생성 상한을 명시함. 2026-08-03 20게임 실행에서 서버(n_ctx 2048 기본값)가
# 게임 후반의 긴 투표 프롬프트에서 추론 도중 출력을 잘라버려 JSON을 못 쓰고
# 몰수패가 나는 사례가 11/20건 발생 -- 서버 컨텍스트를 키우는 게 근본 해결이고
# (서버는 --n_ctx 16384로 재시작함), 클라이언트에서도 상한을 명시해 둠.
MAX_COMPLETION_TOKENS = 4096
_last_call_time: float | None = None
# 게임을 병렬로 돌릴 때(run_game.py --parallel) 여러 스레드가 동시에 들어오므로
# 락으로 감쌈. 락 안에서 sleep해야 "전체 호출 간 최소 간격"이라는 원래 의미가
# 유지됨 -- 안 그러면 N개 스레드가 동시에 통과해서 레이트리밋에 걸림.
_throttle_lock = threading.Lock()


def _throttle() -> None:
    global _last_call_time
    with _throttle_lock:
        if _last_call_time is not None:
            elapsed = time.monotonic() - _last_call_time
            wait = MIN_SECONDS_BETWEEN_CALLS - elapsed
            if wait > 0:
                time.sleep(wait)
        _last_call_time = time.monotonic()


REFORMAT_PROMPT = """The text below was supposed to end with a JSON object wrapped in triple pipes, like |||{{"field": "value"}}|||, but it was not formatted that way (e.g. wrapped in markdown code fences instead, or missing the pipes entirely).

Extract the JSON object that was already intended in the text -- do not invent or change any values, only fix the wrapping/formatting -- and output ONLY that JSON object wrapped in |||...|||, nothing else, no explanation.

If the text does not actually contain the intended JSON content (e.g. it was cut off before any JSON appeared), output exactly NO_JSON without any pipes -- do NOT invent a JSON object and do NOT copy the {{"field": "value"}} example above.

Text:
---
{raw}
---"""


def _chat(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                # 출력이 잘렸다는 뜻 -- JSON이 안 나왔으면 이 게임은 몰수패로
                # 끝나게 되므로, 서버 컨텍스트(n_ctx)/max_tokens 문제임을 바로
                # 알 수 있게 경고를 남김.
                print(f"  [warning: output truncated (finish_reason=length), model={model}]")
            return choice.message.content
        except openai.RateLimitError as e:
            if "PerDay" in str(e):
                # A daily quota, not a per-minute one -- retrying within
                # this process cannot help, so fail fast instead of
                # burning more of the (already exhausted) quota.
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = min(60.0, MIN_SECONDS_BETWEEN_CALLS * (2 ** attempt))
            print(f"  [rate limited, retrying in {backoff:.0f}s ...] {e}")
            time.sleep(backoff)
        except (openai.APIConnectionError, openai.InternalServerError) as e:
            # 순간적인 네트워크 끊김(APITimeoutError는 APIConnectionError의
            # 하위클래스라 여기서 같이 잡힘)이나 서버 5xx 에러 -- rate limit과
            # 똑같이 재시도. 인증 오류(401)/잘못된 요청(400) 등은 재시도해도
            # 의미가 없어서 일부러 여기서 안 잡고 그대로 전파시킴.
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = min(60.0, MIN_SECONDS_BETWEEN_CALLS * (2 ** attempt))
            print(f"  [transient API error, retrying in {backoff:.0f}s ...] {e}")
            time.sleep(backoff)


def call_player(client: OpenAI, model: str, prompt: str, reformatter_model: str | None = None, temperature: float = 1.0) -> dict:
    """Send `prompt` as a single user message and parse the |||...|||-wrapped
    JSON block out of the response, matching the paper's strict-JSON
    protocol (Section 2.5: "Responses must adhere to a strict JSON
    format; any output that fails to do so is deemed invalid.").

    If `reformatter_model` is given and the first attempt doesn't parse,
    one salvage pass is made: a second model is asked to re-wrap the
    same content correctly, and that is parsed instead. This is NOT
    part of the paper's protocol -- it exists because this research
    cares about secret-inference/leakage behavior, not strict-JSON
    compliance, and a model losing an entire game to a markdown-fence
    formatting slip is noise, not signal (see 체크리스트.md). Every
    returned dict is tagged with `needed_reformat` so the raw
    compliance rate can still be recovered from the logs if wanted."""
    raw = _chat(client, model, prompt, temperature)
    try:
        parsed = parse_pipe_json(raw)
        parsed["needed_reformat"] = False
        parsed["raw_response"] = raw
        return parsed
    except InvalidMoveError:
        if not reformatter_model:
            raise
        reformatted_raw = _chat(client, reformatter_model, REFORMAT_PROMPT.format(raw=raw), temperature=0)
        parsed = parse_pipe_json(reformatted_raw)  # still unparseable -> InvalidMoveError propagates (caller downgrades to skip or retries)
        parsed["needed_reformat"] = True
        # 리포맷터가 아니라 실제 행위자(actor)의 원본 응답(사고 과정 포함)을 저장 --
        # 정성 분석의 핵심 데이터라 파싱된 필드만 남기고 버리면 안 됨.
        parsed["raw_response"] = raw
        return parsed


class InvalidMoveError(Exception):
    """Raised when a player's output does not follow the required
    |||...|||-wrapped JSON format. The paper (Section 2.5) treats this
    as an immediate forfeit; this implementation deliberately does NOT --
    the game engine downgrades it to a skip (guess/vote/declare phases)
    or retries the turn and, failing that, aborts the game as invalid
    (question/answer/utterance phases). See 논문과_다르게_간_설계.md."""


def parse_pipe_json(raw: str) -> dict:
    """Extract the |||...|||-wrapped JSON block. The pattern requires an
    actual JSON object ({...}) between the pipes, so a stray "|||"
    mentioned in the model's reasoning text can't pair up with the real
    delimiters and shift the match. Uses the LAST such block in the
    text, matching the "no text after the JSON" instruction."""
    matches = re.findall(r"\|\|\|\s*(\{.*?\})\s*\|\|\|", raw, re.DOTALL)
    if not matches:
        raise InvalidMoveError(f"No |||{{...}}||| JSON block found in model output: {raw!r}")
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError as e:
        raise InvalidMoveError(f"Malformed JSON inside |||...||| block: {raw!r}") from e
