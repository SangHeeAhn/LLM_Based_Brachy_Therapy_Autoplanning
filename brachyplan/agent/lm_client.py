"""LM Studio 클라이언트 (OpenAI 호환 엔드포인트).

표준 라이브러리만 사용한다 — `openai` 패키지를 임상 PC에 깔지 않아도 되도록.

LM Studio 준비
--------------
1. LM Studio 실행 → Developer 탭 → **Start Server** (기본 포트 1234)
2. 모델을 하나 로드 (Structured Output을 쓰므로 7B 이상 권장.
   LM Studio 문서: 7B 미만 모델은 복잡한 구조화 출력에 취약)
3. `python run_phase3.py --check` 로 연결 확인

참고: https://lmstudio.ai/docs/developer/openai-compat
      https://lmstudio.ai/docs/developer/openai-compat/structured-output
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "http://localhost:1234/v1"

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.S | re.I)


class LMStudioError(RuntimeError):
    pass


class EmptyContentError(LMStudioError):
    """모델이 빈 content를 반환. reasoning 모델에서 흔한 실패 모드."""


def strip_think(text: str) -> str:
    """<think>...</think> 블록 제거. 닫히지 않은 <think>도 잘라낸다."""
    if not text:
        return ""
    return _OPEN_THINK_RE.sub("", _THINK_RE.sub("", text)).strip()


@dataclass
class ChatResult:
    content: str
    model: str
    raw: dict = field(default_factory=dict, repr=False)
    elapsed_s: float = 0.0
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    from_reasoning: bool = False   # reasoning_content에서 건져 올렸는가

    def json(self) -> dict:
        """content를 JSON으로 파싱. 코드펜스나 앞뒤 잡텍스트가 있어도 복구한다.

        structured output을 켜면 보통 순수 JSON이 오지만, 모델·양자화에 따라
        ```json 펜스를 붙이거나 설명을 덧붙이는 경우가 있다.
        """
        return parse_json_loose(self.content)


def parse_json_loose(text: str) -> dict:
    """느슨한 JSON 파싱. 실패하면 LMStudioError를 던진다."""
    s = (text or "").strip()
    if not s:
        raise LMStudioError("빈 응답")

    # ```json ... ``` 펜스 제거
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 첫 '{' 부터 균형 잡힌 '}' 까지 잘라내기 (문자열 안의 중괄호 무시)
    start = s.find("{")
    if start < 0:
        raise LMStudioError(f"JSON을 찾을 수 없음: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError as e:
                    raise LMStudioError(f"JSON 파싱 실패: {e}") from e
    raise LMStudioError(f"JSON 괄호가 닫히지 않음: {text[:200]!r}")


class LMStudioClient:
    """LM Studio의 OpenAI 호환 API 클라이언트."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        timeout_s: float = 300.0,
        temperature: float = 0.0,
        # reasoning 모델은 사고 과정에 토큰을 대량으로 쓴다. 부족하면 content가
        # 통째로 비어서 돌아온다 (LM Studio 이슈 #1990). 넉넉하게 잡는다.
        max_tokens: int = 6000,
        api_key: str | None = None,
        retries: int = 2,
        disable_thinking: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.retries = retries
        self.disable_thinking = disable_thinking
        self.n_calls = 0
        self.total_s = 0.0
        self.n_from_reasoning = 0

    # ------------------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None,
                 timeout: float | None = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise LMStudioError(f"HTTP {e.code} {url}\n{body}") from e
        except urllib.error.URLError as e:
            raise LMStudioError(
                f"LM Studio에 연결할 수 없습니다: {url}\n"
                f"  원인: {e.reason}\n"
                f"  확인: LM Studio > Developer 탭 > Start Server (포트 1234), "
                f"모델이 로드되어 있는지"
            ) from e
        except TimeoutError as e:
            raise LMStudioError(f"응답 시간 초과 ({timeout or self.timeout_s}s): {url}") from e

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        d = self._request("/models", timeout=15.0)
        return [m.get("id", "") for m in d.get("data", [])]

    def available(self) -> bool:
        """서버가 떠 있고 모델이 하나 이상 있으면 True. 예외를 던지지 않는다."""
        try:
            return len(self.list_models()) > 0
        except LMStudioError:
            return False

    def resolve_model(self) -> str:
        """model이 지정되지 않았으면 로드된 첫 모델을 쓴다."""
        if self.model:
            return self.model
        models = self.list_models()
        if not models:
            raise LMStudioError("LM Studio에 로드된 모델이 없습니다. 모델을 하나 로드하십시오.")
        self.model = models[0]
        return self.model

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        json_schema: dict | None = None,
        schema_name: str = "response",
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        """chat completion 호출.

        json_schema를 주면 LM Studio의 structured output(문법 기반 샘플링)을 쓴다.

        reasoning 모델 대응
        -------------------
        Qwen3 계열 등 reasoning 모델은 LM Studio에서 다음 실패 모드를 보인다.
          * 사고 과정이 max_tokens를 소진해 content가 빔 (이슈 #1990)
          * reasoning_content만 채워지고 content는 빈 채로 200 응답 (이슈 #1602)
          * json_schema 제약이 reasoning 스트림에 걸려 JSON이 그쪽에 생성됨 (#1773)
        따라서 content가 비면 reasoning_content에서 건져 올리고, <think> 블록을
        제거한다. 그래도 비면 원인을 특정한 예외를 던진다.
        """
        payload = {
            "model": self.resolve_model(),
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.disable_thinking:
            # 서버/템플릿에 따라 무시될 수 있다 (이슈 #1990). 무해하므로 함께 보낸다.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_effort"] = "none"
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True,
                                "schema": json_schema},
            }

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                d = self._request("/chat/completions", payload)
            except LMStudioError as e:
                last = e
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
                    continue
                raise
            dt = time.time() - t0
            self.n_calls += 1
            self.total_s += dt

            try:
                choice = d["choices"][0]
                msg = choice["message"]
            except (KeyError, IndexError) as e:
                raise LMStudioError(f"예상과 다른 응답 형식: {str(d)[:300]}") from e

            finish = str(choice.get("finish_reason", ""))
            usage = d.get("usage", {}) or {}
            content = strip_think(msg.get("content") or "")
            from_reasoning = False

            if not content:
                reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
                reasoning = strip_think(reasoning) or (reasoning or "").strip()
                # JSON이 실제로 들어 있을 때만 회수한다. 잘린 사고 텍스트를
                # 그대로 넘기면 "JSON 없음" 오류로 바뀌어, finish_reason 같은
                # 진짜 원인 정보가 사라진다.
                if "{" in reasoning and "}" in reasoning:
                    content = reasoning
                    from_reasoning = True
                    self.n_from_reasoning += 1

            if not content:
                raise EmptyContentError(
                    self._empty_diagnosis(finish, usage, json_schema)
                )

            return ChatResult(content=content, model=d.get("model", payload["model"]),
                              raw=d, elapsed_s=dt, finish_reason=finish,
                              usage=usage, from_reasoning=from_reasoning)
        raise last if last else LMStudioError("알 수 없는 오류")

    # ------------------------------------------------------------------
    def _empty_diagnosis(self, finish: str, usage: dict, json_schema) -> str:
        comp = usage.get("completion_tokens")
        lines = ["모델이 빈 응답을 반환했습니다.",
                 f"  finish_reason = {finish or '(없음)'}"]
        if comp is not None:
            lines.append(f"  completion_tokens = {comp} / max_tokens = {self.max_tokens}")
        lines.append("")
        if finish == "length" or (comp is not None and comp >= self.max_tokens * 0.95):
            lines += [
                "원인: 사고 과정(reasoning)이 토큰 한도를 다 써서 최종 답이 나오지 못했습니다.",
                "  -> --max-tokens 를 더 올리거나 (예: 12000)",
                "  -> --no-think 로 사고 모드를 끄거나",
                "  -> 비-reasoning 모델(예: qwen2.5-instruct 계열)을 쓰십시오.",
            ]
        elif json_schema is not None:
            lines += [
                "원인 추정: reasoning 모델 + structured output 조합 문제입니다.",
                "  LM Studio에서 json_schema 제약이 reasoning 스트림에 적용되어",
                "  최종 content가 비는 사례가 보고돼 있습니다.",
                "  -> --no-schema 로 스키마 없이(프롬프트로만) 시도하거나",
                "  -> --no-think 로 사고 모드를 끄십시오.",
            ]
        else:
            lines.append("원인 불명. LM Studio 로그를 확인하십시오.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def diagnose(self) -> str:
        """연결 상태를 사람이 읽을 수 있게 요약."""
        lines = [f"base_url : {self.base_url}"]
        try:
            models = self.list_models()
        except LMStudioError as e:
            return "\n".join(lines + [f"상태     : 연결 실패", f"{e}"])
        if not models:
            return "\n".join(lines + ["상태     : 서버는 응답하지만 로드된 모델이 없음"])
        lines.append(f"상태     : 정상 ({len(models)}개 모델)")
        for m in models:
            mark = " <- 사용" if (self.model or models[0]) == m else ""
            lines.append(f"           {m}{mark}")
        return "\n".join(lines)
