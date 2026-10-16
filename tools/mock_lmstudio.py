#!/usr/bin/env python3
"""테스트용 Mock LM Studio 서버.

**실제 LM Studio 없이** HTTP 경로 전체(요청 형식, structured output, 응답 파싱,
오류 처리)를 검증하기 위한 것이다. 모델 추론 품질은 검증하지 않는다.

동작 모드
  good        : 프롬프트를 읽고 그럴듯한 조정을 반환 (루프 수렴 검증)
  sloppy      : ```json 펜스 + 앞뒤 잡텍스트 + 범위를 벗어난 값 (방어 코드 검증)
  broken      : 잘못된 구조물명/필드/NaN (안전장치 검증)

  --- reasoning 모델 실패 모드 재현 (실제로 겪은 문제들) ---
  think_tags  : content 안에 <think>...</think> 를 섞어 보냄
  reasoning   : content는 비고 reasoning_content 에만 JSON  (LM Studio 이슈 #1602)
  schema_dead : json_schema 를 주면 빈 응답, 없으면 정상     (LM Studio 이슈 #1773)
  length      : 사고 과정이 토큰을 다 써서 빈 응답+finish=length (이슈 #1990)

  --- LLM 판단 실패 모드 재현 ---
  overshoot   : 목표를 다 충족하면 OAR 가중치를 전부 10배 낮춰 plan을 무너뜨린다
                (실제 qwen3.6-35b 가 #4에서 99.0 -> #5에서 69.4로 무너뜨린 행동)

사용:
    python tools/mock_lmstudio.py --port 1234 --mode good
    python tools/mock_lmstudio.py --self-test        # 서버 없이 응답 생성만 확인
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ID = "mock-qwen2.5-7b-instruct"


# ---------------------------------------------------------------------------
def _parse_findings(prompt: str) -> list[tuple[str, str, float, bool]]:
    """프롬프트의 '문제점' 절에서 (구조물, 종류, 초과율, 필수여부)를 뽑는다."""
    out = []
    for m in re.finditer(
        r"- (\w+) (\w+) = ([\d.]+) \(기준 ([\d.]+), ([\d.]+)% (미달|초과)\)( \[필수\])?", prompt
    ):
        struct, _metric, _v, _lim, pct, kind, hard = m.groups()
        out.append((struct, "max" if kind == "초과" else "min", float(pct), bool(hard)))
    return out


def _volume_of(prompt: str, struct: str) -> float:
    """DVH 표에서 구조물 부피를 읽는다 (가중치 희석 보정용)."""
    m = re.search(rf"^{re.escape(struct)}\s+([\d.]+)", prompt, re.M)
    return float(m.group(1)) if m else 0.0


def _current_weight(prompt: str, struct: str, field: str) -> float:
    """프롬프트의 '현재 최적화 파라미터' JSON에서 값을 읽는다."""
    m = re.search(r"## 현재 최적화 파라미터\n(\{.*?\n\})\n", prompt, re.S)
    if not m:
        return 1.0
    try:
        p = json.loads(m.group(1))
        v = (p.get("structures", {}).get(struct) or {}).get(field)
        return float(v) if v is not None else 1.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return 1.0


def make_reply(prompt: str, mode: str) -> str:
    """프롬프트를 보고 응답 문자열을 만든다."""
    findings = _parse_findings(prompt)

    if mode == "broken":
        return json.dumps({
            "reasoning": "고의로 잘못된 출력 (안전장치 검증용)",
            "adjustments": [
                {"structure": "BLADDER_TYPO", "field": "weight_over", "value": 5},
                {"structure": "Bladder", "field": "learning_rate", "value": 0.1},
                {"structure": "Bladder", "field": "weight_over", "value": 1e12},
                {"structure": "HR_CTV", "field": "dose_max", "value": 7.5},
                {"structure": "Rectum", "field": "weight_over", "value": 4.0},
            ],
            "lam_smooth": -3.0,
        }, ensure_ascii=False)

    adjustments = []
    # 필수 목표를 먼저, 그다음 초과가 큰 순
    findings.sort(key=lambda f: (not f[3], -f[2]))
    for struct, kind, pct, hard in findings[:4]:
        field = "weight_over" if kind == "max" else "weight_under"
        cur = _current_weight(prompt, struct, field)
        factor = 2.0 if pct < 10 else 4.0
        if hard:
            factor *= 2.0
        if kind == "max":
            # 벌점이 전체 복셀 평균이라 큰 구조물일수록 D2cm3 초과가 희석된다
            vol = _volume_of(prompt, struct)
            factor *= max(1.0, vol / 2.0 / 15.0)
        adjustments.append({"structure": struct, "field": field,
                            "value": round(max(cur, 1e-3) * factor, 4)})

    if not adjustments:
        # 전부 충족 — 여유가 큰 OAR의 압력을 낮춰 표적을 더 밀어본다
        drop = 0.1 if mode == "overshoot" else 0.5   # overshoot: 10배 낮춤
        n_take = 6 if mode == "overshoot" else 2
        m = re.search(r"여유가 (?:가장 )?큰 OAR[^:]*: (.+)", prompt)
        names = ([t.strip().split()[0] for t in m.group(1).split(",")] if m else [])
        if mode == "overshoot" and not names:
            names = re.findall(r"^(\w+)\s+[\d.]+.*?\n", prompt, re.M)
        for name in names[:n_take]:
            cur = _current_weight(prompt, name, "weight_over")
            adjustments.append({"structure": name, "field": "weight_over",
                                "value": round(max(cur, 1e-3) * drop, 6)})

    body = {
        "reasoning": ("미충족 목표의 가중치를 초과 정도에 비례해 올렸습니다."
                      if findings else
                      "모든 목표를 충족하여 여유 있는 OAR의 압력을 낮춰 봅니다."),
        "adjustments": adjustments,
    }

    if mode == "sloppy":
        # 코드펜스 + 앞뒤 잡텍스트 + 범위 초과 값
        body["lam_smooth"] = 99.0
        if body["adjustments"]:
            body["adjustments"][0]["value"] = 1e9
        return ("생각해보니 이렇게 조정하는 게 좋겠습니다.\n\n```json\n"
                + json.dumps(body, ensure_ascii=False, indent=1)
                + "\n```\n추가 설명은 생략합니다.")
    return json.dumps(body, ensure_ascii=False)


# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    mode = "good"
    calls: list[dict] = []

    def log_message(self, *a):    # 조용히
        pass

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "invalid json"}})
            return

        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return

        Handler.calls.append(req)

        # LM Studio가 검사하는 것들을 흉내 낸다
        if "model" not in req or "messages" not in req:
            self._send(400, {"error": {"message": "model, messages 필수"}})
            return
        rf = req.get("response_format")
        if rf is not None and rf.get("type") != "json_schema":
            self._send(400, {"error": {"message": "unsupported response_format"}})
            return

        prompt = "\n".join(m.get("content", "") for m in req["messages"])
        mode = Handler.mode
        has_schema = rf is not None
        max_tokens = int(req.get("max_tokens", 1200))

        message = {"role": "assistant"}
        finish = "stop"
        comp_tokens = 0

        if mode == "length":
            # 사고 과정이 토큰 한도를 다 씀 (이슈 #1990)
            message["content"] = ""
            message["reasoning_content"] = "생각 중... " * 50
            finish = "length"
            comp_tokens = max_tokens

        elif mode == "reasoning":
            # content는 비고 reasoning_content 에만 최종 JSON (이슈 #1602)
            message["content"] = ""
            message["reasoning_content"] = make_reply(prompt, "good")
            comp_tokens = 400

        elif mode == "schema_dead":
            # json_schema 를 주면 제약이 사고 스트림에 걸려 content가 빔 (이슈 #1773)
            if has_schema:
                message["content"] = ""
                message["reasoning_content"] = ""
                finish = "stop"
                comp_tokens = 3
            else:
                message["content"] = make_reply(prompt, "good")
                comp_tokens = 200

        elif mode == "think_tags":
            body = make_reply(prompt, "good")
            message["content"] = ("<think>\n사용자가 무엇을 원하는지 보자. "
                                  "Bladder가 초과했으니 가중치를 올려야겠다.\n</think>\n" + body)
            comp_tokens = 300

        else:
            message["content"] = make_reply(prompt, mode)
            comp_tokens = len(message["content"]) // 4

        self._send(200, {
            "id": "chatcmpl-mock", "object": "chat.completion",
            "model": req["model"],
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": len(prompt) // 4,
                      "completion_tokens": comp_tokens,
                      "total_tokens": len(prompt) // 4 + comp_tokens},
        })


class MockServer:
    """with 문으로 쓰는 테스트용 서버."""

    def __init__(self, mode: str = "good", port: int = 0):
        Handler.mode = mode
        Handler.calls = []
        self.httpd = HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def requests(self) -> list[dict]:
        return Handler.calls

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--mode", default="good",
                    choices=["good", "sloppy", "broken", "think_tags",
                             "reasoning", "schema_dead", "length", "overshoot"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        prompt = """## 문제점
미충족 목표 (초과가 큰 순):
  - Bladder D2cm3_Gy = 5.100 (기준 4.330, 17.8% 초과) [필수]
  - HR_CTV D90_Gy = 4.800 (기준 5.000, 4.0% 미달) [필수]

## 현재 최적화 파라미터
{
 "structures": {"Bladder": {"weight_over": 2.0}, "HR_CTV": {"weight_under": 100.0}}
}
"""
        for m in ("good", "sloppy", "broken"):
            print(f"--- {m} ---")
            print(make_reply(prompt, m)[:400])
            print()
        return 0

    Handler.mode = args.mode
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mock LM Studio: http://127.0.0.1:{args.port}/v1  (mode={args.mode})")
    print("Ctrl+C 로 종료")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
