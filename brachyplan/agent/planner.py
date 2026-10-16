"""Planner — LLM이 다음 최적화 파라미터를 제안한다. + 안전장치.

설계 원칙
---------
LLM은 **dwell time을 직접 만들지 않는다.** 사람이 읽을 수 있는 최적화
파라미터(가중치, 목표선량, 평활 계수)만 조정하고, 실제 dwell time은
결정론적 볼록 최적화기가 계산한다. 따라서
  (a) 물리적 타당성이 코드로 보장되고,
  (b) 목적함수가 볼록이라 최적화 실패가 없으며,
  (c) LLM이 왜 그렇게 했는지 텍스트로 감사(audit)할 수 있다.

출력 형식
---------
구조물 이름을 동적 키로 쓰는 대신 **enum으로 제한된 조정 목록**을 받는다.
문법 기반 샘플링(structured output)에서 훨씬 안정적이고, 작은 모델도
잘 따라온다. 알 수 없는 구조물/필드는 애초에 생성될 수 없다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .evaluator import PlanScore, findings_text, metrics_table
from .lm_client import LMStudioClient, LMStudioError

FIELDS = ("weight_under", "weight_over", "dose_min", "dose_max")

# 안전 범위 — LLM이 무엇을 내놓든 이 범위로 clip 된다
LIMITS = {
    "weight_under": (0.0, 1.0e6),
    "weight_over": (0.0, 1.0e6),
    "dose_min": (0.0, 100.0),
    "dose_max": (0.0, 500.0),
}
LAM_SMOOTH_LIMITS = (0.0, 10.0)


def build_schema(structures: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "무엇이 문제이고 왜 이렇게 조정하는지 2~4문장",
            },
            "adjustments": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "structure": {"type": "string", "enum": structures},
                        "field": {"type": "string", "enum": list(FIELDS)},
                        "value": {"type": "number"},
                    },
                    "required": ["structure", "field", "value"],
                    "additionalProperties": False,
                },
            },
            "lam_smooth": {
                "type": "number",
                "description": "dwell time 평활 계수. 바꾸지 않으려면 현재 값을 그대로",
            },
        },
        "required": ["reasoning", "adjustments"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = """당신은 자궁경부암 HDR 근접치료 계획을 담당하는 의학물리사입니다.

역할: 역최적화기의 **파라미터만** 조정합니다. dwell time을 직접 정하지 않습니다.
최적화기는 볼록 문제를 전역 최적으로 풉니다. 따라서 결과가 나쁘다면 그것은
최적화 실패가 아니라 **당신이 준 가중치/목표선량이 잘못된 것**입니다.

목적함수:
  J(t) = sum_s [ w_under_s * mean(max(0, dose_min_s - d)^2)
               + w_over_s  * mean(max(0, d - dose_max_s)^2) ]
       + lam_smooth * sum(인접 dwell time 차이^2)
  제약: 0 <= t <= t_max

**가중치는 구조물 크기에 따라 의미가 다릅니다 (매우 중요):**
벌점이 구조물 **전체 복셀의 평균**이라, D2cm3 목표처럼 상위 2 cm^3 만
문제인 경우 큰 구조물일수록 심하게 희석됩니다.
  예) Bladder 376 cm^3 에서 2 cm^3 는 전체의 0.5 %
      Rectum   30 cm^3 에서 2 cm^3 는 전체의 6.7 %
즉 같은 정도로 초과했더라도 Bladder는 Rectum보다 **10배 이상 큰 가중치**가
필요합니다. 큰 OAR(100 cm^3 이상)이 계속 초과하면 2배가 아니라
10~100배로 과감하게 올리십시오.

조정 규칙:
- 표적이 미달이면 그 표적의 weight_under를 올리거나, 여유가 있는 OAR의
  weight_over를 낮춥니다.
- OAR이 초과하면 그 OAR의 weight_over를 올립니다.
- 가중치는 배수로 움직이십시오. 1.05배 같은 미세 조정은 무의미합니다.
- [필수] 표시된 목표를 위반하면 점수가 크게 깎입니다. 그것부터 해결하십시오.
- 반대로 목표에 거의 닿았는데(1~2 % 이내) 계속 못 넘기면, 가중치를 더 키우기보다
  경쟁하는 다른 항목의 가중치를 낮추는 편이 나을 수 있습니다.

**변경 폭 규칙 (중요):**
- 필수 목표를 **위반 중**일 때는 과감하게 (10~100배) 움직여도 됩니다.
- 필수 목표를 **모두 충족한 상태**라면 **한 번에 2배 이내**로만 움직이십시오.
  이때는 이미 좋은 plan을 손에 쥐고 있는 상태입니다. 여러 OAR의 가중치를
  동시에 10배씩 낮추면 필수 목표가 한꺼번에 깨져 처음으로 되돌아갑니다.
  소프트 목표(표적 D90 가점)를 조금 더 얻으려다 필수 목표를 잃지 마십시오.
- 한 번에 조정할 항목은 **1~3개**로 좁히십시오. 여러 개를 동시에 바꾸면
  어느 것이 효과가 있었는지 알 수 없습니다.
- **표적에 dose_max를 걸지 마십시오.** 근접치료는 선원이 표적 안에 있어
  V150이 50~60%인 것이 정상입니다. 표적 dose_max를 걸면 커버리지가 무너집니다.
- lam_smooth가 0이면 dwell time이 몇 곳에 몰립니다. 0.01~0.2가 적정합니다.
- 한 번에 8개 이하만 조정하십시오. 모든 것을 동시에 바꾸면 원인을 알 수 없습니다.

절대 하지 말 것:
- 이력에서 이미 시도해 실패한 방향을 반복하지 마십시오.
- 필수([필수] 표시) 목표를 희생해 점수를 올리려 하지 마십시오."""


@dataclass
class Proposal:
    reasoning: str
    params: dict                      # OptimizationProblem.set_weights() 형식
    raw: dict = field(default_factory=dict, repr=False)
    clipped: list[str] = field(default_factory=list)
    source: str = "llm"               # "llm" | "heuristic"
    # --- 계측 -----------------------------------------------------------
    # 이 제안을 얻는 데 실제로 쓴 시간. LLM을 켜면 파이프라인 전체 시간의
    # 거의 전부가 여기다 (최적화 자체는 반복당 0.2초 수준). 눈에 보이지
    # 않으면 "최적화가 느리다"로 오인하게 되므로 반드시 기록한다.
    elapsed_s: float = 0.0
    n_calls: int = 0                  # 이 제안에 쓴 LLM 호출 횟수 (재시도 포함)


# ---------------------------------------------------------------------------
def sanitize(raw: dict, valid_structures: list[str]) -> tuple[dict, list[str]]:
    """LLM 출력을 set_weights() 형식으로 바꾸고 범위를 강제한다.

    structured output을 쓰더라도 모델이 스키마를 벗어날 수 있으므로
    (특히 작은 양자화 모델), 여기서 다시 한 번 방어한다.
    """
    notes: list[str] = []
    structures: dict[str, dict] = {}

    for item in raw.get("adjustments") or []:
        if not isinstance(item, dict):
            notes.append("adjustments 항목이 객체가 아님 — 무시")
            continue
        s = item.get("structure")
        f = item.get("field")
        v = item.get("value")
        if s not in valid_structures:
            notes.append(f"알 수 없는 구조물 '{s}' — 무시")
            continue
        if f not in FIELDS:
            notes.append(f"알 수 없는 필드 '{f}' — 무시")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            notes.append(f"{s}.{f} 값이 숫자가 아님 ({v!r}) — 무시")
            continue
        if v != v or v in (float("inf"), float("-inf")):
            notes.append(f"{s}.{f} 값이 NaN/Inf — 무시")
            continue
        lo, hi = LIMITS[f]
        if not (lo <= v <= hi):
            notes.append(f"{s}.{f} {v:g} -> {min(max(v, lo), hi):g} 로 clip")
            v = min(max(v, lo), hi)
        structures.setdefault(s, {})[f] = v

    out: dict = {}
    if structures:
        out["structures"] = structures

    if "lam_smooth" in raw and raw["lam_smooth"] is not None:
        try:
            v = float(raw["lam_smooth"])
            lo, hi = LAM_SMOOTH_LIMITS
            if v != v:
                notes.append("lam_smooth가 NaN — 무시")
            else:
                if not (lo <= v <= hi):
                    notes.append(f"lam_smooth {v:g} -> {min(max(v, lo), hi):g} 로 clip")
                    v = min(max(v, lo), hi)
                out["lam_smooth"] = v
        except (TypeError, ValueError):
            notes.append(f"lam_smooth가 숫자가 아님 ({raw['lam_smooth']!r}) — 무시")

    return out, notes


def apply_trust_region(
    params: dict,
    current: dict,
    max_factor: float,
) -> list[str]:
    """가중치 변경 폭을 현재값의 [1/f, f] 배로 제한한다 (제자리 수정).

    왜 필요한가 — 실측으로 확인한 문제
    ---------------------------------
    실제 LLM(qwen3.6-35b) 실행에서, 모든 필수 목표를 충족한 plan(score 99.0)을
    얻은 직후 "OAR에 여유가 있으니 표적을 더 밀자"며 **OAR 6개의 가중치를
    동시에 10배 낮췄다.** 판단 방향 자체는 옳았지만 폭이 과해 필수 목표가
    한꺼번에 깨졌고(69.4), 이후 4회를 회복에 낭비했다 (호출당 152초).

    그래서 필수 목표를 충족한 상태에서는 코드로 변경 폭을 좁힌다.
    아직 위반 중이라면 넓게 열어 둔다 (큰 가중치가 실제로 필요하다).
    """
    notes: list[str] = []
    if max_factor <= 0:
        return notes
    cur_s = current.get("structures", {})
    for name, fields in (params.get("structures") or {}).items():
        for f in ("weight_under", "weight_over"):
            if f not in fields:
                continue
            base = (cur_s.get(name) or {}).get(f)
            if base is None:
                continue
            base = float(base)
            if base <= 0:
                continue
            lo, hi = base / max_factor, base * max_factor
            v = float(fields[f])
            if not (lo <= v <= hi):
                clipped = min(max(v, lo), hi)
                notes.append(
                    f"{name}.{f} {v:g} -> {clipped:g} "
                    f"(필수 목표 충족 상태이므로 {max_factor:g}배 이내로 제한)"
                )
                fields[f] = clipped
    return notes


def limit_adjustment_count(params: dict, current: dict, max_items: int) -> list[str]:
    """조정 항목 수를 제한한다. 변화 폭이 큰 것부터 남긴다 (제자리 수정).

    필수 목표를 충족한 상태에서 **여러 OAR을 동시에** 움직이면 한꺼번에
    깨진다. 실측: 6개 OAR 가중치를 동시에 낮춰 99.0 -> 69.4.
    한두 개씩 바꿔야 어느 것이 효과가 있었는지도 알 수 있다.
    """
    notes: list[str] = []
    structs = params.get("structures") or {}
    flat: list[tuple[float, str, str, float]] = []
    cur_s = current.get("structures", {})
    for name, fields in structs.items():
        for f, v in fields.items():
            base = (cur_s.get(name) or {}).get(f)
            if base and float(base) > 0:
                ratio = abs(np.log(max(float(v), 1e-12) / float(base)))
            else:
                ratio = float("inf")
            flat.append((ratio, name, f, v))
    if len(flat) <= max_items:
        return notes

    flat.sort(key=lambda x: -x[0])
    keep = flat[:max_items]
    dropped = flat[max_items:]
    new_structs: dict[str, dict] = {}
    for _, name, f, v in keep:
        new_structs.setdefault(name, {})[f] = v
    params["structures"] = new_structs
    notes.append(
        f"조정 {len(flat)}건 -> {max_items}건으로 축소 "
        f"(동시 변경 과다). 보류: "
        + ", ".join(f"{n}.{f}" for _, n, f, _ in dropped[:5])
    )
    return notes


def enforce_no_target_dose_max(params: dict, target_names: tuple[str, ...]) -> list[str]:
    """표적에 dose_max를 거는 제안은 차단한다 (Phase 2에서 실측으로 확인된 함정)."""
    notes = []
    for s in list(params.get("structures", {})):
        if s in target_names and "dose_max" in params["structures"][s]:
            del params["structures"][s]["dose_max"]
            notes.append(f"{s}.dose_max 제안 차단 (표적 hot spot 제약은 커버리지를 무너뜨림)")
            if not params["structures"][s]:
                del params["structures"][s]
    return notes


# ---------------------------------------------------------------------------
def build_user_message(
    score: PlanScore,
    metrics: dict,
    current_params: dict,
    goals_text: str,
    history_text: str,
    iteration: int,
    max_iter: int,
    status_note: str = "",
) -> str:
    head = f"[반복 {iteration}/{max_iter}]"
    if status_note:
        head += f"\n\n>>> {status_note}"
    return f"""{head}

## 임상 목표
{goals_text}

## 현재 plan의 DVH
{metrics_table(metrics)}

## 목표 달성 현황
{score.table()}

## 문제점
{findings_text(score, metrics)}

## 현재 최적화 파라미터
{json.dumps(current_params, ensure_ascii=False, indent=1)}

## 지금까지의 시도
{history_text}

위를 근거로 다음에 시도할 파라미터 조정을 제안하십시오."""


class Planner:
    """LLM 기반 파라미터 제안기. LM Studio가 없으면 휴리스틱으로 폴백한다."""

    def __init__(
        self,
        client: LMStudioClient | None,
        structures: list[str],
        target_names: tuple[str, ...] = ("GTV", "HR_CTV", "IR_CTV"),
        seed: int | None = 0,
        use_schema: bool = True,
    ):
        self.client = client
        self.structures = list(structures)
        self.target_names = target_names
        self.schema = build_schema(self.structures)
        self.seed = seed
        self.use_schema = use_schema
        self.failures = 0
        self.schema_disabled = False   # #1773 회피로 스키마를 끈 적이 있는가
        self.notes: list[str] = []

    # ------------------------------------------------------------------
    def _schema_instruction(self) -> str:
        """스키마를 쓰지 않을 때 프롬프트로 형식을 지시한다."""
        return (
            "\n\n---\n**반드시 아래 형식의 JSON 객체 하나만** 출력하십시오. "
            "설명이나 코드펜스 없이 JSON만.\n"
            "{\n"
            '  "reasoning": "왜 이렇게 조정하는지 2~4문장",\n'
            '  "adjustments": [\n'
            '    {"structure": "<구조물>", "field": "<필드>", "value": <숫자>}\n'
            "  ],\n"
            '  "lam_smooth": <숫자>\n'
            "}\n"
            f"structure 는 다음 중 하나: {', '.join(self.structures)}\n"
            f"field 는 다음 중 하나: {', '.join(FIELDS)}\n"
            "adjustments 는 8개 이하."
        )

    def _call(self, msg: str, with_schema: bool):
        user = msg if with_schema else msg + self._schema_instruction()
        return self.client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            json_schema=self.schema if with_schema else None,
            schema_name="plan_adjustment",
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    def propose(self, max_change_factor: float = 0.0,
                max_adjustments: int = 0, **kw) -> Proposal:
        """다음 파라미터를 제안한다.

        max_change_factor > 0 이면 가중치 변경 폭을 현재값의 그 배수 이내로
        제한한다 (필수 목표를 이미 충족한 상태에서 쓴다).
        """
        current = kw.get("current_params", {})
        if self.client is None:
            p = self._heuristic(**kw)
            p.clipped += apply_trust_region(p.params, current, max_change_factor)
            if max_adjustments:
                p.clipped += limit_adjustment_count(p.params, current, max_adjustments)
            return p

        msg = build_user_message(**kw)
        attempts: list[bool] = []
        if self.use_schema and not self.schema_disabled:
            attempts.append(True)
        attempts.append(False)      # 스키마 없이 프롬프트로만 (reasoning 모델 회피책)

        raw = None
        errors: list[str] = []
        used_schema = None
        from_reasoning = False
        spent = 0.0          # 재시도까지 포함한 총 소요 [s]
        n_calls = 0
        for with_schema in attempts:
            t_call = time.perf_counter()
            n_calls += 1
            try:
                res = self._call(msg, with_schema)
                raw = res.json()
                used_schema = with_schema
                from_reasoning = res.from_reasoning
                spent += time.perf_counter() - t_call
                break
            except (LMStudioError, ValueError) as e:
                spent += time.perf_counter() - t_call
                errors.append(f"{'스키마 사용' if with_schema else '스키마 없이'}: {e}")
                if with_schema:
                    # structured output이 실패하면 이후 반복에서는 시도하지 않는다
                    # (LM Studio 이슈 #1773 — reasoning 모델에서 흔함)
                    #
                    # 이 첫 실패는 **호출 한 번을 통째로 버린다.** reasoning
                    # 모델에서는 호출당 150초대라 그 자체가 2~3분 손해다.
                    # 다음 실행에서 --no-schema 로 건너뛸 수 있게 안내한다.
                    self.schema_disabled = True
                    self.notes.append(
                        f"structured output 실패 -> 이후 스키마 없이 진행합니다 "
                        f"(LM Studio 이슈 #1773 회피). 이 시도에 "
                        f"{time.perf_counter() - t_call:.0f}초를 버렸습니다 — "
                        f"이 모델을 계속 쓴다면 다음부터 --no-schema 를 주십시오"
                    )

        if raw is None:
            self.failures += 1
            p = self._heuristic(**kw)
            p.reasoning = "[LLM 실패 -> 휴리스틱 폴백]\n" + "\n".join(errors) + \
                          "\n" + p.reasoning
            p.elapsed_s, p.n_calls = spent, n_calls
            return p

        params, notes = sanitize(raw, self.structures)
        notes += enforce_no_target_dose_max(params, self.target_names)
        notes += apply_trust_region(params, current, max_change_factor)
        if max_adjustments:
            notes += limit_adjustment_count(params, current, max_adjustments)
        if used_schema is False and self.use_schema:
            notes.insert(0, "스키마 없이 프롬프트 지시로 응답 받음")
        if from_reasoning:
            notes.insert(0, "content가 비어 reasoning_content에서 JSON을 회수함")

        if not params:
            self.failures += 1
            p = self._heuristic(**kw)
            p.reasoning = ("[LLM이 유효한 조정을 내놓지 못함 -> 휴리스틱 폴백] "
                           + str(raw.get("reasoning", ""))[:200] + "\n" + p.reasoning)
            p.clipped = notes
            p.elapsed_s, p.n_calls = spent, n_calls
            return p
        return Proposal(reasoning=str(raw.get("reasoning", "")).strip(),
                        params=params, raw=raw, clipped=notes, source="llm",
                        elapsed_s=spent, n_calls=n_calls)

    # ------------------------------------------------------------------
    def _heuristic(self, score: PlanScore, current_params: dict,
                   metrics: dict | None = None, **_) -> Proposal:  # noqa: D401
        """LLM 없이도 동작하는 규칙 기반 폴백.

        미충족 목표의 가중치를 올린다. 이때 **구조물 부피로 보정**한다 —
        벌점이 전체 복셀의 평균이라 큰 구조물일수록 D2cm3 초과가 희석되기
        때문이다 (Bladder 376 cm^3 에서 2 cm^3 는 0.5 %에 불과).
        필수 목표 위반은 더 크게 움직인다.
        """
        cur = current_params.get("structures", {})
        metrics = metrics or {}
        adj: dict[str, dict] = {}
        lines = []

        def dilution(name: str) -> float:
            """D2cm3 목표에서 2 cm^3 가 전체 부피에서 차지하는 비율의 역수."""
            vol = float((metrics.get(name) or {}).get("volume_cm3", 0.0))
            return max(1.0, vol / 2.0 / 15.0)   # 30 cm^3 를 기준으로 정규화

        misses = [r for r in score.results if not r.ok]
        for r in sorted(misses, key=lambda r: (not r.goal.hard, r.margin_pct))[:4]:
            s = r.goal.structure
            if s not in self.structures:
                continue
            base = 2.0 if abs(r.margin_pct) < 10 else 4.0
            if r.goal.hard:
                base *= 2.0
            if r.goal.kind == "max":
                factor = base * dilution(s)
                w = float(cur.get(s, {}).get("weight_over", 1.0) or 1.0)
                adj.setdefault(s, {})["weight_over"] = min(w * factor, LIMITS["weight_over"][1])
                lines.append(f"{s} 초과 {abs(r.margin_pct):.1f}% -> weight_over x{factor:.1f}")
            else:
                w = float(cur.get(s, {}).get("weight_under", 1.0) or 1.0)
                adj.setdefault(s, {})["weight_under"] = min(w * base,
                                                            LIMITS["weight_under"][1])
                lines.append(f"{s} 미달 {abs(r.margin_pct):.1f}% -> weight_under x{base:g}")

        if not adj:  # 전부 충족 — 여유가 큰 OAR의 압력을 낮춰 표적을 더 밀어본다
            slack = sorted((r for r in score.results if r.ok and r.goal.kind == "max"),
                           key=lambda r: -r.margin_pct)[:2]
            for r in slack:
                s = r.goal.structure
                if s not in self.structures:
                    continue
                w = float(cur.get(s, {}).get("weight_over", 1.0) or 1.0)
                adj.setdefault(s, {})["weight_over"] = max(w * 0.5, 1e-3)
                lines.append(f"{s} 여유 {r.margin_pct:.0f}% -> weight_over x0.5")

        return Proposal(
            reasoning="휴리스틱: " + ("; ".join(lines) if lines else "조정 없음"),
            params={"structures": adj} if adj else {},
            source="heuristic",
        )
