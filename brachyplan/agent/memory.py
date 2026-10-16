"""Memory — 파라미터 → 결과 궤적 저장.

LLM이 같은 실수를 반복하지 않도록 최근 이력을 프롬프트에 넣는다.
GPT-RadPlan(arXiv:2406.15609)과 DRL 기반 자동계획(arXiv:2506.11957)이
공통으로 쓰는 구조다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Step:
    iteration: int
    params: dict                       # 이 반복에서 사용한 전체 파라미터
    score: float
    n_met: int
    n_goals: int
    hard_violations: list[str]
    key_metrics: dict                  # 요약용 소수 지표
    reasoning: str = ""                # 이 파라미터를 만든 근거 (직전 제안)
    source: str = ""                   # llm | heuristic | initial
    t: list[float] = field(default_factory=list, repr=False)

    @property
    def acceptable_step(self) -> bool:
        """이 스텝이 필수 목표를 모두 지켰는가."""
        return not self.hard_violations

    def summary(self) -> str:
        km = " ".join(f"{k}={v:.2f}" for k, v in self.key_metrics.items())
        flag = " [필수위반]" if self.hard_violations else ""
        return (f"#{self.iteration} score={self.score:.1f} "
                f"({self.n_met}/{self.n_goals}){flag}  {km}")


class Memory:
    def __init__(self, keep: int = 5):
        self.keep = keep
        self.steps: list[Step] = []

    def __len__(self) -> int:
        return len(self.steps)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    @property
    def best(self) -> Step | None:
        if not self.steps:
            return None
        # 필수 목표를 지킨 plan을 우선하고, 그 안에서 점수가 높은 것
        ok = [s for s in self.steps if not s.hard_violations]
        pool = ok or self.steps
        return max(pool, key=lambda s: s.score)

    def improved(self, min_delta: float = 0.1) -> bool:
        """직전 대비 점수가 유의하게 올랐는가."""
        if len(self.steps) < 2:
            return True
        return self.steps[-1].score > self.steps[-2].score + min_delta

    def stagnant_for(self, min_delta: float = 0.1) -> int:
        """최고점 이후 개선 없이 지난 반복 수."""
        if not self.steps:
            return 0
        best_i = max(range(len(self.steps)), key=lambda i: self.steps[i].score)
        return len(self.steps) - 1 - best_i

    # ------------------------------------------------------------------
    def prompt_text(self) -> str:
        """LLM 프롬프트에 넣을 이력 요약."""
        if not self.steps:
            return "(아직 없음 — 첫 시도입니다)"
        recent = self.steps[-self.keep:]
        lines = []
        for s in recent:
            lines.append(s.summary())
            if s.reasoning:
                lines.append(f"    근거: {s.reasoning[:160]}")
            diff = self._param_diff(s)
            if diff:
                lines.append(f"    바꾼 것: {diff}")
        b = self.best
        if b and b is not recent[-1]:
            lines.append(f"최고 기록: #{b.iteration} score={b.score:.1f}")
        return "\n".join(lines)

    def _param_diff(self, step: Step) -> str:
        i = self.steps.index(step)
        if i == 0:
            return ""
        prev = self.steps[i - 1].params.get("structures", {})
        cur = step.params.get("structures", {})
        out = []
        for s, fields in cur.items():
            for f, v in fields.items():
                pv = (prev.get(s) or {}).get(f)
                if pv is None or v is None:
                    continue
                if abs(float(v) - float(pv)) > 1e-9:
                    out.append(f"{s}.{f} {pv:g}->{v:g}")
        return ", ".join(out[:6])

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps([asdict(s) for s in self.steps], ensure_ascii=False, indent=1)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def score_curve(self) -> np.ndarray:
        return np.array([s.score for s in self.steps], dtype=float)
