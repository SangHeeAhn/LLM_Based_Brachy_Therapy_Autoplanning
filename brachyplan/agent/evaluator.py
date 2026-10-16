"""임상 목표 정의, plan 스코어링, LLM에게 줄 자연어 피드백 생성.

스코어 설계
-----------
각 목표를 0~1 부분점수로 환산해 가중평균한다 (백분율로 보고).

  최소 목표 (표적 D90 >= limit)
      v >= limit           -> 1.0
      v <  limit           -> v / limit        (선형, 0에서 0)
  최대 목표 (OAR D2cc <= limit)
      v <= limit           -> 1.0
      limit < v <= 2*limit -> (2*limit - v)/limit
      v >  2*limit         -> 0.0

`hard=True` 목표를 위반하면 부분점수에 `HARD_VIOLATION_FACTOR`(0.25)를 곱한다.

  왜 필요한가 — 실측으로 확인한 문제
  ---------------------------------
  이 항이 없으면 한계를 1 % 넘긴 필수 목표의 부분점수가 0.99가 되어,
  종합 점수가 99.85점으로 나온다. 에이전트 입장에서는 "거의 다 됐다"로 보여
  위반을 고칠 유인이 사라진다. 실제로 mock 실행에서 Bowel이 0.9 % 초과인 채
  8회 반복 내내 방치됐다. 필수 위반에 가파른 벌점을 주면 점수가 크게 떨어져
  에이전트가 그것부터 처리한다.

  아울러 `PlanScore.acceptable`은 점수와 무관하게 위반 유무만 본다.
  LLM이 점수를 올려 필수 제약을 거래하는 것을 막기 위해서다.

수치 출처
---------
EMBRACE-II의 구체적 D2cm3 수치는 여기에 하드코딩하지 않는다.
기본값은 **임상 plan의 실측값**을 상한으로 쓰고 (Phase 2와 동일한 방침),
기관 프로토콜을 쓰려면 goals JSON을 넘긴다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

TARGET_NAMES = ("GTV", "HR_CTV", "IR_CTV")

# 필수(hard) 목표를 위반했을 때 부분점수에 곱하는 계수. 위 docstring 참조.
HARD_VIOLATION_FACTOR = 0.25


@dataclass
class Goal:
    structure: str
    metric: str            # "D90_Gy", "D98_Gy", "D2cm3_Gy", "V150_pct", ...
    kind: str              # "min" | "max"
    limit: float
    weight: float = 1.0
    hard: bool = False
    label: str = ""

    def describe(self) -> str:
        op = ">=" if self.kind == "min" else "<="
        tag = " [필수]" if self.hard else ""
        return f"{self.structure} {self.metric} {op} {self.limit:.3g}{tag}"

    def sub_score(self, value: float) -> float:
        s = self._raw_sub_score(value)
        if self.hard and not self.satisfied(value):
            s *= HARD_VIOLATION_FACTOR
        return s

    def _raw_sub_score(self, value: float) -> float:
        if value != value:                     # NaN
            return 0.0
        lim = self.limit
        if self.kind == "min":
            if lim <= 0:
                return 1.0
            return 1.0 if value >= lim else max(0.0, value / lim)
        if lim <= 0:
            return 1.0 if value <= 0 else 0.0
        if value <= lim:
            return 1.0
        return max(0.0, (2.0 * lim - value) / lim)

    def satisfied(self, value: float) -> bool:
        if value != value:
            return False
        return value >= self.limit if self.kind == "min" else value <= self.limit


@dataclass
class GoalResult:
    goal: Goal
    value: float
    sub_score: float
    ok: bool

    @property
    def margin_pct(self) -> float:
        """목표 대비 여유(+) 또는 초과(-) 비율 [%]."""
        if not self.goal.limit:
            return 0.0
        d = (self.value - self.goal.limit) / self.goal.limit * 100.0
        return d if self.goal.kind == "min" else -d


@dataclass
class PlanScore:
    total: float                  # 0~100
    results: list[GoalResult]
    hard_violations: list[str]

    @property
    def acceptable(self) -> bool:
        return not self.hard_violations

    @property
    def n_met(self) -> int:
        return sum(1 for r in self.results if r.ok)

    def table(self) -> str:
        lines = [f"{'목표':<34}{'값':>9}{'기준':>9}{'여유%':>8}{'점수':>7}  판정",
                 "-" * 76]
        for r in self.results:
            lines.append(
                f"{r.goal.describe():<34}{r.value:9.3f}{r.goal.limit:9.3f}"
                f"{r.margin_pct:8.1f}{r.sub_score:7.2f}  {'OK' if r.ok else 'MISS'}"
            )
        lines.append("-" * 76)
        lines.append(f"{'종합 점수':<34}{self.total:9.1f} / 100      "
                     f"({self.n_met}/{len(self.results)} 충족)")
        if self.hard_violations:
            lines.append(f"필수 목표 위반: {', '.join(self.hard_violations)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def score_plan(metrics: dict[str, dict[str, float]], goals: list[Goal]) -> PlanScore:
    results: list[GoalResult] = []
    hard_viol: list[str] = []
    wsum = 0.0
    acc = 0.0
    for g in goals:
        row = metrics.get(g.structure)
        if row is None or g.metric not in row:
            continue
        v = float(row[g.metric])
        s = g.sub_score(v)
        ok = g.satisfied(v)
        results.append(GoalResult(goal=g, value=v, sub_score=s, ok=ok))
        acc += s * g.weight
        wsum += g.weight
        if g.hard and not ok:
            hard_viol.append(g.describe())
    total = 100.0 * acc / wsum if wsum else 0.0
    return PlanScore(total=total, results=results, hard_violations=hard_viol)


# ---------------------------------------------------------------------------
def goals_from_clinical(
    clinical: dict[str, dict[str, float]],
    prescription_gy: float,
    target: str = "HR_CTV",
    secondary: str | None = "IR_CTV",
    oars: tuple[str, ...] = ("Bladder", "Rectum", "Sigmoid", "Bowel", "Urethra", "Anus"),
) -> list[Goal]:
    """임상 plan 실측값을 기준으로 목표 세트를 만든다.

    - 표적 D90 >= 처방선량        : 필수 (실제 임상 요구)
    - 표적 D90 >= 임상 plan 값    : 가점 (동등 이상인지)
    - OAR D2cm3 <= 임상 plan 값   : 필수 (동등 이하인지)
    - 부표적 D90 >= 임상 plan 값  : 가점
    """
    goals: list[Goal] = [
        Goal(target, "D90_Gy", "min", prescription_gy, weight=3.0, hard=True),
    ]
    if target in clinical and "D90_Gy" in clinical[target]:
        goals.append(Goal(target, "D90_Gy", "min", clinical[target]["D90_Gy"], weight=2.0))
    if secondary and secondary in clinical and "D90_Gy" in clinical[secondary]:
        goals.append(Goal(secondary, "D90_Gy", "min",
                          clinical[secondary]["D90_Gy"], weight=1.0))
    for n in oars:
        row = clinical.get(n)
        if row and "D2cm3_Gy" in row:
            goals.append(Goal(n, "D2cm3_Gy", "max", row["D2cm3_Gy"],
                              weight=1.5, hard=True))
    return goals


def save_goals(goals: list[Goal], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(g) for g in goals], fh, indent=1, ensure_ascii=False)


def load_goals(path: str) -> list[Goal]:
    with open(path, encoding="utf-8") as fh:
        return [Goal(**d) for d in json.load(fh)]


# ---------------------------------------------------------------------------
def findings_text(score: PlanScore, metrics: dict[str, dict[str, float]],
                  max_items: int = 8) -> str:
    """LLM이 읽을 문제점 요약. 위반이 큰 순서로 정렬한다."""
    misses = sorted((r for r in score.results if not r.ok),
                    key=lambda r: r.margin_pct)
    if not misses:
        lines = ["모든 임상 목표를 충족했습니다."]
        slack = sorted((r for r in score.results if r.goal.kind == "max"),
                       key=lambda r: -r.margin_pct)[:3]
        if slack:
            lines.append("여유가 가장 큰 OAR: " + ", ".join(
                f"{r.goal.structure} {r.margin_pct:+.0f}%" for r in slack))
        return "\n".join(lines)

    lines = ["미충족 목표 (초과가 큰 순):"]
    for r in misses[:max_items]:
        op = "미달" if r.goal.kind == "min" else "초과"
        lines.append(
            f"  - {r.goal.structure} {r.goal.metric} = {r.value:.3f} "
            f"(기준 {r.goal.limit:.3f}, {abs(r.margin_pct):.1f}% {op})"
            + (" [필수]" if r.goal.hard else "")
        )
    ok_slack = sorted((r for r in score.results if r.ok and r.goal.kind == "max"),
                      key=lambda r: -r.margin_pct)[:3]
    if ok_slack:
        lines.append("여유가 있는 OAR (가중치를 낮춰도 되는 후보): " + ", ".join(
            f"{r.goal.structure} {r.margin_pct:+.0f}%" for r in ok_slack))
    return "\n".join(lines)


def metrics_table(metrics: dict[str, dict[str, float]],
                  order: list[str] | None = None) -> str:
    """LLM 프롬프트에 넣을 DVH 표 (토큰을 아끼려 필요한 열만)."""
    names = order or list(metrics)
    lines = [f"{'structure':<10}{'vol(cc)':>9}{'D90':>8}{'D98':>8}{'V150%':>8}{'D2cc':>8}"]
    for n in names:
        r = metrics.get(n)
        if not r:
            continue
        g = lambda k: r.get(k, float("nan"))  # noqa: E731
        lines.append(
            f"{n:<10}{r.get('volume_cm3', 0):9.1f}"
            f"{g('D90_Gy'):8.2f}{g('D98_Gy'):8.2f}"
            f"{g('V150_pct'):8.1f}{g('D2cm3_Gy'):8.2f}"
        )
    return "\n".join(lines).replace("nan", "  - ")
