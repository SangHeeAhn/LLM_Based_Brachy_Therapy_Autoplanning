"""에이전트 루프 — observe → analyze → update 반복.

    theta_0 (초기 가중치)
    repeat:
        t       = optimize(problem with theta)      <- 결정론적, 볼록, 전역최적
        metrics = evaluate(t)                       <- kernel만 쓰므로 1초 미만
        score   = score_plan(metrics, goals)
        memory.add(...)
        if 수렴: break
        theta   = guard(planner.propose(...))       <- LLM (또는 휴리스틱)

최종적으로 **메모리에 기록된 best plan**을 반환한다. 마지막 반복이 아니라
best를 쓰므로, LLM이 도중에 나쁜 제안을 해도 결과가 나빠지지 않는다.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from ..optim.objectives import OptimizationProblem
from ..optim.optimizer import optimize_dwell_times
from .evaluator import Goal, PlanScore, score_plan
from .memory import Memory, Step
from .planner import Planner, Proposal

KEY_METRICS = [("HR_CTV", "D90_Gy"), ("Bladder", "D2cm3_Gy"), ("Rectum", "D2cm3_Gy")]


@dataclass
class AgentResult:
    t: np.ndarray
    score: PlanScore
    metrics: dict
    params: dict
    memory: Memory
    iterations: int
    stop_reason: str
    elapsed_s: float
    llm_calls: int = 0
    llm_failures: int = 0
    n_reverts: int = 0
    revert_iterations: list[int] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    # --- 시간 분해 (초) ---------------------------------------------------
    # "최적화가 느리다"는 오인을 막기 위해 어디서 시간이 갔는지 나눠 둔다.
    t_optimize_s: float = 0.0     # 볼록 최적화 (결정론적)
    t_evaluate_s: float = 0.0     # DVH 평가 + 점수
    t_propose_s: float = 0.0      # 제안 (LLM 호출 또는 휴리스틱)
    n_propose_calls: int = 0      # 실제 LLM 호출 횟수 (재시도 포함)

    def timing_lines(self) -> list[str]:
        """사람이 읽을 시간 분해표."""
        other = max(self.elapsed_s - self.t_optimize_s
                    - self.t_evaluate_s - self.t_propose_s, 0.0)
        rows = [("볼록 최적화", self.t_optimize_s),
                ("DVH 평가·점수", self.t_evaluate_s),
                (f"제안(LLM {self.n_propose_calls}회 호출)"
                 if self.n_propose_calls else "제안(휴리스틱)", self.t_propose_s),
                ("기타", other)]
        tot = self.elapsed_s or 1.0
        out = [f"  {'단계':<22}{'시간':>9}{'비중':>8}"]
        for name, v in rows:
            out.append(f"  {name:<22}{v:8.1f}s{v / tot * 100:7.1f}%")
        out.append(f"  {'합계':<22}{self.elapsed_s:8.1f}s{100.0:7.1f}%")
        if self.n_propose_calls and self.iterations:
            out.append(f"  → LLM 호출 평균 "
                       f"{self.t_propose_s / max(self.n_propose_calls, 1):.1f}s/회")
        return out


def _key_metrics(metrics: dict) -> dict:
    out = {}
    for s, m in KEY_METRICS:
        row = metrics.get(s)
        if row and m in row:
            out[f"{s}.{m.split('_')[0]}"] = float(row[m])
    return out


def run_agent(
    ctx,
    problem: OptimizationProblem,
    goals: list[Goal],
    planner: Planner,
    goals_text: str,
    max_iter: int = 12,
    patience: int = 4,
    target_score: float = 99.5,
    t0: np.ndarray | None = None,
    revert_margin: float = 2.0,
    trust_factor: float = 2.0,
    max_adjustments_when_ok: int = 3,
    verbose: bool = True,
    log=print,
) -> AgentResult:
    """에이전트 루프 실행.

    회귀 복구 (revert_margin / trust_factor)
    ---------------------------------------
    실제 LLM 실행에서 관측된 문제: 모든 필수 목표를 충족한 plan을 얻은 뒤
    "OAR에 여유가 있으니 표적을 더 밀자"며 가중치를 과하게 움직여 plan을
    무너뜨리고, 남은 반복을 회복에 낭비했다.

    대응 두 가지
      1) 점수가 최고 기록보다 revert_margin 이상 떨어지면 **최선 파라미터로
         되돌린 뒤** 다음 제안을 받는다. 나쁜 상태 위에 쌓지 않는다.
      2) 필수 목표를 이미 충족한 상태에서는 가중치 변경 폭을 trust_factor 배
         이내로 제한한다 (planner.apply_trust_region).
    """
    memory = Memory()
    proposals: list[Proposal] = []
    started = time.time()
    stop_reason = "max_iter 도달"
    pending: Proposal | None = None
    status_note = ""
    revert_iterations: list[int] = []

    t_opt_total = t_eval_total = t_prop_total = 0.0
    n_prop_calls = 0

    for it in range(1, max_iter + 1):
        params_now = copy.deepcopy(problem.get_weights())

        _t = time.perf_counter()
        r = optimize_dwell_times(problem, t0=t0)
        t_opt = time.perf_counter() - _t
        t_opt_total += t_opt

        _t = time.perf_counter()
        metrics = ctx.evaluate(r.t)
        sc = score_plan(metrics, goals)
        t_eval = time.perf_counter() - _t
        t_eval_total += t_eval

        memory.add(Step(
            iteration=it,
            params=params_now,
            score=sc.total,
            n_met=sc.n_met,
            n_goals=len(sc.results),
            hard_violations=list(sc.hard_violations),
            key_metrics=_key_metrics(metrics),
            reasoning=(pending.reasoning if pending else "초기 파라미터"),
            source=(pending.source if pending else "initial"),
            t=[float(x) for x in r.t],
        ))

        best = memory.best
        if verbose:
            log(f"  #{it:<2} score={sc.total:6.2f} ({sc.n_met}/{len(sc.results)})"
                f"{'  [필수위반]' if sc.hard_violations else ''}"
                f"  KKT={r.kkt_residual:.1e}"
                f"  best={best.score:.2f}@#{best.iteration}"
                f"  [최적화 {t_opt:.2f}s + 평가 {t_eval:.2f}s]")

        if sc.acceptable and sc.total >= target_score:
            stop_reason = f"목표 점수 {target_score} 도달"
            break
        if memory.stagnant_for() >= patience:
            stop_reason = f"{patience}회 연속 개선 없음"
            break
        if it == max_iter:
            break

        # --- 회귀 복구 -------------------------------------------------------
        # **이미 필수 목표를 모두 충족한 plan을 갖고 있을 때만** 되돌린다.
        #
        # 수렴 전 단계의 점수 하락은 탐색의 일부다. 그때까지 되돌리면
        # (a) 결정론적 플래너가 같은 제안을 반복해 교착에 빠지고,
        # (b) 하락을 거쳐 도달했을 더 좋은 해를 놓친다.
        # 실측: 무조건 되돌리게 했더니 99.86에 도달하던 것이 91.22에서 멈췄다.
        #
        # 되돌린 뒤에는 base가 acceptable이므로 trust region(2배 제한)이 걸린다.
        # 따라서 다음 제안은 직전과 다른 폭이 되어 교착이 풀린다.
        status_note = ""
        base_score = sc
        regressed = best.acceptable_step and (
            not sc.acceptable or sc.total < best.score - revert_margin
        )
        if best.iteration != it and regressed:
            problem.load_weights(best.params)
            params_now = copy.deepcopy(problem.get_weights())
            base_metrics = ctx.evaluate(np.asarray(best.t, dtype=float))
            base_score = score_plan(base_metrics, goals)
            metrics = base_metrics
            revert_iterations.append(it)
            status_note = (
                f"직전 변경으로 점수가 {best.score:.1f} -> {sc.total:.1f} 로 "
                f"떨어졌습니다. 그 변경은 되돌렸고, 지금은 최고 기록"
                f"(#{best.iteration}) 상태입니다. 같은 방향으로 다시 크게 움직이지 "
                f"마십시오. 더 작은 폭으로 시도하거나 다른 항목을 건드리십시오."
            )
            if verbose:
                log(f"      [복구] #{best.iteration} 파라미터로 되돌림 "
                    f"({sc.total:.1f} -> {best.score:.1f})")

        # 필수 목표를 충족한 상태라면 변경 폭과 항목 수를 모두 좁힌다
        ok_now = base_score.acceptable
        max_change = trust_factor if ok_now else 0.0
        max_adj = max_adjustments_when_ok if ok_now else 0

        _t = time.perf_counter()
        pending = planner.propose(
            score=base_score,
            metrics=metrics,
            current_params=params_now,
            goals_text=goals_text,
            history_text=memory.prompt_text(),
            iteration=it,
            max_iter=max_iter,
            status_note=status_note,
            max_change_factor=max_change,
            max_adjustments=max_adj,
        )
        t_prop = time.perf_counter() - _t
        t_prop_total += t_prop
        n_prop_calls += pending.n_calls
        proposals.append(pending)
        if verbose:
            tag = "LLM" if pending.source == "llm" else "휴리스틱"
            # 호출이 2회면 스키마 시도가 실패해 재시도한 것이다 — 그 한 번이
            # reasoning 모델에서는 통째로 낭비다. 눈에 보이게 남긴다.
            extra = (f" ({pending.n_calls}회 호출)" if pending.n_calls > 1 else "")
            log(f"      [{tag} {t_prop:.1f}s{extra}] {pending.reasoning[:120]}")
            for n in pending.clipped:
                log(f"      [안전장치] {n}")

        if not pending.params:
            stop_reason = "더 제안할 조정이 없음"
            break
        try:
            problem.set_weights(pending.params)
        except (KeyError, ValueError) as e:
            # sanitize를 통과했는데도 거부되면 그 제안은 버리고 계속한다
            if verbose:
                log(f"      [안전장치] 제안 거부: {e}")
            pending = None

    best = memory.best
    assert best is not None
    t_best = np.asarray(best.t, dtype=float)
    metrics_best = ctx.evaluate(t_best)
    score_best = score_plan(metrics_best, goals)

    client = planner.client
    return AgentResult(
        t=t_best,
        score=score_best,
        metrics=metrics_best,
        params=best.params,
        memory=memory,
        iterations=len(memory),
        stop_reason=stop_reason,
        elapsed_s=time.time() - started,
        llm_calls=(client.n_calls if client else 0),
        llm_failures=planner.failures,
        n_reverts=len(revert_iterations),
        revert_iterations=revert_iterations,
        t_optimize_s=t_opt_total,
        t_evaluate_s=t_eval_total,
        t_propose_s=t_prop_total,
        n_propose_calls=n_prop_calls,
        proposals=proposals,
    )
