"""박스제약 dwell time 역최적화.

    min_t  J(t)    s.t.  0 <= t <= t_max

J는 볼록·C^1 이므로 전역 최적해가 유일한 값을 갖는다 (objectives.py 참조).

백엔드
------
* "lbfgsb"  : scipy.optimize.minimize(method="L-BFGS-B"). 기본값, 가장 빠름.
* "fista"   : 투영 가속경사법 (Beck & Teboulle). numpy만으로 동작하는 폴백.
              scipy가 없는 임상 PC에서도 돌아가야 하므로 유지한다.

두 백엔드는 같은 해로 수렴해야 하며, 테스트에서 이를 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objectives import OptimizationProblem

try:  # scipy는 선택 의존성
    from scipy.optimize import minimize as _scipy_minimize
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


@dataclass
class OptResult:
    t: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    backend: str
    kkt_residual: float
    message: str = ""

    def __str__(self) -> str:
        return (f"[{self.backend}] J={self.objective:.6g}  iter={self.n_iter}  "
                f"KKT={self.kkt_residual:.3e}  "
                f"{'converged' if self.converged else 'NOT converged'}")


# ---------------------------------------------------------------------------
def kkt_residual(problem: OptimizationProblem, t: np.ndarray) -> float:
    """박스제약 1차 최적성 잔차 (projected gradient norm, 무한대 노름).

    내부점(0 < t < t_max) : |g| 가 0이어야 한다
    하한(t = 0)           : g >= 0 이어야 한다  -> 위반량 = max(0, -g)
    상한(t = t_max)       : g <= 0 이어야 한다  -> 위반량 = max(0,  g)
    """
    _, g = problem.value_and_grad(t)
    lo = t <= 1e-12
    hi = t >= problem.t_max - 1e-12
    r = g.copy()
    r[lo] = np.minimum(g[lo], 0.0)   # 하한에서 음의 gradient만 위반
    r[hi] = np.maximum(g[hi], 0.0)
    return float(np.max(np.abs(r))) if len(r) else 0.0


def _project(t: np.ndarray, t_max: float) -> np.ndarray:
    return np.clip(t, 0.0, t_max)


# ---------------------------------------------------------------------------
def _solve_fista(
    problem: OptimizationProblem,
    t0: np.ndarray,
    max_iter: int,
    tol: float,
) -> OptResult:
    """투영 가속경사법 + 역추적 선탐색 (Beck & Teboulle 2009)."""
    t = _project(np.asarray(t0, dtype=float), problem.t_max)
    y = t.copy()
    theta = 1.0
    J, g = problem.value_and_grad(y)

    # Lipschitz 상수 초기 추정
    L = max(float(np.linalg.norm(g)) / max(float(np.linalg.norm(t)), 1.0), 1e-6)

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        Jy, gy = problem.value_and_grad(y)

        # 역추적: J(t_new) <= Jy + gy.(t_new-y) + L/2 |t_new-y|^2
        for _ in range(60):
            t_new = _project(y - gy / L, problem.t_max)
            step = t_new - y
            Jn = problem.value(t_new)
            if Jn <= Jy + float(gy @ step) + 0.5 * L * float(step @ step) + 1e-14:
                break
            L *= 2.0
        else:  # pragma: no cover
            break

        theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta ** 2))
        y = t_new + ((theta - 1.0) / theta_new) * (t_new - t)
        y = _project(y, problem.t_max)

        delta = float(np.max(np.abs(t_new - t)))
        t, theta = t_new, theta_new
        L = max(L * 0.9, 1e-9)   # 너무 커진 L을 서서히 완화

        if delta < tol:
            break

    J = problem.value(t)
    res = kkt_residual(problem, t)
    return OptResult(t=t, objective=J, n_iter=n_iter, converged=res < 1e-4,
                     backend="fista", kkt_residual=res)


def _solve_lbfgsb(
    problem: OptimizationProblem,
    t0: np.ndarray,
    max_iter: int,
    tol: float,
) -> OptResult:
    fun = problem.value_and_grad
    bounds = [(0.0, problem.t_max)] * problem.n_dwell
    r = _scipy_minimize(
        fun, _project(np.asarray(t0, float), problem.t_max),
        jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-16, "gtol": tol, "maxcor": 30},
    )
    res = kkt_residual(problem, r.x)
    return OptResult(t=np.asarray(r.x), objective=float(r.fun), n_iter=int(r.nit),
                     converged=bool(r.success) and res < 1e-3,
                     backend="lbfgsb", kkt_residual=res, message=str(r.message))


# ---------------------------------------------------------------------------
def optimize_dwell_times(
    problem: OptimizationProblem,
    t0: np.ndarray | None = None,
    backend: str = "auto",
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> OptResult:
    """dwell time 최적화.

    Parameters
    ----------
    t0      : 초기값 [s]. None이면 모든 dwell을 t_max/10 으로 시작.
    backend : "auto" | "lbfgsb" | "fista"
    """
    n = problem.n_dwell
    if t0 is None:
        t0 = np.full(n, problem.t_max / 10.0)
    t0 = np.asarray(t0, dtype=float)
    if t0.shape != (n,):
        raise ValueError(f"t0 형상 불일치: {t0.shape} != ({n},)")

    if backend == "auto":
        backend = "lbfgsb" if HAVE_SCIPY else "fista"
    if backend == "lbfgsb" and not HAVE_SCIPY:
        raise RuntimeError("scipy가 없습니다. backend='fista'를 쓰십시오.")

    if backend == "lbfgsb":
        return _solve_lbfgsb(problem, t0, max_iter, tol)
    if backend == "fista":
        return _solve_fista(problem, t0, max_iter, tol)
    raise ValueError(f"알 수 없는 backend: {backend}")


def check_gradient(
    problem: OptimizationProblem,
    t: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """해석적 gradient vs 중심차분. 최대 상대오차를 반환한다."""
    t = np.asarray(t, dtype=float)
    _, g = problem.value_and_grad(t)
    worst = 0.0
    for k in range(len(t)):
        tp, tm = t.copy(), t.copy()
        tp[k] += eps
        tm[k] -= eps
        num = (problem.value(tp) - problem.value(tm)) / (2 * eps)
        denom = max(abs(num), abs(g[k]), 1e-8)
        worst = max(worst, abs(num - g[k]) / denom)
    return worst
