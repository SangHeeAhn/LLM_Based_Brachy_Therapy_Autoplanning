"""Phase 2 파이프라인 — 케이스 로딩부터 최적화 문제 구성까지.

run_phase2.py 와 Phase 3의 LLM 에이전트가 공유하는 준비 코드.
에이전트 루프는 이 모듈이 만든 `PlanContext`를 반복해서 재사용한다
(kernel은 한 번만 계산하면 되므로 반복이 매우 빠르다).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..dose.dvh import evaluate_plan
from ..dose.kernel import DoseGrid, compute_kernel, make_grid
from ..dose.source_data import SourceData
from ..dose.tg43 import TG43Calculator
from ..io.dicom_loader import PlanInfo, find_files, load_plan
from ..io.struct_to_mask import ContourFrame, build_masks, contour_frame
from .objectives import OptimizationProblem, StructureObjective

DEFAULT_TARGETS = ("GTV", "HR_CTV", "IR_CTV")


@dataclass
class PlanContext:
    """한 케이스에 대해 한 번만 준비하면 되는 모든 것."""

    plan: PlanInfo
    calc: TG43Calculator
    grid: DoseGrid
    frame: ContourFrame
    masks: dict[str, np.ndarray]
    sk: float
    prescription_gy: float
    kernels: dict[str, np.ndarray] = field(default_factory=dict)  # 구조물별 (n_dwell, n_vox)
    voxel_index: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def clinical_t(self) -> np.ndarray:
        """RTPLAN에 기록된 임상 dwell time [s]."""
        return np.array([d.time_s for d in self.plan.dwells], dtype=float)

    @property
    def channel_of(self) -> np.ndarray:
        return np.array([d.channel for d in self.plan.dwells], dtype=int)

    # ------------------------------------------------------------------
    def dose_grid_from_t(self, t: np.ndarray) -> np.ndarray:
        """dwell time -> 전체 3D 선량 [Gy]. (평가/리포트용, 느림)"""
        from ..dose.kernel import compute_dose_grid
        dwells = []
        for d, ti in zip(self.plan.dwells, np.asarray(t, dtype=float)):
            dwells.append(type(d)(position=d.position, time_s=float(ti),
                                  axis=d.axis, channel=d.channel))
        return compute_dose_grid(self.calc, self.grid, dwells, self.sk)

    def structure_doses(self, t: np.ndarray) -> dict[str, np.ndarray]:
        """dwell time -> 구조물별 복셀 선량 [Gy]. (최적화 루프용, 빠름)"""
        t = np.asarray(t, dtype=float)
        return {name: K.T @ t for name, K in self.kernels.items()}

    def evaluate(self, t: np.ndarray) -> dict[str, dict[str, float]]:
        """kernel 기반 DVH 지표 — 전체 그리드 재계산 없이 즉시."""
        from ..dose.dvh import DVH
        vv = self.grid.voxel_volume_cm3
        doses = self.structure_doses(t)
        out: dict[str, dict[str, float]] = {}
        for name, d in doses.items():
            if len(d) == 0:
                continue
            dvh = DVH(name, d, vv)
            row = {"volume_cm3": dvh.volume_cm3, "Dmean_Gy": dvh.mean}
            if name in DEFAULT_TARGETS:
                row["D90_Gy"] = dvh.D_percent(90)
                row["D98_Gy"] = dvh.D_percent(98)
                row["V100_pct"] = dvh.V_percent_of(self.prescription_gy, 100)
                row["V150_pct"] = dvh.V_percent_of(self.prescription_gy, 150)
                row["V200_pct"] = dvh.V_percent_of(self.prescription_gy, 200)
            else:
                row["D2cm3_Gy"] = dvh.D_cc(2.0)
                row["D01cm3_Gy"] = dvh.D_cc(0.1)
            out[name] = row
        return out


# ---------------------------------------------------------------------------
def load_context(
    case_dir: str,
    source_name: str = "ir192_mHDR_v2_clrp",
    spacing_mm: float = 2.0,
    margin_mm: float = 15.0,
    decay_to: str | None = None,
    roi_names: list[str] | None = None,
) -> PlanContext:
    """케이스를 읽어 최적화에 필요한 모든 것을 준비한다."""
    files = find_files(case_dir)
    if not files["RTSTRUCT"] or not files["RTPLAN"]:
        raise FileNotFoundError(f"RTSTRUCT/RTPLAN이 없습니다: {case_dir}")
    rs_path = files["RTSTRUCT"][0]
    rd_path = files["RTDOSE"][0] if files["RTDOSE"] else None

    plan = load_plan(files["RTPLAN"][0], rd_path)
    calc = TG43Calculator(SourceData.load(source_name))
    calc_date = datetime.strptime(decay_to, "%Y%m%d") if decay_to else None
    sk = plan.sk_at(calc_date)

    frame = contour_frame(rs_path)

    import pydicom
    rs = pydicom.dcmread(rs_path)
    pts = [np.asarray(c.ContourData, float).reshape(-1, 3)
           for roi in rs.get("ROIContourSequence", [])
           for c in roi.get("ContourSequence", [])]
    pts.append(np.array([d.position for d in plan.dwells]))
    grid = make_grid(np.vstack(pts), rotation=frame.rotation,
                     spacing_mm=spacing_mm, margin_mm=margin_mm)

    masks = build_masks(rs_path, grid, roi_names)

    return PlanContext(
        plan=plan, calc=calc, grid=grid, frame=frame, masks=masks, sk=sk,
        prescription_gy=plan.fraction_dose_gy or 5.0,
    )


def build_kernels(ctx: PlanContext, names: list[str] | None = None) -> None:
    """구조물별 영향행렬 A [Gy/s]를 계산해 ctx에 채운다.

    복셀 수가 크면 메모리를 많이 쓴다 (n_dwell x n_vox x 8 bytes).
    2 mm 그리드 기준 본 케이스는 19 x ~90k = 약 14 MB.

    이미 만들어 둔 구조물은 **다시 계산하지 않는다.** make_problem()이
    반복 호출돼도 kernel 계산이 중복되지 않도록 하기 위해서다.
    """
    for name in (names or list(ctx.masks)):
        if name in ctx.kernels:          # 이미 있음 — 재계산하지 않는다
            continue
        mask = ctx.masks[name]
        idx = np.flatnonzero(mask.ravel())
        if len(idx) == 0:
            continue
        # 전체 그리드 좌표를 만들지 않고 필요한 복셀만 뽑는다.
        # (관심 복셀은 전체의 10% 미만이다)
        pts = ctx.grid.points_at(idx)
        ctx.kernels[name] = compute_kernel(ctx.calc, pts, ctx.plan.dwells, ctx.sk)
        ctx.voxel_index[name] = idx


def kernel_memory_mb(ctx: PlanContext) -> float:
    return sum(K.nbytes for K in ctx.kernels.values()) / 1e6


# ---------------------------------------------------------------------------
def clinical_oar_limits(ctx: PlanContext, t: np.ndarray | None = None,
                        metric: str = "D2cm3_Gy") -> dict[str, float]:
    """임상 plan의 OAR 지표값을 그대로 목표 상한으로 쓴다.

    EMBRACE-II의 구체 수치를 임의로 적어 넣지 않기 위한 방법.
    "임상 plan과 동등 이상"이라는 Phase 2 완료 기준과도 정확히 일치한다.
    """
    t = ctx.clinical_t if t is None else t
    table = ctx.evaluate(t)
    return {name: row[metric] for name, row in table.items()
            if name not in DEFAULT_TARGETS and metric in row}


def iso_target_match(
    ctx: PlanContext,
    problem: OptimizationProblem,
    target: str,
    target_d90_gy: float,
    oar_names: list[str],
    w_lo: float = 1.0,
    w_hi: float = 1e5,
    tol_gy: float = 0.005,
    max_iter: int = 40,
) -> tuple[np.ndarray, float, int]:
    """표적 D90을 지정값에 **맞춘 뒤** OAR을 비교하기 위한 iso-target 탐색.

    왜 필요한가
    -----------
    "최적화 plan의 D90이 임상보다 높은가"는 공정한 비교가 아니다. 임상 plan이
    처방(5 Gy)보다 높은 D90(5.57 Gy)을 준 것은 그만큼 OAR 선량을 지불한
    결과이기 때문이다. Pareto 곡선 위의 서로 다른 점을 비교하는 셈이다.

    표준적인 비교법은 **표적 커버리지를 동일하게 맞춘 뒤 OAR을 보는 것**이다.
    OAR 가중치를 올리면 D90이 단조 감소하므로, 이분법으로 D90을 맞출 수 있다.

    Returns
    -------
    (t, 사용된 oar_weight, 반복 횟수)
    """
    from .optimizer import optimize_dwell_times

    def d90_at(w: float) -> tuple[float, np.ndarray]:
        for o in problem.objectives:
            if o.name in oar_names:
                o.weight_over = w
        r = optimize_dwell_times(problem)
        return ctx.evaluate(r.t)[target]["D90_Gy"], r.t

    d_lo, t_lo = d90_at(w_lo)     # 가중치 작음 -> D90 높음
    d_hi, t_hi = d90_at(w_hi)     # 가중치 큼   -> D90 낮음
    if d_lo < target_d90_gy:
        return t_lo, w_lo, 0      # 가장 느슨하게 해도 목표에 못 미침
    if d_hi > target_d90_gy:
        return t_hi, w_hi, 0      # 가장 빡빡하게 해도 목표를 넘김

    t_best = t_lo
    w_best = w_lo
    n = 0
    for n in range(1, max_iter + 1):
        w_mid = float(np.sqrt(w_lo * w_hi))     # 로그 스케일 이분법
        d_mid, t_mid = d90_at(w_mid)
        # 목표 이상을 유지하는 해 중 가장 가중치가 큰 것을 채택
        if d_mid >= target_d90_gy:
            w_best, t_best = w_mid, t_mid
            w_lo = w_mid
        else:
            w_hi = w_mid
        if abs(d_mid - target_d90_gy) < tol_gy or w_hi / w_lo < 1.001:
            break
    return t_best, w_best, n


def make_problem(
    ctx: PlanContext,
    objectives: list[StructureObjective],
    t_max: float = 60.0,
    lam_smooth: float = 0.0,
    lam_total: float = 0.0,
    total_time_ref: float | None = None,
) -> OptimizationProblem:
    needed = {o.name for o in objectives}
    missing = needed - set(ctx.kernels)
    if missing:
        build_kernels(ctx, sorted(missing))
    return OptimizationProblem(
        kernels={n: ctx.kernels[n] for n in needed},
        objectives=objectives,
        channel_of=ctx.channel_of,
        t_max=t_max,
        lam_smooth=lam_smooth,
        lam_total=lam_total,
        total_time_ref=total_time_ref,
    )
