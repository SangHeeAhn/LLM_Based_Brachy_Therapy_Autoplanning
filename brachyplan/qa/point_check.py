"""Commissioning 검증 — TPS 결과와 엔진 계산 대조.

Phase 1 완료 기준
  1) RTDOSE dose point 7개 : TPS 대비 ±2 %
  2) 구조물 부피          : 폴리곤 기준 부피 대비 ±2 %
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dose.tg43 import TG43Calculator
from ..io.dicom_loader import PlanInfo
from ..io.struct_to_mask import contour_volume_cm3, mask_volume_cm3


@dataclass
class PointResult:
    name: str
    tps_gy: float
    calc_gy: float

    @property
    def diff_pct(self) -> float:
        return 100.0 * (self.calc_gy - self.tps_gy) / self.tps_gy if self.tps_gy else float("nan")


def check_dose_points(
    calc: TG43Calculator,
    plan: PlanInfo,
    sk: float,
    tolerance_pct: float = 2.0,
) -> tuple[list[PointResult], bool]:
    """RTDOSE dose point를 TG-43으로 재계산해 TPS 값과 비교."""
    results: list[PointResult] = []
    for name, pos in plan.dose_points.items():
        tps = plan.dose_point_tps.get(name)
        if tps is None:
            continue
        calc_gy = float(calc.dose_at_points(pos[None, :], plan.dwells, sk)[0])
        results.append(PointResult(name=name, tps_gy=float(tps), calc_gy=calc_gy))

    passed = bool(results) and all(abs(r.diff_pct) <= tolerance_pct for r in results)
    return results, passed


def format_point_table(results: list[PointResult], tolerance_pct: float = 2.0) -> str:
    lines = [f"{'point':<22}{'TPS(Gy)':>10}{'calc(Gy)':>11}{'diff(%)':>10}  판정",
             "-" * 61]
    for r in results:
        flag = "OK " if abs(r.diff_pct) <= tolerance_pct else "FAIL"
        lines.append(f"{r.name:<22}{r.tps_gy:10.3f}{r.calc_gy:11.3f}{r.diff_pct:10.2f}  {flag}")
    if results:
        diffs = [abs(r.diff_pct) for r in results]
        lines.append("-" * 61)
        lines.append(f"{'평균 |차이|':<22}{'':>10}{'':>11}{np.mean(diffs):10.2f}")
        lines.append(f"{'최대 |차이|':<22}{'':>10}{'':>11}{np.max(diffs):10.2f}")
    return "\n".join(lines)


def check_structure_volumes(
    rtstruct_path: str,
    masks: dict[str, np.ndarray],
    spacing: np.ndarray,
    tolerance_pct: float = 2.0,
    frame=None,
    min_volume_cm3: float = 5.0,
) -> tuple[list[tuple[str, float, float, float, bool]], bool]:
    """래스터화 부피 vs 폴리곤(신발끈) 기준 부피.

    기준값의 한계 — 판정에서 제외하는 경우
    -------------------------------------
    1) 작은 구조물 (< min_volume_cm3): 기준 부피는 "평면 면적 x 중앙값 슬래브
       두께"이므로, 컨투어 장수가 적으면 양 끝단 처리에서 수 % 계통오차가 난다.
       구조물이 작을수록 이 오차의 상대적 비중이 커진다.
    2) 구멍(hole)이 있는 구조물: 기준값은 면적을 단순 합산하므로 과대평가된다.
       이 경우 XOR로 처리하는 래스터화 쪽이 옳다.

    Returns
    -------
    (rows, passed) — rows = [(name, v_ref, v_raster, diff_pct, gated)]
    gated=True 인 항목만 합격 판정에 반영된다.
    """
    rows: list[tuple[str, float, float, float, bool]] = []
    for name, mask in masks.items():
        v_raster = mask_volume_cm3(mask, spacing)
        v_ref = contour_volume_cm3(rtstruct_path, name, frame)
        diff = 100.0 * (v_raster - v_ref) / v_ref if v_ref else float("nan")
        gated = bool(v_ref >= min_volume_cm3 and np.isfinite(diff))
        rows.append((name, v_ref, v_raster, diff, gated))

    gated_rows = [r for r in rows if r[4]]
    ok = bool(gated_rows) and all(abs(r[3]) <= tolerance_pct for r in gated_rows)
    return rows, ok


def format_volume_table(rows, tolerance_pct: float = 2.0,
                        min_volume_cm3: float = 5.0) -> str:
    lines = [f"{'structure':<16}{'polygon(cc)':>13}{'raster(cc)':>12}{'diff(%)':>10}  판정",
             "-" * 64]
    for name, v_ref, v_ras, diff, gated in rows:
        if not gated:
            flag = f"참고 (<{min_volume_cm3:g}cc, 기준값 불안정)"
        else:
            flag = "OK " if abs(diff) <= tolerance_pct else "FAIL"
        lines.append(f"{name:<16}{v_ref:13.2f}{v_ras:12.2f}{diff:10.2f}  {flag}")
    return "\n".join(lines)
