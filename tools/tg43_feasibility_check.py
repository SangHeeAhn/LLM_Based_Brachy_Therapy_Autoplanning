#!/usr/bin/env python3
"""
TG-43 타당성 검증 스크립트 (Phase 0)
====================================
Geneva_Case의 RTPLAN에서 dwell position/time과 S_K를 읽어 TG-43 2D formalism으로
RTDOSE에 기록된 7개 dose point 선량을 재계산하고, Oncentra 결과와 대조한다.

주의: 이 스크립트는 **의도적으로 불완전**하다.
      g_L(r) = 1, F(r, theta) = 1 로 두었으므로 (기하함수 + 붕괴보정 + Lambda 만 사용)
      선원 근처나 축방향 점에서는 수 % 오차가 남는다.
      목적은 "좌표계 / 붕괴보정 / 기하함수 / 단위계가 맞는가"를 확인하는 것이다.
      실제 엔진(brachyplan/dose/tg43.py)에서는 consensus 데이터표를 반드시 넣어야 한다.

사용법:
    python tools/tg43_feasibility_check.py [케이스_디렉토리]
    (기본값: ./Geneva_Case)

요구: pydicom, numpy
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime

import numpy as np
import pydicom

# --- 선원 파라미터 (Phase 0에서 반드시 확정할 것) ---------------------------
LAMBDA = 1.109  # cGy h^-1 U^-1  : Ir-192 mHDR-v2 / Flexisource 계열 consensus 값
L_ACTIVE = 0.35  # cm            : 활성 선원 길이. RTPLAN에 ActiveSourceLength가 없어 가정함
# ---------------------------------------------------------------------------


def geometry_function(r_cm: float, theta_rad: float, L: float = L_ACTIVE) -> float:
    """TG-43U1 선형 선원 기하함수 G_L(r, theta). 단위 cm^-2."""
    r = max(float(r_cm), 1e-6)
    sin_t = np.sin(theta_rad)
    if abs(sin_t) < 1e-6:  # theta = 0 or pi (선원 축 상)
        return 1.0 / max(abs(r * r - L * L / 4.0), 1e-9)
    beta1 = np.arctan2(r * sin_t, r * np.cos(theta_rad) - L / 2.0)
    beta2 = np.arctan2(r * sin_t, r * np.cos(theta_rad) + L / 2.0)
    return abs(beta1 - beta2) / (L * r * sin_t)


G0 = geometry_function(1.0, np.pi / 2.0)  # G_L(r0=1cm, theta0=90deg)


def parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:8], "%Y%m%d")
    except ValueError:
        return None


def extract_dwells(rtplan) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """(위치[mm], dwell time[s], 선원 축 단위벡터) 리스트를 반환.

    RTPLAN에는 선원 방향이 기록되지 않으므로 인접 control point로
    채널 접선을 구해 선원 장축을 근사한다 (pyTG43과 동일한 접근).
    """
    dwells: list[tuple[np.ndarray, float, np.ndarray]] = []
    for setup in rtplan.ApplicationSetupSequence:
        for ch in setup.ChannelSequence:
            cps = ch.BrachyControlPointSequence
            total_time = float(ch.ChannelTotalTime)
            final_w = float(ch.FinalCumulativeTimeWeight)
            if final_w <= 0:
                continue
            pos, times = [], []
            # control point는 (도착, 출발) 쌍으로 기록된다
            for i in range(0, len(cps) - 1, 2):
                p = np.asarray(cps[i].ControlPoint3DPosition, dtype=float)
                w0 = float(cps[i].CumulativeTimeWeight)
                w1 = float(cps[i + 1].CumulativeTimeWeight)
                pos.append(p)
                times.append((w1 - w0) / final_w * total_time)
            if not pos:
                continue
            pos_arr = np.asarray(pos)
            n = len(pos_arr)
            for k in range(n):
                a = pos_arr[max(k - 1, 0)]
                b = pos_arr[min(k + 1, n - 1)]
                vec = b - a
                norm = np.linalg.norm(vec)
                axis = vec / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
                dwells.append((pos_arr[k], times[k], axis))
    return dwells


def point_dose(point_mm: np.ndarray, dwells, sk_treat: float) -> float:
    """TG-43 2D (g_L=1, F=1 근사)로 한 점의 총 선량[Gy]을 계산."""
    dose_cgy = 0.0
    for pos_mm, t_s, axis in dwells:
        v = (point_mm - pos_mm) / 10.0  # mm -> cm
        r = float(np.linalg.norm(v))
        if r < 1e-6:
            continue
        cos_t = float(np.dot(v, axis)) / r
        theta = np.arccos(np.clip(cos_t, -1.0, 1.0))
        dose_cgy += sk_treat * LAMBDA * (geometry_function(r, theta) / G0) * (t_s / 3600.0)
    return dose_cgy / 100.0


def main(case_dir: str) -> int:
    rp_files = glob.glob(os.path.join(case_dir, "RP.*.dcm"))
    rd_files = glob.glob(os.path.join(case_dir, "RD.*.dcm"))
    if not rp_files or not rd_files:
        print(f"[ERROR] RTPLAN 또는 RTDOSE를 찾을 수 없습니다: {case_dir}", file=sys.stderr)
        return 1

    rtplan = pydicom.dcmread(rp_files[0])
    rtdose = pydicom.dcmread(rd_files[0])

    src = rtplan.SourceSequence[0]
    sk_ref = float(src.ReferenceAirKermaRate)          # U = uGy m^2 / h
    half_life = float(src.SourceIsotopeHalfLife)       # days

    ref_date = parse_date(getattr(src, "SourceStrengthReferenceDate", None))
    calc_date = parse_date(getattr(rtdose, "ContentDate", None)) or ref_date
    delta_days = (calc_date - ref_date).days if (ref_date and calc_date) else 0
    sk_treat = sk_ref * np.exp(-np.log(2.0) * delta_days / half_life)

    print(f"선원      : {getattr(src, 'SourceIsotopeName', '?')}, T1/2 = {half_life} d")
    print(f"S_K(ref)  : {sk_ref:.2f} U  @ {getattr(src, 'SourceStrengthReferenceDate', '?')}")
    print(f"붕괴보정  : {delta_days} d  ->  S_K(calc) = {sk_treat:.2f} U")
    print(f"Lambda    : {LAMBDA} cGy/h/U   (L_active = {L_ACTIVE} cm)")

    dwells = extract_dwells(rtplan)
    print(f"dwell 수  : {len(dwells)},  총 dwell time = {sum(d[1] for d in dwells):.1f} s\n")

    # RTDOSE의 dose point 좌표 + RTPLAN의 DoseReferenceSequence 선량
    names = {s.ROINumber: s.ROIName for s in rtdose.StructureSetROISequence}
    tps_dose = {
        d.DoseReferenceDescription: float(d.TargetPrescriptionDose)
        for d in rtplan.DoseReferenceSequence
        if "TargetPrescriptionDose" in d
    }

    print(f"{'point':<22}{'TPS(Gy)':>10}{'calc(Gy)':>11}{'diff(%)':>10}")
    print("-" * 53)
    diffs = []
    for roi in rtdose.ROIContourSequence:
        name = names.get(roi.ReferencedROINumber, f"ROI{roi.ReferencedROINumber}")
        p = np.asarray(roi.ContourSequence[0].ContourData, dtype=float)[:3]
        calc = point_dose(p, dwells, sk_treat)
        tps = tps_dose.get(name)
        if tps is None:
            print(f"{name:<22}{'-':>10}{calc:11.3f}{'-':>10}")
            continue
        diff = 100.0 * (calc - tps) / tps
        diffs.append(abs(diff))
        print(f"{name:<22}{tps:10.3f}{calc:11.3f}{diff:10.1f}")

    if diffs:
        print("-" * 53)
        print(f"{'평균 |차이|':<22}{'':>10}{'':>11}{np.mean(diffs):10.1f}")
        print(f"{'최대 |차이|':<22}{'':>10}{'':>11}{np.max(diffs):10.1f}")
        print(
            "\n해석: r ~ 2 cm 점(Point A 등)에서 1 % 이내면 좌표계/붕괴보정/기하함수/단위가 정확한 것.\n"
            "      선원 근처 점의 잔차는 g_L(r), F(r,theta)를 생략한 데서 오는 예상된 오차이다."
        )
    return 0


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_case = os.path.join(root, "Geneva_Case")
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default_case))
