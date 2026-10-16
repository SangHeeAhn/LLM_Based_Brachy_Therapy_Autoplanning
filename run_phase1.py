#!/usr/bin/env python3
"""Phase 1 실행 — TG-43 엔진 commissioning + dose grid/DVH 생성.

사용법:
    python run_phase1.py                              # 기본: ./Geneva_Case, 2mm
    python run_phase1.py --spacing 1.0
    python run_phase1.py --skip-grid                  # dose point 검증만
    python run_phase1.py --axial-grid                 # (비교용) 축방향 그리드 강제
    python run_phase1.py --decay-to 20260715          # 특정 날짜로 붕괴보정
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brachyplan.dose import (  # noqa: E402
    SourceData, TG43Calculator, compute_dose_grid, evaluate_plan, make_grid,
)
from brachyplan.io import (  # noqa: E402
    build_masks, check_frame_of_reference, contour_frame, find_files, load_plan,
)
from brachyplan.qa.point_check import (  # noqa: E402
    check_dose_points, check_structure_volumes, format_point_table, format_volume_table,
)

TARGETS = ("GTV", "HR_CTV", "IR_CTV")


def all_contour_points(rtstruct_path: str) -> np.ndarray:
    import pydicom
    rs = pydicom.dcmread(rtstruct_path)
    pts = []
    for roi in rs.get("ROIContourSequence", []):
        for c in roi.get("ContourSequence", []):
            pts.append(np.asarray(c.ContourData, dtype=float).reshape(-1, 3))
    return np.vstack(pts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Geneva_Case")
    ap.add_argument("--source", default="ir192_mHDR_v2_clrp")
    ap.add_argument("--spacing", type=float, default=2.0, help="dose grid 간격 [mm]")
    ap.add_argument("--margin", type=float, default=15.0, help="구조물 바깥 여유 [mm]")
    ap.add_argument("--tolerance", type=float, default=2.0, help="허용오차 [%%]")
    ap.add_argument("--min-volume", type=float, default=5.0,
                    help="부피 검증에서 합격 판정에 넣을 최소 구조물 크기 [cm^3]")
    ap.add_argument(
        "--decay-to", default=None, metavar="YYYYMMDD",
        help="이 날짜로 S_K를 붕괴보정. 지정하지 않으면 RTPLAN의 "
             "SourceStrengthReferenceDate 강도를 그대로 사용 (TPS 계산 조건 재현).",
    )
    ap.add_argument("--axial-grid", action="store_true",
                    help="컨투어 평면 대신 환자 축방향 그리드를 강제 (오차 비교용)")
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--save-dose", action="store_true",
                    help="선량 그리드를 dose_grid.npz로 저장 (1mm에서 수십 MB)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    case_dir = args.case if os.path.isabs(args.case) else os.path.join(root, args.case)

    print("=" * 64)
    print(" Phase 1 — TG-43 엔진 commissioning")
    print("=" * 64)

    # --- 1. 파일 및 Frame of Reference ---------------------------------
    files = find_files(case_dir)
    for mod in ("RTSTRUCT", "RTPLAN"):
        if not files[mod]:
            print(f"[ERROR] {mod} 파일이 없습니다: {case_dir}", file=sys.stderr)
            return 1
    print(f"\n[파일] CT {len(files['CT'])} / RS {len(files['RTSTRUCT'])} / "
          f"RP {len(files['RTPLAN'])} / RD {len(files['RTDOSE'])}")

    rs_path = files["RTSTRUCT"][0]
    rp_path = files["RTPLAN"][0]
    rd_path = files["RTDOSE"][0] if files["RTDOSE"] else None

    _, for_warnings = check_frame_of_reference(files)
    for w in for_warnings:
        print(f"  [WARN] {w}")

    # --- 2. 선원 데이터 ------------------------------------------------
    src = SourceData.load(args.source)
    print("\n[선원]")
    print("  " + src.summary().replace("\n", "\n  "))
    calc = TG43Calculator(src)

    # --- 3. 플랜 -------------------------------------------------------
    plan = load_plan(rp_path, rd_path)
    # RTPLAN의 ReferenceAirKermaRate는 SourceStrengthReferenceDate 시점 강도이며
    # TPS는 그 강도로 plan을 계산한다. commissioning 시에는 붕괴보정을 하지 않는다.
    calc_date = datetime.strptime(args.decay_to, "%Y%m%d") if args.decay_to else None
    sk = plan.sk_at(calc_date)

    print(f"\n[플랜] dwell {len(plan.dwells)}개, 총 {plan.total_time_s():.1f} s, "
          f"{plan.n_fractions} fx x {plan.fraction_dose_gy} Gy")
    ref_str = f"{plan.ref_date:%Y-%m-%d}" if plan.ref_date else "?"
    print(f"  S_K(ref) = {plan.sk_ref:.2f} U @ {ref_str}")
    if calc_date is None:
        print("  붕괴보정 없음 (TPS 계산 조건 재현)")
    else:
        print(f"  S_K(@ {calc_date:%Y-%m-%d}) = {sk:.2f} U")
    if plan.isotope.replace("-", "").lower() != src.isotope.replace("-", "").lower():
        print(f"  [WARN] RTPLAN 핵종({plan.isotope}) != 데이터셋({src.isotope})")

    # --- 4. dose point 검증 (Phase 1 핵심 완료 기준) --------------------
    print("\n[검증 1] RTDOSE dose point 대조")
    results, points_ok = check_dose_points(calc, plan, sk, args.tolerance)
    if not results:
        print("  dose point가 없어 건너뜁니다.")
    else:
        print(format_point_table(results, args.tolerance))
        print(f"  => {'PASS' if points_ok else 'FAIL'} (허용 +-{args.tolerance}%)")

    if args.skip_grid:
        return 0 if points_ok else 2

    # --- 5. 컨투어 평면 좌표계 ------------------------------------------
    frame = contour_frame(rs_path)
    print("\n[컨투어 좌표계]")
    print(f"  평면 법선   : {frame.rotation[2].round(4)}")
    print(f"  z축과 기울기: {frame.tilt_deg:.2f} deg")
    print(f"  평면 이탈   : {frame.planarity_mm:.4f} mm (0에 가까울수록 정상)")
    if frame.tilt_deg > 1.0 and not args.axial_grid:
        print("  -> oblique 컨투어. 평면 좌표계에 그리드를 정의합니다.")
    tandem = max(
        (d for d in plan.dwells), key=lambda d: sum(1 for x in plan.dwells if x.channel == d.channel)
    )
    ang = np.degrees(np.arccos(np.clip(abs(float(tandem.axis @ frame.rotation[2])), -1, 1)))
    print(f"  applicator 축과의 각: {ang:.2f} deg")

    R = np.eye(3) if args.axial_grid else frame.rotation

    # --- 6. dose grid ----------------------------------------------------
    pts = np.vstack([all_contour_points(rs_path),
                     np.array([d.position for d in plan.dwells])])
    grid = make_grid(pts, rotation=R, spacing_mm=args.spacing, margin_mm=args.margin)
    print(f"\n[선량그리드] {grid.shape} @ {args.spacing} mm = {grid.n_voxels():,} voxel "
          f"({'축방향' if grid.is_axial else 'oblique'})")

    t0 = time.time()
    dose = compute_dose_grid(calc, grid, plan.dwells, sk)
    print(f"  계산 완료 ({time.time() - t0:.1f}s), "
          f"max = {dose.max():.2f} Gy/fx, mean = {dose.mean():.3f} Gy/fx")

    # --- 7. 구조물 마스크 및 부피 검증 ------------------------------------
    t0 = time.time()
    masks = build_masks(rs_path, grid)
    print(f"  {len(masks)}개 ROI 래스터화 ({time.time() - t0:.1f}s)")

    print("\n[검증 2] 구조물 부피 (폴리곤 기준 대비)")
    vol_rows, vol_ok = check_structure_volumes(
        rs_path, masks, grid.spacing, args.tolerance, frame, args.min_volume
    )
    print(format_volume_table(vol_rows, args.tolerance, args.min_volume))
    print(f"  => {'PASS' if vol_ok else 'FAIL'} "
          f"(허용 +-{args.tolerance}%, {args.min_volume:g}cc 이상 구조물만 판정)")

    # --- 8. DVH -----------------------------------------------------------
    rx = plan.fraction_dose_gy or 5.0
    table = evaluate_plan(dose, masks, grid.voxel_volume_cm3, rx, TARGETS)

    print(f"\n[DVH] fraction당 (처방 {rx} Gy 기준)")
    print(f"{'structure':<12}{'vol(cc)':>9}{'Dmean':>8}{'D90':>8}{'D98':>8}"
          f"{'V150%':>8}{'D2cc':>8}{'D0.1cc':>9}")
    print("-" * 70)
    nan = float("nan")
    for name in list(TARGETS) + [n for n in table if n not in TARGETS]:
        if name not in table:
            continue
        r = table[name]
        print(
            f"{name:<12}{r['volume_cm3']:9.1f}{r['Dmean_Gy']:8.2f}"
            f"{r.get('D90_Gy', nan):8.2f}{r.get('D98_Gy', nan):8.2f}"
            f"{r.get('V150_pct', nan):8.1f}{r.get('D2cm3_Gy', nan):8.2f}"
            f"{r.get('D01cm3_Gy', nan):9.2f}"
        )
    print("\n  (D90/D98/V150은 표적, D2cc/D0.1cc는 OAR에만 표시)")

    # --- 9. 요약 -----------------------------------------------------------
    print("\n  [주의] dose grid의 최댓값은 dwell 바로 옆 복셀 값으로, TG-43이"
          " 성립하지 않는 영역입니다.\n"
          "         (선원/애플리케이터 내부). 표적·OAR DVH에는 영향이 없으나"
          " Dmax 자체는 인용하지 마십시오.")

    print("\n" + "=" * 64)
    print(f" 검증 1 dose point : {'PASS' if points_ok else 'FAIL'}")
    print(f" 검증 2 구조물 부피: {'PASS' if vol_ok else 'FAIL'}")
    print("=" * 64)

    if args.save_dose:
        out = os.path.join(root, "dose_grid.npz")
        np.savez_compressed(
            out, dose=dose.astype(np.float32), origin=grid.origin,
            spacing=grid.spacing, rotation=grid.rotation,
        )
        print(f"\n선량 그리드 저장: dose_grid.npz  {dose.shape} (float32 + 그리드 기하)")
    return 0 if (points_ok and vol_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
