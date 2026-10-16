#!/usr/bin/env python3
"""두 RTPLAN(수동 임상 plan vs 생성된 AI plan)을 읽어 비교한다.

무엇을 비교하는가
-----------------
1. plan 자체(RTPLAN만으로 알 수 있는 것)
   - 신원/헤더: label, 승인상태, 참조 RTSTRUCT/FrameOfReference, 환자 ID
   - 선원: 핵종, S_K(ReferenceAirKermaRate), 반감기, 기준일시
   - 기하: 채널 수, 채널별 dwell 수, dwell 위치 일치 여부
   - dwell time: 총 조사시간, 활성 dwell, 최대/평균/변동계수, 채널별 시간
   - TRAK: 파일에 적힌 값 vs S_K x sum(t)/3600 로 재계산한 값
   - 처방: 분할수, BrachyApplicationSetupDose
2. 선량 (--no-dose 로 생략 가능)
   - RTSTRUCT를 읽어 구조물 마스크를 만들고 TG-43으로 두 plan을 **동일 조건에서
     재계산**해 DVH 지표(D90/D98/V150, D2cm3/D0.1cm3)를 비교
   - EMBRACE-II 누적 EQD2 대조 (EBRT 45 Gy/25 fx 가정)

왜 재계산하는가
---------------
RTPLAN에는 DVH가 들어 있지 않다. TPS가 계산한 값을 쓰려면 RTDOSE가 필요한데,
AI plan은 TPS를 거치지 않았으므로 RTDOSE가 없다. 두 plan을 같은 잣대로 보려면
동일한 그리드/마스크/선원 데이터로 양쪽 모두 재계산하는 수밖에 없다.
(따라서 여기 나오는 절대값은 TPS 값과 몇 % 다를 수 있다. 비교는 유효하다.)

★ 연구용입니다. 이 스크립트의 결과는 임상 승인 근거가 될 수 없습니다.

사용법
------
    python compare_plans.py
    python compare_plans.py --ai RP_Brachy_plan_AI.dcm --spacing 2.0
    python compare_plans.py --no-dose            # RTPLAN 헤더/시간만 (수 초)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brachyplan.agent.embrace2 import (  # noqa: E402
    DEFAULT_EBRT_DOSE_GY, DEFAULT_EBRT_FRACTIONS, eqd2_report,
)
from brachyplan.dose.dvh import DVH  # noqa: E402
from brachyplan.dose.kernel import compute_kernel  # noqa: E402
from brachyplan.io import find_files, load_plan  # noqa: E402
from brachyplan.optim.pipeline import DEFAULT_TARGETS, load_context  # noqa: E402

POS_TOL_MM = 0.01          # dwell 위치 동일 판정 허용오차
TRAK_REL_TOL = 1e-4        # TRAK 재계산 상대 허용오차
ACTIVE_MIN_S = 0.5         # '활성 dwell' 기준. visualize_dwell.py와 동일


# ---------------------------------------------------------------------------
# 표시 유틸
# ---------------------------------------------------------------------------
def head(title: str) -> None:
    print(f"\n{title}")


def mark(ok: bool) -> str:
    return "일치" if ok else "다름"


def cv(x: np.ndarray) -> float:
    """변동계수. 평균이 0이면 nan."""
    m = float(np.mean(x))
    return float(np.std(x) / m) if m > 0 else float("nan")


def fmt_delta(new: float, old: float, unit: str = "") -> str:
    d = new - old
    if old:
        return f"{d:+.2f}{unit} ({d / old * 100:+.1f}%)"
    return f"{d:+.2f}{unit}"


# ---------------------------------------------------------------------------
# RTPLAN 헤더에서 직접 읽는 값
# ---------------------------------------------------------------------------
def read_header(path: str) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    ref_ss = ""
    seq = ds.get("ReferencedStructureSetSequence", [])
    if len(seq):
        ref_ss = str(getattr(seq[0], "ReferencedSOPInstanceUID", ""))

    setups = ds.get("ApplicationSetupSequence", [])
    trak = sum(float(getattr(s, "TotalReferenceAirKerma", 0.0)) for s in setups)
    channels = [ch for s in setups for ch in s.get("ChannelSequence", [])]

    return {
        "path": path,
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "label": str(getattr(ds, "RTPlanLabel", "")),
        "name": str(getattr(ds, "RTPlanName", "")),
        "approval": str(getattr(ds, "ApprovalStatus", "")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "station": str(getattr(ds, "StationName", "")),
        "plan_date": f"{getattr(ds, 'RTPlanDate', '')} {getattr(ds, 'RTPlanTime', '')}".strip(),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "frame_uid": str(getattr(ds, "FrameOfReferenceUID", "")),
        "ref_structure_set": ref_ss,
        "trak_file": trak,
        "n_setups": len(setups),
        "n_channels": len(channels),
        "applicator": [str(getattr(ch, "ApplicatorSequence", [{}])[0].get(
            "ApplicatorType", "")) if "ApplicatorSequence" in ch else ""
            for ch in channels],
        "channel_total_time": {int(ch.ChannelNumber): float(ch.ChannelTotalTime)
                               for ch in channels},
    }


# ---------------------------------------------------------------------------
# 선량 재계산
# ---------------------------------------------------------------------------
def metrics_row(name: str, doses: np.ndarray, vv_cm3: float,
                prescription_gy: float) -> dict[str, float]:
    """ctx.evaluate()와 동일한 지표 정의를 쓴다 (비교 가능성 유지)."""
    dvh = DVH(name, np.asarray(doses, dtype=float), vv_cm3)
    row = {"volume_cm3": dvh.volume_cm3, "Dmean_Gy": dvh.mean}
    if name in DEFAULT_TARGETS:
        row["D90_Gy"] = dvh.D_percent(90)
        row["D98_Gy"] = dvh.D_percent(98)
        row["V100_pct"] = dvh.V_percent_of(prescription_gy, 100)
        row["V150_pct"] = dvh.V_percent_of(prescription_gy, 150)
    else:
        row["D2cm3_Gy"] = dvh.D_cc(2.0)
        row["D01cm3_Gy"] = dvh.D_cc(0.1)
    return row


def evaluate_both(ctx, plan_a, plan_b, same_positions: bool,
                  prescription_gy: float, skip: set[str] = frozenset()) -> tuple[dict, dict]:
    """두 plan을 같은 그리드/마스크로 TG-43 재계산한다.

    메모리 절약을 위해 구조물 하나씩 kernel을 만들고 즉시 소비한 뒤 버린다
    (전체 kernel을 들고 있으면 2 mm 그리드에서 1 GB를 넘는다).
    dwell 위치가 같으면 kernel을 한 번만 만들어 두 시간벡터에 재사용한다.
    """
    grid_pts = ctx.grid.points()
    vv = ctx.grid.voxel_volume_cm3
    t_a = np.array([d.time_s for d in plan_a.dwells], dtype=float)
    t_b = np.array([d.time_s for d in plan_b.dwells], dtype=float)
    sk_a, sk_b = plan_a.sk_ref, plan_b.sk_ref

    out_a: dict[str, dict[str, float]] = {}
    out_b: dict[str, dict[str, float]] = {}
    for name, mask in ctx.masks.items():
        if name in skip:
            continue
        idx = np.flatnonzero(mask.ravel())
        if len(idx) == 0:
            continue
        pts = grid_pts[idx]
        K = compute_kernel(ctx.calc, pts, plan_a.dwells, sk_a)
        out_a[name] = metrics_row(name, K.T @ t_a, vv, prescription_gy)
        if same_positions:
            # 위치가 같으면 kernel도 같다. S_K만 다르면 선형 배율로 보정된다.
            out_b[name] = metrics_row(name, (K.T @ t_b) * (sk_b / sk_a),
                                      vv, prescription_gy)
            del K
        else:
            del K
            K2 = compute_kernel(ctx.calc, pts, plan_b.dwells, sk_b)
            out_b[name] = metrics_row(name, K2.T @ t_b, vv, prescription_gy)
            del K2
    return out_a, out_b


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Geneva_Case",
                    help="RTSTRUCT가 있는 케이스 폴더. 수동 plan의 기본 출처")
    ap.add_argument("--manual", default=None,
                    help="수동(임상) RTPLAN 경로. 기본은 케이스 폴더의 RTPLAN")
    ap.add_argument("--ai", default="RP_1B_Cervix_AI.dcm",
                    help="생성된 RTPLAN 경로")
    ap.add_argument("--spacing", type=float, default=2.0,
                    help="재계산 그리드 [mm]")
    ap.add_argument("--source", default="ir192_mHDR_v2_clrp")
    ap.add_argument("--no-dose", action="store_true",
                    help="TG-43 재계산을 생략하고 RTPLAN 내용만 비교")
    ap.add_argument("--force-dose", action="store_true",
                    help="기준계(FrameOfReference)나 dwell 개수가 달라도 DVH를 "
                         "계산한다. 결과는 신뢰할 수 없다")
    ap.add_argument("--top", type=int, default=8,
                    help="dwell time 차이 상위 몇 개를 보일지")
    ap.add_argument("--include-body", action="store_true",
                    help="BODY(외곽) 구조물도 DVH 표에 넣는다. 기본은 제외")
    ap.add_argument("--ebrt-dose", type=float, default=DEFAULT_EBRT_DOSE_GY)
    ap.add_argument("--ebrt-fractions", type=int, default=DEFAULT_EBRT_FRACTIONS)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(root, p)

    case_dir = resolve(args.case)
    files = find_files(case_dir)
    if not files["RTPLAN"]:
        print(f"[오류] {case_dir} 에 RTPLAN이 없습니다")
        return 2
    manual_path = resolve(args.manual) if args.manual else files["RTPLAN"][0]
    ai_path = resolve(args.ai)
    if not os.path.isfile(ai_path):
        print(f"[오류] AI plan을 찾을 수 없습니다: {ai_path}")
        return 2

    print("=" * 78)
    print(" RTPLAN 비교 — 수동(임상) vs 생성(AI)")
    print("=" * 78)
    print(f" 수동: {os.path.basename(manual_path)}")
    print(f" AI  : {os.path.basename(ai_path)}")

    hm, ha = read_header(manual_path), read_header(ai_path)
    pm, pa = load_plan(manual_path), load_plan(ai_path)
    problems: list[str] = []
    notes: list[str] = []

    # --- 1. 신원 / 헤더 ---------------------------------------------------
    head("[1] 신원 및 헤더")
    rows = [
        ("RTPlanLabel", hm["label"], ha["label"], None),
        ("RTPlanName", hm["name"], ha["name"], None),
        ("계획일시", hm["plan_date"], ha["plan_date"], None),
        ("Manufacturer", hm["manufacturer"], ha["manufacturer"], None),
        ("ApprovalStatus", hm["approval"], ha["approval"], None),
        ("PatientID", hm["patient_id"], ha["patient_id"], True),
        ("FrameOfReferenceUID", hm["frame_uid"][-16:], ha["frame_uid"][-16:], True),
        ("참조 RTSTRUCT UID", hm["ref_structure_set"][-16:],
         ha["ref_structure_set"][-16:], True),
        ("SOPInstanceUID", hm["sop_uid"][-16:], ha["sop_uid"][-16:], False),
    ]
    print(f"  {'항목':<22}{'수동':>26}{'AI':>26}   판정")
    print("  " + "-" * 76)
    for label, a, b, must_match in rows:
        verdict = ""
        if must_match is True:
            verdict = mark(a == b)
            if a != b:
                problems.append(f"{label}이(가) 서로 다릅니다 — 같은 환자/기준계가 "
                                f"아닐 수 있습니다")
        elif must_match is False:
            verdict = "달라야 정상" if a != b else "동일(위험)"
            if a == b:
                problems.append("두 plan의 SOPInstanceUID가 같습니다 — "
                                "별개 plan으로 취급되지 않습니다")
        print(f"  {label:<22}{a[:26]:>26}{b[:26]:>26}   {verdict}")

    if ha["approval"].upper() != "UNAPPROVED":
        problems.append(f"AI plan의 ApprovalStatus가 {ha['approval']} 입니다 "
                        f"— 연구용 plan은 UNAPPROVED여야 합니다")
    else:
        notes.append("AI plan ApprovalStatus = UNAPPROVED (정상)")

    # --- 2. 선원 ----------------------------------------------------------
    head("[2] 선원")
    print(f"  {'항목':<22}{'수동':>22}{'AI':>22}   판정")
    print("  " + "-" * 70)
    src_rows = [
        ("핵종", pm.isotope, pa.isotope),
        ("S_K [U]", f"{pm.sk_ref:.4f}", f"{pa.sk_ref:.4f}"),
        ("반감기 [d]", f"{pm.half_life_days:.4f}", f"{pa.half_life_days:.4f}"),
        ("S_K 기준일", str(pm.ref_date), str(pa.ref_date)),
    ]
    for label, a, b in src_rows:
        print(f"  {label:<22}{a:>22}{b:>22}   {mark(a == b)}")
    if abs(pm.sk_ref - pa.sk_ref) > 1e-9:
        problems.append("S_K가 서로 다릅니다 — dwell time을 직접 비교할 수 없습니다 "
                        "(선량은 S_K x t 에 비례)")

    # --- 3. 기하 (채널 / dwell 위치) ---------------------------------------
    head("[3] 기하 — 채널과 dwell 위치")
    same_count = len(pm.dwells) == len(pa.dwells)
    print(f"  채널 수          수동 {hm['n_channels']}    AI {ha['n_channels']}"
          f"   {mark(hm['n_channels'] == ha['n_channels'])}")
    print(f"  dwell 수         수동 {len(pm.dwells)}   AI {len(pa.dwells)}"
          f"   {mark(same_count)}")

    same_positions = False
    if same_count and len(pm.dwells):
        Pm = np.array([d.position for d in pm.dwells])
        Pa = np.array([d.position for d in pa.dwells])
        dev = np.linalg.norm(Pm - Pa, axis=1)
        same_positions = bool(dev.max() <= POS_TOL_MM)
        print(f"  dwell 위치 최대 편차  {dev.max():.4f} mm   "
              f"{'동일 위치' if same_positions else '위치가 다름'}")
        if not same_positions:
            notes.append("dwell 위치가 다릅니다 — 시간뿐 아니라 카테터/정지위치도 "
                         "바뀐 plan입니다. kernel을 각각 계산합니다")
    else:
        problems.append("dwell 개수가 달라 위치를 대응시킬 수 없습니다")

    ch_m = sorted({d.channel for d in pm.dwells})
    ch_a = sorted({d.channel for d in pa.dwells})
    if ch_m != ch_a:
        problems.append(f"채널 번호 집합이 다릅니다: {ch_m} vs {ch_a}")

    # --- 4. dwell time -----------------------------------------------------
    head("[4] dwell time")
    tm = np.array([d.time_s for d in pm.dwells], dtype=float)
    ta = np.array([d.time_s for d in pa.dwells], dtype=float)
    print(f"  {'지표':<22}{'수동':>12}{'AI':>12}   차이")
    print("  " + "-" * 62)
    stat_rows = [
        ("총 조사시간 [s]", tm.sum(), ta.sum(), "s"),
        (f"활성 dwell (>{ACTIVE_MIN_S:g}s)", float((tm > ACTIVE_MIN_S).sum()),
         float((ta > ACTIVE_MIN_S).sum()), ""),
        ("0이 아닌 dwell", float((tm > 0).sum()), float((ta > 0).sum()), ""),
        ("최대 [s]", tm.max() if len(tm) else 0.0,
         ta.max() if len(ta) else 0.0, "s"),
        ("평균 [s]", tm.mean() if len(tm) else 0.0,
         ta.mean() if len(ta) else 0.0, "s"),
        ("변동계수 CV", cv(tm), cv(ta), ""),
    ]
    for label, a, b, unit in stat_rows:
        print(f"  {label:<22}{a:12.2f}{b:12.2f}   {fmt_delta(b, a, unit)}")

    print(f"\n  채널별 총 시간 [s]")
    print(f"  {'채널':<8}{'수동':>10}{'AI':>10}{'차이':>12}{'비율':>10}")
    print("  " + "-" * 50)
    for c in sorted(set(ch_m) | set(ch_a)):
        a = float(sum(d.time_s for d in pm.dwells if d.channel == c))
        b = float(sum(d.time_s for d in pa.dwells if d.channel == c))
        ratio = f"{b / a:.3f}" if a > 0 else "-"
        print(f"  {c:<8}{a:10.2f}{b:10.2f}{b - a:+12.2f}{ratio:>10}")

    if same_count and args.top > 0:
        diff = ta - tm
        order = np.argsort(-np.abs(diff))[:args.top]
        print(f"\n  dwell 단위 차이 상위 {len(order)}개")
        print(f"  {'#':<5}{'채널':<7}{'수동[s]':>10}{'AI[s]':>10}{'차이[s]':>10}")
        print("  " + "-" * 44)
        for i in order:
            print(f"  {i:<5}{pm.dwells[i].channel:<7}{tm[i]:10.2f}"
                  f"{ta[i]:10.2f}{diff[i]:+10.2f}")

    # --- 5. TRAK -----------------------------------------------------------
    head("[5] TRAK (총 기준공기커마)")
    print(f"  {'plan':<8}{'파일값':>12}{'S_K·Σt/3600':>14}{'상대오차':>12}   판정")
    print("  " + "-" * 52)
    for tag, h, p, t in (("수동", hm, pm, tm), ("AI", ha, pa, ta)):
        calc = p.sk_ref * float(t.sum()) / 3600.0
        rel = abs(h["trak_file"] - calc) / calc if calc > 0 else float("nan")
        ok = rel <= TRAK_REL_TOL
        print(f"  {tag:<8}{h['trak_file']:12.4f}{calc:14.4f}{rel:12.2e}   {'OK' if ok else 'FAIL'}")
        if not ok:
            problems.append(f"{tag} plan의 TRAK이 dwell time 합과 맞지 않습니다 "
                            f"(파일 {h['trak_file']:.4f} vs 계산 {calc:.4f})")

    # --- 6. 처방 -----------------------------------------------------------
    head("[6] 처방")
    print(f"  {'항목':<22}{'수동':>14}{'AI':>14}   판정")
    print("  " + "-" * 56)
    fm = pm.fraction_dose_gy if pm.fraction_dose_gy is not None else float("nan")
    fa = pa.fraction_dose_gy if pa.fraction_dose_gy is not None else float("nan")
    print(f"  {'분할수':<22}{pm.n_fractions:>14}{pa.n_fractions:>14}"
          f"   {mark(pm.n_fractions == pa.n_fractions)}")
    print(f"  {'fraction당 [Gy]':<22}{fm:14.3f}{fa:14.3f}"
          f"   {mark(abs(fm - fa) < 1e-6 or (fm != fm and fa != fa))}")
    print(f"  {'총 처방 [Gy]':<22}{fm * pm.n_fractions:14.3f}"
          f"{fa * pa.n_fractions:14.3f}")
    if pm.n_fractions != pa.n_fractions:
        problems.append("분할수가 다릅니다 — 누적 EQD2 비교가 성립하지 않습니다")

    # --- 7. 선량 재계산 ------------------------------------------------------
    # 두 plan이 같은 기준계에 있지 않으면 DVH 비교는 의미가 없다.
    # 케이스의 RTSTRUCT로 만든 마스크는 AI plan의 해부구조가 아니기 때문이다.
    # 값이 그럴듯하게 나오기 때문에 오히려 위험하다 — 기본적으로 막는다.
    same_frame = hm["frame_uid"] == ha["frame_uid"]
    if args.no_dose:
        notes.append("--no-dose 지정 — DVH 비교는 생략했습니다")
    elif not files["RTSTRUCT"]:
        notes.append("RTSTRUCT가 없어 DVH 비교를 생략했습니다")
    elif not (same_frame and same_count) and not args.force_dose:
        head("[7] TG-43 재계산 DVH — 생략")
        print("  두 plan이 같은 기준계/기하가 아닙니다. 케이스의 RTSTRUCT로 만든")
        print("  마스크는 AI plan의 해부구조가 아니므로, 재계산해도 숫자만 그럴듯한")
        print("  무의미한 DVH가 나옵니다.")
        print(f"    FrameOfReferenceUID 일치: {same_frame}")
        print(f"    dwell 개수 일치        : {same_count}")
        print("  같은 케이스에서 내보낸 plan으로 비교하거나 (export_rtplan.py),")
        print("  의도적으로 강행하려면 --force-dose 를 주십시오.")
        notes.append("기준계 불일치로 DVH 비교를 생략했습니다 (--force-dose 로 강행 가능)")
    else:
        if not same_frame:
            problems.append("--force-dose 로 기준계가 다른 plan의 DVH를 계산했습니다 "
                            "— 결과를 신뢰하지 마십시오")
        head("[7] TG-43 재계산 DVH (동일 그리드·마스크·선원 데이터)")
        t0 = time.time()
        ctx = load_context(case_dir, args.source, spacing_mm=args.spacing)
        rx = pm.fraction_dose_gy or ctx.prescription_gy
        print(f"  그리드 {ctx.grid.shape} @ {args.spacing}mm, "
              f"구조물 {len(ctx.masks)}개, 처방 {rx:g} Gy/fx ({time.time() - t0:.1f}s)")

        # AI plan의 dwell이 그리드 밖으로 나가지 않는지 확인
        pts = ctx.grid.points()
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        Pa = np.array([d.position for d in pa.dwells])
        if len(Pa) and (np.any(Pa < lo - 1e-6) or np.any(Pa > hi + 1e-6)):
            problems.append("AI plan의 dwell 일부가 재계산 그리드 밖에 있습니다 "
                            "— 선량 비교가 부정확할 수 있습니다")

        t0 = time.time()
        # BODY/External은 외곽 윤곽이라 D2cm3가 선원 바로 옆 값이 되어 의미가 없다
        skip = set() if args.include_body else {"BODY"}
        mm, ma = evaluate_both(ctx, pm, pa, same_positions, rx, skip)
        print(f"  두 plan 재계산 완료 ({time.time() - t0:.1f}s)"
              f"{'  [kernel 공유]' if same_positions else '  [kernel 개별 계산]'}")

        nan = float("nan")
        print(f"\n  {'':<10}{'':>8}{'------- 수동(임상) -------':^34}"
              f"{'-------- AI plan --------':^34}")
        print(f"  {'구조물':<10}{'vol(cc)':>8}{'Dmean':>8}{'D90':>8}{'D98':>8}"
              f"{'V150%':>8}{'D2cc':>8}   {'Dmean':>8}{'D90':>8}{'D98':>8}"
              f"{'V150%':>8}{'D2cc':>8}   {'판정':>6}")
        print("  " + "-" * 110)
        order = [n for n in DEFAULT_TARGETS if n in mm]
        order += [n for n in mm if n not in order]
        for name in order:
            a, b = mm[name], ma.get(name)
            if b is None:
                continue
            if name in DEFAULT_TARGETS:
                better = b.get("D90_Gy", 0) >= a.get("D90_Gy", 0) - 1e-6
                verdict = "동등+" if better else "저하"
            else:
                better = b.get("D2cm3_Gy", 1e9) <= a.get("D2cm3_Gy", 1e9) + 1e-6
                verdict = "동등+" if better else "증가"
            print(
                f"  {name:<10}{a['volume_cm3']:8.1f}"
                f"{a['Dmean_Gy']:8.2f}{a.get('D90_Gy', nan):8.2f}"
                f"{a.get('D98_Gy', nan):8.2f}{a.get('V150_pct', nan):8.1f}"
                f"{a.get('D2cm3_Gy', nan):8.2f}   "
                f"{b['Dmean_Gy']:8.2f}{b.get('D90_Gy', nan):8.2f}"
                f"{b.get('D98_Gy', nan):8.2f}{b.get('V150_pct', nan):8.1f}"
                f"{b.get('D2cm3_Gy', nan):8.2f}   {verdict:>6}"
            )
        print("\n  (D90/D98/V150은 표적, D2cc는 OAR에만 의미가 있습니다. "
              "절대값은 TPS 계산과 다를 수 있습니다)")

        n_bt = pm.n_fractions or 1
        head(f"[8] EMBRACE-II 누적 EQD2 대조 "
             f"(EBRT {args.ebrt_dose:g} Gy/{args.ebrt_fractions} fx + BT {n_bt} fx)")
        avail = set(mm)
        print("  수동(임상) plan")
        print("  " + eqd2_report(mm, n_bt, args.ebrt_dose, args.ebrt_fractions,
                                 avail).replace("\n", "\n  "))
        print("\n  AI plan")
        print("  " + eqd2_report(ma, n_bt, args.ebrt_dose, args.ebrt_fractions,
                                 avail).replace("\n", "\n  "))

    # --- 요약 ---------------------------------------------------------------
    print("\n" + "=" * 78)
    if problems:
        print(f" 확인 필요 {len(problems)}건")
        for p in problems:
            print(f"   - {p}")
    else:
        print(" 구조적 문제 없음 — 두 plan은 동일 환자/기준계/선원의 "
              "dwell time 변형입니다")
    for n in notes:
        print(f"   · {n}")
    print("=" * 78)
    print(" ※ 연구용 비교입니다. 임상 사용 전 TPS 재계산과 독립 검증이 필요합니다.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
