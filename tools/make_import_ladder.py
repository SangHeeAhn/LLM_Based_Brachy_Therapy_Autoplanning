#!/usr/bin/env python3
"""수신 TPS(Oncentra 등) import 문제를 이분 탐색하기 위한 시험 파일 생성기.

왜 이 방식인가
--------------
확인된 사실:
  1. pydicom으로 **아무것도 바꾸지 않고 재저장하면 원본과 바이트 단위로 동일**하다.
     -> 인코딩/재직렬화는 원인이 아니다.
  2. 실제 내보내기 파일에서 원본 대비 사라진 요소는 ReviewDate/ReviewTime/
     ReviewerName 3개뿐이고, ControlPoint3DPosition을 포함한 기하 정보와
     벤더 私有 태그는 전부 보존된다.
     -> "정보가 빠져서" 안 보이는 것이 아니다.

따라서 남은 원인은 **우리가 의도적으로 바꾼 값 중 하나**에 수신 TPS가
반응하는 것이다. 한 번에 하나씩만 더해 가며 어느 단계에서 깨지는지 찾는다.

사다리 (A 계열, 누적)
---------------------
  A0  SOPInstanceUID만 신규        <- 그 외 100% 원본
  A1  + ApprovalStatus=UNAPPROVED, Review* 삭제
  A2  + RTPlanLabel/Name/Description 변경
  A3  + SeriesInstanceUID 신규
  A4  + dwell time 변경 (ChannelTotalTime, CumulativeTimeWeight, TRAK)
  A5  + DoseReferenceSequence 처방선량 스케일

별도 변형 (B 계열, 가설 검증용)
-------------------------------
  B_noprivate   A5에서 벤더 私有 태그를 제거 — TPS가 私有 데이터와 표준 태그의
                불일치 때문에 렌더링을 포기하는지 본다
  B_srcuid      A5의 UID를 원본 루트/길이(<=61자)로 — 64자 UID를 못 다루는
                구형 파서 가설
  B_keepseries  A5에서 SeriesInstanceUID만 원본 유지

읽는 법
-------
  * A0이 안 보인다  -> 파일 내용과 무관. 수신 TPS가 "자기가 만들지 않은 plan"을
                      재구성하지 않는 정책. DICOM 수정으로는 해결 불가.
  * A0~A3까지 정상, A4에서 깨짐 -> dwell time 태그 기록 방식 문제.
                      CumulativeTimeWeight 스케일/자릿수를 원본 관례에 더
                      가깝게 맞춰야 한다.
  * B_noprivate가 오히려 정상 -> 私有 블록과 표준 태그의 불일치가 원인.
  * B_srcuid만 정상 -> UID 길이/루트 문제.

사용법
------
    python tools/make_import_ladder.py --case Geneva_Case --out ladder/
    python tools/make_import_ladder.py --scale 1.2      # dwell time 배율

★ 생성물은 전부 연구용 시험 파일입니다. ApprovalStatus는 A1부터 UNAPPROVED가
  되며, A0은 원본의 승인상태를 그대로 가지므로 **환자 치료에 쓰일 수 있는
  경로에 두지 마십시오.**
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brachyplan.io import find_files, load_plan  # noqa: E402
from brachyplan.io.rtplan_writer import (  # noqa: E402
    _ds, _weight_scale, quantize_times, short_uid, uid_root_of,
)

# 벤더 私有 블록 (B_noprivate에서 제거할 대상)
PRIVATE_GROUPS = (0x0021, 0x3005, 0x3007, 0x300B, 0x300F)


# ---------------------------------------------------------------------------
def new_uid(style: str, ref: str) -> str:
    if style == "source":
        return short_uid(uid_root_of(ref), max_len=max(len(ref), 32))
    from pydicom.uid import generate_uid
    return generate_uid()


def apply_times(ds, times_s: np.ndarray) -> None:
    """dwell time 관련 태그를 갱신한다 (write_rtplan과 동일한 규칙)."""
    scale = _weight_scale(ds)
    k = 0
    total = 0.0
    for setup in ds.ApplicationSetupSequence:
        for ch in setup.ChannelSequence:
            cps = ch.BrachyControlPointSequence
            n = len(cps) // 2
            t = times_s[k:k + n]
            k += n
            total += float(np.sum(t))
            cum = 0.0
            for i, dt in enumerate(t):
                cps[2 * i].CumulativeTimeWeight = _ds(cum * scale)
                cum += float(dt)
                cps[2 * i + 1].CumulativeTimeWeight = _ds(cum * scale)
            ch.ChannelTotalTime = _ds(float(np.sum(t)))
            ch.FinalCumulativeTimeWeight = _ds(cum * scale if cum > 0 else 1.0)
    sk = float(ds.SourceSequence[0].ReferenceAirKermaRate)
    if len(ds.ApplicationSetupSequence) == 1:
        ds.ApplicationSetupSequence[0].TotalReferenceAirKerma = _ds(sk * total / 3600.0)


def strip_private(ds) -> int:
    """벤더 私有 태그를 재귀적으로 제거하고 개수를 돌려준다."""
    n = 0
    for e in list(ds):
        if e.VR == "SQ":
            for item in e.value:
                n += strip_private(item)
        if e.tag.group in PRIVATE_GROUPS and e.tag.is_private:
            del ds[e.tag]
            n += 1
    return n


def save(ds, path: str) -> None:
    if ds.file_meta is not None:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:                       # pydicom 2.x
        ds.save_as(path)


# ---------------------------------------------------------------------------
def build(src: str, out_dir: str, times: np.ndarray, dose_scale: float,
          uid_style: str, label_suffix: str) -> list[tuple[str, str, str]]:
    """사다리 파일을 만들고 (이름, 경로, 설명) 목록을 돌려준다."""
    os.makedirs(out_dir, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    now = datetime.datetime.now()

    def fresh():
        return pydicom.dcmread(src)

    def step_a0(ds):
        ds.SOPInstanceUID = new_uid(uid_style, str(ds.SOPInstanceUID))

    def step_a1(ds):
        ds.ApprovalStatus = "UNAPPROVED"
        for tag in ("ReviewDate", "ReviewTime", "ReviewerName"):
            if tag in ds:
                delattr(ds, tag)

    def step_a2(ds):
        base = str(getattr(ds, "RTPlanLabel", "PLAN"))
        ds.RTPlanLabel = (base + label_suffix)[:16]
        ds.RTPlanName = (str(getattr(ds, "RTPlanName", base)) + label_suffix)[:64]
        ds.RTPlanDescription = "brachyplan test (RESEARCH)"[:64]
        ds.InstanceCreationDate = ds.RTPlanDate = now.strftime("%Y%m%d")
        ds.InstanceCreationTime = ds.RTPlanTime = now.strftime("%H%M%S")

    def step_a3(ds):
        ds.SeriesInstanceUID = new_uid(uid_style, str(ds.SeriesInstanceUID))

    def step_a4(ds):
        apply_times(ds, times)

    def step_a5(ds):
        for dref in ds.get("DoseReferenceSequence", []):
            if "TargetPrescriptionDose" in dref:
                dref.TargetPrescriptionDose = _ds(
                    float(dref.TargetPrescriptionDose) * dose_scale)

    steps = [
        ("A0_sopuid", [step_a0],
         "SOPInstanceUID만 신규 (그 외 100% 원본, 승인상태도 원본)"),
        ("A1_unapproved", [step_a0, step_a1],
         "+ ApprovalStatus=UNAPPROVED, Review* 삭제"),
        ("A2_label", [step_a0, step_a1, step_a2],
         "+ RTPlanLabel/Name/Description/일시 변경"),
        ("A3_seriesuid", [step_a0, step_a1, step_a2, step_a3],
         "+ SeriesInstanceUID 신규"),
        ("A4_dwelltime", [step_a0, step_a1, step_a2, step_a3, step_a4],
         "+ dwell time 변경 (ChannelTotalTime/CumulativeTimeWeight/TRAK)"),
        ("A5_dosepoint", [step_a0, step_a1, step_a2, step_a3, step_a4, step_a5],
         "+ DoseReference 처방선량 스케일  = 실제 내보내기와 동등"),
    ]
    for name, fns, desc in steps:
        ds = fresh()
        for fn in fns:
            fn(ds)
        p = os.path.join(out_dir, f"RP_{name}.dcm")
        save(ds, p)
        rows.append((name, p, desc))

    # --- B 계열 -----------------------------------------------------------
    full = [step_a0, step_a1, step_a2, step_a3, step_a4, step_a5]

    ds = fresh()
    for fn in full:
        fn(ds)
    n_priv = strip_private(ds)
    p = os.path.join(out_dir, "RP_B_noprivate.dcm")
    save(ds, p)
    rows.append(("B_noprivate", p,
                 f"A5에서 벤더 私有 태그 {n_priv}개 제거 (표준 태그만 남김)"))

    ds = fresh()
    old_sop, old_ser = str(ds.SOPInstanceUID), str(ds.SeriesInstanceUID)
    for fn in full:
        fn(ds)
    ds.SOPInstanceUID = short_uid(uid_root_of(old_sop), max_len=len(old_sop))
    ds.SeriesInstanceUID = short_uid(uid_root_of(old_ser), max_len=len(old_ser))
    p = os.path.join(out_dir, "RP_B_srcuid.dcm")
    save(ds, p)
    rows.append(("B_srcuid", p, "A5 + UID를 원본 루트/길이로 (64자 회피)"))

    ds = fresh()
    keep_ser = str(ds.SeriesInstanceUID)
    for fn in full:
        fn(ds)
    ds.SeriesInstanceUID = keep_ser
    save(ds, os.path.join(out_dir, "RP_B_keepseries.dcm"))
    rows.append(("B_keepseries", os.path.join(out_dir, "RP_B_keepseries.dcm"),
                 "A5 + SeriesInstanceUID만 원본 유지"))

    return rows


# ---------------------------------------------------------------------------
def check(path: str, src_plan, expect_times: np.ndarray | None) -> str:
    """생성 파일이 읽히고 기하가 보존됐는지 확인."""
    try:
        p = load_plan(path)
    except Exception as e:                                   # noqa: BLE001
        return f"읽기 실패: {e}"
    if len(p.dwells) != len(src_plan.dwells):
        return f"dwell 수 {len(p.dwells)} != {len(src_plan.dwells)}"
    P0 = np.array([d.position for d in src_plan.dwells])
    P1 = np.array([d.position for d in p.dwells])
    dev = float(np.abs(P0 - P1).max())
    if dev > 1e-9:
        return f"위치 편차 {dev:.4g} mm"
    if expect_times is not None:
        t = np.array([d.time_s for d in p.dwells])
        err = float(np.abs(t - expect_times).max())
        if err > 1e-6:
            return f"시간 오차 {err:.3g} s"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Geneva_Case")
    ap.add_argument("--src", default=None, help="원본 RTPLAN 경로 (기본: 케이스에서 탐색)")
    ap.add_argument("--out", default="ladder", help="출력 폴더")
    ap.add_argument("--dwell", default=None,
                    help="dwell time .npy. 미지정 시 원본 시간 x --scale")
    ap.add_argument("--scale", type=float, default=1.2,
                    help="--dwell 미지정 시 원본 시간에 곱할 배율")
    ap.add_argument("--dose-scale", type=float, default=1.05,
                    help="DoseReference 처방선량 배율 (A5 단계)")
    ap.add_argument("--time-resolution", type=float, default=0.1)
    ap.add_argument("--uid-style", default="pydicom", choices=["pydicom", "source"])
    ap.add_argument("--suffix", default="_T")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    case_dir = args.case if os.path.isabs(args.case) else os.path.join(root, args.case)
    src = args.src or (find_files(case_dir)["RTPLAN"] or [None])[0]
    if not src or not os.path.isfile(src):
        print(f"[오류] RTPLAN을 찾을 수 없습니다: {case_dir}", file=sys.stderr)
        return 2
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    plan = load_plan(src)
    if args.dwell:
        t = np.load(args.dwell).astype(float).ravel()
        if len(t) != len(plan.dwells):
            print(f"[오류] dwell 수 불일치: {len(t)} vs {len(plan.dwells)}", file=sys.stderr)
            return 2
    else:
        t = np.array([d.time_s for d in plan.dwells]) * args.scale
    t = quantize_times(t, args.time_resolution)

    print("=" * 78)
    print(" import 이분 탐색용 시험 파일 생성")
    print("=" * 78)
    print(f" 원본   : {os.path.basename(src)}")
    print(f" dwell  : {len(plan.dwells)}개, 총 {sum(d.time_s for d in plan.dwells):.1f}s "
          f"-> {t.sum():.1f}s")
    print(f" 출력   : {out_dir}")

    rows = build(src, out_dir, t, args.dose_scale, args.uid_style, args.suffix)

    print(f"\n{'파일':<26}{'검증':<8}설명")
    print("-" * 78)
    for name, path, desc in rows:
        expect = t if ("A4" in name or "A5" in name or name.startswith("B_")) else None
        print(f"{os.path.basename(path):<26}{check(path, plan, expect):<8}{desc}")

    print("\n" + "=" * 78)
    print(" 확인 절차")
    print("=" * 78)
    print(" 1. A0부터 순서대로 하나씩 import 하고, dwell position이 보이는지 기록")
    print(" 2. 처음으로 안 보이는 단계 = 원인이 되는 변경")
    print(" 3. A0부터 이미 안 보이면 -> 파일 내용 문제가 아니라 수신 TPS의")
    print("    plan 재구성 정책 문제. B_noprivate / B_srcuid 도 시험")
    print(" 4. import 전에 이전 시험 plan을 지워 이름 충돌을 피하십시오")
    print("\n ※ 전부 연구용 시험 파일입니다. 치료 경로에 두지 마십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
