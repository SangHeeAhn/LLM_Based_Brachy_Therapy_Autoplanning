#!/usr/bin/env python3
"""최적화된 dwell time을 RTPLAN DICOM(.dcm)으로 내보낸다.

dwell **position**은 원본 RTPLAN 그대로이고 시간만 바뀌므로, 원본을 복사해
시간 관련 태그만 갱신한다. 자세한 근거는 `brachyplan/io/rtplan_writer.py` 참조.

사용법:
    python export_rtplan.py                                   # dwell_times_agent.npy
    python export_rtplan.py --dwell dwell_times_optimized.npy
    python export_rtplan.py --time-resolution 0.1 --out RP_AI.dcm
    python export_rtplan.py --no-quantize                     # 양자화 없이

★ 이 파일은 연구용입니다. TPS로 다시 불러 선량을 재계산하고 독립 검증을 거쳐야
  하며, 그대로 조사에 사용해서는 안 됩니다. ApprovalStatus는 UNAPPROVED로
  강제됩니다.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brachyplan.io import find_files  # noqa: E402
from brachyplan.io.rtplan_writer import (  # noqa: E402
    dose_change_from_quantization, quantize_times, verify_rtplan, write_rtplan,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Geneva_Case")
    ap.add_argument("--dwell", default="dwell_times_agent.npy",
                    help="최적화된 dwell time [s] (.npy)")
    ap.add_argument("--out", default=None,
                    help="출력 .dcm 경로 (기본: RP_<원본label><suffix>.dcm)")
    ap.add_argument("--suffix", default="_AI", help="plan label에 붙일 접미사")
    ap.add_argument("--time-resolution", type=float, default=0.1,
                    help="조사기 시간 분해능 [s]. 기본 0.1")
    ap.add_argument("--no-quantize", action="store_true",
                    help="시간 양자화를 하지 않는다 (실제 조사와 달라질 수 있음)")
    ap.add_argument("--spacing", type=float, default=2.0,
                    help="양자화 선량영향 평가용 그리드 [mm]")
    ap.add_argument("--no-dose-check", action="store_true",
                    help="양자화 선량영향 평가 및 dose point 재계산 생략 (빠름)")

    g = ap.add_argument_group(
        "수신 TPS 호환 (import 시 카테터/plan 정보가 안 보일 때)",
        "생성 파일에는 기하 정보가 모두 들어 있다. 그래도 수신 TPS가 plan을 "
        "재구성하지 못하면 식별자 처리가 원인일 수 있으므로 아래를 조합해 본다.")
    g.add_argument("--keep-series-uid", action="store_true",
                   help="SeriesInstanceUID를 원본 그대로 둔다 (같은 시리즈로 취급)")
    g.add_argument("--uid-style", default="pydicom", choices=["pydicom", "source"],
                   help="새 UID 생성 방식. source = 원본 UID 루트를 잇고 길이도 "
                        "원본에 맞춤(64자 UID를 못 다루는 구형 시스템용). "
                        "타 벤더 루트를 쓰므로 연구용 한정")
    g.add_argument("--new-private-uid", action="store_true",
                   help="벤더 私有 plan 식별 UID (0021,1000)도 새로 발급")
    g.add_argument("--compat", action="store_true",
                   help="위 세 옵션의 조합 프리셋: --uid-style source + "
                        "--new-private-uid (시리즈 UID는 새로 발급)")
    args = ap.parse_args()

    if args.compat:
        args.uid_style = "source"
        args.new_private_uid = True

    root = os.path.dirname(os.path.abspath(__file__))
    case_dir = args.case if os.path.isabs(args.case) else os.path.join(root, args.case)
    dwell_path = args.dwell if os.path.isabs(args.dwell) \
        else os.path.join(root, args.dwell)

    print("=" * 72)
    print(" RTPLAN 내보내기")
    print("=" * 72)

    files = find_files(case_dir)
    if not files["RTPLAN"]:
        print(f"[ERROR] RTPLAN이 없습니다: {case_dir}", file=sys.stderr)
        return 1
    src = files["RTPLAN"][0]

    if not os.path.isfile(dwell_path):
        print(f"[ERROR] {dwell_path} 가 없습니다.\n"
              f"        먼저 최적화를 실행하십시오: python run_phase3.py "
              f"(또는 run_phase2.py)", file=sys.stderr)
        return 1
    t_opt = np.load(dwell_path).astype(float).ravel()

    resolution = 0.0 if args.no_quantize else args.time_resolution
    t_q = quantize_times(t_opt, resolution)

    out = args.out
    if out is None:
        import pydicom
        label = str(getattr(pydicom.dcmread(src, stop_before_pixels=True),
                            "RTPlanLabel", "PLAN"))
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        out = os.path.join(root, f"RP_{safe}{args.suffix}.dcm")
    elif not os.path.isabs(out):
        out = os.path.join(root, out)

    print(f"\n원본 : {os.path.basename(src)}")
    print(f"dwell: {os.path.basename(dwell_path)}  ({len(t_opt)}개)")
    print(f"UID  : style={args.uid_style}, "
          f"series={'원본유지' if args.keep_series_uid else '신규'}, "
          f"私有(0021,1000)={'신규' if args.new_private_uid else '원본유지'}")

    # --- 양자화 영향 및 dose point 재계산 준비 --------------------------------
    recompute = None
    ctx = None
    if not args.no_dose_check:
        from brachyplan.optim.pipeline import build_kernels, load_context
        ctx = load_context(case_dir, spacing_mm=args.spacing)
        build_kernels(ctx)
        recompute = (ctx.calc, ctx.sk)

    # --- 쓰기 --------------------------------------------------------------
    report = write_rtplan(
        src, out, t_q,
        time_resolution_s=0.0,          # 이미 위에서 양자화했다
        label_suffix=args.suffix,
        recompute_dose_points_with=recompute,
        keep_series_uid=args.keep_series_uid,
        uid_style=args.uid_style,
        new_private_uid=args.new_private_uid,
    )
    report.time_resolution_s = resolution
    report.max_quantization_shift_s = float(np.max(np.abs(t_q - t_opt))) \
        if len(t_opt) else 0.0

    print("\n" + report.summary())

    # --- 양자화가 DVH에 준 영향 ---------------------------------------------
    if ctx is not None and resolution > 0:
        delta = dose_change_from_quantization(ctx, t_opt, t_q)
        big = sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:6]
        print(f"\n[양자화 영향] {resolution:g} s 반올림으로 인한 지표 변화 (Gy/fx)")
        for k, v in big:
            print(f"  {k:<22}{v:+8.4f}")
        worst = max((abs(v) for v in delta.values()), default=0.0)
        print(f"  최대 변화 {worst:.4f} Gy/fx "
              f"({'무시 가능' if worst < 0.01 else '확인 필요'})")

    # --- 재읽기 검증 --------------------------------------------------------
    print("\n[검증] 저장한 파일을 다시 읽어 확인")
    ok, msgs = verify_rtplan(out, t_q)
    for m in msgs:
        print(f"  {m}")

    print("\n" + "=" * 72)
    print(f" 파일 정합성 : {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    print("\n★ 이 RTPLAN은 연구용입니다.")
    print("  - ApprovalStatus = UNAPPROVED (기계 생성 계획)")
    print("  - dwell position은 원본과 동일, 시간만 변경됨")
    print("  - TPS로 다시 불러 선량을 재계산하고, 독립 검증(2차 계산)을 거친 뒤에만")
    print("    임상적 판단에 사용하십시오. 그대로 조사에 쓰면 안 됩니다.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
