#!/usr/bin/env python3
"""Phase 2 실행 — dwell time 역최적화.

완료 기준
  1) 임상 dwell time을 초기값으로 넣어도 DVH가 나빠지지 않는다 (sanity check)
  2) 무작위/균일 초기값에서 시작해도 임상 plan **동등 이상**의 DVH에 도달한다
  3) 볼록성: 초기값이 달라도 같은 목적함수 값에 수렴한다

목표 설정 방침
  OAR 상한은 **임상 plan의 D2cm3 값**을 그대로 쓴다. EMBRACE-II의 구체 수치를
  임의로 적어 넣지 않기 위함이며, "임상 plan 동등 이상"이라는 기준과도 일치한다.
  기관 프로토콜 값을 쓰려면 --goals 로 JSON을 넘긴다.

사용법:
    python run_phase2.py
    python run_phase2.py --spacing 1.0 --smooth 0.02
    python run_phase2.py --backend fista
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brachyplan.optim.objectives import (  # noqa: E402
    StructureObjective, default_objectives, load_objectives,
)
from brachyplan.optim.optimizer import (  # noqa: E402
    HAVE_SCIPY, check_gradient, optimize_dwell_times,
)
from brachyplan.optim.pipeline import (  # noqa: E402
    DEFAULT_TARGETS, build_kernels, clinical_oar_limits, iso_target_match,
    kernel_memory_mb, load_context, make_problem,
)

OARS = ("Bladder", "Rectum", "Sigmoid", "Bowel", "Urethra", "Anus")


def dvh_row(name: str, r: dict) -> str:
    nan = float("nan")
    return (f"{name:<10}{r['volume_cm3']:8.1f}{r['Dmean_Gy']:8.2f}"
            f"{r.get('D90_Gy', nan):8.2f}{r.get('D98_Gy', nan):8.2f}"
            f"{r.get('V150_pct', nan):8.1f}{r.get('D2cm3_Gy', nan):8.2f}")


def print_comparison(clinical: dict, optimized: dict, order: list[str]) -> None:
    print(f"{'':<10}{'':>8}{'  ---------- 임상 plan ----------':<40}"
          f"{'  --------- 최적화 plan ---------':<40}")
    print(f"{'structure':<10}{'vol(cc)':>8}{'Dmean':>8}{'D90':>8}{'D98':>8}"
          f"{'V150%':>8}{'D2cc':>8}   {'Dmean':>8}{'D90':>8}{'D98':>8}"
          f"{'V150%':>8}{'D2cc':>8}   {'판정':>6}")
    print("-" * 112)
    nan = float("nan")
    for name in order:
        if name not in clinical:
            continue
        c, o = clinical[name], optimized[name]
        if name in DEFAULT_TARGETS:
            better = o.get("D90_Gy", 0) >= c.get("D90_Gy", 0) - 1e-6
            verdict = "동등+" if better else "저하"
        else:
            better = o.get("D2cm3_Gy", 1e9) <= c.get("D2cm3_Gy", 1e9) + 1e-6
            verdict = "동등+" if better else "증가"
        print(
            f"{name:<10}{c['volume_cm3']:8.1f}"
            f"{c['Dmean_Gy']:8.2f}{c.get('D90_Gy', nan):8.2f}{c.get('D98_Gy', nan):8.2f}"
            f"{c.get('V150_pct', nan):8.1f}{c.get('D2cm3_Gy', nan):8.2f}   "
            f"{o['Dmean_Gy']:8.2f}{o.get('D90_Gy', nan):8.2f}{o.get('D98_Gy', nan):8.2f}"
            f"{o.get('V150_pct', nan):8.1f}{o.get('D2cm3_Gy', nan):8.2f}   {verdict:>6}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Geneva_Case")
    ap.add_argument("--source", default="ir192_mHDR_v2_clrp")
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--target", default="HR_CTV")
    ap.add_argument("--t-max", type=float, default=200.0, help="dwell당 최대 시간 [s]")
    ap.add_argument("--smooth", type=float, default=0.05,
                    help="lam_smooth. 0이면 dwell time이 몇 개 위치에 몰린다 "
                         "(IPSA의 알려진 단점). 실측: 0 -> 19개 중 7개만 활성")
    ap.add_argument("--secondary-target", default="IR_CTV",
                    help="부목표 구조물. 'none'이면 사용 안 함")
    ap.add_argument("--secondary-weight", type=float, default=50.0)
    ap.add_argument("--hot-spot", type=float, default=None,
                    help="표적 dose_max = 처방 x 이 값. 기본은 비활성 "
                         "(근접치료에서는 표적 내 고선량이 정상이므로 걸면 D90이 나빠짐)")
    ap.add_argument("--oar-weight", type=float, default=1.0)
    ap.add_argument("--target-weight", type=float, default=100.0)
    ap.add_argument("--backend", default="auto", choices=["auto", "lbfgsb", "fista"])
    ap.add_argument("--goals", default=None, help="목표 JSON 경로 (지정 시 임상값 대신 사용)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    case_dir = args.case if os.path.isabs(args.case) else os.path.join(root, args.case)

    print("=" * 72)
    print(" Phase 2 — dwell time 역최적화")
    print("=" * 72)

    # --- 1. 컨텍스트 ----------------------------------------------------
    t0 = time.time()
    ctx = load_context(case_dir, args.source, spacing_mm=args.spacing)
    print(f"\n[준비] 그리드 {ctx.grid.shape} @ {args.spacing}mm, "
          f"dwell {len(ctx.plan.dwells)}개, 처방 {ctx.prescription_gy} Gy/fx "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    build_kernels(ctx)
    print(f"[kernel] {len(ctx.kernels)}개 구조물, "
          f"{sum(K.shape[1] for K in ctx.kernels.values()):,} voxel, "
          f"{kernel_memory_mb(ctx):.1f} MB ({time.time() - t0:.1f}s)")

    t_clin = ctx.clinical_t
    clinical = ctx.evaluate(t_clin)

    # --- 2. 목표 --------------------------------------------------------
    if args.goals:
        objectives = load_objectives(args.goals)
        print(f"\n[목표] {args.goals} 에서 로드")
    else:
        limits = clinical_oar_limits(ctx)
        limits = {k: v for k, v in limits.items() if k in OARS}
        objectives = default_objectives(
            prescription_gy=ctx.prescription_gy,
            target_names=(args.target,),
            oar_limits=limits,
            oar_weight=args.oar_weight,
            target_weight_under=args.target_weight,
            hot_spot_factor=args.hot_spot,
        )
        sec = args.secondary_target
        if sec and sec.lower() != "none" and sec in clinical:
            # 부목표를 넣지 않으면 최적화기가 IR_CTV 커버리지를 희생한다 (실측 -14%).
            objectives.append(StructureObjective(
                name=sec, dose_min=clinical[sec]["D90_Gy"],
                weight_under=args.secondary_weight,
            ))
        print("\n[목표] OAR 상한 = 임상 plan의 D2cm3 (동등 이상 달성 여부를 봄)")
    for o in objectives:
        lo = f"{o.dose_min:.2f}" if o.dose_min is not None else "-"
        hi = f"{o.dose_max:.2f}" if o.dose_max is not None else "-"
        print(f"   {o.name:<10} min={lo:>6} max={hi:>6}  "
              f"w_under={o.weight_under:<6g} w_over={o.weight_over:g}")

    problem = make_problem(ctx, objectives, t_max=args.t_max, lam_smooth=args.smooth)

    # --- 3. gradient 검증 ------------------------------------------------
    err = check_gradient(problem, t_clin)
    print(f"\n[검증 A] gradient 상대오차 (수치미분 대비): {err:.3e} "
          f"=> {'OK' if err < 1e-5 else 'FAIL'}")
    grad_ok = err < 1e-5

    # --- 4. 최적화 -------------------------------------------------------
    print(f"\n[최적화] backend={args.backend}"
          f"{' (scipy 없음 -> fista)' if not HAVE_SCIPY else ''}")
    rng = np.random.default_rng(args.seed)
    starts = {
        "임상 dwell time": t_clin,
        "균일 (전부 10s)": np.full(problem.n_dwell, 10.0),
        "무작위 #1": rng.uniform(0, args.t_max, problem.n_dwell),
        "무작위 #2": rng.uniform(0, args.t_max, problem.n_dwell),
        "전부 0": np.zeros(problem.n_dwell),
    }
    results = {}
    for label, s in starts.items():
        t0 = time.time()
        r = optimize_dwell_times(problem, t0=s, backend=args.backend)
        results[label] = r
        print(f"   {label:<18} J={r.objective:12.6g}  iter={r.n_iter:<5} "
              f"KKT={r.kkt_residual:.2e}  총시간={r.t.sum():7.1f}s  ({time.time() - t0:.2f}s)")

    Js = np.array([r.objective for r in results.values()])
    spread = float(Js.max() - Js.min())
    rel = spread / max(abs(float(Js.mean())), 1e-12)
    convex_ok = rel < 1e-4
    print(f"\n[검증 B] 초기값 5종의 목적함수 편차: {spread:.3e} "
          f"(상대 {rel:.2e}) => {'OK (볼록성 확인)' if convex_ok else 'FAIL'}")

    best = min(results.values(), key=lambda r: r.objective)
    tgt = args.target
    d90_c = clinical[tgt]["D90_Gy"]
    rx = ctx.prescription_gy

    # --- 5. iso-target 비교 -------------------------------------------------
    # 임상 plan은 처방(5 Gy)보다 높은 D90을 주었고, 그 대가로 OAR 선량을 지불했다.
    # Pareto 곡선 위 다른 점끼리 비교하지 않도록 표적 커버리지를 맞춘 뒤 비교한다.
    oar_present = [n for n in OARS if n in clinical]
    print(f"\n[iso-target 탐색] {tgt} D90을 임상값 {d90_c:.3f} Gy에 맞추는 중...")
    t_iso, w_used, n_bisect = iso_target_match(
        ctx, problem, tgt, d90_c, oar_present
    )
    optimized = ctx.evaluate(t_iso)
    d90_o = optimized[tgt]["D90_Gy"]
    print(f"   oar_weight = {w_used:.4g} (이분법 {n_bisect}회) -> D90 = {d90_o:.3f} Gy")

    print(f"\n[결과] 임상 plan vs 최적화 plan — 동일 표적 커버리지 "
          f"(fraction당, 처방 {rx} Gy)")
    order = [n for n in DEFAULT_TARGETS if n in clinical] + oar_present
    print_comparison(clinical, optimized, order)

    target_ok = d90_o >= d90_c - 0.01
    oar_ok = all(
        optimized[n]["D2cm3_Gy"] <= clinical[n]["D2cm3_Gy"] + 1e-6
        for n in oar_present
    )
    goal_ok = d90_o >= rx

    print(f"\n[검증 C] 임상 목표 {tgt} D90 >= 처방 {rx} Gy : "
          f"{d90_o:.3f} Gy => {'OK' if goal_ok else 'FAIL'}")
    print(f"[검증 D] 동일 D90({d90_c:.3f} Gy)에서 모든 OAR D2cm3 동등 이하 "
          f"=> {'OK' if oar_ok else 'FAIL'}")
    if oar_ok:
        gains = [(n, 100 * (optimized[n]['D2cm3_Gy'] - clinical[n]['D2cm3_Gy'])
                  / clinical[n]['D2cm3_Gy']) for n in oar_present]
        gains.sort(key=lambda x: x[1])
        print("   OAR 개선폭: " + ", ".join(f"{n} {g:+.0f}%" for n, g in gains))
    t_opt = t_iso

    # --- 6. dwell time 분포 -------------------------------------------------
    def stats(t):
        act = int(np.sum(t > 0.5))
        cv = float(np.std(t) / max(np.mean(t), 1e-9))
        return act, cv, float(t.max()), float(t.sum())

    ac, cc_, mc, sc = stats(t_clin)
    ao, co, mo, so = stats(t_opt)
    print(f"\n[dwell time 분포]")
    print(f"{'':<12}{'활성/전체':>10}{'변동계수':>10}{'최대(s)':>9}{'총합(s)':>9}")
    print("-" * 50)
    print(f"{'임상':<12}{f'{ac}/{len(t_clin)}':>10}{cc_:10.2f}{mc:9.1f}{sc:9.1f}")
    print(f"{'최적화':<12}{f'{ao}/{len(t_opt)}':>10}{co:10.2f}{mo:9.1f}{so:9.1f}")
    dwell_ok = (ao >= 0.8 * len(t_opt)) and (co <= cc_ * 1.2)
    print(f"  => {'OK' if dwell_ok else 'CHECK'} "
          f"(활성 dwell 80% 이상, 변동계수가 임상의 1.2배 이하)")

    print(f"\n{'ch':>3}{'#':>4}{'임상(s)':>10}{'최적화(s)':>11}")
    print("-" * 28)
    ch = ctx.channel_of
    for k in range(len(t_opt)):
        print(f"{ch[k]:>3}{k:>4}{t_clin[k]:10.2f}{t_opt[k]:11.2f}")
    n_at_max = int(np.sum(t_opt >= args.t_max - 1e-6))
    if n_at_max:
        print(f"  [WARN] {n_at_max}개 dwell이 t_max({args.t_max}s)에 붙어 있습니다. "
              f"--t-max를 올려 확인하십시오.")

    # 부목표 손해를 정직하게 보고
    sec = args.secondary_target
    if sec and sec.lower() != "none" and sec in clinical:
        sc_c, sc_o = clinical[sec]["D90_Gy"], optimized[sec]["D90_Gy"]
        print(f"\n[트레이드오프] {sec} D90: 임상 {sc_c:.2f} -> 최적화 {sc_o:.2f} Gy "
              f"({100 * (sc_o - sc_c) / sc_c:+.1f}%)")
        if sc_o < sc_c - 1e-6:
            print(f"   HR_CTV 커버리지와 OAR을 동시에 지키면서 {sec}까지 완전히 지키지는"
                  f" 못했습니다.\n"
                  f"   --secondary-weight 를 올리면 개선되지만 OAR이 다시 올라갑니다"
                  f" (3자 Pareto).")

    # --- 7. 요약 ------------------------------------------------------------
    all_ok = grad_ok and convex_ok and goal_ok and oar_ok and target_ok and dwell_ok
    print("\n" + "=" * 72)
    print(f" A gradient 정확성   : {'PASS' if grad_ok else 'FAIL'}")
    print(f" B 볼록성/전역수렴   : {'PASS' if convex_ok else 'FAIL'}")
    print(f" C 임상목표 D90>=처방: {'PASS' if goal_ok else 'FAIL'}")
    print(f" D iso-D90 OAR 우위  : {'PASS' if oar_ok else 'FAIL'}")
    print(f" E dwell 분포 임상급 : {'PASS' if dwell_ok else 'CHECK'}")
    print("=" * 72)

    np.save(os.path.join(root, "dwell_times_optimized.npy"), t_opt)
    print("\n최적화 dwell time 저장: dwell_times_optimized.npy")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
