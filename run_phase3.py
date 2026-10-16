#!/usr/bin/env python3
"""Phase 3 실행 — LM Studio LLM 에이전트가 최적화 파라미터를 튜닝한다.

LM Studio 준비
  1. LM Studio 실행 → Developer 탭 → Start Server (기본 포트 1234)
  2. 모델 로드 (structured output을 쓰므로 7B 이상 권장)
  3. python run_phase3.py --check      ← 연결 확인
  4. python run_phase3.py              ← 실행

LM Studio 없이도 돌아간다
  --no-llm      내장 휴리스틱으로 같은 루프를 실행 (루프 자체 검증용)
  --mock good   테스트용 가짜 서버를 띄워 HTTP 경로까지 검증

핵심 설계
  LLM은 dwell time을 직접 만들지 않는다. 가중치·목표선량만 조정하고
  실제 dwell time은 결정론적 볼록 최적화기가 계산한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brachyplan.agent import (  # noqa: E402
    DEFAULT_BASE_URL, DEFAULT_EBRT_DOSE_GY, DEFAULT_EBRT_FRACTIONS, LMStudioClient,
    NOT_IMPLEMENTED, Planner, eqd2_report, goals_from_clinical, goals_from_embrace2,
    load_goals, metrics_table, run_agent, save_goals, score_plan,
)
from brachyplan.optim.objectives import (  # noqa: E402
    StructureObjective, default_objectives, objectives_from_goals,
)
from brachyplan.optim.pipeline import (  # noqa: E402
    DEFAULT_TARGETS, build_kernels, clinical_oar_limits, iso_target_match,
    kernel_memory_mb, load_context, make_problem,
)

OARS = ("Bladder", "Rectum", "Sigmoid", "Bowel", "Urethra", "Anus")


def print_comparison(clinical, agent_m, order):
    nan = float("nan")
    print(f"{'structure':<10}{'vol(cc)':>8}  {'--- 임상 ---':<26}{'--- 에이전트 ---':<28}{'판정':>6}")
    print(f"{'':<10}{'':>8}  {'D90':>8}{'V150%':>8}{'D2cc':>8}  {'D90':>8}{'V150%':>8}{'D2cc':>8}")
    print("-" * 88)
    for n in order:
        if n not in clinical or n not in agent_m:
            continue
        c, o = clinical[n], agent_m[n]
        if n in DEFAULT_TARGETS:
            v = "동등+" if o.get("D90_Gy", 0) >= c.get("D90_Gy", 0) - 1e-6 else "저하"
        else:
            v = "동등+" if o.get("D2cm3_Gy", 1e9) <= c.get("D2cm3_Gy", 1e9) + 1e-6 else "증가"
        print(f"{n:<10}{c['volume_cm3']:8.1f}  "
              f"{c.get('D90_Gy', nan):8.2f}{c.get('V150_pct', nan):8.1f}"
              f"{c.get('D2cm3_Gy', nan):8.2f}  "
              f"{o.get('D90_Gy', nan):8.2f}{o.get('V150_pct', nan):8.1f}"
              f"{o.get('D2cm3_Gy', nan):8.2f}{v:>8}")


PING_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"},
                   "value": {"type": "number"}},
    "required": ["reasoning", "value"],
    "additionalProperties": False,
}
PING_PROMPT = ('테스트입니다. 아래 형식의 JSON 객체 하나만 출력하십시오.\n'
               '{"reasoning": "짧은 설명", "value": 42}')


def run_check(client, mock_srv, args) -> int:
    """LM Studio 연결 + 두 경로(스키마 사용/미사용)를 실제로 시험한다.

    reasoning 모델에서 structured output이 깨지는 사례가 많아
    (LM Studio 이슈 #1773/#1990/#1602), 어느 경로가 동작하는지 알려준다.
    """
    print("=" * 72)
    print(" LM Studio 연결 확인")
    print("=" * 72)
    if client is None:
        print("--no-llm 이므로 확인할 것이 없습니다.")
        return 0

    print(client.diagnose())
    print(f"max_tokens = {client.max_tokens}, timeout = {client.timeout_s:.0f}s, "
          f"no-think = {client.disable_thinking}")
    if not client.available():
        if mock_srv:
            mock_srv.__exit__()
        return 1

    results = {}
    for label, schema in (("structured output (json_schema)", PING_SCHEMA),
                          ("스키마 없이 (프롬프트 지시만)", None)):
        print(f"\n[{label}]")
        try:
            r = client.chat([{"role": "user", "content": PING_PROMPT}],
                            json_schema=schema, schema_name="ping")
            parsed = r.json()
            ok = "value" in parsed
            extra = " (reasoning_content에서 회수)" if r.from_reasoning else ""
            print(f"  응답 {r.elapsed_s:5.1f}s, finish={r.finish_reason}, "
                  f"tokens={r.usage.get('completion_tokens', '?')}{extra}")
            print(f"  파싱 결과: {str(parsed)[:120]}")
            print(f"  => {'OK' if ok else '형식 불일치'}")
            results[label] = ok
        except Exception as e:
            print(f"  실패: {e}")
            results[label] = False

    print("\n" + "=" * 72)
    if results.get("structured output (json_schema)"):
        print(" 권장 실행:  python run_phase3.py")
    elif results.get("스키마 없이 (프롬프트 지시만)"):
        print(" structured output은 실패했지만 프롬프트 방식은 동작합니다.")
        print(" 권장 실행:  python run_phase3.py --no-schema")
        print(" (에이전트는 자동으로도 이 경로로 넘어가지만, 처음부터 지정하면 빠릅니다)")
    else:
        print(" 두 경로 모두 실패했습니다. 시도해 볼 것:")
        print("   python run_phase3.py --check --max-tokens 12000")
        print("   python run_phase3.py --check --no-think")
        print("   비-reasoning 모델(예: qwen2.5-*-instruct) 로드 후 재시도")
    print("=" * 72)

    if mock_srv:
        mock_srv.__exit__()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Geneva_Case")
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--target", default="HR_CTV")
    ap.add_argument("--secondary", default="IR_CTV")

    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=None, help="미지정 시 로드된 첫 모델")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="재현성을 위해 기본 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="reasoning 모델은 사고 과정에 토큰을 많이 쓴다. "
                         "부족하면 content가 비어서 돌아온다")
    ap.add_argument("--no-think", action="store_true",
                    help="사고 모드 끄기 시도 (chat_template_kwargs.enable_thinking=false). "
                         "모델/템플릿에 따라 무시될 수 있음")
    ap.add_argument("--no-schema", action="store_true",
                    help="structured output을 쓰지 않고 프롬프트 지시로만 JSON을 받는다. "
                         "reasoning 모델에서 스키마가 사고 스트림에 걸리는 문제 회피")

    ap.add_argument("--check", action="store_true", help="LM Studio 연결만 확인하고 종료")
    ap.add_argument("--no-llm", action="store_true", help="휴리스틱만 사용")
    ap.add_argument("--mock", default=None,
                    choices=["good", "sloppy", "broken", "think_tags",
                             "reasoning", "schema_dead", "length", "overshoot"],
                    help="테스트용 가짜 LM Studio 서버를 띄워 실행 "
                         "(reasoning 모델 실패 모드 재현 포함)")

    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--revert-margin", type=float, default=2.0,
                    help="최고점 대비 이만큼 떨어지면 최선 파라미터로 되돌린다")
    ap.add_argument("--trust-factor", type=float, default=2.0,
                    help="필수 목표 충족 상태에서 허용할 가중치 변경 배수")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--target-score", type=float, default=99.5)
    ap.add_argument("--t-max", type=float, default=200.0)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--goals-source", default="embrace2",
                    choices=["embrace2", "clinical"],
                    help="embrace2: EMBRACE-II 프로토콜 누적 EQD2를 fraction당으로 환산 "
                         "(기본). clinical: 임상 plan 실측값을 상한으로 사용")
    ap.add_argument("--ebrt-dose", type=float, default=DEFAULT_EBRT_DOSE_GY,
                    help="EBRT 총선량 [Gy]. EMBRACE-II 규정값 45 Gy")
    ap.add_argument("--ebrt-fractions", type=int, default=DEFAULT_EBRT_FRACTIONS,
                    help="EBRT 분할수. EMBRACE-II 규정값 25 fx")
    ap.add_argument("--objectives", default="goals", choices=["goals", "clinical"],
                    help="최적화 파라미터(dose_min/dose_max)의 출처. "
                         "goals: 평가 목표에서 유도 (기본, 신환 적용 가능). "
                         "clinical: 임상 plan 실측 D2cm3 (구버전 동작)")
    ap.add_argument("--goals", default=None, help="목표 JSON (지정 시 위 설정보다 우선)")
    ap.add_argument("--save-goals", default=None)
    ap.add_argument("--no-reference-probe", action="store_true",
                    help="달성 가능성 기준선(iso-target 탐색) 계산을 생략")
    ap.add_argument("--save-trace", default="agent_trace.json")
    ap.add_argument("--save-dwell", default="dwell_times_agent.npy")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    case_dir = args.case if os.path.isabs(args.case) else os.path.join(root, args.case)

    # --- LM Studio 연결 -------------------------------------------------
    mock_srv = None
    base_url = args.base_url
    if args.mock:
        sys.path.insert(0, os.path.join(root, "tools"))
        from mock_lmstudio import MockServer
        mock_srv = MockServer(mode=args.mock).__enter__()
        base_url = mock_srv.base_url
        print(f"[mock] 가짜 LM Studio 서버 (mode={args.mock}) → {base_url}")

    client = None
    if not args.no_llm:
        client = LMStudioClient(base_url=base_url, model=args.model,
                                timeout_s=args.timeout, temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                disable_thinking=args.no_think)

    if args.check:
        return run_check(client, mock_srv, args)

    print("=" * 72)
    print(" Phase 3 — LLM 에이전트 자동계획")
    print("=" * 72)

    if client is not None and not client.available():
        print(f"\n[WARN] LM Studio에 연결할 수 없습니다 ({base_url}).")
        print("       휴리스틱 모드로 계속합니다. 연결 진단: python run_phase3.py --check")
        client = None

    # --- 준비 -----------------------------------------------------------
    t0 = time.time()
    ctx = load_context(case_dir, spacing_mm=args.spacing)
    t_load = time.time() - t0
    build_kernels(ctx)
    t_kernel = time.time() - t0 - t_load
    clinical = ctx.evaluate(ctx.clinical_t)
    print(f"\n[준비] 그리드 {ctx.grid.shape} @ {args.spacing}mm, dwell {len(ctx.plan.dwells)}개, "
          f"kernel {kernel_memory_mb(ctx):.1f} MB "
          f"(로드 {t_load:.1f}s + kernel {t_kernel:.1f}s)")
    print(f"       처방 {ctx.prescription_gy} Gy/fx x {ctx.plan.n_fractions} fx")

    if client is not None:
        print(f"\n[LLM]  {base_url}")
        try:
            print(f"       모델: {client.resolve_model()}  "
                  f"temperature={args.temperature} seed={args.seed} "
                  f"max_tokens={args.max_tokens}")
            if args.no_schema:
                print("       structured output 미사용 (--no-schema)")
            if args.no_think:
                print("       사고 모드 끄기 시도 (--no-think)")
        except Exception as e:
            print(f"       모델 확인 실패: {e}")
    else:
        print("\n[LLM]  사용 안 함 — 내장 휴리스틱")

    # --- 목표 -------------------------------------------------------------
    n_bt = ctx.plan.n_fractions or 1
    if args.goals:
        goals = load_goals(args.goals)
        print(f"\n[목표] {args.goals}")
    elif args.goals_source == "embrace2":
        goals, expl = goals_from_embrace2(
            n_bt, available_structures=set(clinical),
            ebrt_dose_gy=args.ebrt_dose, ebrt_fractions=args.ebrt_fractions)
        print("\n[목표] EMBRACE-II 프로토콜 (원문 인용, 누적 EQD2 -> fraction당 환산)")
        print("  " + "\n  ".join(expl))
        print("  미구현 항목: " + "; ".join(NOT_IMPLEMENTED))
    else:
        goals = goals_from_clinical(clinical, ctx.prescription_gy,
                                    target=args.target, secondary=args.secondary,
                                    oars=OARS)
        print("\n[목표] 임상 plan 실측값 기준")
    goals_text = "\n".join(
        f"  - {g.describe()} (가중치 {g.weight:g})"
        + (f"  [{g.label}]" if g.label else "") for g in goals)
    print("\n[fraction당 목표]")
    print(goals_text)
    if args.save_goals:
        save_goals(goals, args.save_goals)
        print(f"       목표를 {args.save_goals} 에 저장 (수정 후 --goals 로 재사용)")

    # --- 최적화 문제 -------------------------------------------------------
    # 최적화 파라미터는 **평가 목표에서 유도**한다. 목표가 EMBRACE-II면 상한도
    # EMBRACE-II에서 나오고, 목표가 임상 실측이면 상한도 거기서 나온다.
    # 이렇게 하면 임상 plan 의존이 기하(dwell 위치)+선원+처방만 남아 신환에도
    # 그대로 적용된다. --objectives clinical 로 예전 동작을 복원할 수 있다.
    if args.objectives == "goals":
        objectives = objectives_from_goals(
            goals, primary_target=args.target,
            secondary_target=args.secondary or None,
            available=set(ctx.masks),
            target_weight_under=100.0, secondary_weight_under=50.0,
            oar_weight=1.0)
        src_txt = "평가 목표에서 유도 (임상 dwell time 비의존)"
    else:
        limits = {k: v for k, v in clinical_oar_limits(ctx).items() if k in OARS}
        objectives = default_objectives(ctx.prescription_gy, (args.target,), limits,
                                        oar_weight=1.0, target_weight_under=100.0)
        if args.secondary and args.secondary in clinical:
            objectives.append(StructureObjective(
                name=args.secondary, dose_min=clinical[args.secondary]["D90_Gy"],
                weight_under=50.0))
        src_txt = "임상 plan 실측 D2cm3 (구버전 동작)"

    # 평가 목표에 있는데 최적화 objective에 없는 구조물은 비활성(가중치 0)으로
    # 넣어 둔다. 목적함수 값은 그대로지만, 에이전트가 그 지표를 고치려 할 때
    # 손댈 수 있는 손잡이가 생긴다. (없으면 프롬프트는 "GTV D98 미달"을 문제로
    # 제시하면서 GTV 조정 제안은 sanitize()가 '알 수 없는 구조물'로 버린다.)
    have = {o.name for o in objectives}
    for name in dict.fromkeys(g.structure for g in goals):
        if name in have or name not in ctx.masks:
            continue
        mins = [g.limit for g in goals if g.structure == name and g.kind == "min"]
        maxs = [g.limit for g in goals if g.structure == name and g.kind == "max"]
        objectives.append(StructureObjective(
            name=name,
            dose_min=max(mins) if mins else None,
            dose_max=min(maxs) if maxs else None))
        have.add(name)

    print(f"\n[최적화 파라미터] {src_txt}")
    print(f"   {'구조물':<10}{'dose_min':>10}{'dose_max':>10}{'w_under':>10}{'w_over':>9}")
    for o in objectives:
        lo = f"{o.dose_min:.3f}" if o.dose_min is not None else "-"
        hi = f"{o.dose_max:.3f}" if o.dose_max is not None else "-"
        print(f"   {o.name:<10}{lo:>10}{hi:>10}{o.weight_under:>10g}{o.weight_over:>9g}")

    problem = make_problem(ctx, objectives, t_max=args.t_max, lam_smooth=args.smooth)

    planner = Planner(client, [o.name for o in objectives],
                      target_names=DEFAULT_TARGETS, seed=args.seed,
                      use_schema=not args.no_schema)

    # --- 기준선: 초기 파라미터 그대로 ----------------------------------------
    from brachyplan.optim.optimizer import optimize_dwell_times
    t_base = optimize_dwell_times(problem).t
    base_score = score_plan(ctx.evaluate(t_base), goals)
    print(f"\n[기준선] 초기 파라미터만으로 최적화: score {base_score.total:.2f} "
          f"({base_score.n_met}/{len(base_score.results)})"
          f"{'  [필수위반]' if base_score.hard_violations else ''}")

    # --- 달성 가능성 기준선 -----------------------------------------------------
    # 목표가 이 케이스에서 애초에 달성 가능한지, 에이전트가 못 찾은 것인지 구분한다.
    # iso-target 이분법으로 표적 최소치에 맞춘 뒤 OAR을 본다 (약 1초).
    ref_score = None
    hard_min = next((g for g in goals
                     if g.hard and g.kind == "min" and g.structure == args.target), None)
    if not args.no_reference_probe and hard_min is not None:
        oar_names = [g.structure for g in goals if g.kind == "max"]
        probe = make_problem(ctx, [
            StructureObjective(g.structure, dose_min=g.limit, weight_under=100.0)
            if g.kind == "min" else
            StructureObjective(g.structure, dose_max=g.limit, weight_over=1.0)
            for g in goals if g.structure in ctx.masks
        ], t_max=args.t_max, lam_smooth=args.smooth)
        t_ref, w_ref, _ = iso_target_match(ctx, probe, args.target,
                                           hard_min.limit, oar_names)
        ref_score = score_plan(ctx.evaluate(t_ref), goals)
        print(f"\n[달성 가능성] 표적을 목표 최소치에 맞춘 기준해: "
              f"score {ref_score.total:.2f} "
              f"({'전 필수목표 충족 가능' if ref_score.acceptable else '필수목표 동시 충족 불가'})")
        if not ref_score.acceptable:
            print("  " + "; ".join(ref_score.hard_violations))
            print("  -> 목표 자체가 이 해부학/애플리케이터로는 무리일 수 있습니다 "
                  "(EMBRACE-II는 이런 경우 IC/IS 침 사용을 권고).")

    # --- 에이전트 루프 --------------------------------------------------------
    print(f"\n[에이전트] 최대 {args.max_iter}회, patience {args.patience}, "
          f"목표 점수 {args.target_score}")
    res = run_agent(ctx, problem, goals, planner, goals_text,
                    max_iter=args.max_iter, patience=args.patience,
                    target_score=args.target_score,
                    revert_margin=args.revert_margin,
                    trust_factor=args.trust_factor)

    print(f"\n  종료: {res.stop_reason}  ({res.iterations}회, {res.elapsed_s:.1f}s)")
    if res.llm_calls:
        c = planner.client
        print(f"  LLM 호출 {res.llm_calls}회, 평균 {c.total_s / max(c.n_calls, 1):.1f}s"
              f"{f', 폴백 {res.llm_failures}회' if res.llm_failures else ''}"
              f"{f', reasoning에서 회수 {c.n_from_reasoning}회' if c.n_from_reasoning else ''}")
    if res.n_reverts:
        print(f"  회귀 복구 {res.n_reverts}회 (나쁜 제안 후 최선 상태로 되돌림)")
    for n in planner.notes:
        print(f"  [참고] {n}")

    # --- 시간 분해 ------------------------------------------------------------
    # "최적화가 오래 걸린다"는 오해를 없애기 위해 어디서 시간이 갔는지 보여준다.
    # LLM을 켜면 거의 전부가 LLM 대기 시간이고, 최적화 자체는 반복당 0.2초 수준이다.
    print("\n[시간 분해]")
    print(f"  {'준비(로드+kernel)':<22}{t_load + t_kernel:8.1f}s")
    for line in res.timing_lines():
        print(line)
    if res.t_propose_s > 0.5 * max(res.elapsed_s, 1e-9):
        print("  → 병목은 LLM 응답 대기입니다. 최적화 계산이 아닙니다.")
        print("     줄이려면: --max-iter 축소, --max-tokens 축소(reasoning 모델), "
              "--no-schema, 또는 더 작은 모델")

    # --- 결과 ---------------------------------------------------------------
    print(f"\n[최종 plan] score {res.score.total:.2f}  "
          f"(기준선 {base_score.total:.2f}, {res.score.total - base_score.total:+.2f})")
    print(res.score.table())

    print(f"\n[DVH] 임상 vs 에이전트 (fraction당, 처방 {ctx.prescription_gy} Gy)")
    order = [n for n in DEFAULT_TARGETS if n in clinical] + \
            [n for n in OARS if n in clinical]
    print_comparison(clinical, res.metrics, order)

    t_clin = ctx.clinical_t
    cv = lambda t: float(np.std(t) / max(np.mean(t), 1e-9))  # noqa: E731
    if args.goals_source == "embrace2" and not args.goals:
        print(f"\n[EMBRACE-II 누적 EQD2 대조] (EBRT {args.ebrt_dose:g} Gy/"
              f"{args.ebrt_fractions} fx + BT {n_bt} fx)")
        print(eqd2_report(res.metrics, n_bt, args.ebrt_dose, args.ebrt_fractions,
                          set(clinical)))
        print("\n  [참고] 임상 plan을 같은 기준으로 보면:")
        print("  " + eqd2_report(clinical, n_bt, args.ebrt_dose,
                                 args.ebrt_fractions, set(clinical)
                                 ).replace("\n", "\n  "))

    print(f"\n[dwell time] 활성 {int((res.t > 0.5).sum())}/{len(res.t)}"
          f" (임상 {int((t_clin > 0.5).sum())}/{len(t_clin)}), "
          f"CV {cv(res.t):.2f} (임상 {cv(t_clin):.2f}), "
          f"총 {res.t.sum():.1f}s (임상 {t_clin.sum():.1f}s)")

    print("\n[에이전트 궤적]")
    for s in res.memory.steps:
        mark = " <- 채택" if s.iteration == res.memory.best.iteration else ""
        print(f"  {s.summary()}{mark}")
        if s.reasoning and s.source != "initial":
            print(f"      [{s.source}] {s.reasoning[:150]}")

    # --- 저장 ---------------------------------------------------------------
    if args.save_trace:
        res.memory.save(os.path.join(root, args.save_trace))
        print(f"\n궤적 저장: {args.save_trace}")
    if args.save_dwell:
        np.save(os.path.join(root, args.save_dwell), res.t)
        print(f"dwell time 저장: {args.save_dwell}  "
              f"(python visualize_dwell.py --dwell {args.save_dwell} 로 시각화)")

    hard_ok = res.score.acceptable
    improved = res.score.total >= base_score.total - 1e-9
    if ref_score is not None:
        gap = ref_score.total - res.score.total
        print(f"\n[기준해 대비] 에이전트 {res.score.total:.2f} vs 기준해 "
              f"{ref_score.total:.2f} ({-gap:+.2f})")
        if gap > 1.0:
            print("  에이전트가 달성 가능한 수준에 못 미쳤습니다 "
                  "(--max-iter를 늘리거나 다른 모델을 시도해 보십시오).")
    print("\n" + "=" * 72)
    print(f" 필수 목표 준수     : {'PASS' if hard_ok else 'FAIL'}")
    print(f" 기준선 대비 개선   : {'PASS' if improved else 'FAIL'} "
          f"({base_score.total:.2f} -> {res.score.total:.2f})")
    print("=" * 72)

    if mock_srv:
        mock_srv.__exit__()
    return 0 if (hard_ok and improved) else 2


if __name__ == "__main__":
    raise SystemExit(main())
