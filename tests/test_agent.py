"""Phase 3 테스트 — LM Studio 클라이언트, 스코어링, 안전장치, 에이전트 루프.

실제 LM Studio 없이 **Mock 서버로 HTTP 경로 전체**를 검증한다.
환자 데이터가 필요한 테스트는 Geneva_Case가 없으면 skip 된다.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from brachyplan.agent.evaluator import (  # noqa: E402
    HARD_VIOLATION_FACTOR, Goal, findings_text, metrics_table, score_plan,
)
from brachyplan.agent.lm_client import (  # noqa: E402
    EmptyContentError, LMStudioClient, LMStudioError, parse_json_loose, strip_think,
)
from brachyplan.agent.memory import Memory, Step  # noqa: E402
from brachyplan.agent.planner import (  # noqa: E402
    LIMITS, Planner, apply_trust_region, build_schema, enforce_no_target_dose_max,
    limit_adjustment_count, sanitize,
)
from mock_lmstudio import MockServer  # noqa: E402

CASE = os.path.join(ROOT, "Geneva_Case")
HAS_CASE = os.path.isdir(CASE) and any(f.endswith(".dcm") for f in os.listdir(CASE))


# ---------------------------------------------------------------------------
class TestJsonParsing(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json_loose('{"a":1}'), {"a": 1})

    def test_code_fence(self):
        self.assertEqual(parse_json_loose('```json\n{"a":1}\n```'), {"a": 1})

    def test_fence_without_lang(self):
        self.assertEqual(parse_json_loose('```\n{"a":1}\n```'), {"a": 1})

    def test_surrounding_prose(self):
        txt = '생각해보니 이렇습니다.\n{"a": 1, "b": [2,3]}\n이상입니다.'
        self.assertEqual(parse_json_loose(txt), {"a": 1, "b": [2, 3]})

    def test_nested_braces(self):
        txt = 'x {"a": {"b": {"c": 1}}} y'
        self.assertEqual(parse_json_loose(txt), {"a": {"b": {"c": 1}}})

    def test_braces_inside_string(self):
        """문자열 안의 중괄호에 속으면 안 된다."""
        txt = 'pre {"note": "값은 {이렇게} 입니다", "n": 2} post'
        self.assertEqual(parse_json_loose(txt)["n"], 2)

    def test_escaped_quote_in_string(self):
        txt = r'{"note": "그는 \"안녕\" 이라 했다", "n": 3}'
        self.assertEqual(parse_json_loose(txt)["n"], 3)

    def test_empty_raises(self):
        with self.assertRaises(LMStudioError):
            parse_json_loose("   ")

    def test_no_json_raises(self):
        with self.assertRaises(LMStudioError):
            parse_json_loose("죄송하지만 그 요청은 수행할 수 없습니다.")

    def test_unbalanced_raises(self):
        with self.assertRaises(LMStudioError):
            parse_json_loose('{"a": 1')


# ---------------------------------------------------------------------------
class TestLMStudioClient(unittest.TestCase):
    """Mock 서버로 실제 HTTP 왕복을 검증한다."""

    def test_list_models_and_available(self):
        with MockServer() as srv:
            c = LMStudioClient(base_url=srv.base_url)
            self.assertTrue(c.available())
            self.assertEqual(c.list_models(), ["mock-qwen2.5-7b-instruct"])

    def test_resolve_model_picks_first(self):
        with MockServer() as srv:
            c = LMStudioClient(base_url=srv.base_url)
            self.assertEqual(c.resolve_model(), "mock-qwen2.5-7b-instruct")

    def test_chat_sends_structured_output(self):
        schema = build_schema(["HR_CTV", "Bladder"])
        with MockServer() as srv:
            c = LMStudioClient(base_url=srv.base_url)
            c.chat([{"role": "user", "content": "hi"}], json_schema=schema,
                   schema_name="plan_adjustment", seed=7)
            req = srv.requests[-1]
            self.assertEqual(req["response_format"]["type"], "json_schema")
            self.assertEqual(req["response_format"]["json_schema"]["name"],
                             "plan_adjustment")
            self.assertEqual(req["response_format"]["json_schema"]["schema"], schema)
            self.assertFalse(req["stream"])
            self.assertEqual(req["seed"], 7)
            self.assertEqual(req["temperature"], 0.0)

    def test_chat_returns_parseable_json(self):
        with MockServer() as srv:
            c = LMStudioClient(base_url=srv.base_url)
            r = c.chat([{"role": "user", "content": "문제점\n(없음)"}])
            self.assertIn("adjustments", r.json())
            self.assertGreater(r.elapsed_s, -1)
            self.assertEqual(c.n_calls, 1)

    def test_sloppy_output_still_parses(self):
        with MockServer(mode="sloppy") as srv:
            c = LMStudioClient(base_url=srv.base_url)
            r = c.chat([{"role": "user", "content": "x"}])
            self.assertIn("```", r.content)          # 펜스가 실제로 들어 있고
            self.assertIn("adjustments", r.json())   # 그래도 파싱된다

    def test_unreachable_server_message_is_actionable(self):
        c = LMStudioClient(base_url="http://127.0.0.1:9/v1", retries=0)
        self.assertFalse(c.available())
        with self.assertRaises(LMStudioError) as cm:
            c.list_models()
        self.assertIn("Start Server", str(cm.exception))

    def test_diagnose_when_down(self):
        c = LMStudioClient(base_url="http://127.0.0.1:9/v1", retries=0)
        self.assertIn("연결 실패", c.diagnose())

    def test_diagnose_when_up(self):
        with MockServer() as srv:
            self.assertIn("정상", LMStudioClient(base_url=srv.base_url).diagnose())


# ---------------------------------------------------------------------------
class TestReasoningModels(unittest.TestCase):
    """reasoning 모델(Qwen3 등)의 실패 모드 — 실제로 겪은 문제들.

    LM Studio 이슈 #1602 / #1773 / #1990 참조.
    """

    PROMPT = [{"role": "user", "content": """## 문제점
미충족 목표 (초과가 큰 순):
  - Bladder D2cm3_Gy = 5.100 (기준 4.330, 17.8% 초과) [필수]

## 현재 최적화 파라미터
{
 "structures": {"Bladder": {"weight_over": 2.0}}
}
"""}]

    def test_strip_think_closed(self):
        self.assertEqual(strip_think("<think>생각</think>{\"a\":1}"), '{"a":1}')

    def test_strip_think_unclosed(self):
        """닫히지 않은 <think>도 잘라낸다 (토큰 부족으로 잘린 경우)."""
        self.assertEqual(strip_think('{"a":1}\n<think>생각하다 잘림'), '{"a":1}')

    def test_strip_think_multiple(self):
        self.assertEqual(strip_think("<think>a</think>x<think>b</think>y"), "xy")

    def test_strip_think_none(self):
        self.assertEqual(strip_think('{"a":1}'), '{"a":1}')

    def test_think_tags_in_content_parsed(self):
        with MockServer(mode="think_tags") as srv:
            r = LMStudioClient(base_url=srv.base_url).chat(self.PROMPT)
            self.assertNotIn("<think>", r.content)
            self.assertIn("adjustments", r.json())

    def test_reasoning_content_recovered(self):
        """content가 비고 reasoning_content에만 JSON이 있는 경우 (이슈 #1602)."""
        with MockServer(mode="reasoning") as srv:
            c = LMStudioClient(base_url=srv.base_url)
            r = c.chat(self.PROMPT)
            self.assertTrue(r.from_reasoning)
            self.assertIn("adjustments", r.json())
            self.assertEqual(c.n_from_reasoning, 1)

    def test_schema_kills_content_but_no_schema_works(self):
        """json_schema를 주면 빈 응답, 없으면 정상 (이슈 #1773)."""
        with MockServer(mode="schema_dead") as srv:
            c = LMStudioClient(base_url=srv.base_url, retries=0)
            with self.assertRaises(EmptyContentError):
                c.chat(self.PROMPT, json_schema=build_schema(["Bladder"]))
            self.assertIn("adjustments", c.chat(self.PROMPT).json())

    def test_schema_dead_diagnosis_mentions_no_schema(self):
        with MockServer(mode="schema_dead") as srv:
            c = LMStudioClient(base_url=srv.base_url, retries=0)
            with self.assertRaises(EmptyContentError) as cm:
                c.chat(self.PROMPT, json_schema=build_schema(["Bladder"]))
            self.assertIn("--no-schema", str(cm.exception))

    def test_token_exhaustion_diagnosis(self):
        """사고 과정이 토큰을 다 쓴 경우 원인과 해결책을 알려야 한다 (이슈 #1990).

        잘린 사고 텍스트를 content로 회수해 'JSON 없음'으로 바꿔치기하면
        정작 중요한 finish_reason 정보가 사라진다 — 그러면 안 된다.
        """
        with MockServer(mode="length") as srv:
            c = LMStudioClient(base_url=srv.base_url, retries=0, max_tokens=6000)
            with self.assertRaises(EmptyContentError) as cm:
                c.chat(self.PROMPT)
            msg = str(cm.exception)
            self.assertIn("finish_reason = length", msg)
            self.assertIn("--max-tokens", msg)
            self.assertIn("6000", msg)

    def test_planner_auto_falls_back_to_no_schema(self):
        """schema가 죽어 있으면 planner가 스스로 스키마 없는 경로로 전환한다."""
        kw = dict(
            score=score_plan({"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 300.0}},
                             [Goal("Bladder", "D2cm3_Gy", "max", 4.0, hard=True)]),
            metrics={"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 300.0}},
            current_params={"structures": {"Bladder": {"weight_over": 2.0}}},
            goals_text="", history_text="", iteration=1, max_iter=5)
        with MockServer(mode="schema_dead") as srv:
            p = Planner(LMStudioClient(base_url=srv.base_url, retries=0), ["Bladder"])
            prop = p.propose(**kw)
            self.assertEqual(prop.source, "llm")           # 휴리스틱이 아니라 LLM
            self.assertTrue(p.schema_disabled)
            self.assertTrue(any("스키마 없이" in n for n in prop.clipped))
            # 두 번째 호출부터는 스키마를 아예 시도하지 않는다
            n_before = len(srv.requests)
            p.propose(**kw)
            new_reqs = srv.requests[n_before:]
            self.assertTrue(all("response_format" not in r for r in new_reqs))

    def test_planner_falls_back_to_heuristic_when_all_paths_dead(self):
        with MockServer(mode="length") as srv:
            p = Planner(LMStudioClient(base_url=srv.base_url, retries=0), ["Bladder"])
            prop = p.propose(
                score=score_plan({"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 300.0}},
                                 [Goal("Bladder", "D2cm3_Gy", "max", 4.0, hard=True)]),
                metrics={"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 300.0}},
                current_params={}, goals_text="", history_text="",
                iteration=1, max_iter=5)
            self.assertEqual(prop.source, "heuristic")
            self.assertIn("max-tokens", prop.reasoning)

    def test_disable_thinking_sends_flags(self):
        with MockServer() as srv:
            c = LMStudioClient(base_url=srv.base_url, disable_thinking=True)
            c.chat(self.PROMPT)
            req = srv.requests[-1]
            self.assertEqual(req["chat_template_kwargs"]["enable_thinking"], False)
            self.assertEqual(req["reasoning_effort"], "none")

    def test_default_max_tokens_is_generous(self):
        """reasoning 모델을 감당할 수 있어야 한다."""
        self.assertGreaterEqual(LMStudioClient().max_tokens, 4000)


# ---------------------------------------------------------------------------
class TestSanitize(unittest.TestCase):
    VALID = ["HR_CTV", "IR_CTV", "Bladder", "Rectum"]

    def test_valid_passthrough(self):
        p, notes = sanitize(
            {"adjustments": [{"structure": "Bladder", "field": "weight_over", "value": 8}]},
            self.VALID)
        self.assertEqual(p["structures"]["Bladder"]["weight_over"], 8.0)
        self.assertEqual(notes, [])

    def test_unknown_structure_dropped(self):
        p, notes = sanitize(
            {"adjustments": [{"structure": "BLADDER_TYPO", "field": "weight_over",
                              "value": 8}]}, self.VALID)
        self.assertEqual(p, {})
        self.assertTrue(any("알 수 없는 구조물" in n for n in notes))

    def test_unknown_field_dropped(self):
        p, notes = sanitize(
            {"adjustments": [{"structure": "Bladder", "field": "learning_rate",
                              "value": 0.1}]}, self.VALID)
        self.assertEqual(p, {})
        self.assertTrue(any("알 수 없는 필드" in n for n in notes))

    def test_value_clipped_high(self):
        p, notes = sanitize(
            {"adjustments": [{"structure": "Bladder", "field": "weight_over",
                              "value": 1e12}]}, self.VALID)
        self.assertEqual(p["structures"]["Bladder"]["weight_over"],
                         LIMITS["weight_over"][1])
        self.assertTrue(any("clip" in n for n in notes))

    def test_negative_weight_clipped_to_zero(self):
        p, _ = sanitize(
            {"adjustments": [{"structure": "Bladder", "field": "weight_over",
                              "value": -5}]}, self.VALID)
        self.assertEqual(p["structures"]["Bladder"]["weight_over"], 0.0)

    def test_nan_and_inf_dropped(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            p, notes = sanitize(
                {"adjustments": [{"structure": "Bladder", "field": "weight_over",
                                  "value": bad}]}, self.VALID)
            self.assertEqual(p, {})
            self.assertTrue(notes)

    def test_non_numeric_dropped(self):
        p, notes = sanitize(
            {"adjustments": [{"structure": "Bladder", "field": "weight_over",
                              "value": "많이"}]}, self.VALID)
        self.assertEqual(p, {})
        self.assertTrue(any("숫자가 아님" in n for n in notes))

    def test_lam_smooth_clipped(self):
        p, _ = sanitize({"adjustments": [], "lam_smooth": 99.0}, self.VALID)
        self.assertEqual(p["lam_smooth"], 10.0)
        p, _ = sanitize({"adjustments": [], "lam_smooth": -3.0}, self.VALID)
        self.assertEqual(p["lam_smooth"], 0.0)

    def test_malformed_adjustment_item(self):
        p, notes = sanitize({"adjustments": ["문자열", 42, None]}, self.VALID)
        self.assertEqual(p, {})
        self.assertEqual(len(notes), 3)

    def test_missing_adjustments_key(self):
        p, notes = sanitize({"reasoning": "..."}, self.VALID)
        self.assertEqual(p, {})

    def test_target_dose_max_blocked(self):
        p = {"structures": {"HR_CTV": {"dose_max": 7.5, "weight_under": 200.0},
                            "Bladder": {"dose_max": 4.0}}}
        notes = enforce_no_target_dose_max(p, ("GTV", "HR_CTV", "IR_CTV"))
        self.assertNotIn("dose_max", p["structures"]["HR_CTV"])
        self.assertEqual(p["structures"]["HR_CTV"]["weight_under"], 200.0)
        self.assertEqual(p["structures"]["Bladder"]["dose_max"], 4.0)  # OAR은 허용
        self.assertTrue(notes)

    def test_target_entry_removed_when_empty(self):
        p = {"structures": {"HR_CTV": {"dose_max": 7.5}}}
        enforce_no_target_dose_max(p, ("HR_CTV",))
        self.assertNotIn("HR_CTV", p["structures"])


# ---------------------------------------------------------------------------
class TestOvershootGuards(unittest.TestCase):
    """과도한 변경 방지 — 실제 LLM이 좋은 plan을 무너뜨린 사례에서 나온 장치."""

    CUR = {"structures": {
        "Bladder": {"weight_over": 100.0},
        "Rectum": {"weight_over": 10.0},
        "HR_CTV": {"weight_under": 100.0},
    }}

    def test_trust_region_clips_large_decrease(self):
        p = {"structures": {"Bladder": {"weight_over": 10.0}}}   # 10배 감소 시도
        notes = apply_trust_region(p, self.CUR, 2.0)
        self.assertEqual(p["structures"]["Bladder"]["weight_over"], 50.0)
        self.assertTrue(notes)

    def test_trust_region_clips_large_increase(self):
        p = {"structures": {"Rectum": {"weight_over": 1000.0}}}
        apply_trust_region(p, self.CUR, 2.0)
        self.assertEqual(p["structures"]["Rectum"]["weight_over"], 20.0)

    def test_trust_region_allows_small_change(self):
        p = {"structures": {"Rectum": {"weight_over": 15.0}}}
        notes = apply_trust_region(p, self.CUR, 2.0)
        self.assertEqual(p["structures"]["Rectum"]["weight_over"], 15.0)
        self.assertEqual(notes, [])

    def test_trust_region_disabled_when_factor_zero(self):
        p = {"structures": {"Bladder": {"weight_over": 1.0}}}
        self.assertEqual(apply_trust_region(p, self.CUR, 0.0), [])
        self.assertEqual(p["structures"]["Bladder"]["weight_over"], 1.0)

    def test_trust_region_ignores_dose_fields(self):
        """dose_min/dose_max는 가중치가 아니므로 배수 제한 대상이 아니다."""
        cur = {"structures": {"HR_CTV": {"dose_min": 5.0}}}
        p = {"structures": {"HR_CTV": {"dose_min": 50.0}}}
        apply_trust_region(p, cur, 2.0)
        self.assertEqual(p["structures"]["HR_CTV"]["dose_min"], 50.0)

    def test_limit_count_keeps_largest_changes(self):
        p = {"structures": {
            "Bladder": {"weight_over": 10.0},     # 10배 감소 (가장 큼)
            "Rectum": {"weight_over": 11.0},      # 1.1배 (가장 작음)
            "HR_CTV": {"weight_under": 400.0},    # 4배
        }}
        notes = limit_adjustment_count(p, self.CUR, 2)
        kept = {n for n in p["structures"]}
        self.assertEqual(kept, {"Bladder", "HR_CTV"})
        self.assertTrue(any("축소" in n for n in notes))

    def test_limit_count_noop_when_few(self):
        p = {"structures": {"Bladder": {"weight_over": 10.0}}}
        self.assertEqual(limit_adjustment_count(p, self.CUR, 3), [])
        self.assertEqual(len(p["structures"]), 1)

    def test_planner_applies_guards(self):
        """propose()가 두 장치를 실제로 적용해야 한다."""
        with MockServer(mode="overshoot") as srv:
            p = Planner(LMStudioClient(base_url=srv.base_url),
                        ["Bladder", "Rectum", "HR_CTV"])
            sc = score_plan({"Bladder": {"D2cm3_Gy": 1.0, "volume_cm3": 300.0}},
                            [Goal("Bladder", "D2cm3_Gy", "max", 4.0, hard=True)])
            self.assertTrue(sc.acceptable)
            prop = p.propose(
                score=sc, metrics={"Bladder": {"D2cm3_Gy": 1.0, "volume_cm3": 300.0}},
                current_params=self.CUR, goals_text="", history_text="",
                iteration=1, max_iter=5,
                max_change_factor=2.0, max_adjustments=2)
            for fields in prop.params.get("structures", {}).values():
                for f, v in fields.items():
                    if f.startswith("weight"):
                        self.assertGreater(v, 0)
            self.assertLessEqual(
                sum(len(f) for f in prop.params.get("structures", {}).values()), 2)


class TestLoadWeights(unittest.TestCase):
    """최선 상태로 되돌리기 — set_weights는 병합이라 복구에 쓸 수 없다."""

    def _problem(self):
        from brachyplan.optim.objectives import (
            OptimizationProblem, StructureObjective,
        )
        rng = np.random.default_rng(0)
        return OptimizationProblem(
            kernels={"A": rng.random((3, 20)), "B": rng.random((3, 20))},
            objectives=[
                StructureObjective("A", dose_min=5.0, weight_under=100.0),
                StructureObjective("B", dose_max=2.0, weight_over=1.0),
            ],
            channel_of=np.ones(3), t_max=60.0, lam_smooth=0.05,
        )

    def test_roundtrip_restores_exactly(self):
        p = self._problem()
        snapshot = p.get_weights()
        p.set_weights({"structures": {"B": {"weight_over": 5000.0}},
                       "lam_smooth": 1.5})
        self.assertNotEqual(p.get_weights(), snapshot)
        p.load_weights(snapshot)
        self.assertEqual(p.get_weights(), snapshot)

    def test_load_ignores_unknown_structure(self):
        p = self._problem()
        p.load_weights({"structures": {"NOPE": {"weight_over": 1.0}}})
        self.assertEqual(len(p.objectives), 2)


# ---------------------------------------------------------------------------
class TestScoring(unittest.TestCase):
    def test_min_goal(self):
        g = Goal("HR_CTV", "D90_Gy", "min", 5.0)
        self.assertEqual(g.sub_score(5.0), 1.0)
        self.assertEqual(g.sub_score(6.0), 1.0)
        self.assertAlmostEqual(g.sub_score(2.5), 0.5)
        self.assertTrue(g.satisfied(5.0))
        self.assertFalse(g.satisfied(4.99))

    def test_max_goal(self):
        g = Goal("Bladder", "D2cm3_Gy", "max", 4.0)
        self.assertEqual(g.sub_score(3.0), 1.0)
        self.assertAlmostEqual(g.sub_score(6.0), 0.5)
        self.assertEqual(g.sub_score(8.0), 0.0)
        self.assertEqual(g.sub_score(100.0), 0.0)

    def test_nan_scores_zero(self):
        self.assertEqual(Goal("X", "D90_Gy", "min", 5.0).sub_score(float("nan")), 0.0)

    def test_hard_violation_penalised(self):
        """필수 목표를 1% 넘기면 부분점수가 크게 깎여야 한다 (실측 문제 재발 방지)."""
        soft = Goal("Bowel", "D2cm3_Gy", "max", 0.384, hard=False)
        hard = Goal("Bowel", "D2cm3_Gy", "max", 0.384, hard=True)
        v = 0.388
        self.assertGreater(soft.sub_score(v), 0.98)
        self.assertAlmostEqual(hard.sub_score(v),
                               soft.sub_score(v) * HARD_VIOLATION_FACTOR)
        # 충족하면 계수가 적용되지 않는다
        self.assertEqual(hard.sub_score(0.3), 1.0)

    def test_score_plan_aggregation(self):
        metrics = {"HR_CTV": {"D90_Gy": 5.5}, "Bladder": {"D2cm3_Gy": 3.0}}
        goals = [Goal("HR_CTV", "D90_Gy", "min", 5.0, weight=3.0, hard=True),
                 Goal("Bladder", "D2cm3_Gy", "max", 4.0, weight=1.0, hard=True)]
        s = score_plan(metrics, goals)
        self.assertAlmostEqual(s.total, 100.0)
        self.assertTrue(s.acceptable)
        self.assertEqual(s.n_met, 2)

    def test_score_plan_flags_hard_violation(self):
        metrics = {"Bladder": {"D2cm3_Gy": 5.0}}
        goals = [Goal("Bladder", "D2cm3_Gy", "max", 4.0, hard=True)]
        s = score_plan(metrics, goals)
        self.assertFalse(s.acceptable)
        self.assertEqual(len(s.hard_violations), 1)

    def test_missing_metric_skipped(self):
        s = score_plan({}, [Goal("Nope", "D90_Gy", "min", 5.0)])
        self.assertEqual(s.results, [])
        self.assertEqual(s.total, 0.0)

    def test_margin_pct_sign(self):
        metrics = {"A": {"D90_Gy": 4.5}, "B": {"D2cm3_Gy": 5.0}}
        goals = [Goal("A", "D90_Gy", "min", 5.0), Goal("B", "D2cm3_Gy", "max", 4.0)]
        s = score_plan(metrics, goals)
        self.assertLess(s.results[0].margin_pct, 0)   # 미달
        self.assertLess(s.results[1].margin_pct, 0)   # 초과
        self.assertGreater(score_plan({"A": {"D90_Gy": 6.0}},
                                      [goals[0]]).results[0].margin_pct, 0)

    def test_findings_text_lists_misses(self):
        metrics = {"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 300}}
        s = score_plan(metrics, [Goal("Bladder", "D2cm3_Gy", "max", 4.0, hard=True)])
        txt = findings_text(s, metrics)
        self.assertIn("Bladder", txt)
        self.assertIn("초과", txt)

    def test_metrics_table_no_nan_text(self):
        txt = metrics_table({"A": {"volume_cm3": 1.0, "D90_Gy": 2.0}})
        self.assertNotIn("nan", txt)


# ---------------------------------------------------------------------------
def _step(i, score, hard=False, params=None):
    return Step(iteration=i, params=params or {}, score=score, n_met=1, n_goals=2,
                hard_violations=["x"] if hard else [], key_metrics={},
                t=[1.0, 2.0])


class TestMemory(unittest.TestCase):
    def test_best_prefers_acceptable(self):
        m = Memory()
        m.add(_step(1, 99.0, hard=True))     # 점수는 높지만 필수 위반
        m.add(_step(2, 80.0, hard=False))    # 점수는 낮지만 합격
        self.assertEqual(m.best.iteration, 2)

    def test_best_falls_back_when_all_violate(self):
        m = Memory()
        m.add(_step(1, 50.0, hard=True))
        m.add(_step(2, 70.0, hard=True))
        self.assertEqual(m.best.iteration, 2)

    def test_stagnant_for(self):
        m = Memory()
        for i, s in enumerate([10.0, 50.0, 40.0, 45.0], start=1):
            m.add(_step(i, s))
        self.assertEqual(m.stagnant_for(), 2)

    def test_improved(self):
        m = Memory()
        m.add(_step(1, 10.0))
        self.assertTrue(m.improved())        # 첫 스텝
        m.add(_step(2, 10.05))
        self.assertFalse(m.improved())       # 0.1 미만 개선

    def test_prompt_text_includes_diff(self):
        m = Memory()
        m.add(_step(1, 10.0, params={"structures": {"Bladder": {"weight_over": 1.0}}}))
        m.add(_step(2, 20.0, params={"structures": {"Bladder": {"weight_over": 8.0}}}))
        txt = m.prompt_text()
        self.assertIn("Bladder.weight_over 1->8", txt)

    def test_prompt_text_empty(self):
        self.assertIn("아직 없음", Memory().prompt_text())

    def test_json_roundtrip(self):
        import json
        m = Memory()
        m.add(_step(1, 10.0))
        self.assertEqual(len(json.loads(m.to_json())), 1)


# ---------------------------------------------------------------------------
class TestPlannerFallback(unittest.TestCase):
    """LLM이 없거나 망가져도 루프가 계속되어야 한다."""

    def _score(self):
        metrics = {"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 300.0}}
        return score_plan(metrics, [Goal("Bladder", "D2cm3_Gy", "max", 4.0, hard=True)])

    def _kw(self, sc):
        return dict(score=sc, metrics={"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 300.0}},
                    current_params={"structures": {"Bladder": {"weight_over": 1.0}}},
                    goals_text="", history_text="", iteration=1, max_iter=5)

    def test_heuristic_without_client(self):
        p = Planner(None, ["Bladder"])
        prop = p.propose(**self._kw(self._score()))
        self.assertEqual(prop.source, "heuristic")
        self.assertGreater(prop.params["structures"]["Bladder"]["weight_over"], 1.0)

    def test_heuristic_scales_with_volume(self):
        """큰 구조물일수록 가중치를 더 크게 올려야 한다 (mean() 희석 보정)."""
        p = Planner(None, ["Bladder"])
        kw_small = self._kw(self._score())
        kw_small["metrics"] = {"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 30.0}}
        kw_big = self._kw(self._score())
        kw_big["metrics"] = {"Bladder": {"D2cm3_Gy": 5.0, "volume_cm3": 400.0}}
        w_small = p.propose(**kw_small).params["structures"]["Bladder"]["weight_over"]
        w_big = p.propose(**kw_big).params["structures"]["Bladder"]["weight_over"]
        self.assertGreater(w_big, w_small * 5)

    def test_falls_back_when_server_down(self):
        c = LMStudioClient(base_url="http://127.0.0.1:9/v1", retries=0)
        p = Planner(c, ["Bladder"])
        prop = p.propose(**self._kw(self._score()))
        self.assertEqual(prop.source, "heuristic")
        self.assertIn("폴백", prop.reasoning)
        self.assertEqual(p.failures, 1)

    def test_broken_llm_output_falls_back(self):
        with MockServer(mode="broken") as srv:
            p = Planner(LMStudioClient(base_url=srv.base_url), ["Bladder"])
            prop = p.propose(**self._kw(self._score()))
            # broken 모드는 Bladder/Rectum 조정을 포함하므로 llm 경로가 유지되지만
            # 잘못된 항목은 모두 걸러져야 한다
            self.assertNotIn("BLADDER_TYPO", prop.params.get("structures", {}))
            self.assertTrue(prop.clipped)


# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
class TestAgentLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from brachyplan.agent.evaluator import goals_from_clinical
        from brachyplan.optim.pipeline import (
            build_kernels, clinical_oar_limits, load_context,
        )
        cls.ctx = load_context(CASE, spacing_mm=3.0)
        build_kernels(cls.ctx)
        cls.clin = cls.ctx.evaluate(cls.ctx.clinical_t)
        cls.oars = ("Bladder", "Rectum", "Sigmoid", "Bowel", "Urethra", "Anus")
        cls.goals = goals_from_clinical(cls.clin, cls.ctx.prescription_gy,
                                        oars=cls.oars)
        cls.limits = {k: v for k, v in clinical_oar_limits(cls.ctx).items()
                      if k in cls.oars}

    def _objs(self):
        """매번 새 목표 객체를 만든다 (set_weights가 제자리에서 수정하므로)."""
        from brachyplan.optim.objectives import StructureObjective, default_objectives
        return (
            default_objectives(self.ctx.prescription_gy, ("HR_CTV",), self.limits,
                               oar_weight=1.0, target_weight_under=100.0)
            + [StructureObjective("IR_CTV", dose_min=self.clin["IR_CTV"]["D90_Gy"],
                                  weight_under=50.0)]
        )

    def _problem(self):
        from brachyplan.optim.pipeline import make_problem
        return make_problem(self.ctx, self._objs(), t_max=200.0, lam_smooth=0.05)

    def _run(self, client, max_iter=8):
        from brachyplan.agent.loop import run_agent
        pr = self._problem()
        planner = Planner(client, [o.name for o in self._objs()])
        return pr, run_agent(self.ctx, pr, self.goals, planner, "",
                             max_iter=max_iter, patience=4, verbose=False)

    def test_heuristic_loop_improves(self):
        from brachyplan.optim.optimizer import optimize_dwell_times
        pr, res = self._run(None)
        base = score_plan(self.ctx.evaluate(optimize_dwell_times(self._problem()).t),
                          self.goals)
        self.assertGreaterEqual(res.score.total, base.total)
        self.assertGreater(res.iterations, 1)

    def test_mock_llm_loop_reaches_acceptable(self):
        with MockServer(mode="good") as srv:
            _, res = self._run(LMStudioClient(base_url=srv.base_url), max_iter=14)
        self.assertTrue(res.score.acceptable,
                        f"필수 위반: {res.score.hard_violations}")
        self.assertGreater(res.llm_calls, 0)

    def test_hostile_llm_never_returns_worse_than_baseline(self):
        """LLM이 계속 잘못된 제안을 해도 결과가 기준선보다 나빠지면 안 된다."""
        from brachyplan.optim.optimizer import optimize_dwell_times
        base = score_plan(self.ctx.evaluate(optimize_dwell_times(self._problem()).t),
                          self.goals)
        with MockServer(mode="broken") as srv:
            _, res = self._run(LMStudioClient(base_url=srv.base_url), max_iter=4)
        self.assertGreaterEqual(res.score.total, base.total - 1e-9)

    def test_result_t_is_valid(self):
        pr, res = self._run(None, max_iter=4)
        self.assertEqual(len(res.t), len(self.ctx.plan.dwells))
        self.assertTrue(np.all(res.t >= -1e-9))
        self.assertTrue(np.all(res.t <= pr.t_max + 1e-9))

    def test_returned_metrics_match_returned_t(self):
        """반환된 지표가 반환된 dwell time으로 재계산한 값과 일치해야 한다."""
        _, res = self._run(None, max_iter=3)
        recomputed = self.ctx.evaluate(res.t)
        self.assertAlmostEqual(recomputed["HR_CTV"]["D90_Gy"],
                               res.metrics["HR_CTV"]["D90_Gy"], places=9)

    def test_overshoot_after_success_does_not_lose_best(self):
        """좋은 plan을 얻은 뒤 LLM이 과도하게 움직여도 결과를 잃지 않는다.

        실제 qwen3.6-35b 가 #4에서 99.0을 얻고 #5에서 69.4로 무너뜨린 상황.
        """
        with MockServer(mode="overshoot") as srv:
            from brachyplan.agent.loop import run_agent
            pr = self._problem()
            planner = Planner(LMStudioClient(base_url=srv.base_url),
                              [o.name for o in self._objs()])
            res = run_agent(self.ctx, pr, self.goals, planner, "",
                            max_iter=12, patience=4, target_score=100.0,
                            verbose=False)
        peak = max(s.score for s in res.memory.steps)
        self.assertAlmostEqual(res.score.total, peak, places=6)
        self.assertGreater(res.n_reverts, 0, "회귀 복구가 한 번은 일어나야 한다")

    def test_no_revert_before_first_acceptable_plan(self):
        """필수 목표를 한 번도 못 지킨 단계에서는 되돌리지 않는다.

        탐색 중의 하락까지 되돌리면 결정론적 플래너가 교착에 빠지고,
        하락을 거쳐 도달했을 더 좋은 해를 놓친다 (실측: 99.86 -> 91.22).
        """
        from brachyplan.agent.loop import run_agent
        for mode in ("good", "overshoot"):
            with self.subTest(mode=mode), MockServer(mode=mode) as srv:
                pr = self._problem()
                planner = Planner(LMStudioClient(base_url=srv.base_url),
                                  [o.name for o in self._objs()])
                res = run_agent(self.ctx, pr, self.goals, planner, "",
                                max_iter=14, patience=4, target_score=100.0,
                                verbose=False)
                first_ok = next((s.iteration for s in res.memory.steps
                                 if s.acceptable_step), None)
                for it in res.revert_iterations:
                    self.assertIsNotNone(
                        first_ok, "합격 plan이 없는데 되돌림이 일어났다")
                    self.assertGreater(
                        it, first_ok,
                        f"{mode}: 첫 합격(#{first_ok}) 이전 #{it}에서 되돌림")

    def test_dips_during_exploration_are_allowed(self):
        """탐색 중 하락이 있어도 최종적으로 최고점에 도달해야 한다."""
        from brachyplan.agent.loop import run_agent
        with MockServer(mode="good") as srv:
            pr = self._problem()
            planner = Planner(LMStudioClient(base_url=srv.base_url),
                              [o.name for o in self._objs()])
            res = run_agent(self.ctx, pr, self.goals, planner, "",
                            max_iter=14, patience=4, verbose=False)
        peak = max(s.score for s in res.memory.steps)
        self.assertAlmostEqual(res.score.total, peak, places=6)
        self.assertTrue(res.score.acceptable)

    def test_hard_constraints_never_relaxed_by_agent(self):
        """에이전트가 무엇을 하든 t의 박스 제약은 유지된다."""
        with MockServer(mode="broken") as srv:
            pr, res = self._run(LMStudioClient(base_url=srv.base_url), max_iter=3)
        self.assertEqual(pr.t_max, 200.0)
        self.assertTrue(np.all(res.t >= 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
