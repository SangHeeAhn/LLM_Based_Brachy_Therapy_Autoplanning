"""EMBRACE-II 목표 변환 테스트 — EQD2 수식과 프로토콜 값 매핑.

EMBRACE-II의 모든 수치는 **EBRT + 근접치료 누적 EQD2**다. 이를 근접치료
fraction당 물리선량으로 환산하는 것이 여기서 검증할 핵심이다.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brachyplan.agent.embrace2 import (  # noqa: E402
    AB_OAR, AB_TUMOR, CONSTRAINTS, bt_limit_per_fraction, cumulative_eqd2_from_bt,
    ebrt_eqd2, eqd2, eqd2_report, goals_from_embrace2,
)


class TestEQD2(unittest.TestCase):
    def test_two_gy_fractions_are_identity(self):
        """2 Gy/fx 는 정의상 EQD2 = 물리선량."""
        for ab in (3.0, 10.0):
            self.assertAlmostEqual(eqd2(60.0, 2.0, ab), 60.0, places=9)

    def test_hypofractionation_increases_eqd2_for_low_ab(self):
        """분할선량이 크면 a/b가 낮은 조직에서 EQD2가 더 커진다."""
        hi = eqd2(30.0, 6.0, AB_OAR)
        lo = eqd2(30.0, 6.0, AB_TUMOR)
        self.assertGreater(hi, 30.0)
        self.assertGreater(hi, lo)

    def test_ebrt_45_in_25fx(self):
        """EMBRACE-II 규정 EBRT (§4.1.6: 45 Gy / 25 fx = 1.8 Gy/fx)."""
        self.assertAlmostEqual(ebrt_eqd2(45.0, 25, AB_OAR), 43.2, places=6)
        self.assertAlmostEqual(ebrt_eqd2(45.0, 25, AB_TUMOR), 44.25, places=6)

    def test_zero_fractions_is_zero(self):
        self.assertEqual(ebrt_eqd2(45.0, 0, AB_OAR), 0.0)


class TestLimitConversion(unittest.TestCase):
    def test_roundtrip(self):
        """한계 -> fraction당 -> 다시 누적 EQD2 = 원래 한계."""
        for limit, ab in ((80.0, AB_OAR), (65.0, AB_OAR), (85.0, AB_TUMOR)):
            d = bt_limit_per_fraction(limit, 6, ab)
            back = cumulative_eqd2_from_bt(d, 6, ab)
            self.assertAlmostEqual(back, limit, places=6)

    def test_known_values(self):
        """손계산 대조 (EBRT 45/25, BT 6 fx)."""
        self.assertAlmostEqual(bt_limit_per_fraction(80.0, 6, AB_OAR), 4.237, places=3)
        self.assertAlmostEqual(bt_limit_per_fraction(65.0, 6, AB_OAR), 3.018, places=3)
        self.assertAlmostEqual(bt_limit_per_fraction(70.0, 6, AB_OAR), 3.458, places=3)
        self.assertAlmostEqual(bt_limit_per_fraction(85.0, 6, AB_TUMOR), 5.320, places=3)

    def test_more_fractions_lower_per_fraction_dose(self):
        a = bt_limit_per_fraction(80.0, 4, AB_OAR)
        b = bt_limit_per_fraction(80.0, 6, AB_OAR)
        self.assertGreater(a, b)

    def test_higher_ebrt_leaves_less_room(self):
        low = bt_limit_per_fraction(80.0, 6, AB_OAR, ebrt_dose_gy=45.0)
        high = bt_limit_per_fraction(80.0, 6, AB_OAR, ebrt_dose_gy=50.4, ebrt_fractions=28)
        self.assertLess(high, low)

    def test_ebrt_alone_exceeding_limit_gives_zero(self):
        self.assertEqual(bt_limit_per_fraction(30.0, 6, AB_OAR), 0.0)

    def test_invalid_fractions_raises(self):
        with self.assertRaises(ValueError):
            bt_limit_per_fraction(80.0, 0, AB_OAR)


class TestGoals(unittest.TestCase):
    ALL = {"HR_CTV", "IR_CTV", "GTV", "Bladder", "Rectum", "Sigmoid", "Bowel"}

    def test_all_constraints_mapped(self):
        goals, _ = goals_from_embrace2(6, self.ALL)
        self.assertEqual({g.structure for g in goals}, self.ALL)

    def test_missing_structures_skipped(self):
        goals, _ = goals_from_embrace2(6, {"HR_CTV", "Bladder"})
        self.assertEqual({g.structure for g in goals}, {"HR_CTV", "Bladder"})

    def test_kinds(self):
        by = {(g.structure, g.metric, g.hard): g
              for g in goals_from_embrace2(6, self.ALL)[0]}
        self.assertEqual(by[("HR_CTV", "D90_Gy", False)].kind, "min")
        self.assertEqual(by[("Bladder", "D2cm3_Gy", False)].kind, "max")

    def test_aim_is_soft_and_limit_is_hard(self):
        """Table 4: planning aim = soft constraint, limit = hard constraint.

        aim 을 hard 로 걸면 임상적으로 수용 가능한 plan도 '필수 위반'이 되어
        에이전트가 달성 불가능한 문제를 풀게 된다 (Phase 3 Bowel 사례).
        """
        goals = goals_from_embrace2(6, self.ALL)[0]
        for g in goals:
            if "[aim]" in g.label:
                self.assertFalse(g.hard, g.label)
            elif "[limit]" in g.label:
                self.assertTrue(g.hard, g.label)
            else:
                self.fail(f"aim/limit 표시가 없는 목표: {g.label}")

    def test_every_hard_goal_is_looser_than_its_aim(self):
        """hard(limit) 목표는 정의상 aim보다 느슨해야 한다."""
        by: dict[tuple[str, str], dict[bool, float]] = {}
        for g in goals_from_embrace2(6, self.ALL)[0]:
            by.setdefault((g.structure, g.metric), {})[g.hard] = g.limit
        for key, pair in by.items():
            if len(pair) < 2:
                continue
            aim, lim = pair[False], pair[True]
            kind = "min" if key[1] != "D2cm3_Gy" else "max"
            if kind == "min":
                self.assertLess(lim, aim, key)     # 표적: 더 낮은 선량이 느슨
            else:
                self.assertGreater(lim, aim, key)  # OAR: 더 높은 선량이 느슨

    def test_bowel_aim_not_hard(self):
        """회귀 테스트 — bowel 70 Gy는 aim이므로 필수가 아니다.

        임상 승인 plan조차 70 Gy를 넘는 경우가 있다 (실측 70.2 Gy).
        """
        goals = [g for g in goals_from_embrace2(6, self.ALL)[0]
                 if g.structure == "Bowel"]
        hard = [g for g in goals if g.hard]
        self.assertEqual(len(hard), 1)
        # hard 목표는 75 Gy 환산값이어야 한다
        self.assertAlmostEqual(hard[0].limit,
                               round(bt_limit_per_fraction(75.0, 6, AB_OAR), 4),
                               places=4)

    def test_labels_carry_protocol_reference(self):
        for g in goals_from_embrace2(6, self.ALL)[0]:
            self.assertTrue(g.label.startswith("§") or g.label.startswith("Table"),
                            g.label)

    def test_explanation_mentions_eqd2_and_ebrt(self):
        _, lines = goals_from_embrace2(6, self.ALL)
        txt = "\n".join(lines)
        self.assertIn("EQD2", txt)
        self.assertIn("45", txt)

    def test_constraint_table_matches_protocol_quotes(self):
        """Pötter et al. CTRO 2018 Table 4 의 aim/limit 값 그대로.

        (aim, limit) — limit 이 없는 항목은 None.
        """
        expect = {
            ("HR_CTV", "D90_Gy"): (90.0, 85.0),
            ("HR_CTV", "D98_Gy"): (75.0, None),
            ("IR_CTV", "D98_Gy"): (60.0, None),
            ("GTV", "D98_Gy"): (95.0, 90.0),
            ("Bladder", "D2cm3_Gy"): (80.0, 90.0),
            ("Rectum", "D2cm3_Gy"): (65.0, 75.0),
            ("Sigmoid", "D2cm3_Gy"): (70.0, 75.0),
            ("Bowel", "D2cm3_Gy"): (70.0, 75.0),
        }
        by = {(c.structure, c.metric): c for c in CONSTRAINTS}
        self.assertEqual(set(by), set(expect))
        for key, (aim, lim) in expect.items():
            self.assertEqual(by[key].aim_eqd2_gy, aim, key)
            self.assertEqual(by[key].limit_eqd2_gy, lim, key)
            self.assertEqual(by[key].eqd2_gy, aim, key)   # 하위호환 별칭

    def test_alpha_beta_assignment(self):
        for c in CONSTRAINTS:
            expected = AB_TUMOR if c.kind == "min" else AB_OAR
            self.assertEqual(c.alpha_beta, expected, c.structure)


class TestReport(unittest.TestCase):
    def test_report_flags_violation(self):
        """임상 plan의 Bladder 4.33 Gy/fx 는 누적 81.3 Gy로 80 Gy 초과."""
        txt = eqd2_report({"Bladder": {"D2cm3_Gy": 4.33}}, 6)
        self.assertIn("MISS", txt)
        self.assertIn("81.3", txt)

    def test_report_ok_when_within(self):
        txt = eqd2_report({"Rectum": {"D2cm3_Gy": 2.11}}, 6)
        self.assertIn("OK", txt)
        self.assertNotIn("MISS", txt)

    def test_report_skips_missing_metrics(self):
        self.assertEqual(len(eqd2_report({}, 6).splitlines()), 2)   # 헤더만


if __name__ == "__main__":
    unittest.main(verbosity=2)
