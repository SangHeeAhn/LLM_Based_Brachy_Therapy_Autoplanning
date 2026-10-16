"""실제 케이스 통합 테스트 (Geneva_Case가 있을 때만 실행).

Phase 1/2에서 어렵게 찾아낸 사실들이 코드 변경으로 되돌아가지 않도록 잠근다.
케이스 폴더가 없으면 전체가 skip 되므로 데이터 없는 환경에서도 안전하다.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CASE = os.path.join(ROOT, "Geneva_Case")
HAS_CASE = os.path.isdir(CASE) and any(f.endswith(".dcm") for f in os.listdir(CASE))

if HAS_CASE:
    from brachyplan.io import check_frame_of_reference, contour_frame, find_files
    from brachyplan.optim.objectives import StructureObjective, default_objectives
    from brachyplan.optim.optimizer import check_gradient, optimize_dwell_times
    from brachyplan.optim.pipeline import (
        build_kernels, clinical_oar_limits, iso_target_match, load_context,
        make_problem,
    )
    from brachyplan.qa.point_check import check_dose_points

OARS = ["Bladder", "Rectum", "Sigmoid", "Bowel", "Urethra", "Anus"]


@unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
class TestCaseFacts(unittest.TestCase):
    """데이터 자체에 대해 확인된 사실 — 로더가 이를 계속 올바로 읽어야 한다."""

    @classmethod
    def setUpClass(cls):
        cls.files = find_files(CASE)
        cls.frame = contour_frame(cls.files["RTSTRUCT"][0])

    def test_contours_are_oblique(self):
        """컨투어 평면이 축방향이 아니다 (para-transverse MR). 약 7.9도."""
        self.assertGreater(self.frame.tilt_deg, 5.0)
        self.assertLess(self.frame.tilt_deg, 10.0)

    def test_contours_are_planar(self):
        """각 컨투어는 평면 위에 있다 (SVD 잔차가 0에 가까움)."""
        self.assertLess(self.frame.planarity_mm, 0.01)

    def test_frame_of_reference_check_matches_data(self):
        """FoR 불일치 검사가 실제 데이터와 일치해야 한다.

        케이스에 따라 영상이 구조물과 다른 FoR일 수도(등록 필요), 같을 수도
        있다. 어느 쪽이든 `check_frame_of_reference`의 판정이 실제 UID
        집합과 어긋나지 않아야 한다 — 특정 케이스의 사실을 박아 두면
        데이터가 바뀔 때마다 깨진다.
        """
        fors, warnings = check_frame_of_reference(self.files)
        self.assertIn("RTSTRUCT", fors)
        self.assertIn("RTPLAN", fors)
        has_warning = any("FrameOfReference" in w for w in warnings)
        self.assertEqual(has_warning, len(set(fors.values())) > 1,
                         f"FoR = {fors}, 경고 = {warnings}")

    def test_contour_normal_matches_applicator_axis(self):
        """평면 법선이 tandem 축과 거의 평행 (applicator 기준 촬영)."""
        from brachyplan.io import load_plan
        plan = load_plan(self.files["RTPLAN"][0])
        ch = np.array([d.channel for d in plan.dwells])
        # dwell이 가장 많은 채널 = tandem
        tandem_ch = np.bincount(ch).argmax()
        idx = np.flatnonzero(ch == tandem_ch)
        p0 = plan.dwells[idx[0]].position
        p1 = plan.dwells[idx[-1]].position
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        ang = np.degrees(np.arccos(abs(float(axis @ self.frame.normal))))
        self.assertLess(ang, 5.0)


@unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
class TestPhase1Commissioning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = load_context(CASE, spacing_mm=3.0)   # 테스트는 거친 격자로 빠르게

    def test_dose_points_within_tolerance(self):
        """TPS dose point 7개가 ±2% 안에 들어야 한다 (Phase 1 완료 기준)."""
        results, ok = check_dose_points(self.ctx.calc, self.ctx.plan, self.ctx.sk, 2.0)
        self.assertEqual(len(results), 7)
        for r in results:
            self.assertLess(abs(r.diff_pct), 2.0, f"{r.name}: {r.diff_pct:.2f}%")
        self.assertTrue(ok)

    def test_no_decay_correction_by_default(self):
        """붕괴보정을 하지 않는 것이 기본. 하면 계통오차가 생긴다."""
        self.assertAlmostEqual(self.ctx.sk, self.ctx.plan.sk_ref, places=6)

    def test_decay_correction_worsens_agreement(self):
        """ContentDate로 붕괴보정하면 오차가 커진다 — 되돌아가지 않도록 잠금."""
        ctx2 = load_context(CASE, spacing_mm=3.0, decay_to="20260709")
        r_no, _ = check_dose_points(self.ctx.calc, self.ctx.plan, self.ctx.sk, 2.0)
        r_dc, _ = check_dose_points(ctx2.calc, ctx2.plan, ctx2.sk, 2.0)
        self.assertLess(np.mean([abs(r.diff_pct) for r in r_no]),
                        np.mean([abs(r.diff_pct) for r in r_dc]))

    def test_plan_basics(self):
        self.assertEqual(len(self.ctx.plan.dwells), 19)
        self.assertAlmostEqual(self.ctx.plan.total_time_s(), 245.0, places=3)
        self.assertAlmostEqual(self.ctx.prescription_gy, 5.0, places=6)
        self.assertEqual(len(set(d.channel for d in self.ctx.plan.dwells)), 3)


@unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
class TestPhase2Optimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = load_context(CASE, spacing_mm=3.0)
        build_kernels(cls.ctx)
        cls.clin = cls.ctx.evaluate(cls.ctx.clinical_t)
        cls.limits = {k: v for k, v in clinical_oar_limits(cls.ctx).items() if k in OARS}

    def _problem(self, smooth=0.05, secondary=True, hot=None):
        objs = default_objectives(5.0, ("HR_CTV",), self.limits,
                                  oar_weight=1.0, target_weight_under=100.0,
                                  hot_spot_factor=hot)
        if secondary:
            objs.append(StructureObjective("IR_CTV",
                                           dose_min=self.clin["IR_CTV"]["D90_Gy"],
                                           weight_under=50.0))
        return make_problem(self.ctx, objs, t_max=200.0, lam_smooth=smooth)

    def test_kernel_reproduces_dose(self):
        """A^T t 로 계산한 구조물 선량이 직접 TG-43 계산과 일치."""
        t = self.ctx.clinical_t
        from brachyplan.dose.kernel import compute_kernel
        name = "HR_CTV"
        idx = self.ctx.voxel_index[name]
        pts = self.ctx.grid.points()[idx]
        direct = self.ctx.calc.dose_at_points(pts, self.ctx.plan.dwells, self.ctx.sk)
        via = self.ctx.kernels[name].T @ t
        np.testing.assert_allclose(via, direct, rtol=1e-10)

    def test_gradient_accurate(self):
        p = self._problem()
        self.assertLess(check_gradient(p, self.ctx.clinical_t), 1e-5)

    def test_convex_global_optimum(self):
        """초기값이 달라도 같은 목적함수 값에 수렴해야 한다."""
        p = self._problem()
        rng = np.random.default_rng(0)
        vals = [
            optimize_dwell_times(p, t0=t0).objective
            for t0 in [self.ctx.clinical_t, np.zeros(19), np.full(19, 10.0),
                       rng.uniform(0, 200, 19)]
        ]
        self.assertLess(max(vals) - min(vals), 1e-6 * max(1.0, abs(vals[0])))

    def test_iso_d90_beats_clinical_on_oars(self):
        """표적 커버리지를 임상과 맞추면 모든 OAR이 임상 이하여야 한다."""
        p = self._problem()
        d90c = self.clin["HR_CTV"]["D90_Gy"]
        t, _, _ = iso_target_match(self.ctx, p, "HR_CTV", d90c, OARS)
        ev = self.ctx.evaluate(t)
        self.assertGreaterEqual(ev["HR_CTV"]["D90_Gy"], d90c - 0.01)
        for n in OARS:
            self.assertLessEqual(ev[n]["D2cm3_Gy"], self.clin[n]["D2cm3_Gy"] + 1e-6,
                                 f"{n} D2cc 악화")

    def test_hot_spot_penalty_hurts_target(self):
        """표적 dose_max(처방 150%)를 걸면 D90이 오히려 나빠진다 — 재발 방지."""
        d90c = self.clin["HR_CTV"]["D90_Gy"]
        with_hot = optimize_dwell_times(self._problem(hot=1.5, secondary=False)).t
        no_hot = optimize_dwell_times(self._problem(hot=None, secondary=False)).t
        d_hot = self.ctx.evaluate(with_hot)["HR_CTV"]["D90_Gy"]
        d_no = self.ctx.evaluate(no_hot)["HR_CTV"]["D90_Gy"]
        self.assertLess(d_hot, d_no)
        self.assertLess(d_hot, d90c)

    def test_smoothness_keeps_dwells_active(self):
        """평활항이 없으면 dwell이 몇 개 위치에 몰린다 (IPSA의 알려진 단점)."""
        t_rough = optimize_dwell_times(self._problem(smooth=0.0)).t
        t_smooth = optimize_dwell_times(self._problem(smooth=0.05)).t
        self.assertLess(int(np.sum(t_rough > 0.5)), int(np.sum(t_smooth > 0.5)))
        cv = lambda t: float(np.std(t) / max(np.mean(t), 1e-9))  # noqa: E731
        self.assertLess(cv(t_smooth), cv(t_rough))

    def test_secondary_target_protects_ir_ctv(self):
        """부목표를 넣으면 IR_CTV 커버리지 손실이 줄어야 한다."""
        t_wo = optimize_dwell_times(self._problem(secondary=False)).t
        t_w = optimize_dwell_times(self._problem(secondary=True)).t
        self.assertGreater(self.ctx.evaluate(t_w)["IR_CTV"]["D90_Gy"],
                           self.ctx.evaluate(t_wo)["IR_CTV"]["D90_Gy"])


@unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
class TestVisualization(unittest.TestCase):
    """시각화 도구 — 데이터 수집과 HTML 생성."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import visualize_dwell as viz
        cls.viz = viz
        cls.tmp = tempfile.mkdtemp()
        # 실제 최적화 결과가 없어도 돌도록 합성 dwell time을 만든다
        from brachyplan.io import find_files, load_plan
        plan = load_plan(find_files(CASE)["RTPLAN"][0])
        cls.n = len(plan.dwells)
        cls.dwell_path = os.path.join(cls.tmp, "t.npy")
        np.save(cls.dwell_path, np.linspace(1.0, 20.0, cls.n))

    def test_collect_data_shape(self):
        d = self.viz.collect_data(CASE, self.dwell_path, ["HR_CTV", "Bladder"], 48)
        self.assertEqual(len(d["dwells"]), self.n)
        self.assertEqual(len(d["channels"]), 3)
        self.assertEqual(set(d["structures"]), {"HR_CTV", "Bladder"})
        for dw in d["dwells"]:
            self.assertEqual(len(dw["p"]), 3)
            self.assertIn("tc", dw)
            self.assertIn("to", dw)

    def test_contour_points_are_subsampled(self):
        d = self.viz.collect_data(CASE, self.dwell_path, ["Bladder"], 16)
        for loop in d["structures"]["Bladder"]["loops"]:
            self.assertLessEqual(len(loop), 16)

    def test_stats_match_input(self):
        d = self.viz.collect_data(CASE, self.dwell_path, [], 48)
        t = np.load(self.dwell_path)
        self.assertAlmostEqual(d["statsOpt"]["total"], round(float(t.sum()), 1), places=6)
        self.assertEqual(d["statsOpt"]["active"], int(np.sum(t > 0.5)))
        self.assertAlmostEqual(d["statsClin"]["total"], 245.0, places=3)

    def test_missing_file_raises_with_hint(self):
        with self.assertRaises(FileNotFoundError) as cm:
            self.viz.collect_data(CASE, os.path.join(self.tmp, "nope.npy"), [], 48)
        self.assertIn("run_phase2.py", str(cm.exception))

    def test_dwell_count_mismatch_raises(self):
        bad = os.path.join(self.tmp, "bad.npy")
        np.save(bad, np.ones(self.n + 3))
        with self.assertRaises(ValueError) as cm:
            self.viz.collect_data(CASE, bad, [], 48)
        self.assertIn("불일치", str(cm.exception))

    def test_html_is_self_contained(self):
        import json as _json
        import re
        d = self.viz.collect_data(CASE, self.dwell_path, ["HR_CTV"], 48)
        html = self.viz.build_html(d)
        # 치환되지 않은 자리표시자가 남아 있으면 안 된다
        self.assertNotIn("__DATA__", html)
        self.assertNotIn("__CASE__", html)
        # 외부 리소스를 불러오지 않아야 한다
        self.assertNotRegex(html, r'<(script|link)[^>]+(src|href)\s*=\s*["\']https?:')
        # 내장 데이터가 유효한 JSON이어야 한다
        m = re.search(r"const D = (\{.*?\});\n", html, re.S)
        self.assertIsNotNone(m)
        parsed = _json.loads(m.group(1))
        self.assertEqual(len(parsed["dwells"]), self.n)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
