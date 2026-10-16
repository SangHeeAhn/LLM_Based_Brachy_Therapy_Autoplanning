"""Phase 2 단위 테스트 — 목적함수와 최적화기.

합성 문제만 사용하므로 환자 데이터 없이 실행된다.
실행:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brachyplan.optim.objectives import (  # noqa: E402
    OptimizationProblem, StructureObjective, default_objectives,
)
from brachyplan.optim.optimizer import (  # noqa: E402
    HAVE_SCIPY, check_gradient, kkt_residual, optimize_dwell_times,
)


def make_problem(seed: int = 0, n_dwell: int = 6, **kw) -> OptimizationProblem:
    """합성 최적화 문제. kernel은 양수(선량은 항상 양수)."""
    rng = np.random.default_rng(seed)
    kernels = {
        "TARGET": rng.uniform(0.002, 0.02, size=(n_dwell, 120)),
        "OAR": rng.uniform(0.001, 0.010, size=(n_dwell, 200)),
    }
    objs = [
        StructureObjective("TARGET", dose_min=5.0, dose_max=7.5,
                           weight_under=100.0, weight_over=1.0),
        StructureObjective("OAR", dose_max=3.0, weight_over=5.0),
    ]
    ch = np.array([1] * (n_dwell // 2) + [2] * (n_dwell - n_dwell // 2))
    return OptimizationProblem(kernels=kernels, objectives=objs, channel_of=ch,
                               t_max=60.0, **kw)


class TestObjective(unittest.TestCase):
    def test_nonnegative(self):
        p = make_problem()
        for t in [np.zeros(6), np.full(6, 10.0), np.full(6, 60.0)]:
            self.assertGreaterEqual(p.value(t), 0.0)

    def test_gradient_matches_numeric(self):
        """해석적 gradient가 중심차분과 일치해야 한다."""
        p = make_problem(lam_smooth=0.05, lam_total=0.01, total_time_ref=200.0)
        rng = np.random.default_rng(3)
        for _ in range(5):
            t = rng.uniform(1.0, 40.0, size=6)
            self.assertLess(check_gradient(p, t), 1e-5)

    def test_gradient_at_kinks(self):
        """max(0,·)^2 는 C^1 이므로 꺾이는 지점에서도 gradient가 맞아야 한다."""
        p = make_problem()
        # 목표선량을 정확히 만족시키는 t 근처
        r = optimize_dwell_times(p, backend="fista", max_iter=300)
        self.assertLess(check_gradient(p, r.t), 1e-4)

    def test_convexity_midpoint(self):
        """볼록성: J((a+b)/2) <= (J(a)+J(b))/2."""
        p = make_problem(lam_smooth=0.02)
        rng = np.random.default_rng(11)
        for _ in range(30):
            a = rng.uniform(0, 60, size=6)
            b = rng.uniform(0, 60, size=6)
            lhs = p.value(0.5 * (a + b))
            rhs = 0.5 * (p.value(a) + p.value(b))
            self.assertLessEqual(lhs, rhs + 1e-9)

    def test_smoothness_penalty_only_within_channel(self):
        """채널 경계를 넘는 dwell 쌍에는 smoothness 벌점이 없어야 한다."""
        p = make_problem(lam_smooth=1.0)
        pairs = p._smooth_pairs()
        for i, j in pairs:
            self.assertEqual(p.channel_of[i], p.channel_of[j])
        # 채널이 2개, dwell 6개 -> (3-1) + (3-1) = 4 쌍
        self.assertEqual(len(pairs), 4)

    def test_total_time_penalty(self):
        p = make_problem(lam_total=1.0, total_time_ref=60.0)
        t = np.full(6, 10.0)   # 합 = 60 -> 벌점 0
        p_no = make_problem()
        self.assertAlmostEqual(p.value(t), p_no.value(t), places=9)

    def test_zero_weight_is_ignored(self):
        p = make_problem()
        p.set_weights({"structures": {"OAR": {"weight_over": 0.0}}})
        t = np.full(6, 30.0)
        only_target = OptimizationProblem(
            kernels={"TARGET": p.kernels["TARGET"]},
            objectives=[o for o in p.objectives if o.name == "TARGET"],
            channel_of=p.channel_of, t_max=60.0,
        )
        self.assertAlmostEqual(p.value(t), only_target.value(t), places=12)


class TestSetWeights(unittest.TestCase):
    """LLM(Phase 3) 출력 검증 — 잘못된 파라미터는 조용히 넘어가면 안 된다."""

    def test_valid_update(self):
        p = make_problem()
        p.set_weights({"structures": {"TARGET": {"weight_under": 250.0}},
                       "lam_smooth": 0.1})
        self.assertEqual(p._obj_by_name["TARGET"].weight_under, 250.0)
        self.assertEqual(p.lam_smooth, 0.1)

    def test_unknown_structure_raises(self):
        p = make_problem()
        with self.assertRaises(KeyError):
            p.set_weights({"structures": {"BLADDER_TYPO": {"dose_max": 4.0}}})

    def test_unknown_field_raises(self):
        p = make_problem()
        with self.assertRaises(KeyError):
            p.set_weights({"structures": {"OAR": {"dose_limit": 4.0}}})

    def test_unknown_top_level_raises(self):
        p = make_problem()
        with self.assertRaises(KeyError):
            p.set_weights({"learning_rate": 0.1})

    def test_negative_weight_raises(self):
        p = make_problem()
        with self.assertRaises(ValueError):
            p.set_weights({"structures": {"OAR": {"weight_over": -1.0}}})

    def test_inverted_dose_range_raises(self):
        p = make_problem()
        with self.assertRaises(ValueError):
            p.set_weights({"structures": {"TARGET": {"dose_min": 9.0, "dose_max": 2.0}}})

    def test_roundtrip(self):
        p = make_problem()
        w = p.get_weights()
        p.set_weights(w)
        self.assertEqual(p.get_weights(), w)


class TestOptimizer(unittest.TestCase):
    def test_respects_box_constraints(self):
        p = make_problem()
        r = optimize_dwell_times(p)
        self.assertTrue(np.all(r.t >= -1e-9))
        self.assertTrue(np.all(r.t <= p.t_max + 1e-9))

    def test_reduces_objective(self):
        p = make_problem()
        t0 = np.full(p.n_dwell, 1.0)
        r = optimize_dwell_times(p, t0=t0)
        self.assertLess(r.objective, p.value(t0))

    def test_kkt_satisfied(self):
        p = make_problem(lam_smooth=0.01)
        r = optimize_dwell_times(p)
        self.assertLess(r.kkt_residual, 1e-3)

    def test_global_optimum_from_any_start(self):
        """볼록 문제이므로 초기값과 무관하게 같은 목적함수 값에 도달해야 한다."""
        p = make_problem(lam_smooth=0.01)
        rng = np.random.default_rng(7)
        vals = []
        for _ in range(6):
            t0 = rng.uniform(0, 60, size=p.n_dwell)
            vals.append(optimize_dwell_times(p, t0=t0).objective)
        self.assertLess(max(vals) - min(vals), 1e-6 * max(1.0, abs(vals[0])))

    @unittest.skipUnless(HAVE_SCIPY, "scipy 없음")
    def test_backends_agree(self):
        """L-BFGS-B와 FISTA가 같은 최적값에 도달해야 한다."""
        p = make_problem(lam_smooth=0.01)
        a = optimize_dwell_times(p, backend="lbfgsb")
        b = optimize_dwell_times(p, backend="fista", max_iter=20000, tol=1e-11)
        self.assertAlmostEqual(a.objective, b.objective,
                               delta=1e-4 * max(1.0, abs(a.objective)))

    def test_t_max_binding(self):
        """t_max를 아주 작게 하면 모든 dwell이 상한에 붙어야 한다
        (표적 커버리지가 부족한 상황)."""
        p = make_problem()
        p.t_max = 0.5
        r = optimize_dwell_times(p)
        np.testing.assert_allclose(r.t, 0.5, atol=1e-6)

    def test_oar_only_gives_zero_time(self):
        """표적 목표를 없애고 OAR 한계를 0으로 두면 유일한 최적해는 t=0 이다.

        (한계를 0보다 크게 두면 벌점이 0인 t가 여러 개라 해가 유일하지 않다.)
        """
        p = make_problem()
        p.set_weights({
            "structures": {
                "TARGET": {"weight_under": 0.0, "weight_over": 0.0},
                "OAR": {"dose_max": 0.0, "weight_over": 5.0},
            }
        })
        r = optimize_dwell_times(p, t0=np.full(p.n_dwell, 20.0))
        np.testing.assert_allclose(r.t, 0.0, atol=1e-5)

    def test_slack_solution_is_already_optimal(self):
        """모든 목표가 이미 충족된 t에서는 최적화가 J=0을 유지해야 한다.

        (해가 유일하지 않은 경우 — 최적화기가 불필요하게 t를 움직이면 안 된다.)
        """
        p = make_problem()
        p.set_weights({"structures": {"TARGET": {"weight_under": 0.0,
                                                 "weight_over": 0.0}}})
        t0 = np.full(p.n_dwell, 20.0)
        self.assertAlmostEqual(p.value(t0), 0.0, places=12)
        r = optimize_dwell_times(p, t0=t0)
        self.assertAlmostEqual(r.objective, 0.0, places=12)

    def test_smoothness_reduces_variation(self):
        """smoothness 가중치를 올리면 채널 내 dwell time 변동이 줄어야 한다."""
        def variation(lam):
            p = make_problem(lam_smooth=lam)
            t = optimize_dwell_times(p).t
            return sum(abs(t[i] - t[j]) for i, j in p._smooth_pairs())

        self.assertLess(variation(1.0), variation(0.0) + 1e-9)

    def test_rejects_bad_t0_shape(self):
        p = make_problem()
        with self.assertRaises(ValueError):
            optimize_dwell_times(p, t0=np.zeros(3))

    def test_kernel_dwell_mismatch_raises(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            OptimizationProblem(
                kernels={"A": rng.random((5, 10)), "B": rng.random((4, 10))},
                objectives=[StructureObjective("A", dose_max=1.0, weight_over=1.0)],
                channel_of=np.ones(5),
            )

    def test_objective_without_kernel_raises(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(KeyError):
            OptimizationProblem(
                kernels={"A": rng.random((5, 10))},
                objectives=[StructureObjective("MISSING", dose_max=1.0, weight_over=1.0)],
                channel_of=np.ones(5),
            )


class TestKKT(unittest.TestCase):
    def test_residual_zero_at_solution(self):
        p = make_problem()
        r = optimize_dwell_times(p)
        self.assertLess(kkt_residual(p, r.t), 1e-3)

    def test_residual_positive_away_from_solution(self):
        p = make_problem()
        self.assertGreater(kkt_residual(p, np.full(p.n_dwell, 0.1)), 1e-6)

    def test_lower_bound_active_is_optimal(self):
        """하한 t=0 에서 gradient가 양수여도 KKT 잔차는 0이어야 한다.

        OAR 한계를 0으로 두면 t를 늘릴수록 벌점이 늘어 gradient > 0 이지만,
        t=0 은 하한이라 더 줄일 수 없으므로 최적이다.
        """
        p = make_problem()
        p.set_weights({
            "structures": {
                "TARGET": {"weight_under": 0.0, "weight_over": 0.0},
                "OAR": {"dose_max": 0.0, "weight_over": 5.0},
            }
        })
        t = np.full(p.n_dwell, 1e-3)
        _, g = p.value_and_grad(t)
        self.assertTrue(np.all(g > 0), "gradient가 양수여야 하는 설정")
        self.assertLess(kkt_residual(p, np.zeros(p.n_dwell)), 1e-12)


class TestDefaultObjectives(unittest.TestCase):
    def test_structure(self):
        objs = default_objectives(5.0, ("HR_CTV",), {"Bladder": 4.3, "Rectum": 2.1})
        by = {o.name: o for o in objs}
        self.assertEqual(by["HR_CTV"].dose_min, 5.0)
        self.assertEqual(by["Bladder"].dose_max, 4.3)
        self.assertIsNone(by["Bladder"].dose_min)

    def test_hot_spot_disabled_by_default(self):
        """근접치료에서는 표적 내 고선량이 정상이므로 dose_max를 걸지 않는다."""
        by = {o.name: o for o in default_objectives(5.0, ("HR_CTV",))}
        self.assertIsNone(by["HR_CTV"].dose_max)
        self.assertEqual(by["HR_CTV"].weight_over, 0.0)

    def test_hot_spot_when_requested(self):
        by = {o.name: o for o in
              default_objectives(5.0, ("HR_CTV",), hot_spot_factor=4.0)}
        self.assertAlmostEqual(by["HR_CTV"].dose_max, 20.0)
        self.assertGreater(by["HR_CTV"].weight_over, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
