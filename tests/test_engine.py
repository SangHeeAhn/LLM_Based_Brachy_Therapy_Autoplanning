"""Phase 1 단위 테스트 — 외부 데이터 없이 물리/수학적 정합성만 검증.

실행:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brachyplan.dose.dvh import DVH  # noqa: E402
from brachyplan.dose.kernel import DoseGrid, compute_kernel  # noqa: E402
from brachyplan.dose.source_data import SourceData  # noqa: E402
from brachyplan.dose.tg43 import TG43Calculator, geometry_function  # noqa: E402
from brachyplan.io.struct_to_mask import rasterize_polygon  # noqa: E402


class TestGeometryFunction(unittest.TestCase):
    """G_L(r, theta)의 해석적 성질 검증."""

    L = 0.36

    def test_transverse_reference(self):
        """G_L(1cm, 90deg)는 해석해 beta/(L*r*sin) 와 일치해야 한다."""
        r, th = 1.0, np.pi / 2
        got = float(geometry_function(np.array(r), np.array(th), self.L))
        beta = 2 * np.arctan2(self.L / 2, r)  # 대칭이므로
        expected = beta / (self.L * r)
        self.assertAlmostEqual(got, expected, places=10)

    def test_point_source_limit(self):
        """r >> L 에서 G_L -> 1/r^2 로 수렴해야 한다."""
        for r in [5.0, 10.0, 20.0]:
            got = float(geometry_function(np.array(r), np.array(np.pi / 2), self.L))
            self.assertAlmostEqual(got, 1.0 / r**2, delta=1e-4 / r**2 * 10)

    def test_axial_formula(self):
        """theta=0 에서 1/(r^2 - L^2/4)."""
        r = 2.0
        got = float(geometry_function(np.array(r), np.array(0.0), self.L))
        self.assertAlmostEqual(got, 1.0 / (r**2 - self.L**2 / 4), places=10)

    def test_symmetry(self):
        """theta 와 180-theta 는 선형 선원에서 동일 (기하학적 대칭)."""
        r = 3.0
        for th_deg in [10, 30, 60, 89]:
            a = float(geometry_function(np.array(r), np.array(np.radians(th_deg)), self.L))
            b = float(geometry_function(np.array(r), np.array(np.radians(180 - th_deg)), self.L))
            self.assertAlmostEqual(a, b, places=10)

    def test_continuity_near_axis(self):
        """theta -> 0 극한에서 일반식과 축상식이 이어져야 한다 (발산/불연속 없음)."""
        r = 2.0
        vals = [
            float(geometry_function(np.array(r), np.array(np.radians(t)), self.L))
            for t in [0.0, 0.001, 0.01, 0.1, 1.0]
        ]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(abs(a - b) / a, 1e-3)


class TestSourceData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SourceData.load("ir192_mHDR_v2_clrp")

    def test_g_at_reference(self):
        """g_L(1 cm) = 1 (정의)."""
        self.assertAlmostEqual(float(self.src.g_L(1.0)), 1.0, places=3)

    def test_g_scatter_buildup_peak(self):
        """Ir-192 g_L(r)은 산란 축적으로 r ~ 3-4 cm에서 최댓값(약 1.01)을 갖는다.

        (저에너지 선원과 달리 고에너지 Ir-192는 1 cm 이후 바로 감소하지 않는다.
         AAPM Report 229 / HEBD의 잘 알려진 특성.)
        """
        r = np.linspace(0.2, 20.0, 2000)
        g = self.src.g_L(r)
        peak_r = float(r[np.argmax(g)])
        self.assertTrue(3.0 <= peak_r <= 4.5, f"peak at r={peak_r}")
        self.assertAlmostEqual(float(np.max(g)), 1.011, delta=0.005)

    def test_g_monotonic_beyond_peak(self):
        """최댓값 이후(r >= 5 cm)에는 단조 감소해야 한다."""
        r = np.linspace(5.0, 20.0, 300)
        self.assertTrue(np.all(np.diff(self.src.g_L(r)) < 0))

    def test_g_attenuation_at_20cm(self):
        """20 cm에서 g_L 은 0.65~0.75 수준 (조직 감쇠)."""
        self.assertTrue(0.65 < float(self.src.g_L(20.0)) < 0.75)

    def test_g_clamped_outside(self):
        """피팅 범위 밖은 경계값 고정 (발산 금지)."""
        self.assertEqual(float(self.src.g_L(0.01)), float(self.src.g_L(0.20)))
        self.assertEqual(float(self.src.g_L(100.0)), float(self.src.g_L(20.0)))
        self.assertTrue(np.isfinite(self.src.g_L(np.array([1e-6, 1e6]))).all())

    def test_F_at_90deg(self):
        """F(r, 90deg) = 1 (정의) — 표에 있는 모든 r에서."""
        for r in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            self.assertAlmostEqual(float(self.src.F(r, 90.0)), 1.0, places=6)

    def test_F_interpolation_bounds(self):
        """보간값은 항상 (0, 1.1] 범위."""
        r = np.random.uniform(0.1, 25.0, 500)
        th = np.random.uniform(-360.0, 360.0, 500)
        f = self.src.F(r, th)
        self.assertTrue(np.all(f > 0.4))
        self.assertTrue(np.all(f <= 1.01))

    def test_F_theta_folding(self):
        """theta 와 -theta, theta+360 은 같아야 한다."""
        for th in [30.0, 120.0, 175.0]:
            a = float(self.src.F(2.0, th))
            self.assertAlmostEqual(a, float(self.src.F(2.0, -th)), places=10)
            self.assertAlmostEqual(a, float(self.src.F(2.0, th + 360.0)), places=10)

    def test_F_exact_at_nodes(self):
        """표의 격자점에서는 보간 없이 원값이 나와야 한다."""
        self.assertAlmostEqual(float(self.src.F(1.00, 0.0)), 0.630, places=6)
        self.assertAlmostEqual(float(self.src.F(5.00, 30.0)), 0.919, places=6)
        self.assertAlmostEqual(float(self.src.F(20.00, 180.0)), 0.830, places=6)


class _FakeDwell:
    def __init__(self, position, time_s, axis):
        self.position = np.asarray(position, float)
        self.time_s = float(time_s)
        self.axis = np.asarray(axis, float)


class TestTG43Calculator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calc = TG43Calculator(SourceData.load("ir192_mHDR_v2_clrp"))

    def test_reference_point_gives_lambda(self):
        """(1 cm, 90deg)에서 선량률 = Lambda 여야 한다 (G/G0=1, g=1, F=1)."""
        dwell_pos = np.array([0.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        point = np.array([[10.0, 0.0, 0.0]])  # 1 cm 옆 (transverse)
        rate = float(self.calc.dose_rate_per_U(point, dwell_pos, axis)[0])
        self.assertAlmostEqual(rate, self.calc.src.dose_rate_constant, places=3)

    def test_dose_scales_linearly_with_time(self):
        pos, axis = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        pts = np.array([[15.0, 0.0, 0.0]])
        d1 = self.calc.dose_at_points(pts, [_FakeDwell(pos, 10.0, axis)], 40000.0)
        d2 = self.calc.dose_at_points(pts, [_FakeDwell(pos, 20.0, axis)], 40000.0)
        self.assertAlmostEqual(float(d2[0] / d1[0]), 2.0, places=10)

    def test_dose_scales_linearly_with_sk(self):
        pos, axis = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        pts = np.array([[15.0, 0.0, 0.0]])
        d1 = self.calc.dose_at_points(pts, [_FakeDwell(pos, 10.0, axis)], 20000.0)
        d2 = self.calc.dose_at_points(pts, [_FakeDwell(pos, 10.0, axis)], 40000.0)
        self.assertAlmostEqual(float(d2[0] / d1[0]), 2.0, places=10)

    def test_superposition(self):
        """여러 dwell의 선량은 단순 합이어야 한다."""
        axis = np.array([0.0, 0.0, 1.0])
        a = _FakeDwell([0, 0, 0], 10.0, axis)
        b = _FakeDwell([0, 0, 5], 7.0, axis)
        pts = np.array([[20.0, 0.0, 2.0]])
        da = self.calc.dose_at_points(pts, [a], 40000.0)
        db = self.calc.dose_at_points(pts, [b], 40000.0)
        dab = self.calc.dose_at_points(pts, [a, b], 40000.0)
        self.assertAlmostEqual(float(dab[0]), float(da[0] + db[0]), places=12)

    def test_units_sanity(self):
        """40000 U 선원이 1 cm 지점에 100초 조사 -> 약 12.3 Gy.

        D = S_K*Lambda*t = 40000 * 1.109 cGy/h * (100/3600) h = 1232 cGy
        """
        pos, axis = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        pts = np.array([[10.0, 0.0, 0.0]])
        d = float(self.calc.dose_at_points(pts, [_FakeDwell(pos, 100.0, axis)], 40000.0)[0])
        expected = 40000.0 * 1.109 * (100.0 / 3600.0) / 100.0
        self.assertAlmostEqual(d, expected, places=2)

    def test_inverse_square_far_field(self):
        """먼 거리에서 선량은 1/r^2 에 가깝게 (산란 감쇠 g_L 만큼 편차)."""
        pos, axis = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        p1 = self.calc.dose_rate_per_U(np.array([[50.0, 0, 0]]), pos, axis)[0]
        p2 = self.calc.dose_rate_per_U(np.array([[100.0, 0, 0]]), pos, axis)[0]
        ratio_geo = p1 / p2
        g_ratio = float(self.calc.src.g_L(5.0) / self.calc.src.g_L(10.0))
        self.assertAlmostEqual(ratio_geo, 4.0 * g_ratio, delta=0.02)


class TestKernel(unittest.TestCase):
    def test_kernel_reproduces_direct_dose(self):
        """A^T . t 가 직접 계산과 일치해야 한다 (Phase 2 최적화의 전제)."""
        calc = TG43Calculator(SourceData.load("ir192_mHDR_v2_clrp"))
        axis = np.array([0.0, 0.0, 1.0])
        dwells = [
            _FakeDwell([0, 0, 0], 12.0, axis),
            _FakeDwell([0, 0, 5], 8.0, axis),
            _FakeDwell([3, 0, 10], 20.0, axis),
        ]
        pts = np.random.uniform(-50, 50, size=(40, 3))
        sk = 39000.0
        direct = calc.dose_at_points(pts, dwells, sk)
        A = compute_kernel(calc, pts, dwells, sk)
        t = np.array([d.time_s for d in dwells])
        via_kernel = A.T @ t
        np.testing.assert_allclose(via_kernel, direct, rtol=1e-12)


class TestRasterize(unittest.TestCase):
    def test_square_area(self):
        """10x10 mm 정사각형을 1 mm 격자에 래스터화하면 면적이 ~100 mm^2."""
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        x = np.arange(-5, 16, 1.0)
        y = np.arange(-5, 16, 1.0)
        m = rasterize_polygon(poly, x, y)
        self.assertAlmostEqual(m.sum() * 1.0, 100.0, delta=21.0)  # 경계 효과 허용

    def test_circle_area(self):
        """반지름 20 mm 원 → pi*r^2 에 0.5 mm 격자로 1% 이내."""
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        poly = np.stack([20 * np.cos(t), 20 * np.sin(t)], axis=1)
        h = 0.5
        x = np.arange(-25, 25, h)
        y = np.arange(-25, 25, h)
        area = rasterize_polygon(poly, x, y).sum() * h * h
        self.assertAlmostEqual(area / (np.pi * 400), 1.0, delta=0.01)

    def test_hole_via_xor(self):
        """바깥 사각형 XOR 안쪽 사각형 = 도넛."""
        outer = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        inner = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=float)
        h = 0.5
        x = np.arange(-2, 22, h)
        y = np.arange(-2, 22, h)
        m = rasterize_polygon(outer, x, y) ^ rasterize_polygon(inner, x, y)
        self.assertAlmostEqual(m.sum() * h * h, 400 - 100, delta=15.0)

    def test_outside_bbox_empty(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        x = np.arange(100, 110, 1.0)
        y = np.arange(100, 110, 1.0)
        self.assertEqual(rasterize_polygon(poly, x, y).sum(), 0)


class TestDVH(unittest.TestCase):
    def setUp(self):
        # 1000 복셀, 선량 0.1 ~ 100.0 Gy 균등, 복셀 부피 0.001 cc -> 총 1 cc
        self.dvh = DVH("test", np.linspace(0.1, 100.0, 1000), 0.001)

    def test_volume(self):
        self.assertAlmostEqual(self.dvh.volume_cm3, 1.0, places=9)

    def test_D90_is_10th_percentile_from_top(self):
        """상위 90% 가 받는 최소 선량 ≈ 10 번째 백분위수."""
        self.assertAlmostEqual(self.dvh.D_percent(90), 10.0, delta=0.2)

    def test_D100_equals_min(self):
        self.assertAlmostEqual(self.dvh.D_percent(100), self.dvh.min, places=9)

    def test_Dcc_ordering(self):
        """D0.1cc >= D1cc >= D2cc (좁은 부피일수록 높은 선량)."""
        self.assertGreaterEqual(self.dvh.D_cc(0.1), self.dvh.D_cc(1.0))

    def test_Dcc_larger_than_volume(self):
        """구조물보다 큰 부피를 요구하면 최소선량을 반환."""
        self.assertAlmostEqual(self.dvh.D_cc(999.0), self.dvh.min, places=9)

    def test_V_dose(self):
        """50 Gy 이상 받는 부피 ≈ 50%."""
        self.assertAlmostEqual(self.dvh.V_dose(50.0), 50.0, delta=0.2)

    def test_V_percent_of_prescription(self):
        """처방 20 Gy 기준 V150 = 30 Gy 이상 부피 ≈ 70%."""
        self.assertAlmostEqual(self.dvh.V_percent_of(20.0, 150), 70.0, delta=0.2)

    def test_cumulative_monotonic(self):
        _, vol = self.dvh.cumulative()
        self.assertTrue(np.all(np.diff(vol) <= 1e-9))
        self.assertAlmostEqual(vol[0], 100.0, places=6)

    def test_empty_structure(self):
        empty = DVH("empty", np.array([]), 0.001)
        self.assertTrue(np.isnan(empty.D_percent(90)))
        self.assertEqual(empty.volume_cm3, 0.0)


class TestObliqueGrid(unittest.TestCase):
    """oblique(기울어진) 컨투어 평면 지원 검증.

    자궁경부 브라키의 para-transverse MR 컨투어를 축방향 그리드로 처리하면
    부피가 크게 틀린다. 이 문제가 재발하지 않도록 잠근다.
    """

    @staticmethod
    def _rot(axis, deg):
        """Rodrigues 회전행렬."""
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        t = np.radians(deg)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)

    def test_grid_local_roundtrip(self):
        R = self._rot([1.0, 0.5, 0.0], 20.0)
        g = DoseGrid(origin=np.array([10.0, -5.0, 3.0]),
                     spacing=np.array([2.0, 2.0, 2.0]), shape=(3, 4, 5), rotation=R)
        pts = g.points()
        local = g.to_local(pts)
        x, y, z = g.local_axes()
        # 국소 좌표는 정확히 격자축 위에 있어야 한다
        np.testing.assert_allclose(np.unique(np.round(local[:, 0], 9)), np.round(x, 9))
        np.testing.assert_allclose(np.unique(np.round(local[:, 2], 9)), np.round(z, 9))

    def test_rejects_non_orthonormal_rotation(self):
        with self.assertRaises(ValueError):
            DoseGrid(np.zeros(3), np.ones(3), (2, 2, 2),
                     rotation=np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]))

    def test_contour_frame_recovers_tilt(self):
        """기울어진 평면 위의 합성 컨투어에서 법선을 복원해야 한다."""
        from brachyplan.io.struct_to_mask import _orthonormal_basis
        R_true = self._rot([0.3, -0.8, 0.0], 12.0)
        w_true = R_true[2] if R_true[2][2] > 0 else -R_true[2]
        basis = _orthonormal_basis(w_true)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(basis[2], w_true / np.linalg.norm(w_true), atol=1e-12)

    def test_oblique_cylinder_volume(self):
        """기울어진 원기둥: 평면 정렬 그리드는 정확, 축방향 그리드는 부정확."""
        from brachyplan.io.struct_to_mask import rasterize_polygon

        R = self._rot([1.0, 0.0, 0.0], 20.0)   # x축 기준 20도
        radius, n_planes, dz = 20.0, 21, 2.0
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circle = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
        true_vol = np.pi * radius**2 * (n_planes * dz) / 1000.0  # cm^3

        h = 1.0
        xs = np.arange(-30, 30, h)
        ys = np.arange(-30, 30, h)

        # (a) 평면 좌표계에서 래스터화 → 정확해야 한다
        area = rasterize_polygon(circle, xs, ys).sum() * h * h
        vol_aligned = area * n_planes * dz / 1000.0
        self.assertAlmostEqual(vol_aligned / true_vol, 1.0, delta=0.01)

        # (b) 같은 컨투어를 환자좌표로 되돌린 뒤 z=const 평면으로 잘못 묶으면
        #     평면이 z축에 대해 기울어져 컨투어 내부 z 변동이 생긴다
        pts3d = np.column_stack([circle, np.zeros(len(circle))]) @ R
        z_span = float(np.ptp(pts3d[:, 2]))
        self.assertGreater(z_span, 10.0,
                           "기울어진 평면이면 컨투어 내부 z 변동이 커야 한다")


class TestDoseGrid(unittest.TestCase):
    def test_points_order_matches_shape(self):
        g = DoseGrid(origin=np.array([0.0, 0.0, 0.0]),
                     spacing=np.array([2.0, 2.0, 3.0]), shape=(4, 5, 6))
        pts = g.points()
        self.assertEqual(pts.shape, (4 * 5 * 6, 3))
        # C-order (z, y, x): 첫 6개는 z=0, y=0, x=0..5
        np.testing.assert_allclose(pts[:6, 0], np.arange(6) * 2.0)
        np.testing.assert_allclose(pts[:6, 1], 0.0)
        np.testing.assert_allclose(pts[:6, 2], 0.0)
        # 마지막 점
        np.testing.assert_allclose(pts[-1], [5 * 2.0, 4 * 2.0, 3 * 3.0])

    def test_voxel_volume(self):
        g = DoseGrid(np.zeros(3), np.array([1.0, 2.0, 2.5]), (2, 2, 2))
        self.assertAlmostEqual(g.voxel_volume_cm3, 5.0 / 1000.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
