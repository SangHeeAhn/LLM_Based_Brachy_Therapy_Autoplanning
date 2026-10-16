"""AAPM TG-43U1 2D formalism.

    D_dot(r, theta) = S_K * Lambda * [G_L(r,theta) / G_L(r0,theta0)] * g_L(r) * F(r,theta)

    r0 = 1 cm, theta0 = 90 deg
    S_K   : air-kerma strength  [U = uGy m^2 / h  =  cGy cm^2 / h]
    Lambda: dose-rate constant  [cGy h^-1 U^-1]
    -> D_dot [cGy/h]

한 dwell의 선량은 D_dot * t (t는 시간[h]).

참고: Rivard MJ et al., Med Phys 31(3):633-674, 2004 (TG-43U1)
      https://pubmed.ncbi.nlm.nih.gov/15070264/
"""

from __future__ import annotations

import numpy as np

from .source_data import SourceData


def geometry_function(r_cm: np.ndarray, theta_rad: np.ndarray, L_cm: float) -> np.ndarray:
    """선형 선원 기하함수 G_L(r, theta)  [cm^-2]. (TG-43U1 Eq. 4)

        theta != 0, pi :  beta / (L * r * sin(theta))
        theta == 0, pi :  1 / (r^2 - L^2/4)

    beta는 선원 양 끝에서 관심점을 바라보는 각도 차이.
    """
    r = np.maximum(np.asarray(r_cm, dtype=float), 1e-6)
    th = np.asarray(theta_rad, dtype=float)
    sin_t = np.sin(th)
    cos_t = np.cos(th)

    # 축상(axial) 근사식. r^2 - L^2/4 <= 0 이면 선원 내부이므로 발산 방지.
    denom = r * r - (L_cm ** 2) / 4.0
    axial = 1.0 / np.where(np.abs(denom) < 1e-9, 1e-9, denom)

    # 일반식
    safe_sin = np.where(np.abs(sin_t) < 1e-9, 1.0, sin_t)
    beta1 = np.arctan2(r * safe_sin, r * cos_t - L_cm / 2.0)
    beta2 = np.arctan2(r * safe_sin, r * cos_t + L_cm / 2.0)
    general = np.abs(beta1 - beta2) / (L_cm * r * safe_sin)

    return np.where(np.abs(sin_t) < 1e-6, np.abs(axial), general)


class TG43Calculator:
    """단일 선원 모델에 대한 TG-43 선량률 계산기."""

    def __init__(self, source: SourceData):
        self.src = source
        self.L = source.active_length_cm
        self.G0 = float(geometry_function(np.array(1.0), np.array(np.pi / 2.0), self.L))

    # ------------------------------------------------------------------
    def dose_rate_per_U(
        self,
        points_mm: np.ndarray,
        dwell_pos_mm: np.ndarray,
        dwell_axis: np.ndarray,
    ) -> np.ndarray:
        """S_K = 1 U 일 때의 선량률 [cGy/h] — 한 dwell이 여러 점에 주는 값.

        Parameters
        ----------
        points_mm  : (N, 3) 관심점 [mm]
        dwell_pos_mm : (3,) dwell 위치 [mm]
        dwell_axis : (3,) 선원 장축 단위벡터

        Returns
        -------
        (N,) 선량률 [cGy/h/U]
        """
        v = (np.asarray(points_mm, dtype=float) - np.asarray(dwell_pos_mm, dtype=float)) / 10.0
        r = np.linalg.norm(v, axis=-1)
        r = np.maximum(r, 1e-6)

        axis = np.asarray(dwell_axis, dtype=float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        cos_t = np.clip((v @ axis) / r, -1.0, 1.0)
        theta = np.arccos(cos_t)

        geo = geometry_function(r, theta, self.L) / self.G0
        g = self.src.g_L(r)
        f = self.src.F(r, np.degrees(theta))
        return self.src.dose_rate_constant * geo * g * f

    # ------------------------------------------------------------------
    def dose_at_points(
        self,
        points_mm: np.ndarray,
        dwells,
        sk: float,
    ) -> np.ndarray:
        """모든 dwell을 합산한 총 선량 [Gy].

        Parameters
        ----------
        points_mm : (N, 3) [mm]
        dwells    : Dwell 리스트 (position [mm], time_s, axis)
        sk        : 붕괴보정된 air-kerma strength [U]
        """
        pts = np.atleast_2d(np.asarray(points_mm, dtype=float))
        total = np.zeros(len(pts), dtype=float)
        for d in dwells:
            if d.time_s <= 0:
                continue
            total += self.dose_rate_per_U(pts, d.position, d.axis) * (d.time_s / 3600.0)
        return total * sk / 100.0  # cGy -> Gy
