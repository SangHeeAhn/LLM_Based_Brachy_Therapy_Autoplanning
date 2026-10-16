"""DVH 및 선량-체적 지표.

정의 (EMBRACE-II / GEC-ESTRO 관례)
  D_x%   : 구조물 부피의 x% 가 받는 최소 선량
  D_x cc : 구조물 중 가장 많이 받는 x cm^3 의 최소 선량  (예: D2cm3)
  V_x%   : 처방선량의 x% 이상을 받는 부피 (%, 또는 cm^3)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DVH:
    name: str
    doses: np.ndarray          # 정렬되지 않은 복셀 선량 [Gy]
    voxel_volume_cm3: float

    @property
    def volume_cm3(self) -> float:
        return len(self.doses) * self.voxel_volume_cm3

    @property
    def n_voxels(self) -> int:
        return len(self.doses)

    # ------------------------------------------------------------------
    def D_percent(self, pct: float) -> float:
        """D_x% [Gy]. 부피의 상위 x%가 받는 최소 선량."""
        if self.n_voxels == 0:
            return float("nan")
        s = np.sort(self.doses)[::-1]
        k = int(np.ceil(self.n_voxels * pct / 100.0))
        k = min(max(k, 1), self.n_voxels)
        return float(s[k - 1])

    def D_cc(self, cc: float) -> float:
        """D_x cm^3 [Gy]. 가장 높은 x cm^3 가 받는 최소 선량."""
        if self.n_voxels == 0:
            return float("nan")
        k = int(np.ceil(cc / self.voxel_volume_cm3))
        if k > self.n_voxels:
            return float(np.min(self.doses))
        k = max(k, 1)
        s = np.sort(self.doses)[::-1]
        return float(s[k - 1])

    def V_dose(self, dose_gy: float, as_percent: bool = True) -> float:
        """dose_gy 이상을 받는 부피."""
        if self.n_voxels == 0:
            return float("nan")
        n = int(np.count_nonzero(self.doses >= dose_gy))
        return 100.0 * n / self.n_voxels if as_percent else n * self.voxel_volume_cm3

    def V_percent_of(self, prescription_gy: float, pct: float, as_percent: bool = True) -> float:
        """V150 등: 처방선량의 pct% 이상을 받는 부피."""
        return self.V_dose(prescription_gy * pct / 100.0, as_percent)

    @property
    def mean(self) -> float:
        return float(np.mean(self.doses)) if self.n_voxels else float("nan")

    @property
    def max(self) -> float:
        return float(np.max(self.doses)) if self.n_voxels else float("nan")

    @property
    def min(self) -> float:
        return float(np.min(self.doses)) if self.n_voxels else float("nan")

    def cumulative(self, n_bins: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """누적 DVH 곡선 (dose [Gy], volume [%])."""
        if self.n_voxels == 0:
            return np.array([]), np.array([])
        dmax = float(np.max(self.doses))
        edges = np.linspace(0.0, dmax * 1.001, n_bins + 1)
        counts, _ = np.histogram(self.doses, bins=edges)
        cum = np.cumsum(counts[::-1])[::-1]
        return edges[:-1], 100.0 * cum / self.n_voxels


def dvh_from_mask(name: str, dose: np.ndarray, mask: np.ndarray,
                  voxel_volume_cm3: float) -> DVH:
    if dose.shape != mask.shape:
        raise ValueError(f"shape 불일치: dose {dose.shape} vs mask {mask.shape}")
    return DVH(name=name, doses=dose[mask].astype(float),
               voxel_volume_cm3=voxel_volume_cm3)


def evaluate_plan(
    dose: np.ndarray,
    masks: dict[str, np.ndarray],
    voxel_volume_cm3: float,
    prescription_gy: float,
    target_names: tuple[str, ...] = ("HR_CTV", "IR_CTV", "GTV"),
) -> dict[str, dict[str, float]]:
    """구조물별 핵심 지표를 계산해 표로 반환."""
    result: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        d = dvh_from_mask(name, dose, mask, voxel_volume_cm3)
        if d.n_voxels == 0:
            continue
        row = {
            "volume_cm3": d.volume_cm3,
            "Dmean_Gy": d.mean,
            "Dmax_Gy": d.max,
        }
        if name in target_names:
            row["D90_Gy"] = d.D_percent(90)
            row["D98_Gy"] = d.D_percent(98)
            row["D100_Gy"] = d.min
            row["V100_pct"] = d.V_percent_of(prescription_gy, 100)
            row["V150_pct"] = d.V_percent_of(prescription_gy, 150)
            row["V200_pct"] = d.V_percent_of(prescription_gy, 200)
        else:
            row["D2cm3_Gy"] = d.D_cc(2.0)
            row["D1cm3_Gy"] = d.D_cc(1.0)
            row["D01cm3_Gy"] = d.D_cc(0.1)
        result[name] = row
    return result
