"""선원 TG-43 데이터셋 로딩 및 보간.

데이터 출처와 한계는 JSON의 `provenance` 필드에 기록되어 있다.
새 선원을 추가하려면 같은 스키마의 JSON을 `brachyplan/data/`에 넣으면 된다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


@dataclass
class SourceData:
    name: str
    isotope: str
    half_life_days: float
    active_length_cm: float
    dose_rate_constant: float          # Lambda [cGy h^-1 U^-1]
    provenance: dict

    # radial dose function (Taylor & Rogers modified polynomial)
    _g_coef: dict
    _g_rmin: float
    _g_rmax: float

    # anisotropy function
    _f_r: np.ndarray        # (nr,)
    _f_theta: np.ndarray    # (nt,) deg
    _f_val: np.ndarray      # (nt, nr)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path_or_name: str) -> "SourceData":
        path = path_or_name
        if not os.path.isfile(path):
            cand = os.path.join(DATA_DIR, path_or_name)
            if not cand.endswith(".json"):
                cand += ".json"
            path = os.path.abspath(cand)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"선원 데이터를 찾을 수 없습니다: {path_or_name}\n"
                f"사용 가능: {sorted(os.listdir(os.path.abspath(DATA_DIR)))}"
            )
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)

        rdf = d["radial_dose_function"]
        if rdf["form"] != "taylor_rogers_modified_polynomial":
            raise NotImplementedError(f"미지원 g_L 형식: {rdf['form']}")
        af = d["anisotropy_function"]

        return cls(
            name=d["name"],
            isotope=d["isotope"],
            half_life_days=float(d["half_life_days"]),
            active_length_cm=float(d["active_length_cm"]),
            dose_rate_constant=float(d["dose_rate_constant_cGy_h_U"]),
            provenance=d.get("provenance", {}),
            _g_coef=rdf["coefficients"],
            _g_rmin=float(rdf["r_min_cm"]),
            _g_rmax=float(rdf["r_max_cm"]),
            _f_r=np.asarray(af["radii_cm"], dtype=float),
            _f_theta=np.asarray(af["theta_deg"], dtype=float),
            _f_val=np.asarray(af["F"], dtype=float),
        )

    # ------------------------------------------------------------------
    def g_L(self, r_cm: np.ndarray | float) -> np.ndarray:
        """방사상 선량함수 g_L(r).

        피팅 범위(0.2~20 cm) 밖에서는 TG-43U1 권고에 따라 경계값으로
        고정(clamp)한다. 외삽 다항식은 발산하므로 사용하지 않는다.
        """
        scalar = np.isscalar(r_cm) or np.ndim(r_cm) == 0
        r = np.clip(np.asarray(r_cm, dtype=float), self._g_rmin, self._g_rmax)
        c = self._g_coef
        poly = (
            c["a0"] * r ** -2
            + c["a1"] * r ** -1
            + c["a2"]
            + c["a3"] * r
            + c["a4"] * r ** 2
            + c["a5"] * r ** 3
        )
        out = poly * np.exp(-c["a6"] * r)
        return float(out) if scalar else out

    def F(self, r_cm: np.ndarray | float, theta_deg: np.ndarray | float) -> np.ndarray:
        """비등방성 함수 F(r, theta). (r, theta) 양방향 선형보간.

        표 범위 밖은 가장 가까운 경계값으로 고정한다 (TG-43U1 §4.2 권고와 동일한
        보수적 처리). theta는 0~180도로 접어 넣는다.
        """
        scalar = (np.isscalar(r_cm) or np.ndim(r_cm) == 0) and (
            np.isscalar(theta_deg) or np.ndim(theta_deg) == 0
        )
        r = np.atleast_1d(np.asarray(r_cm, dtype=float))
        th = np.atleast_1d(np.asarray(theta_deg, dtype=float))
        th = np.abs(((th + 180.0) % 360.0) - 180.0)  # -> [0, 180]

        r_c = np.clip(r, self._f_r[0], self._f_r[-1])
        t_c = np.clip(th, self._f_theta[0], self._f_theta[-1])

        # 인덱스와 가중치
        ir = np.clip(np.searchsorted(self._f_r, r_c, "right") - 1, 0, len(self._f_r) - 2)
        it = np.clip(np.searchsorted(self._f_theta, t_c, "right") - 1, 0, len(self._f_theta) - 2)

        r0, r1 = self._f_r[ir], self._f_r[ir + 1]
        t0, t1 = self._f_theta[it], self._f_theta[it + 1]
        wr = np.where(r1 > r0, (r_c - r0) / (r1 - r0), 0.0)
        wt = np.where(t1 > t0, (t_c - t0) / (t1 - t0), 0.0)

        f00 = self._f_val[it, ir]
        f01 = self._f_val[it, ir + 1]
        f10 = self._f_val[it + 1, ir]
        f11 = self._f_val[it + 1, ir + 1]

        val = (
            f00 * (1 - wt) * (1 - wr)
            + f01 * (1 - wt) * wr
            + f10 * wt * (1 - wr)
            + f11 * wt * wr
        )
        return float(val[0]) if scalar else val

    # ------------------------------------------------------------------
    def summary(self) -> str:
        p = self.provenance
        return (
            f"{self.name}\n"
            f"  isotope        : {self.isotope}, T1/2 = {self.half_life_days} d\n"
            f"  active length  : {self.active_length_cm} cm\n"
            f"  Lambda         : {self.dose_rate_constant} cGy/h/U\n"
            f"  g_L(r) range   : {self._g_rmin}-{self._g_rmax} cm (범위 밖 clamp)\n"
            f"  F(r,theta)     : {len(self._f_r)} radii x {len(self._f_theta)} angles\n"
            f"  provenance     : {p.get('source', '?')}\n"
            f"  WARNING        : {p.get('warning', '')}"
        )
