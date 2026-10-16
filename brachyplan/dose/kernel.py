"""선량 그리드 정의, 3D 선량 계산, dwell 영향행렬(kernel) 생성.

그리드는 **회전(oblique)** 을 지원한다. 자궁경부 브라키에서는 GEC-ESTRO 권고에
따라 applicator 축에 수직인 para-transverse 영상에 컨투어를 그리는 경우가 많아,
구조물 평면이 환자 축방향(axial)과 기울어져 있다. 이때 축방향 그리드로
래스터화하면 부피가 크게 틀어진다 (본 케이스에서 최대 -32%).

TG-43은 임의 점에서 선량을 계산하므로 그리드가 축정렬일 필요가 전혀 없다.
따라서 컨투어 평면 좌표계에 그리드를 직접 정의하는 것이 정확하고 단순하다.

Phase 2의 역최적화는  dose = A^T . t  형태를 쓴다.
  A[k, i] = dwell k 가 복셀 i 에 단위시간(1초)당 주는 선량 [Gy/s]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tg43 import TG43Calculator


@dataclass
class DoseGrid:
    """직교 그리드. rotation의 행이 그리드 x/y/z축의 환자좌표계 방향이다.

    복셀 (i, j, k) 중심의 환자좌표
        p = origin + i*spacing[0]*u + j*spacing[1]*v + k*spacing[2]*w
    여기서 u, v, w = rotation[0], rotation[1], rotation[2] (정규직교).
    """

    origin: np.ndarray            # (3,) [mm] 환자좌표
    spacing: np.ndarray           # (3,) (dx, dy, dz) [mm]
    shape: tuple[int, int, int]   # (nz, ny, nx)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        self.spacing = np.asarray(self.spacing, dtype=float)
        self.rotation = np.asarray(self.rotation, dtype=float)
        if self.rotation.shape != (3, 3):
            raise ValueError("rotation은 (3,3) 이어야 합니다.")
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("rotation이 정규직교가 아닙니다.")

    @property
    def is_axial(self) -> bool:
        return bool(np.allclose(self.rotation, np.eye(3), atol=1e-9))

    @property
    def voxel_volume_cm3(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0

    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    # ------------------------------------------------------------------
    def local_origin(self) -> np.ndarray:
        """origin을 그리드 축(u, v, w)에 투영한 국소 좌표 (3,)."""
        return self.rotation @ self.origin

    def local_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """국소 좌표계에서의 축 좌표값 (x, y, z). 래스터화에 쓴다."""
        nz, ny, nx = self.shape
        o = self.local_origin()
        return (
            o[0] + np.arange(nx) * self.spacing[0],
            o[1] + np.arange(ny) * self.spacing[1],
            o[2] + np.arange(nz) * self.spacing[2],
        )

    def to_local(self, points_mm: np.ndarray) -> np.ndarray:
        """환자좌표 → 그리드 국소좌표."""
        return np.atleast_2d(np.asarray(points_mm, dtype=float)) @ self.rotation.T

    def points(self) -> np.ndarray:
        """(nz*ny*nx, 3) 복셀 중심의 **환자좌표**. 순서는 C-order (z, y, x)."""
        x, y, z = self.local_axes()
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        local = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        return local @ self.rotation  # 국소 -> 환자

    def points_at(self, flat_idx: np.ndarray) -> np.ndarray:
        """지정한 평탄 인덱스의 복셀 중심 좌표만 (M, 3).

        `points()[idx]` 와 결과가 같지만 **전체 그리드를 만들지 않는다.**
        관심 복셀은 보통 전체의 10 % 미만(본 케이스 9.1 %)이라, 전체를
        만들면 쓰지도 않을 좌표에 메모리를 낭비한다 — 1 mm 격자에서
        729만 복셀 x 3 x 8 byte = 175 MB.
        """
        nz, ny, nx = self.shape
        idx = np.asarray(flat_idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= nz * ny * nx):
            raise IndexError("flat_idx가 그리드 범위를 벗어났습니다.")
        k, rem = np.divmod(idx, ny * nx)      # z
        j, i = np.divmod(rem, nx)             # y, x
        o = self.local_origin()
        local = np.stack([o[0] + i * self.spacing[0],
                          o[1] + j * self.spacing[1],
                          o[2] + k * self.spacing[2]], axis=1)
        return local @ self.rotation


# ---------------------------------------------------------------------------
def make_grid(
    points_mm: np.ndarray,
    rotation: np.ndarray | None = None,
    spacing_mm: float | tuple[float, float, float] = 2.0,
    margin_mm: float = 15.0,
) -> DoseGrid:
    """주어진 점들을 모두 포함하는 그리드를 만든다 (회전 좌표계 기준)."""
    R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    sp = np.asarray((spacing_mm,) * 3 if np.isscalar(spacing_mm) else spacing_mm, dtype=float)

    pts = np.atleast_2d(np.asarray(points_mm, dtype=float))
    local = pts @ R.T
    lo = local.min(axis=0) - margin_mm
    hi = local.max(axis=0) + margin_mm
    n = np.maximum(np.ceil((hi - lo) / sp).astype(int) + 1, 1)

    origin = lo @ R  # 국소 -> 환자
    return DoseGrid(origin=origin, spacing=sp,
                    shape=(int(n[2]), int(n[1]), int(n[0])), rotation=R)


# ---------------------------------------------------------------------------
def compute_dose_grid(
    calc: TG43Calculator,
    grid: DoseGrid,
    dwells,
    sk: float,
    chunk: int = 200_000,
) -> np.ndarray:
    """전체 그리드 선량 [Gy]. shape = grid.shape."""
    pts = grid.points()
    out = np.zeros(len(pts), dtype=np.float64)
    for s in range(0, len(pts), chunk):
        out[s:s + chunk] = calc.dose_at_points(pts[s:s + chunk], dwells, sk)
    return out.reshape(grid.shape)


def compute_kernel(
    calc: TG43Calculator,
    points_mm: np.ndarray,
    dwells,
    sk: float,
) -> np.ndarray:
    """영향행렬 A [Gy/s]. shape = (n_dwell, n_point).

    dose[i] = sum_k A[k, i] * t[k]   (t [s])
    """
    pts = np.atleast_2d(np.asarray(points_mm, dtype=float))
    A = np.zeros((len(dwells), len(pts)), dtype=np.float64)
    for k, d in enumerate(dwells):
        # cGy/h/U -> Gy/s : / 100 / 3600
        A[k] = calc.dose_rate_per_U(pts, d.position, d.axis) * sk / 100.0 / 3600.0
    return A
