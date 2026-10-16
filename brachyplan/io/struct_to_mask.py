"""RTSTRUCT contour → 3D binary mask (rasterize).

핵심: 컨투어 평면이 **환자 축방향(axial)이 아닐 수 있다.**
자궁경부 브라키에서는 GEC-ESTRO 권고에 따라 applicator 축에 수직인
para-transverse MR에 컨투어를 그리는 경우가 흔하다. 이때 평면 법선은
z축과 수 도 기울어져 있고, 축방향 그리드로 래스터화하면 부피가 크게 틀린다.

따라서 본 모듈은
  1) RTSTRUCT 전체 점으로 컨투어 평면의 법선 w를 SVD로 추정하고,
  2) 정규직교 기저 (u, v, w)를 만들어
  3) 그 좌표계에 정의된 그리드에 래스터화한다.
법선이 z축이면 자동으로 기존 축방향 동작과 같아진다.

의존성은 numpy만 사용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pydicom


# ---------------------------------------------------------------------------
# ROI 이름 정규화
# ---------------------------------------------------------------------------
# RTSTRUCT의 ROIName은 기관/플래너마다 다르다 (CTV_HR, HRCTV, HR-CTV, sigmoid ...).
# 파이프라인 전체는 아래 표준 이름을 쓰므로, 마스크를 만들 때 한 번 매핑한다.
# 키는 영숫자만 남기고 소문자로 만든 형태다.
ROI_ALIASES: dict[str, str] = {
    # 표적
    "hrctv": "HR_CTV", "ctvhr": "HR_CTV", "hrctvfinal": "HR_CTV",
    "ctvhrt": "HR_CTV", "hrctv1": "HR_CTV",
    "irctv": "IR_CTV", "ctvir": "IR_CTV", "ctvir1": "IR_CTV",
    "gtv": "GTV", "gtvres": "GTV", "gtvt": "GTV",
    # OAR
    "bladder": "Bladder", "vesica": "Bladder",
    "rectum": "Rectum",
    "sigmoid": "Sigmoid", "colonsigmoid": "Sigmoid", "sigmoidcolon": "Sigmoid",
    "bowel": "Bowel", "bowelloops": "Bowel", "smallbowel": "Bowel",
    "urethra": "Urethra",
    "anus": "Anus", "analcanal": "Anus",
    # 기타
    "body": "BODY", "external": "BODY",
}


def canonical_roi_name(name: str) -> str:
    """RTSTRUCT ROIName -> 파이프라인 표준 이름. 모르는 이름은 원본 그대로."""
    key = re.sub(r"[^a-z0-9]", "", str(name).strip().lower())
    return ROI_ALIASES.get(key, str(name).strip())


# ---------------------------------------------------------------------------
# 컨투어 평면 좌표계
# ---------------------------------------------------------------------------
@dataclass
class ContourFrame:
    """컨투어가 놓인 평면들의 좌표계."""

    rotation: np.ndarray   # (3,3) 행 = u, v, w (정규직교). w = 평면 법선
    tilt_deg: float        # z축과 w 사이의 각도
    planarity_mm: float    # 각 컨투어의 평면 이탈 최대치 (0에 가까워야 정상)

    @property
    def normal(self) -> np.ndarray:
        return self.rotation[2]


def _orthonormal_basis(w: np.ndarray) -> np.ndarray:
    """법선 w로부터 정규직교 기저 (u, v, w)를 만든다 (행렬의 행으로)."""
    w = np.asarray(w, dtype=float)
    w = w / np.linalg.norm(w)
    # w와 가장 덜 평행한 축을 골라 u를 만든다
    seed = np.eye(3)[int(np.argmin(np.abs(w)))]
    u = np.cross(seed, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    R = np.stack([u, v, w])
    if np.linalg.det(R) < 0:      # 오른손 좌표계 유지
        R[1] = -R[1]
    return R


def contour_frame(rtstruct_path: str, roi_names: list[str] | None = None) -> ContourFrame:
    """RTSTRUCT의 모든 컨투어에서 공통 평면 법선을 추정한다."""
    rs = pydicom.dcmread(rtstruct_path)
    name_of = {int(s.ROINumber): canonical_roi_name(s.ROIName)
               for s in rs.StructureSetROISequence}
    wanted = None if roi_names is None else {canonical_roi_name(n) for n in roi_names}

    normals, weights, planarity = [], [], 0.0
    for roi in rs.get("ROIContourSequence", []):
        nm = name_of.get(int(roi.ReferencedROINumber))
        if wanted is not None and nm not in wanted:
            continue
        for c in roi.get("ContourSequence", []):
            d = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            if len(d) < 3:
                continue
            cen = d - d.mean(axis=0)
            n = np.linalg.svd(cen, full_matrices=False)[2][-1]
            if n[2] < 0:
                n = -n
            normals.append(n)
            weights.append(len(d))
            planarity = max(planarity, float(np.max(np.abs(cen @ n))))

    if not normals:
        raise ValueError("컨투어를 찾을 수 없습니다.")

    w = np.average(np.asarray(normals), axis=0, weights=np.asarray(weights, dtype=float))
    w /= np.linalg.norm(w)
    tilt = float(np.degrees(np.arccos(np.clip(abs(w[2]), -1.0, 1.0))))
    return ContourFrame(rotation=_orthonormal_basis(w), tilt_deg=tilt,
                        planarity_mm=planarity)


# ---------------------------------------------------------------------------
# 폴리곤 래스터화
# ---------------------------------------------------------------------------
def rasterize_polygon(
    poly_xy: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """폴리곤 내부 판정 (even-odd crossing number).

    Parameters
    ----------
    poly_xy : (N, 2) 폴리곤 정점 (그리드 국소 좌표) [mm]
    x_coords, y_coords : 그리드 국소 좌표 [mm], 오름차순

    Returns
    -------
    (ny, nx) bool 마스크
    """
    ny, nx = len(y_coords), len(x_coords)
    mask = np.zeros((ny, nx), dtype=bool)
    if len(poly_xy) < 3:
        return mask

    xmin, ymin = poly_xy.min(axis=0)
    xmax, ymax = poly_xy.max(axis=0)
    ix0, ix1 = np.searchsorted(x_coords, xmin, "left"), np.searchsorted(x_coords, xmax, "right")
    iy0, iy1 = np.searchsorted(y_coords, ymin, "left"), np.searchsorted(y_coords, ymax, "right")
    if ix0 >= ix1 or iy0 >= iy1:
        return mask

    X, Y = np.meshgrid(x_coords[ix0:ix1], y_coords[iy0:iy1])
    inside = np.zeros(X.shape, dtype=bool)

    x1, y1 = poly_xy[:, 0], poly_xy[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    for a_x, a_y, b_x, b_y in zip(x1, y1, x2, y2):
        if a_y == b_y:                       # 수평 변 제외
            continue
        cond = (a_y > Y) != (b_y > Y)        # +x 반직선이 이 변을 지나는가
        if not cond.any():
            continue
        t = (Y - a_y) / (b_y - a_y)
        inside ^= cond & (X < a_x + t * (b_x - a_x))

    mask[iy0:iy1, ix0:ix1] = inside
    return mask


# ---------------------------------------------------------------------------
# 3D 마스크
# ---------------------------------------------------------------------------
def build_masks(
    rtstruct_path: str,
    grid,
    roi_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """RTSTRUCT의 ROI들을 grid 형상의 bool 마스크로 변환.

    Parameters
    ----------
    grid : DoseGrid  (rotation 포함. 컨투어 평면 좌표계와 일치해야 정확하다)
    """
    rs = pydicom.dcmread(rtstruct_path)
    name_of = {int(s.ROINumber): canonical_roi_name(s.ROIName)
               for s in rs.StructureSetROISequence}
    wanted = None if roi_names is None else {canonical_roi_name(n) for n in roi_names}

    R = grid.rotation
    x_coords, y_coords, z_coords = grid.local_axes()
    nz, ny, nx = grid.shape
    out: dict[str, np.ndarray] = {}

    for roi in rs.get("ROIContourSequence", []):
        name = name_of.get(int(roi.ReferencedROINumber))
        if name is None or (wanted is not None and name not in wanted):
            continue
        if name in out:      # 별칭 충돌 — 먼저 나온 ROI를 유지한다
            continue
        contours = roi.get("ContourSequence", [])
        if not len(contours):
            continue

        # 국소 좌표로 변환 후 평면(w 오프셋)별로 묶기
        planes: dict[float, list[np.ndarray]] = {}
        for c in contours:
            if str(getattr(c, "ContourGeometricType", "")) != "CLOSED_PLANAR":
                continue
            d = np.asarray(c.ContourData, dtype=float).reshape(-1, 3) @ R.T  # -> 국소
            key = round(float(np.mean(d[:, 2])), 2)
            planes.setdefault(key, []).append(d[:, :2])
        if not planes:
            continue

        plane_w = np.array(sorted(planes.keys()), dtype=float)
        slab = float(np.median(np.diff(plane_w))) if len(plane_w) > 1 else float(grid.spacing[2])
        half = max(slab, float(grid.spacing[2])) / 2.0

        plane_masks: dict[float, np.ndarray] = {}
        for key, polys in planes.items():
            m = np.zeros((ny, nx), dtype=bool)
            for poly in polys:
                m ^= rasterize_polygon(poly, x_coords, y_coords)
            plane_masks[key] = m

        vol = np.zeros((nz, ny, nx), dtype=bool)
        for k, wz in enumerate(z_coords):
            idx = int(np.argmin(np.abs(plane_w - wz)))
            if abs(plane_w[idx] - wz) <= half + 1e-6:
                vol[k] = plane_masks[round(float(plane_w[idx]), 2)]
        out[name] = vol

    return out


def mask_volume_cm3(mask: np.ndarray, spacing: np.ndarray) -> float:
    return float(mask.sum()) * float(np.prod(np.asarray(spacing, dtype=float))) / 1000.0


def contour_volume_cm3(rtstruct_path: str, roi_name: str,
                       frame: ContourFrame | None = None) -> float:
    """폴리곤 면적(신발끈) x 평면 간격으로 계산한 기준 부피 [cm^3].

    래스터화 부피의 검증용 — 그리드 해상도에 의존하지 않는다.
    면적은 컨투어 평면 안에서 계산하므로 oblique 평면에서도 정확하다.

    주의: 같은 평면의 폴리곤 면적을 단순 합산하므로 구멍(hole)이 있는
    구조물에서는 과대평가된다. 그런 경우 XOR로 처리하는 래스터화 쪽이 옳다.
    """
    if frame is None:
        frame = contour_frame(rtstruct_path)
    R = frame.rotation

    rs = pydicom.dcmread(rtstruct_path)
    want = canonical_roi_name(roi_name)
    num = next((int(s.ROINumber) for s in rs.StructureSetROISequence
                if canonical_roi_name(s.ROIName) == want), None)
    if num is None:
        raise KeyError(f"ROI '{roi_name}' 없음")

    for roi in rs.ROIContourSequence:
        if int(roi.ReferencedROINumber) != num:
            continue
        planes: dict[float, float] = {}
        for c in roi.get("ContourSequence", []):
            d = np.asarray(c.ContourData, dtype=float).reshape(-1, 3) @ R.T
            x, y = d[:, 0], d[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            key = round(float(np.mean(d[:, 2])), 2)
            planes[key] = planes.get(key, 0.0) + area
        if not planes:
            return 0.0
        ws = np.array(sorted(planes.keys()))
        slab = float(np.median(np.diff(ws))) if len(ws) > 1 else 1.0
        return float(sum(planes.values()) * slab) / 1000.0
    return 0.0
