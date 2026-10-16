"""DICOM 로딩 — CT / RTSTRUCT / RTPLAN / RTDOSE.

좌표계 정책
-----------
본 패키지는 **DICOM 환자 좌표계(mm)** 를 내부 표준으로 사용한다.
RTPLAN의 ControlPoint3DPosition, RTSTRUCT의 ContourData, CT의
ImagePositionPatient가 모두 같은 Frame of Reference를 공유하므로
별도 변환 없이 그대로 쓸 수 있다. (pyTG43은 IEC 61217로 변환하지만,
TG-43은 선원 국소 좌표계만 필요하므로 전역 좌표계 선택은 무관하다.)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pydicom


# ---------------------------------------------------------------------------
# 자료구조
# ---------------------------------------------------------------------------
@dataclass
class CTGeometry:
    """CT 시리즈의 기하 정보. 선량 그리드 정의와 구조물 rasterize에 쓰인다."""

    origin: np.ndarray          # (3,) 첫 슬라이스 ImagePositionPatient [mm]
    spacing: np.ndarray         # (3,) (dx, dy, dz) [mm]
    shape: tuple[int, int, int]  # (nz, ny, nx)
    z_positions: np.ndarray     # (nz,) 오름차순 슬라이스 z [mm]
    hu: np.ndarray | None = None  # (nz, ny, nx) HU. load_pixels=False면 None

    def voxel_volume_cm3(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0


@dataclass
class Dwell:
    """단일 dwell position."""

    position: np.ndarray   # (3,) [mm], DICOM 환자 좌표
    time_s: float
    axis: np.ndarray       # (3,) 단위벡터, 선원 장축 근사
    channel: int


@dataclass
class PlanInfo:
    dwells: list[Dwell]
    sk_ref: float                 # ReferenceAirKermaRate [U = uGy m^2 / h]
    half_life_days: float
    isotope: str
    ref_date: datetime | None
    n_fractions: int
    fraction_dose_gy: float | None
    dose_points: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (3,) mm
    dose_point_tps: dict[str, float] = field(default_factory=dict)     # name -> Gy/fx

    def total_time_s(self) -> float:
        return float(sum(d.time_s for d in self.dwells))

    def sk_at(self, when: datetime | None) -> float:
        """붕괴보정된 air-kerma strength [U]."""
        if when is None or self.ref_date is None:
            return self.sk_ref
        days = (when - self.ref_date).total_seconds() / 86400.0
        return self.sk_ref * float(np.exp(-np.log(2.0) * days / self.half_life_days))


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def _parse_dt(date_value, time_value=None) -> datetime | None:
    if not date_value:
        return None
    s = str(date_value)[:8]
    try:
        dt = datetime.strptime(s, "%Y%m%d")
    except ValueError:
        return None
    if time_value:
        t = str(time_value).split(".")[0].ljust(6, "0")[:6]
        try:
            dt = dt.replace(hour=int(t[0:2]), minute=int(t[2:4]), second=int(t[4:6]))
        except ValueError:
            pass
    return dt


def find_files(case_dir: str) -> dict[str, list[str]]:
    """디렉토리에서 modality별 파일 경로를 찾는다 (파일명이 아닌 태그 기준)."""
    out: dict[str, list[str]] = {"CT": [], "RTSTRUCT": [], "RTPLAN": [], "RTDOSE": []}
    for path in sorted(glob.glob(os.path.join(case_dir, "*.dcm"))):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["Modality"])
        except Exception:
            continue
        mod = str(getattr(ds, "Modality", ""))
        if mod in out:
            out[mod].append(path)
    return out


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------
def load_ct(paths: list[str], load_pixels: bool = False) -> CTGeometry:
    if not paths:
        raise ValueError("CT 파일이 없습니다.")
    headers = []
    for p in paths:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        headers.append((float(ds.ImagePositionPatient[2]), p, ds))
    headers.sort(key=lambda x: x[0])

    z = np.array([h[0] for h in headers], dtype=float)
    first = headers[0][2]

    orient = np.asarray(first.ImageOrientationPatient, dtype=float)
    if not (np.allclose(orient[:3], [1, 0, 0]) and np.allclose(orient[3:], [0, 1, 0])):
        raise NotImplementedError(
            "축방향(axial) 정렬 CT만 지원합니다. ImageOrientationPatient = "
            f"{orient.tolist()}"
        )

    dz = np.diff(z)
    if len(dz) and (np.max(dz) - np.min(dz)) > 1e-3:
        raise ValueError(f"슬라이스 간격이 불균일합니다: {np.min(dz):.4f} ~ {np.max(dz):.4f} mm")
    dz_val = float(dz[0]) if len(dz) else float(getattr(first, "SliceThickness", 1.0))

    px = np.asarray(first.PixelSpacing, dtype=float)  # [row(y), col(x)]
    origin = np.asarray(first.ImagePositionPatient, dtype=float)
    shape = (len(headers), int(first.Rows), int(first.Columns))
    spacing = np.array([float(px[1]), float(px[0]), dz_val], dtype=float)  # (dx, dy, dz)

    hu = None
    if load_pixels:
        vol = np.empty(shape, dtype=np.int16)
        for i, (_, p, _) in enumerate(headers):
            ds = pydicom.dcmread(p)
            arr = ds.pixel_array.astype(np.int32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            vol[i] = (arr * slope + intercept).astype(np.int16)
        hu = vol

    return CTGeometry(origin=origin, spacing=spacing, shape=shape, z_positions=z, hu=hu)


def load_plan(path: str, rtdose_path: str | None = None) -> PlanInfo:
    """RTPLAN에서 dwell / 선원 정보를 추출. RTDOSE가 있으면 dose point도 붙인다."""
    rp = pydicom.dcmread(path)

    if not hasattr(rp, "SourceSequence") or len(rp.SourceSequence) == 0:
        raise ValueError("RTPLAN에 SourceSequence가 없습니다 (brachytherapy plan이 아님).")
    src = rp.SourceSequence[0]
    sk_ref = float(src.ReferenceAirKermaRate)
    half_life = float(src.SourceIsotopeHalfLife)
    isotope = str(getattr(src, "SourceIsotopeName", "?"))
    ref_date = _parse_dt(getattr(src, "SourceStrengthReferenceDate", None),
                         getattr(src, "SourceStrengthReferenceTime", None))

    dwells: list[Dwell] = []
    for setup in rp.ApplicationSetupSequence:
        for ch in setup.ChannelSequence:
            cps = ch.BrachyControlPointSequence
            total_time = float(ch.ChannelTotalTime)
            final_w = float(ch.FinalCumulativeTimeWeight)
            if final_w <= 0:
                continue          # 가중치 정보가 없으면 시간을 복원할 수 없다
            # ChannelTotalTime == 0 인 채널(전 dwell time 0)도 **건너뛰지 않는다.**
            # 위치는 유효하고, 내보낸 RTPLAN을 다시 읽을 때 dwell 개수가
            # 달라지면 안 되기 때문이다 (rtplan_writer 왕복 검증).
            positions, times = [], []
            # control point는 (도착, 출발) 쌍. 홀수 개면 마지막은 무시.
            for i in range(0, len(cps) - 1, 2):
                p = np.asarray(cps[i].ControlPoint3DPosition, dtype=float)
                w0 = float(cps[i].CumulativeTimeWeight)
                w1 = float(cps[i + 1].CumulativeTimeWeight)
                positions.append(p)
                times.append(0.0 if total_time <= 0
                             else (w1 - w0) / final_w * total_time)
            if not positions:
                continue
            arr = np.asarray(positions)
            n = len(arr)
            ch_num = int(ch.ChannelNumber)
            for k in range(n):
                a = arr[max(k - 1, 0)]
                b = arr[min(k + 1, n - 1)]
                vec = b - a
                norm = float(np.linalg.norm(vec))
                axis = vec / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
                dwells.append(Dwell(position=arr[k], time_s=float(times[k]),
                                    axis=axis, channel=ch_num))

    n_fx, fx_dose = 1, None
    if "FractionGroupSequence" in rp and len(rp.FractionGroupSequence):
        fg = rp.FractionGroupSequence[0]
        n_fx = int(getattr(fg, "NumberOfFractionsPlanned", 1) or 1)
        rbs = fg.get("ReferencedBrachyApplicationSetupSequence", [])
        if len(rbs) and "BrachyApplicationSetupDose" in rbs[0]:
            fx_dose = float(rbs[0].BrachyApplicationSetupDose)

    info = PlanInfo(
        dwells=dwells, sk_ref=sk_ref, half_life_days=half_life, isotope=isotope,
        ref_date=ref_date, n_fractions=n_fx, fraction_dose_gy=fx_dose,
    )

    # RTPLAN의 DoseReferenceSequence = TPS가 계산한 point 선량
    for dref in rp.get("DoseReferenceSequence", []):
        name = str(getattr(dref, "DoseReferenceDescription", "")).strip()
        if name and "TargetPrescriptionDose" in dref:
            info.dose_point_tps[name] = float(dref.TargetPrescriptionDose)

    # RTDOSE의 ROI = dose point 좌표
    if rtdose_path:
        rd = pydicom.dcmread(rtdose_path)
        names = {int(s.ROINumber): str(s.ROIName).strip()
                 for s in rd.get("StructureSetROISequence", [])}
        for roi in rd.get("ROIContourSequence", []):
            nm = names.get(int(roi.ReferencedROINumber))
            cs = roi.get("ContourSequence", [])
            if nm and len(cs):
                info.dose_points[nm] = np.asarray(cs[0].ContourData, dtype=float)[:3]

    return info


def dose_grid_present(rtdose_path: str) -> bool:
    """RTDOSE에 실제 3D dose grid가 들어 있는지 확인."""
    rd = pydicom.dcmread(rtdose_path, stop_before_pixels=True)
    return "PixelData" in rd or "GridFrameOffsetVector" in rd


def check_frame_of_reference(files: dict[str, list[str]]) -> tuple[dict[str, str], list[str]]:
    """modality별 FrameOfReferenceUID를 모아 불일치를 찾는다.

    RTSTRUCT/RTPLAN/RTDOSE가 같은 FoR을 공유해야 선량-구조물 계산이 성립한다.
    CT가 다른 FoR이면 등록(registration) 없이는 CT를 함께 쓸 수 없다.

    Returns
    -------
    (modality -> FoR UID, 경고 메시지 리스트)
    """
    fors: dict[str, str] = {}
    for mod, paths in files.items():
        if not paths:
            continue
        ds = pydicom.dcmread(paths[0], stop_before_pixels=True)
        if mod == "RTSTRUCT":
            seq = ds.get("ReferencedFrameOfReferenceSequence", [])
            uid = str(seq[0].FrameOfReferenceUID) if len(seq) else ""
        else:
            uid = str(getattr(ds, "FrameOfReferenceUID", ""))
        if uid:
            fors[mod] = uid

    warnings: list[str] = []
    plan_group = [m for m in ("RTSTRUCT", "RTPLAN", "RTDOSE") if m in fors]
    if plan_group:
        ref = fors[plan_group[0]]
        for m in plan_group[1:]:
            if fors[m] != ref:
                warnings.append(f"{plan_group[0]}와 {m}의 FrameOfReference가 다릅니다.")
        if "CT" in fors and fors["CT"] != ref:
            warnings.append(
                "CT의 FrameOfReference가 RTSTRUCT/RTPLAN과 다릅니다 — "
                "구조물은 다른 영상(예: MR)에 그려졌습니다. "
                "영상 등록(registration) 없이 CT를 선량/구조물과 함께 쓰면 안 됩니다."
            )
    return fors, warnings
