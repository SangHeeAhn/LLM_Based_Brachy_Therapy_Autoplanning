"""최적화된 dwell time을 RTPLAN DICOM으로 내보내기.

전략 — 원본을 복사해 **시간 관련 태그만** 수정한다
----------------------------------------------------
본 도구는 dwell **position**을 바꾸지 않는다 (원본 RTPLAN의 채널·제어점 기하를
그대로 쓴다). 따라서 RTPLAN을 처음부터 만들 필요가 없고, 그래서도 안 된다.
원본을 복사해 다음만 갱신하는 것이 가장 안전하다.

  갱신하는 것
    - ChannelSequence[].ChannelTotalTime
    - BrachyControlPointSequence[].CumulativeTimeWeight  (도착/출발 쌍)
    - ChannelSequence[].FinalCumulativeTimeWeight
    - ApplicationSetupSequence[].TotalReferenceAirKerma  = S_K x sum(t) / 3600
    - SOPInstanceUID / SeriesInstanceUID (새 인스턴스이므로 필수)
    - ApprovalStatus = UNAPPROVED (아래 안전 항목 참조)
    - RTPlanLabel / RTPlanName / RTPlanDescription (기계 생성 표시)
    - (선택) DoseReferenceSequence[].TargetPrescriptionDose 를 재계산

  건드리지 않는 것
    - ControlPoint3DPosition, ControlPointRelativePosition (기하 불변)
    - SourceSequence (선원, S_K, 기준일)
    - StudyInstanceUID, FrameOfReferenceUID (같은 검사/좌표계)
    - PatientID 등 환자 식별자

가중치 스케일
-------------
DICOM에서 CumulativeTimeWeight는 상대값이고 절대시간은 ChannelTotalTime이 준다.
본 케이스 원본은 plan 전체에서 (총가중치 / 총시간)이 일정했다
(13.730131 s per weight unit). 그 스케일을 그대로 유지해 파일을 원본 관례에
맞춘다. 스케일을 알 수 없으면 weight = seconds 로 둔다 (표준 위반 아님).

★ 안전
-------
- **ApprovalStatus는 강제로 UNAPPROVED**로 만든다. 기계가 만든 계획이
  "승인됨"으로 보이면 안 된다. 원본의 리뷰 정보(ReviewerName 등)도 지운다.
- 라벨과 설명에 기계 생성 표시를 남긴다.
- 조사기 시간 분해능(기본 0.1 s)으로 **양자화**하고, 그로 인한 선량 변화를
  보고한다. 양자화를 무시하면 계산 선량과 실제 조사 선량이 달라진다.
- 이 파일은 **연구용**이다. TPS로 다시 불러 선량을 재계산하고 독립 검증을
  거쳐야 하며, 그대로 조사에 쓰면 안 된다.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field

import numpy as np
import pydicom
from pydicom.uid import generate_uid

# DICOM DS(Decimal String)는 16자 제한
_DS_MAX = 16


def _ds(value: float) -> str:
    """DS VR에 맞는 문자열. 16자를 넘지 않게 유효숫자를 줄인다."""
    v = float(value)
    if v == 0:
        return "0.0"
    for prec in range(12, -1, -1):
        s = f"{v:.{prec}f}"
        if len(s) <= _DS_MAX:
            return s
    return f"{v:.6e}"[:_DS_MAX]


def quantize_times(times_s, resolution_s: float = 0.1) -> np.ndarray:
    """조사기 시간 분해능으로 반올림. resolution_s <= 0 이면 그대로 둔다."""
    t = np.asarray(times_s, dtype=float)
    if resolution_s and resolution_s > 0:
        t = np.round(t / resolution_s) * resolution_s
    return np.maximum(t, 0.0)


@dataclass
class ExportReport:
    out_path: str
    sop_instance_uid: str
    series_instance_uid: str
    n_dwells: int
    total_time_s: float
    total_time_original_s: float
    trak: float
    trak_original: float
    time_resolution_s: float
    max_quantization_shift_s: float
    channels: list[dict] = field(default_factory=list)
    empty_channels: list[int] = field(default_factory=list)
    dose_points_updated: dict[str, float] = field(default_factory=dict)
    dose_points_original: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"출력       : {self.out_path}",
            f"SOP UID    : {self.sop_instance_uid}",
            f"dwell      : {self.n_dwells}개",
            f"총 조사시간: {self.total_time_s:.2f} s "
            f"(원본 {self.total_time_original_s:.2f} s)",
            f"TRAK       : {self.trak:.4f} (원본 {self.trak_original:.4f})",
            f"시간 양자화: {self.time_resolution_s:g} s, "
            f"최대 이동 {self.max_quantization_shift_s:.4f} s",
            "",
            f"{'ch':>3}{'dwell':>7}{'원본(s)':>10}{'최적화(s)':>11}",
            "-" * 31,
        ]
        for c in self.channels:
            lines.append(f"{c['channel']:>3}{c['n_dwells']:>7}"
                         f"{c['total_time_original_s']:10.2f}{c['total_time_s']:11.2f}")
        if self.dose_points_updated:
            lines.append("")
            lines.append("재계산된 dose point (Gy/fx) — 원본 대비")
            lines.append(f"  {'point':<22}{'원본':>8}{'신규':>8}{'변화%':>8}")
            lines.append("  " + "-" * 46)
            worst = ("", 0.0)
            for k, v in self.dose_points_updated.items():
                o = self.dose_points_original.get(k)
                if o:
                    pct = 100.0 * (v - o) / o
                    if abs(pct) > abs(worst[1]):
                        worst = (k, pct)
                    lines.append(f"  {k:<22}{o:8.3f}{v:8.3f}{pct:+8.1f}")
                else:
                    lines.append(f"  {k:<22}{'-':>8}{v:8.3f}{'-':>8}")
            if worst[0]:
                lines.append(f"  -> 최대 변화: {worst[0]} {worst[1]:+.0f}% "
                             f"— 리뷰 시 반드시 확인하십시오")
        for w in self.warnings:
            lines.append(f"[WARN] {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def uid_root_of(uid: str) -> str:
    """UID의 조직 루트를 추정한다 — 마지막 3개 성분을 뺀 앞부분.

    Nucletron/Oncentra는 `1.3.6.1.4.1.2452.6.<a>.<b>.<c>.<d>` 형태를 쓴다.
    같은 루트를 유지하면 수신 시스템의 UID 화이트리스트/길이 가정에 걸릴
    가능성이 줄어든다.
    """
    parts = str(uid).split(".")
    return ".".join(parts[:-3]) if len(parts) > 4 else str(uid)


def short_uid(root: str, max_len: int = 60) -> str:
    """`root` 아래에 난수 UID를 만들되 길이를 `max_len` 이하로 맞춘다.

    왜 필요한가
    -----------
    pydicom의 `generate_uid()`는 정확히 64자 UID를 만든다. DICOM 상한이 64라
    규격 위반은 아니지만, 널 종단 포함 65바이트를 가정하지 않은 구형 파서에서
    깨지는 사례가 알려져 있다. 원본 plan의 UID가 60~61자라면 같은 길이대로
    맞춰 주는 편이 안전하다.
    """
    import random
    root = str(root).rstrip(".")
    budget = max_len - len(root) - 1
    if budget < 4:
        raise ValueError(f"UID 루트가 너무 깁니다: {root!r} (max_len={max_len})")
    # 첫 자리는 0이 되지 않게 (선행 0 금지)
    digits = [str(random.randint(1, 9))]
    digits += [str(random.randint(0, 9)) for _ in range(budget - 1)]
    return root + "." + "".join(digits)


def _weight_scale(rtplan) -> float:
    """원본의 (총 CumulativeTimeWeight / 총 ChannelTotalTime) [weight per second].

    구할 수 없으면 1.0 (weight = seconds).
    """
    w_sum = 0.0
    t_sum = 0.0
    for setup in rtplan.ApplicationSetupSequence:
        for ch in setup.ChannelSequence:
            f = float(getattr(ch, "FinalCumulativeTimeWeight", 0) or 0)
            t = float(getattr(ch, "ChannelTotalTime", 0) or 0)
            if f > 0 and t > 0:
                w_sum += f
                t_sum += t
    return (w_sum / t_sum) if (w_sum > 0 and t_sum > 0) else 1.0


def write_rtplan(
    src_rtplan_path: str,
    out_path: str,
    dwell_times_s,
    *,
    time_resolution_s: float = 0.1,
    label_suffix: str = "_AI",
    description: str = "brachyplan optimized (RESEARCH, NOT FOR TREATMENT)",
    recompute_dose_points_with=None,
    keep_series_uid: bool = False,
    uid_style: str = "pydicom",
    new_private_uid: bool = False,
) -> ExportReport:
    """dwell time을 갱신한 RTPLAN을 저장한다.

    Parameters
    ----------
    dwell_times_s : (n_dwell,) 초 단위. **원본 RTPLAN의 dwell 순서**와 같아야 한다
        (`brachyplan.io.load_plan`이 만드는 순서: 채널 순 -> 제어점 쌍 순).
    time_resolution_s : 조사기 시간 분해능. 0이면 양자화하지 않음.
    recompute_dose_points_with : (calc, sk) 튜플을 주면 DoseReferenceSequence의
        point 선량을 TG-43으로 재계산해 갱신한다.
    keep_series_uid : True면 SeriesInstanceUID를 원본 그대로 둔다. 수신 TPS가
        "자기가 만든 시리즈"의 plan만 온전히 재구성하는 경우에 쓴다.
    uid_style : 새 UID 생성 방식.
        "pydicom" — `generate_uid()` (64자, pydicom 등록 루트). 표준 준수.
        "source"  — 원본 UID의 조직 루트를 이어받고 길이도 원본에 맞춘다.
                    구형 수신 시스템 호환용. **다른 벤더의 UID 루트를 빌려
                    쓰는 것이므로 연구 목적 외 배포/전송에는 쓰지 말 것.**
    new_private_uid : True면 벤더 私有 plan 식별 UID (0021,1000)도 새로
        발급한다. 수신 TPS가 이 태그로 plan 동일성을 판정해 import를 중복으로
        보고 무시하는 경우의 회피책이다.
    """
    ds = pydicom.dcmread(src_rtplan_path)

    if not hasattr(ds, "ApplicationSetupSequence"):
        raise ValueError("brachytherapy RTPLAN이 아닙니다 (ApplicationSetupSequence 없음)")

    # --- dwell 수 확인 --------------------------------------------------
    pairs: list[tuple] = []          # (setup, channel, [(cp_arrive, cp_depart), ...])
    for setup in ds.ApplicationSetupSequence:
        for ch in setup.ChannelSequence:
            cps = ch.BrachyControlPointSequence
            cp_pairs = [(cps[i], cps[i + 1]) for i in range(0, len(cps) - 1, 2)]
            pairs.append((setup, ch, cp_pairs))

    n_expected = sum(len(p[2]) for p in pairs)
    t_new = quantize_times(dwell_times_s, time_resolution_s)
    if t_new.shape != (n_expected,):
        raise ValueError(
            f"dwell 수 불일치: 입력 {np.asarray(dwell_times_s).shape} vs "
            f"RTPLAN {n_expected}개. 같은 케이스의 RTPLAN인지 확인하십시오."
        )
    t_raw = np.asarray(dwell_times_s, dtype=float)
    max_shift = float(np.max(np.abs(t_new - t_raw))) if len(t_new) else 0.0

    scale = _weight_scale(ds)
    warnings: list[str] = []
    channels: list[dict] = []
    empty_channels: list[int] = []

    # --- 채널별 갱신 -----------------------------------------------------
    k = 0
    total_time = 0.0
    total_time_orig = 0.0
    for setup, ch, cp_pairs in pairs:
        n = len(cp_pairs)
        times = t_new[k:k + n]
        k += n
        t_orig = float(getattr(ch, "ChannelTotalTime", 0) or 0)
        t_sum = float(np.sum(times))
        total_time += t_sum
        total_time_orig += t_orig

        cum = 0.0
        for (cp_a, cp_d), dt in zip(cp_pairs, times):
            cp_a.CumulativeTimeWeight = _ds(cum * scale)
            cum += float(dt)
            cp_d.CumulativeTimeWeight = _ds(cum * scale)

        ch.ChannelTotalTime = _ds(t_sum)
        if t_sum > 0:
            ch.FinalCumulativeTimeWeight = _ds(cum * scale)
        else:
            # 0/0 을 피한다: 모든 가중치 0, Final은 1
            ch.FinalCumulativeTimeWeight = _ds(1.0)
            empty_channels.append(int(ch.ChannelNumber))
            warnings.append(
                f"채널 {ch.ChannelNumber}의 dwell time이 전부 0입니다 "
                f"(FinalCumulativeTimeWeight=1, 가중치 0으로 기록). "
                f"TPS/조사기가 빈 채널을 거부할 수 있습니다."
            )
        channels.append({
            "channel": int(ch.ChannelNumber),
            "n_dwells": n,
            "total_time_s": t_sum,
            "total_time_original_s": t_orig,
        })

    # --- TRAK 재계산 -----------------------------------------------------
    sk = float(ds.SourceSequence[0].ReferenceAirKermaRate)
    trak_orig = float(getattr(ds.ApplicationSetupSequence[0],
                              "TotalReferenceAirKerma", 0) or 0)
    trak = sk * total_time / 3600.0
    # 여러 application setup이면 채널 시간을 setup별로 합산해야 하지만,
    # 본 도구가 다루는 자궁경부 IC/IS plan은 setup이 하나다.
    if len(ds.ApplicationSetupSequence) == 1:
        ds.ApplicationSetupSequence[0].TotalReferenceAirKerma = _ds(trak)
    else:
        warnings.append(
            f"ApplicationSetup이 {len(ds.ApplicationSetupSequence)}개입니다 — "
            f"TotalReferenceAirKerma를 갱신하지 않았습니다. 직접 확인하십시오."
        )

    # --- dose point 재계산 (선택) -----------------------------------------
    updated_points: dict[str, float] = {}
    original_points: dict[str, float] = {}
    if recompute_dose_points_with is not None and "DoseReferenceSequence" in ds:
        calc, sk_calc = recompute_dose_points_with
        from .dicom_loader import load_plan
        # 갱신된 시간으로 dwell 재구성 (위치는 원본과 동일)
        tmp = load_plan(src_rtplan_path)
        for d, ti in zip(tmp.dwells, t_new):
            d.time_s = float(ti)
        # dose point 좌표는 RTDOSE에 있으므로, RTPLAN에 좌표가 없으면 건너뛴다
        for dref in ds.DoseReferenceSequence:
            name = str(getattr(dref, "DoseReferenceDescription", "")).strip()
            pos = getattr(dref, "DoseReferencePointCoordinates", None)
            if not name or pos is None:
                continue
            p = np.asarray(pos, dtype=float)[None, :3]
            prev = getattr(dref, "TargetPrescriptionDose", None)
            if prev is not None:
                original_points[name] = float(prev)
            gy = float(calc.dose_at_points(p, tmp.dwells, sk_calc)[0])
            dref.TargetPrescriptionDose = _ds(gy)
            updated_points[name] = gy
        if not updated_points:
            warnings.append(
                "DoseReferenceSequence에 좌표(DoseReferencePointCoordinates)가 없어 "
                "point 선량을 재계산하지 못했습니다. 기록된 값은 원본(구버전) 값입니다."
            )

    # --- 식별자 / 승인상태 -------------------------------------------------
    now = datetime.datetime.now()
    old_sop = str(getattr(ds, "SOPInstanceUID", ""))
    old_series = str(getattr(ds, "SeriesInstanceUID", ""))

    if uid_style == "source" and old_sop:
        root = uid_root_of(old_sop)
        make_uid = lambda ref: short_uid(root, max_len=max(len(ref), 32))  # noqa: E731
        warnings.append(
            f"uid_style=source — 새 UID를 원본 루트({root})에 만들었습니다. "
            f"타 벤더 UID 루트를 사용하므로 연구용으로만 쓰십시오."
        )
    elif uid_style == "source":
        make_uid = lambda ref: generate_uid()                              # noqa: E731
        warnings.append("원본 SOPInstanceUID가 없어 uid_style=source를 적용하지 못했습니다")
    elif uid_style == "pydicom":
        make_uid = lambda ref: generate_uid()                              # noqa: E731
    else:
        raise ValueError(f"uid_style은 'pydicom' 또는 'source'여야 합니다: {uid_style!r}")

    ds.SOPInstanceUID = make_uid(old_sop)
    if "file_meta" in dir(ds) and ds.file_meta is not None:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    if not keep_series_uid:
        ds.SeriesInstanceUID = make_uid(old_series or old_sop)

    # 벤더 私有 plan 식별 UID. 값의 의미는 비공개 사양이므로 기본은 손대지 않는다.
    if new_private_uid:
        tag = pydicom.tag.Tag(0x0021, 0x1000)
        if tag in ds and str(getattr(ds[tag], "VR", "")) == "UI":
            ds[tag].value = make_uid(str(ds[tag].value))
            warnings.append("私有 태그 (0021,1000)에 새 UID를 발급했습니다 "
                            "— 의미가 비공개 사양이므로 import 결과를 확인하십시오")
        else:
            warnings.append("私有 태그 (0021,1000)이 없거나 UI가 아니어서 "
                            "new_private_uid를 적용하지 못했습니다")
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.RTPlanDate = now.strftime("%Y%m%d")
    ds.RTPlanTime = now.strftime("%H%M%S")

    # 기계 생성 계획은 절대 '승인됨'이어서는 안 된다
    ds.ApprovalStatus = "UNAPPROVED"
    for tag in ("ReviewDate", "ReviewTime", "ReviewerName"):
        if tag in ds:
            delattr(ds, tag)

    base_label = str(getattr(ds, "RTPlanLabel", "PLAN"))
    ds.RTPlanLabel = (base_label + label_suffix)[:16]          # SH: 16자
    ds.RTPlanName = (str(getattr(ds, "RTPlanName", base_label)) + label_suffix)[:64]
    ds.RTPlanDescription = description[:64]
    ds.SeriesDescription = description[:64]

    ds.save_as(out_path, enforce_file_format=True) \
        if _supports_enforce() else ds.save_as(out_path)

    return ExportReport(
        out_path=out_path,
        sop_instance_uid=str(ds.SOPInstanceUID),
        series_instance_uid=str(ds.SeriesInstanceUID),
        n_dwells=n_expected,
        total_time_s=total_time,
        total_time_original_s=total_time_orig,
        trak=trak,
        trak_original=trak_orig,
        time_resolution_s=time_resolution_s,
        max_quantization_shift_s=max_shift,
        channels=channels,
        empty_channels=empty_channels,
        dose_points_updated=updated_points,
        dose_points_original=original_points,
        warnings=warnings,
    )


def _supports_enforce() -> bool:
    """pydicom 3.x 의 save_as(enforce_file_format=...) 지원 여부."""
    import inspect
    try:
        return "enforce_file_format" in inspect.signature(
            pydicom.Dataset.save_as).parameters
    except (TypeError, ValueError):   # pragma: no cover
        return False


# ---------------------------------------------------------------------------
def verify_rtplan(out_path: str, expected_times_s,
                  tol_s: float = 1e-6) -> tuple[bool, list[str]]:
    """저장한 파일을 **다시 읽어** dwell time과 TRAK 정합성을 확인한다.

    쓰기 코드만 검증하는 게 아니라, DICOM 왕복(DS 문자열 변환 포함)에서
    값이 살아남는지 본다.
    """
    from .dicom_loader import load_plan
    msgs: list[str] = []
    plan = load_plan(out_path)
    got = np.array([d.time_s for d in plan.dwells], dtype=float)
    want = np.asarray(expected_times_s, dtype=float)

    ok = True
    if got.shape != want.shape:
        return False, [f"dwell 수 불일치: {got.shape} vs {want.shape}"]
    err = float(np.max(np.abs(got - want))) if len(got) else 0.0
    if err > tol_s:
        ok = False
        msgs.append(f"dwell time 왕복 오차 {err:.3e} s > {tol_s:g} s")
    else:
        msgs.append(f"dwell time 왕복 오차 {err:.3e} s  OK")

    ds = pydicom.dcmread(out_path, stop_before_pixels=True)
    sk = float(ds.SourceSequence[0].ReferenceAirKermaRate)
    trak_file = float(ds.ApplicationSetupSequence[0].TotalReferenceAirKerma)
    trak_calc = sk * float(np.sum(got)) / 3600.0
    if trak_calc > 0 and abs(trak_file - trak_calc) / trak_calc > 1e-4:
        ok = False
        msgs.append(f"TRAK 불일치: 파일 {trak_file:.4f} vs 계산 {trak_calc:.4f}")
    else:
        msgs.append(f"TRAK {trak_file:.4f} = S_K x sum(t)/3600  OK")

    if str(ds.ApprovalStatus).upper() != "UNAPPROVED":
        ok = False
        msgs.append(f"ApprovalStatus가 {ds.ApprovalStatus} — UNAPPROVED여야 합니다")
    else:
        msgs.append("ApprovalStatus = UNAPPROVED  OK")

    # 가중치 단조성
    for setup in ds.ApplicationSetupSequence:
        for ch in setup.ChannelSequence:
            w = [float(cp.CumulativeTimeWeight) for cp in ch.BrachyControlPointSequence]
            if any(b < a - 1e-9 for a, b in zip(w, w[1:])):
                ok = False
                msgs.append(f"채널 {ch.ChannelNumber}: CumulativeTimeWeight가 감소함")
            if w and abs(w[0]) > 1e-9:
                ok = False
                msgs.append(f"채널 {ch.ChannelNumber}: 첫 가중치가 0이 아님 ({w[0]})")
            f = float(ch.FinalCumulativeTimeWeight)
            if w and float(ch.ChannelTotalTime) > 0 and abs(w[-1] - f) > 1e-6 * max(f, 1):
                ok = False
                msgs.append(f"채널 {ch.ChannelNumber}: 마지막 가중치 {w[-1]} != "
                            f"FinalCumulativeTimeWeight {f}")
    if ok:
        msgs.append("CumulativeTimeWeight 단조성 / 시작 0 / Final 일치  OK")
    return ok, msgs


def dose_change_from_quantization(ctx, t_raw, t_quantized) -> dict[str, float]:
    """양자화로 인한 구조물 지표 변화 [Gy]. (보고용)"""
    a = ctx.evaluate(np.asarray(t_raw, dtype=float))
    b = ctx.evaluate(np.asarray(t_quantized, dtype=float))
    out: dict[str, float] = {}
    for name in a:
        for metric in ("D90_Gy", "D98_Gy", "D2cm3_Gy"):
            if metric in a[name] and metric in b[name]:
                d = b[name][metric] - a[name][metric]
                if not math.isnan(d):
                    out[f"{name}.{metric.replace('_Gy', '')}"] = d
    return out
