"""EMBRACE-II 프로토콜 기반 임상 목표.

출처
----
EMBRACE II study protocol v.1.0 (원문 PDF에서 직접 인용)
https://www.embracestudy.dk/UserUpload/PublicDocuments/EMBRACE%20II%20Protocol.pdf

인용문 (프로토콜 본문 그대로)
  §3.3.1 "The data supports a dose constraint of >=85Gy EQD2 to the CTVHR D90
          which is predicted to lead to a 3-year actuarial local control of >96%
          in tumours <=30cc and >91% in tumours >30cc. Dose planning aims for
          CTVIR and GTVres proposed for similar levels of local control are:
          CTVIR D98>=60Gy and GTVres D98>=95Gy."
  §2    "planning aims ... for 2cm3 reference volumes for OARs
          (e.g. <80 Gy for bladder, <65 Gy for rectum) and for vaginal reference
          points (recto-vaginal point < 65 Gy, PIBS)"
  §3.3.5 "it may be appropriate to aim for sigmoid and bowel dose planning aim
          of 70Gy in case there are adherences."
  §4.1.6 "EBRT has to be delivered as IMRT/VMAT with daily cone beam CT (IGRT)
          in 25 fractions with 1.8 Gy to a total dose of 45 Gy"

★ 중요 — planning aim 과 limit 은 다른 것이다
---------------------------------------------
EMBRACE-II는 목표를 두 단계로 나눈다 (Pötter et al., Clin Transl Radiat Oncol
2018;9:48-60, Table 4 "Planning aims (soft constraints) and limits for
prescribed dose (hard constraints)"; https://doi.org/10.1016/j.ctro.2018.01.001).

  지표                     planning aim    limit
  CTVHR D90  (EQD2_10)       > 90 Gy      > 85 Gy
  CTVHR D98  (EQD2_10)       > 75 Gy        -
  GTVres D98 (EQD2_10)       > 95 Gy      > 90 Gy
  CTVIR D98  (EQD2_10)       > 60 Gy        -
  Bladder D2cm3 (EQD2_3)     < 80 Gy      < 90 Gy
  Rectum  D2cm3 (EQD2_3)     < 65 Gy      < 75 Gy
  Sigmoid D2cm3 (EQD2_3)     < 70 Gy*     < 75 Gy*
  Bowel   D2cm3 (EQD2_3)     < 70 Gy*     < 75 Gy*
  (* 유착 등으로 장 루프가 고정되어 최대선량 부위가 매번 같을 때 적용)

본문(§3.3.7)은 "Hard constraints must be fulfilled for 80-90% of the patients,
and soft constraints should be fulfilled as often as possible"라고 규정한다.
따라서 aim 은 soft goal, limit 은 hard goal 로 넣는다. aim 을 hard 로 걸면
임상적으로 수용 가능한 plan까지 '필수 위반'으로 판정되어 에이전트가 달성
불가능한 문제를 풀게 된다 (실측: 승인된 임상 plan조차 bowel aim 70 Gy를
0.2 Gy 초과 — limit 75 Gy 기준으로는 합격).

★ 중요 — 이 수치는 **EBRT + 근접치료 누적 EQD2**다
------------------------------------------------------
프로토콜의 모든 값은 외부방사선과 근접치료를 합산한 EQD2이지, 근접치료
fraction당 물리선량이 아니다. 따라서 이 도구에서 쓰려면
  1) EBRT 처방(기본 45 Gy / 25 fx)을 EQD2로 환산해 빼고,
  2) 남은 EQD2를 근접치료 fraction 수로 나눠 fraction당 물리선량으로 되돌려야
     한다.
그 변환을 `bt_limit_per_fraction()`이 수행한다.

가정 (모두 EMBRACE/GEC-ESTRO 관례이며, 보수적인 쪽이다)
  - alpha/beta = 10 Gy (종양), 3 Gy (만기반응 정상조직)
  - OAR의 D2cm3 위치가 EBRT 전량을 받는다 (중앙차폐 미사용 전제)
  - 분할 간 DVH 최악치 합산 (같은 위치가 매번 최대선량을 받는다고 가정)
  - 전체 치료기간(OTT) 보정은 하지 않는다
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .evaluator import Goal

AB_TUMOR = 10.0
AB_OAR = 3.0

# 기본 EBRT (프로토콜 §4.1.6에서 고정)
DEFAULT_EBRT_DOSE_GY = 45.0
DEFAULT_EBRT_FRACTIONS = 25


@dataclass(frozen=True)
class Embrace2Constraint:
    structure: str
    metric: str                    # 우리 DVH 지표 이름
    kind: str                      # "min" | "max"
    aim_eqd2_gy: float             # planning aim (soft) — 누적 EQD2 [Gy]
    limit_eqd2_gy: float | None    # limit for prescribed dose (hard). 없으면 None
    alpha_beta: float
    weight: float
    source: str                    # 프로토콜 인용 위치

    @property
    def eqd2_gy(self) -> float:
        """하위호환 — 예전 단일 값은 planning aim에 해당한다."""
        return self.aim_eqd2_gy


# 프로토콜에서 직접 확인한 값만 넣는다 (Table 4 + 본문 인용).
CONSTRAINTS: tuple[Embrace2Constraint, ...] = (
    Embrace2Constraint("HR_CTV", "D90_Gy", "min", 90.0, 85.0, AB_TUMOR, 3.0,
                       "§3.3.1/Table 4 CTVHR D90 aim>=90, limit>=85 Gy"),
    Embrace2Constraint("HR_CTV", "D98_Gy", "min", 75.0, None, AB_TUMOR, 1.0,
                       "Table 4 CTVHR D98 aim>=75 Gy"),
    Embrace2Constraint("IR_CTV", "D98_Gy", "min", 60.0, None, AB_TUMOR, 1.0,
                       "§3.3.1/Table 4 CTVIR D98 aim>=60 Gy"),
    Embrace2Constraint("GTV", "D98_Gy", "min", 95.0, 90.0, AB_TUMOR, 1.0,
                       "§3.3.1/Table 4 GTVres D98 aim>=95, limit>=90 Gy"),
    Embrace2Constraint("Bladder", "D2cm3_Gy", "max", 80.0, 90.0, AB_OAR, 1.5,
                       "§2/Table 4 bladder D2cm3 aim<80, limit<90 Gy"),
    Embrace2Constraint("Rectum", "D2cm3_Gy", "max", 65.0, 75.0, AB_OAR, 1.5,
                       "§2/Table 4 rectum D2cm3 aim<65, limit<75 Gy"),
    Embrace2Constraint("Sigmoid", "D2cm3_Gy", "max", 70.0, 75.0, AB_OAR, 1.5,
                       "§3.3.5/Table 4 sigmoid D2cm3 aim<70, limit<75 Gy"),
    Embrace2Constraint("Bowel", "D2cm3_Gy", "max", 70.0, 75.0, AB_OAR, 1.5,
                       "§3.3.5/Table 4 bowel D2cm3 aim<70, limit<75 Gy"),
)

# 프로토콜에 있으나 본 도구의 지표로 직접 매핑되지 않는 것 (참고용)
NOT_IMPLEMENTED = (
    "recto-vaginal point < 65 Gy (§2) — dose point 기반이라 DVH 지표가 아님",
    "PIBS, 질 점막 선량 — 별도 reference point 필요",
    "point A — 본 도구는 volumetric 처방을 전제",
    "CTVHR D90 planning aim의 상한 <95 Gy (Table 4) — 하한 목표와 방향이 반대라 "
    "동시에 걸면 목적함수가 상충하므로 미반영",
)


# ---------------------------------------------------------------------------
def eqd2(total_dose_gy: float, dose_per_fraction_gy: float,
         alpha_beta: float) -> float:
    """선형이차 모델 EQD2.

        EQD2 = D * (d + a/b) / (2 + a/b)
    """
    return total_dose_gy * (dose_per_fraction_gy + alpha_beta) / (2.0 + alpha_beta)


def ebrt_eqd2(dose_gy: float = DEFAULT_EBRT_DOSE_GY,
              fractions: int = DEFAULT_EBRT_FRACTIONS,
              alpha_beta: float = AB_OAR) -> float:
    if fractions <= 0:
        return 0.0
    return eqd2(dose_gy, dose_gy / fractions, alpha_beta)


def bt_limit_per_fraction(
    cumulative_eqd2_gy: float,
    n_bt_fractions: int,
    alpha_beta: float,
    ebrt_dose_gy: float = DEFAULT_EBRT_DOSE_GY,
    ebrt_fractions: int = DEFAULT_EBRT_FRACTIONS,
) -> float:
    """누적 EQD2 목표 -> 근접치료 **fraction당 물리선량** [Gy].

    남은 EQD2를 n회로 나눠 받으려면 fraction당 얼마여야 하는가:
        n * d * (d + ab) / (2 + ab) = R        (R = 누적목표 - EBRT EQD2)
        => n*d^2 + n*ab*d - R*(2+ab) = 0
        => d = [-n*ab + sqrt((n*ab)^2 + 4*n*R*(2+ab))] / (2n)

    EBRT만으로 이미 목표를 넘으면 0을 반환한다 (근접치료 여유 없음).
    """
    if n_bt_fractions <= 0:
        raise ValueError("n_bt_fractions는 1 이상이어야 합니다")
    remaining = cumulative_eqd2_gy - ebrt_eqd2(ebrt_dose_gy, ebrt_fractions, alpha_beta)
    if remaining <= 0:
        return 0.0
    n, ab = float(n_bt_fractions), float(alpha_beta)
    disc = (n * ab) ** 2 + 4.0 * n * remaining * (2.0 + ab)
    return (-n * ab + math.sqrt(disc)) / (2.0 * n)


def cumulative_eqd2_from_bt(
    bt_dose_per_fraction_gy: float,
    n_bt_fractions: int,
    alpha_beta: float,
    ebrt_dose_gy: float = DEFAULT_EBRT_DOSE_GY,
    ebrt_fractions: int = DEFAULT_EBRT_FRACTIONS,
) -> float:
    """fraction당 근접치료 물리선량 -> EBRT 포함 누적 EQD2 [Gy]. (리포트용)"""
    d = float(bt_dose_per_fraction_gy)
    bt = eqd2(d * n_bt_fractions, d, alpha_beta)
    return ebrt_eqd2(ebrt_dose_gy, ebrt_fractions, alpha_beta) + bt


# ---------------------------------------------------------------------------
def goals_from_embrace2(
    n_bt_fractions: int,
    available_structures: set[str] | None = None,
    ebrt_dose_gy: float = DEFAULT_EBRT_DOSE_GY,
    ebrt_fractions: int = DEFAULT_EBRT_FRACTIONS,
) -> tuple[list[Goal], list[str]]:
    """EMBRACE-II 누적 EQD2 목표를 fraction당 물리선량 목표로 환산한다.

    planning aim 은 soft goal, limit for prescribed dose 는 hard goal 로
    각각 만든다 (Table 4). limit 이 없는 지표는 soft goal 하나만 나온다.

    Returns
    -------
    (goals, 설명 줄 리스트)
    """
    goals: list[Goal] = []
    lines: list[str] = [
        f"EBRT {ebrt_dose_gy:g} Gy / {ebrt_fractions} fx, 근접치료 {n_bt_fractions} fx 기준",
        f"  EBRT EQD2: 종양(a/b={AB_TUMOR:g}) "
        f"{ebrt_eqd2(ebrt_dose_gy, ebrt_fractions, AB_TUMOR):.2f} Gy, "
        f"OAR(a/b={AB_OAR:g}) {ebrt_eqd2(ebrt_dose_gy, ebrt_fractions, AB_OAR):.2f} Gy",
        "",
        f"{'구조물':<10}{'지표':<8}{'aim(EQD2)':>10}{'BT/fx':>8}"
        f"{'limit(EQD2)':>12}{'BT/fx':>8}  출처",
        "-" * 96,
    ]
    for c in CONSTRAINTS:
        if available_structures is not None and c.structure not in available_structures:
            continue
        op = ">=" if c.kind == "min" else "<="

        def per_fx(eqd2_gy: float) -> float:
            return bt_limit_per_fraction(eqd2_gy, n_bt_fractions, c.alpha_beta,
                                         ebrt_dose_gy, ebrt_fractions)

        d_aim = per_fx(c.aim_eqd2_gy)
        d_lim = per_fx(c.limit_eqd2_gy) if c.limit_eqd2_gy is not None else None
        lines.append(
            f"{c.structure:<10}{c.metric.replace('_Gy', ''):<8}"
            f"{op + str(c.aim_eqd2_gy):>10}{d_aim:8.3f}"
            f"{(op + str(c.limit_eqd2_gy)) if d_lim is not None else '-':>12}"
            f"{d_lim if d_lim is not None else float('nan'):8.3f}  {c.source}"
        )
        if d_aim > 0:
            goals.append(Goal(structure=c.structure, metric=c.metric, kind=c.kind,
                              limit=round(d_aim, 4), weight=c.weight, hard=False,
                              label=c.source + " [aim]"))
        else:
            lines.append("           [경고] EBRT만으로 이미 aim 도달 — 목표 제외")
        if d_lim is not None:
            if d_lim > 0:
                goals.append(Goal(structure=c.structure, metric=c.metric, kind=c.kind,
                                  limit=round(d_lim, 4), weight=c.weight, hard=True,
                                  label=c.source + " [limit]"))
            else:
                lines.append("           [경고] EBRT만으로 이미 limit 도달 — 목표 제외")
    lines.append("-" * 96)
    lines.append("EMBRACE-II 값은 EBRT+BT 누적 EQD2이며, 위 표는 이를 "
                 "fraction당 물리선량으로 환산한 것입니다.")
    lines.append("aim = planning aim (soft), limit = limit for prescribed dose "
                 "(hard). 필수 판정은 limit 기준입니다.")
    return goals, lines


def eqd2_report(
    metrics: dict,
    n_bt_fractions: int,
    ebrt_dose_gy: float = DEFAULT_EBRT_DOSE_GY,
    ebrt_fractions: int = DEFAULT_EBRT_FRACTIONS,
    available_structures: set[str] | None = None,
) -> str:
    """계산된 fraction당 선량을 누적 EQD2로 되돌려 프로토콜 값과 대조.

    aim(soft)과 limit(hard)을 함께 보여준다. limit 위반만이 프로토콜상
    '지켜야 하는 선'을 넘은 것이다.
    """
    lines = [f"{'구조물':<10}{'지표':<8}{'BT/fx':>8}{'누적EQD2':>10}"
             f"{'aim':>9}{'여유':>8}{'limit':>9}{'여유':>8}  판정",
             "-" * 80]
    for c in CONSTRAINTS:
        if available_structures is not None and c.structure not in available_structures:
            continue
        row = metrics.get(c.structure)
        if not row or c.metric not in row:
            continue
        d = float(row[c.metric])
        cum = cumulative_eqd2_from_bt(d, n_bt_fractions, c.alpha_beta,
                                      ebrt_dose_gy, ebrt_fractions)
        sign = 1.0 if c.kind == "min" else -1.0

        def margin_of(target: float) -> float:
            return sign * (cum - target)

        op = ">=" if c.kind == "min" else "<="
        m_aim = margin_of(c.aim_eqd2_gy)
        aim_ok = m_aim >= 0.0
        if c.limit_eqd2_gy is None:
            lim_txt, lim_margin_txt = "-", "-"
            verdict = "OK" if aim_ok else "MISS(aim)"
        else:
            m_lim = margin_of(c.limit_eqd2_gy)
            lim_txt = op + str(c.limit_eqd2_gy)
            lim_margin_txt = f"{m_lim:+.1f}"
            verdict = "OK" if aim_ok else ("MISS(aim)" if m_lim >= 0.0
                                           else "MISS(limit)")
        lines.append(f"{c.structure:<10}{c.metric.replace('_Gy', ''):<8}{d:8.2f}"
                     f"{cum:10.1f}{op + str(c.aim_eqd2_gy):>9}{m_aim:+8.1f}"
                     f"{lim_txt:>9}{lim_margin_txt:>8}  {verdict}")
    return "\n".join(lines)
