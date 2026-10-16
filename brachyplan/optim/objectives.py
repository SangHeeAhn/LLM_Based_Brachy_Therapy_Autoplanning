"""dwell time 역최적화 목적함수.

수학적 형태
-----------
dose = A^T t   (A: (n_dwell, n_voxel) [Gy/s],  t: dwell time [s])

    J(t) = sum_s [ w_s^under * mean( max(0, Dmin_s - d_i)^2 )
                 + w_s^over  * mean( max(0, d_i - Dmax_s)^2 ) ]
         + lam_smooth * sum_over_channels sum_k (t_k - t_{k+1})^2
         + lam_total  * (sum_k t_k - T_ref)^2
    s.t.  0 <= t_k <= t_max

성질
----
* max(0, ·)^2 는 아핀함수의 볼록·미분가능 합성이므로 **J는 볼록이고 C^1**이다.
  박스 제약도 볼록이므로 전역 최적해가 보장된다 (국소 최소 없음).
* 따라서 투영 경사법 계열이면 어떤 초기값에서도 같은 해로 수렴한다.
  Phase 3에서 LLM이 가중치를 바꿔도 "최적화가 실패해서 나쁜 plan"이 나올 일은
  없고, 오직 가중치 선택의 문제로 환원된다 — 이게 LLM을 파라미터 튜너로만
  쓰는 설계의 핵심 근거다.

smoothness 항의 의미
--------------------
IPSA의 알려진 단점은 특정 dwell에 과도하게 긴 시간을 몰아주는 것이다
(Poder 2016, Yoshio 2018). 인접 dwell time 차이에 2차 벌점을 주면
임상적으로 더 매끄러운 분포가 나온다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class StructureObjective:
    """한 구조물에 대한 선량 목표.

    dose_min : 이 값보다 낮은 복셀에 벌점 (표적 커버리지)
    dose_max : 이 값보다 높은 복셀에 벌점 (OAR 보호 / 표적 hot spot 억제)
    """

    name: str
    dose_min: float | None = None
    dose_max: float | None = None
    weight_under: float = 0.0
    weight_over: float = 0.0

    def validate(self) -> None:
        if self.dose_min is not None and self.dose_min < 0:
            raise ValueError(f"{self.name}: dose_min < 0")
        if self.dose_max is not None and self.dose_max < 0:
            raise ValueError(f"{self.name}: dose_max < 0")
        if (self.dose_min is not None and self.dose_max is not None
                and self.dose_min > self.dose_max):
            raise ValueError(f"{self.name}: dose_min > dose_max")
        if self.weight_under < 0 or self.weight_over < 0:
            raise ValueError(f"{self.name}: 가중치는 음수일 수 없습니다")


@dataclass
class OptimizationProblem:
    """구조물별 kernel과 목표를 묶은 최적화 문제.

    kernels[name] : (n_dwell, n_voxel_s) [Gy/s]
    """

    kernels: dict[str, np.ndarray]
    objectives: list[StructureObjective]
    channel_of: np.ndarray                    # (n_dwell,) 채널 번호
    t_max: float = 60.0                       # dwell당 최대 시간 [s]
    lam_smooth: float = 0.0
    lam_total: float = 0.0
    total_time_ref: float | None = None       # T_ref [s]
    _obj_by_name: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for o in self.objectives:
            o.validate()
            if o.name not in self.kernels:
                raise KeyError(f"목표 '{o.name}'에 해당하는 kernel이 없습니다")
        self._obj_by_name = {o.name: o for o in self.objectives}
        n = self.n_dwell
        for name, K in self.kernels.items():
            if K.shape[0] != n:
                raise ValueError(f"kernel '{name}'의 dwell 수 불일치: {K.shape[0]} != {n}")
        if len(self.channel_of) != n:
            raise ValueError("channel_of 길이가 dwell 수와 다릅니다")

    @property
    def n_dwell(self) -> int:
        return int(next(iter(self.kernels.values())).shape[0])

    # ------------------------------------------------------------------
    def _smooth_pairs(self) -> list[tuple[int, int]]:
        """같은 채널 안에서 인접한 dwell 인덱스 쌍.

        dwell은 채널별로 순서대로 저장되어 있다고 가정한다
        (dicom_loader가 그렇게 만든다).
        """
        pairs: list[tuple[int, int]] = []
        ch = np.asarray(self.channel_of)
        for c in np.unique(ch):
            idx = np.flatnonzero(ch == c)
            pairs.extend(zip(idx[:-1], idx[1:]))
        return pairs

    # ------------------------------------------------------------------
    def value_and_grad(self, t: np.ndarray) -> tuple[float, np.ndarray]:
        """J(t) 와 dJ/dt 를 함께 계산."""
        t = np.asarray(t, dtype=float)
        J = 0.0
        grad = np.zeros_like(t)

        for o in self.objectives:
            K = self.kernels[o.name]          # (n_dwell, n_vox)
            n_vox = K.shape[1]
            if n_vox == 0:
                continue
            d = K.T @ t                        # (n_vox,) [Gy]

            if o.dose_min is not None and o.weight_under > 0:
                short = np.maximum(0.0, o.dose_min - d)
                J += o.weight_under * float(np.mean(short ** 2))
                # dJ/dd = -2w/n * short
                grad += K @ (-2.0 * o.weight_under / n_vox * short)

            if o.dose_max is not None and o.weight_over > 0:
                over = np.maximum(0.0, d - o.dose_max)
                J += o.weight_over * float(np.mean(over ** 2))
                grad += K @ (2.0 * o.weight_over / n_vox * over)

        if self.lam_smooth > 0:
            for i, j in self._smooth_pairs():
                diff = t[i] - t[j]
                J += self.lam_smooth * diff ** 2
                grad[i] += 2.0 * self.lam_smooth * diff
                grad[j] -= 2.0 * self.lam_smooth * diff

        if self.lam_total > 0 and self.total_time_ref is not None:
            diff = float(np.sum(t)) - self.total_time_ref
            J += self.lam_total * diff ** 2
            grad += 2.0 * self.lam_total * diff

        return J, grad

    def value(self, t: np.ndarray) -> float:
        return self.value_and_grad(t)[0]

    # ------------------------------------------------------------------
    def set_weights(self, params: dict) -> None:
        """LLM(Phase 3)이 내놓은 파라미터를 적용한다.

        형식:
            {"structures": {"HR_CTV": {"dose_min": 5.0, "weight_under": 100}, ...},
             "lam_smooth": 0.01, "lam_total": 0.0}
        알 수 없는 키는 무시하지 않고 예외를 던진다 (LLM 오출력 조기 발견).
        """
        allowed_struct = {"dose_min", "dose_max", "weight_under", "weight_over"}
        allowed_top = {"structures", "lam_smooth", "lam_total", "total_time_ref"}

        unknown = set(params) - allowed_top
        if unknown:
            raise KeyError(f"알 수 없는 최상위 키: {sorted(unknown)}")

        for name, upd in (params.get("structures") or {}).items():
            if name not in self._obj_by_name:
                raise KeyError(f"알 수 없는 구조물: {name}")
            bad = set(upd) - allowed_struct
            if bad:
                raise KeyError(f"{name}: 알 수 없는 필드 {sorted(bad)}")
            o = self._obj_by_name[name]
            for k, v in upd.items():
                setattr(o, k, None if v is None else float(v))
            o.validate()

        for k in ("lam_smooth", "lam_total", "total_time_ref"):
            if k in params and params[k] is not None:
                v = float(params[k])
                if k != "total_time_ref" and v < 0:
                    raise ValueError(f"{k} < 0")
                setattr(self, k, v)

    def load_weights(self, full: dict) -> None:
        """get_weights() 덤프를 **전체 교체**로 되돌린다.

        set_weights()는 병합이라 이전 상태로 되돌릴 수 없다. 에이전트가
        나쁜 제안으로 plan을 망가뜨렸을 때 최선 상태로 복구하는 데 쓴다.
        """
        for name, vals in (full.get("structures") or {}).items():
            o = self._obj_by_name.get(name)
            if o is None:
                continue
            for k in ("dose_min", "dose_max", "weight_under", "weight_over"):
                if k in vals:
                    v = vals[k]
                    setattr(o, k, None if v is None else float(v))
            o.validate()
        for k in ("lam_smooth", "lam_total", "total_time_ref"):
            if k in full and full[k] is not None:
                setattr(self, k, float(full[k]))

    def get_weights(self) -> dict:
        return {
            "structures": {o.name: {
                "dose_min": o.dose_min, "dose_max": o.dose_max,
                "weight_under": o.weight_under, "weight_over": o.weight_over,
            } for o in self.objectives},
            "lam_smooth": self.lam_smooth,
            "lam_total": self.lam_total,
            "total_time_ref": self.total_time_ref,
        }


# ---------------------------------------------------------------------------
def default_objectives(
    prescription_gy: float,
    target_names: tuple[str, ...] = ("HR_CTV",),
    oar_limits: dict[str, float] | None = None,
    oar_weight: float = 1.0,
    target_weight_under: float = 100.0,
    target_weight_over: float = 1.0,
    hot_spot_factor: float | None = None,
) -> list[StructureObjective]:
    """기본 목표 세트.

    표적: dose_min = 처방선량. dose_max는 **기본적으로 걸지 않는다.**
    OAR : dose_max = oar_limits[name]  (임상 plan 값 또는 기관 프로토콜 값)

    hot_spot_factor를 걸면 안 되는 이유 (실측으로 확인)
    ---------------------------------------------------
    근접치료는 선원이 표적 **안에** 있으므로 고선량 영역을 피할 수 없다.
    본 케이스 임상 plan의 HR-CTV는 D50 = 8.0 Gy, V150 = 57 %, V200 = 29 %,
    Dmax = 771 Gy로, 표적의 절반 이상이 처방의 150 %를 넘는다. 이것이 정상이다.

    외부방사선(EBRT)식으로 dose_max = 처방 x 1.5 를 걸면 최적화기가 dwell time을
    억눌러 D90이 오히려 나빠진다. 실측: hot_spot 1.5 -> D90 4.87 Gy (임상 5.57),
    OAR도 전부 악화. hot_spot을 없애면 D90 5.70 Gy, OAR 대부분 개선.

    hot spot을 굳이 통제하려면 처방의 400~600 % 같은 훨씬 높은 값에
    약한 가중치로 걸어야 한다 (그래도 D90은 손해를 본다).

    NOTE: EMBRACE-II의 구체적 D2cm3 수치는 여기에 하드코딩하지 않는다.
          프로토콜 원문을 확인해 goals JSON에 넣거나, 임상 plan 값을 기준으로 쓴다.
    """
    objs = [
        StructureObjective(
            name=n,
            dose_min=prescription_gy,
            dose_max=(None if hot_spot_factor is None
                      else prescription_gy * hot_spot_factor),
            weight_under=target_weight_under,
            weight_over=(0.0 if hot_spot_factor is None else target_weight_over),
        )
        for n in target_names
    ]
    for name, limit in (oar_limits or {}).items():
        objs.append(StructureObjective(name=name, dose_max=float(limit),
                                       weight_over=oar_weight))
    return objs


def objectives_from_goals(
    goals,
    primary_target: str,
    secondary_target: str | None = None,
    available: set[str] | None = None,
    target_weight_under: float = 100.0,
    secondary_weight_under: float = 50.0,
    oar_weight: float = 1.0,
) -> list[StructureObjective]:
    """평가 목표(goals)에서 최적화 파라미터를 유도한다.

    왜 필요한가
    -----------
    이전에는 목표는 EMBRACE-II 프로토콜에서, 최적화기의 OAR 상한은 **임상
    plan의 실측 D2cm3**에서 가져왔다. 출처가 섞여 있어서

      * 신환(임상 plan의 dwell time이 아직 의미 없는 경우)에 적용할 수 없고,
      * "무엇을 기준으로 최적화했는가"를 한 줄로 설명할 수 없다.

    실측: 같은 케이스에서 상한만 임상값 대신 균일시간 plan 값으로 바꾸면
    HR_CTV D90이 6.81 -> 7.51 Gy로 움직였다. 즉 임상 dwell time이 결과에
    직접 영향을 준다. 목표에서 유도하면 이 경로가 사라지고, 임상 plan 의존은
    **기하(dwell 위치) + 선원 + 처방**만 남는다.

    규칙
    ----
    구조물별로 goals를 모아
      dose_min = max(min류 목표의 limit)   (가장 엄격한 하한 = planning aim)
      dose_max = min(max류 목표의 limit)   (가장 엄격한 상한 = planning aim)
    가중치는
      주표적   -> weight_under = target_weight_under
      부표적   -> weight_under = secondary_weight_under
      그 외 표적(예: GTV) -> 0  (에이전트가 올릴 수 있는 '손잡이')
      상한이 있는 구조물 -> weight_over = oar_weight

    표적에 dose_max를 걸지 않는 원칙은 그대로다 (default_objectives 참조).
    """
    by_name: dict[str, dict[str, list[float]]] = {}
    for g in goals:
        if available is not None and g.structure not in available:
            continue
        slot = by_name.setdefault(g.structure, {"min": [], "max": []})
        if g.kind in slot:
            slot[g.kind].append(float(g.limit))

    objs: list[StructureObjective] = []
    for name, lim in by_name.items():
        dose_min = max(lim["min"]) if lim["min"] else None
        dose_max = min(lim["max"]) if lim["max"] else None
        if name == primary_target:
            w_under = target_weight_under
        elif secondary_target and name == secondary_target:
            w_under = secondary_weight_under
        else:
            w_under = 0.0
        if dose_min is None:
            w_under = 0.0
        w_over = oar_weight if dose_max is not None else 0.0
        # 표적에는 상한을 걸지 않는다 (근접치료 hot spot은 정상)
        if dose_min is not None and dose_max is not None:
            dose_max, w_over = None, 0.0
        objs.append(StructureObjective(name=name, dose_min=dose_min,
                                       dose_max=dose_max,
                                       weight_under=w_under, weight_over=w_over))
    objs.sort(key=lambda o: (o.name != primary_target, o.name))
    return objs


def save_objectives(objs: list[StructureObjective], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(o) for o in objs], fh, indent=1, ensure_ascii=False)


def load_objectives(path: str) -> list[StructureObjective]:
    with open(path, encoding="utf-8") as fh:
        return [StructureObjective(**d) for d in json.load(fh)]
