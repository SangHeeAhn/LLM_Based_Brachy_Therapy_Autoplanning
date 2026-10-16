#!/usr/bin/env python3
"""
Ir-192 microSelectron-v2 (mHDR-v2) TG-43 데이터셋 JSON 생성기
=============================================================
출처: CLRP TG-43 Parameter Database (Taylor & Rogers, EGSnrc)
      https://physics.carleton.ca/clrp/seed_database/Ir192_HDR/microSelectron_v2
      - F(r,theta) : microSelectron_v2_fofrt.htm  (L = 0.36 cm, 12 radii x 47 angles)
      - g_L(r)     : 상기 페이지의 modified polynomial fit 계수 (0.20 - 20.00 cm)
      - Lambda     : 1.109 cGy h^-1 U^-1 (Taylor&Rogers 2008; HEBD consensus와 동일)

이 스크립트는 표를 텍스트로 보관하고 파싱하여 JSON을 만든다.
숫자를 손으로 재입력하지 않고 표 원문을 그대로 두는 것이 목적이다.

검증: 생성 직후 F(r,90deg)==1.0, g_L(1)==1.0 을 assert 한다.
"""

from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# F(r, theta) 원표 (CLRP, L = 0.36 cm). '-' 는 미제공 (선원 캡슐 내부/근처)
# 열 순서: r = 0.25 0.50 1.00 2.00 3.00 4.00 5.00 7.50 10.00 12.50 15.00 20.00
# ---------------------------------------------------------------------------
F_RADII = [0.25, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00, 12.50, 15.00, 20.00]

F_TABLE_TEXT = """
0    -     0.667 0.630 0.635 0.658 0.677 0.694 0.741 0.775 0.800 0.819 0.848
1    -     0.665 0.628 0.638 0.659 0.681 0.699 0.744 0.780 0.803 0.822 0.850
2    -     0.665 0.632 0.645 0.669 0.691 0.709 0.751 0.783 0.807 0.826 0.851
3    -     0.666 0.639 0.654 0.677 0.699 0.717 0.757 0.786 0.811 0.830 0.854
5    -     0.675 0.660 0.677 0.698 0.716 0.734 0.773 0.801 0.823 0.837 0.863
7    -     0.694 0.687 0.703 0.721 0.738 0.751 0.787 0.812 0.833 0.846 0.869
10   -     0.733 0.727 0.742 0.759 0.773 0.785 0.815 0.837 0.852 0.863 0.879
12   0.807 0.759 0.754 0.766 0.780 0.794 0.804 0.831 0.851 0.865 0.874 0.890
15   0.838 0.793 0.788 0.799 0.811 0.822 0.829 0.853 0.868 0.880 0.887 0.900
20   0.881 0.843 0.839 0.845 0.854 0.860 0.865 0.882 0.896 0.902 0.907 0.917
25   0.912 0.880 0.875 0.881 0.887 0.893 0.894 0.907 0.915 0.921 0.925 0.934
30   0.934 0.908 0.904 0.908 0.913 0.918 0.919 0.927 0.932 0.938 0.939 0.945
35   0.949 0.929 0.929 0.928 0.932 0.936 0.935 0.944 0.948 0.950 0.952 0.955
40   0.961 0.946 0.948 0.945 0.947 0.948 0.948 0.954 0.957 0.961 0.961 0.965
45   0.970 0.959 0.958 0.959 0.961 0.962 0.960 0.966 0.968 0.971 0.969 0.974
50   0.977 0.969 0.969 0.969 0.970 0.970 0.969 0.973 0.977 0.977 0.977 0.980
55   0.984 0.978 0.977 0.978 0.980 0.980 0.977 0.980 0.985 0.983 0.982 0.988
60   0.989 0.984 0.983 0.985 0.984 0.982 0.984 0.986 0.989 0.988 0.988 0.992
65   0.993 0.990 0.991 0.990 0.990 0.990 0.990 0.991 0.992 0.994 0.992 0.996
70   0.995 0.993 0.995 0.994 0.995 0.996 0.995 0.995 0.995 0.995 0.996 0.994
75   0.998 0.996 0.994 0.996 0.996 0.996 0.995 0.997 0.995 0.998 0.997 0.996
80   0.999 0.999 0.999 0.999 0.998 0.996 0.994 0.997 0.998 0.999 0.996 0.999
85   1.000 1.001 1.002 1.000 1.001 1.000 0.997 1.000 1.000 1.001 0.999 1.001
90   1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000
95   1.000 0.999 1.001 0.999 1.000 1.000 0.999 1.001 1.003 1.000 0.999 0.998
100  1.000 1.000 1.000 0.999 0.999 0.998 0.995 0.998 0.998 0.999 0.999 1.002
105  0.998 0.997 1.002 0.997 0.997 0.996 0.995 0.997 0.996 0.997 0.997 0.998
110  0.996 0.994 0.993 0.993 0.994 0.993 0.991 0.992 0.994 0.995 0.995 0.997
115  0.994 0.990 0.991 0.989 0.989 0.988 0.986 0.990 0.991 0.991 0.991 0.994
120  0.989 0.986 0.987 0.983 0.984 0.984 0.980 0.986 0.987 0.987 0.986 0.989
125  0.986 0.979 0.978 0.977 0.979 0.979 0.978 0.979 0.980 0.982 0.979 0.986
130  0.978 0.969 0.971 0.969 0.970 0.969 0.969 0.973 0.975 0.975 0.975 0.978
135  0.971 0.960 0.961 0.958 0.960 0.960 0.959 0.964 0.968 0.969 0.968 0.973
140  0.962 0.946 0.948 0.944 0.947 0.948 0.948 0.953 0.957 0.960 0.960 0.964
145  0.950 0.930 0.927 0.926 0.931 0.934 0.934 0.942 0.947 0.950 0.949 0.956
150  0.934 0.908 0.903 0.906 0.911 0.914 0.915 0.925 0.930 0.934 0.939 0.944
155  0.912 0.877 0.874 0.878 0.884 0.888 0.893 0.904 0.912 0.920 0.924 0.931
160  0.881 0.841 0.834 0.840 0.849 0.857 0.861 0.879 0.890 0.899 0.904 0.914
165  0.834 0.785 0.779 0.791 0.803 0.815 0.824 0.845 0.862 0.874 0.883 0.896
168  0.797 0.745 0.740 0.754 0.769 0.782 0.791 0.821 0.841 0.857 0.868 0.884
170  -     0.715 0.711 0.725 0.743 0.759 0.772 0.803 0.825 0.842 0.857 0.876
173  -     0.667 0.657 0.678 0.699 0.717 0.734 0.772 0.800 0.821 0.836 0.860
175  -     -     0.621 0.646 0.671 0.692 0.711 0.753 0.782 0.808 0.824 0.850
177  -     -     0.588 0.616 0.642 0.668 0.688 0.733 0.767 0.793 0.813 0.842
178  -     -     0.579 0.603 0.630 0.654 0.676 0.725 0.761 0.788 0.808 0.838
179  -     -     0.579 0.595 0.621 0.645 0.669 0.718 0.755 0.785 0.804 0.836
180  -     -     0.579 0.592 0.618 0.640 0.661 0.711 0.746 0.779 0.797 0.830
"""


def parse_f_table() -> tuple[list[float], list[list[float | None]]]:
    thetas: list[float] = []
    rows: list[list[float | None]] = []
    for line in F_TABLE_TEXT.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        thetas.append(float(parts[0]))
        vals = parts[1:]
        if len(vals) != len(F_RADII):
            raise ValueError(f"열 개수 불일치 (theta={parts[0]}): {len(vals)} != {len(F_RADII)}")
        rows.append([None if v == "-" else float(v) for v in vals])
    return thetas, rows


def fill_missing(thetas: list[float], rows: list[list[float | None]]) -> list[list[float]]:
    """미제공(None) 항목을 같은 r 열에서 가장 가까운 유효 theta 값으로 채운다.

    TG-43U1은 표 범위 밖 외삽을 허용하며, 선원 캡슐에 가려 MC 값이 없는
    극각 영역은 인접 유효값으로 상수 확장하는 것이 관례이다.
    (이 근사는 r <= 0.5 cm, theta < 12deg / theta > 168deg 영역에만 적용됨)
    """
    n_theta, n_r = len(thetas), len(rows[0])
    filled = [[0.0] * n_r for _ in range(n_theta)]
    for j in range(n_r):
        col = [rows[i][j] for i in range(n_theta)]
        valid = [i for i, v in enumerate(col) if v is not None]
        if not valid:
            raise ValueError(f"열 {j} 전체가 비어 있음")
        for i in range(n_theta):
            if col[i] is not None:
                filled[i][j] = col[i]
            else:
                nearest = min(valid, key=lambda k: abs(k - i))
                filled[i][j] = col[nearest]
    return filled


def main() -> int:
    thetas, raw = parse_f_table()
    if len(thetas) != 47:
        raise ValueError(f"theta 개수가 47이 아님: {len(thetas)}")

    # 검증 1: F(r, 90deg) == 1.0
    i90 = thetas.index(90.0)
    for j, v in enumerate(raw[i90]):
        assert v == 1.0, f"F({F_RADII[j]}, 90deg) = {v} != 1.0"

    f_filled = fill_missing(thetas, raw)

    data = {
        "name": "Ir-192 microSelectron-v2 (mHDR-v2)",
        "isotope": "Ir-192",
        "half_life_days": 73.81,
        "active_length_cm": 0.36,
        "dose_rate_constant_cGy_h_U": 1.109,
        "dose_rate_constant_uncertainty": 0.012,
        "provenance": {
            "source": "CLRP TG-43 Parameter Database (Taylor & Rogers, EGSnrc)",
            "url": "https://physics.carleton.ca/clrp/seed_database/Ir192_HDR/microSelectron_v2",
            "f_table_url": "https://physics.carleton.ca/sites/default/files/clrp/databases/seed/Ir192_HDR/data_files/microSelectron_v2_fofrt.htm",
            "lambda_note": "1.109 cGy/h/U — Taylor&Rogers 2008 및 Perez-Calatayud 2012 (HEBD) consensus 값과 동일",
            "warning": "임상 사용 전 반드시 자원(自院) TPS의 선원 데이터와 대조 검증할 것. "
                       "본 케이스의 RTPLAN에는 선원 모델이 명시돼 있지 않으므로 mHDR-v2 가정임.",
        },
        # g_L(r) = (a0 r^-2 + a1 r^-1 + a2 + a3 r + a4 r^2 + a5 r^3) * exp(-a6 r)
        "radial_dose_function": {
            "form": "taylor_rogers_modified_polynomial",
            "r_min_cm": 0.20,
            "r_max_cm": 20.00,
            "coefficients": {
                "a0": -8.2166e-04,
                "a1": 6.3598e-03,
                "a2": 9.8048e-01,
                "a3": 1.3689e-01,
                "a4": 6.4290e-03,
                "a5": 1.9392e-04,
                "a6": 1.2180e-01,
            },
        },
        "anisotropy_function": {
            "geometry": "line_source",
            "L_cm": 0.36,
            "radii_cm": F_RADII,
            "theta_deg": thetas,
            "F": f_filled,  # shape (n_theta, n_r)
            "missing_fill": "nearest valid theta at same r (constant extension)",
        },
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "brachyplan",
        "data",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ir192_mHDR_v2_clrp.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)

    # 검증 2: g_L(1.0) == 1.0
    c = data["radial_dose_function"]["coefficients"]
    r = 1.0
    g1 = (
        c["a0"] * r**-2 + c["a1"] * r**-1 + c["a2"] + c["a3"] * r + c["a4"] * r**2 + c["a5"] * r**3
    ) * pow(2.718281828459045, -c["a6"] * r)
    assert abs(g1 - 1.0) < 2e-4, f"g_L(1cm) = {g1}, 1.0에서 벗어남"

    print(f"생성 완료: {out_path}")
    print(f"  theta {len(thetas)}개 x radius {len(F_RADII)}개")
    print(f"  검증 F(r,90) = 1.0  OK")
    print(f"  검증 g_L(1cm) = {g1:.6f}  OK")
    n_missing = sum(1 for row in raw for v in row if v is None)
    print(f"  미제공 항목 {n_missing}개를 인접 theta 값으로 채움")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
