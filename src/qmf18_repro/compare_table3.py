#!/usr/bin/env python3
"""
QMF18 Table 3 逐点 EOS 比对脚本
比较论文 Table 3 (cgs) 与代码输出 eos.csv (MeV/fm³)
"""
import csv
import json
import sys
from pathlib import Path

# ── 单位转换因子 ──
# 1 MeV/fm³ = 1.6022e33 erg/cm³
MEVFM3_TO_ERGCM3 = 1.6022e33
# 1 MeV/fm³ / c² = 1.7827e12 g/cm³
MEVFM3_TO_GCM3 = 1.7827e12

# ── 论文 Table 3 数据 (cgs) ──
# (epsilon [g/cm³], P [erg/cm³], rho_N [fm⁻³])
TABLE3_PAPER = [
    (0.13855e15, 0.79586e33, 0.082),
    (0.14365e15, 0.85234e33, 0.085),
    (0.15216e15, 0.95144e33, 0.090),
    (0.16920e15, 0.11706e34, 0.100),
    (0.18626e15, 0.14226e34, 0.110),
    (0.20336e15, 0.17145e34, 0.120),
    (0.22047e15, 0.20433e34, 0.130),
    (0.27203e15, 0.33950e34, 0.160),
    (0.32393e15, 0.55426e34, 0.190),
    (0.37631e15, 0.87679e34, 0.220),
    (0.42926e15, 0.13315e35, 0.250),
    (0.48293e15, 0.19385e35, 0.280),
    (0.53741e15, 0.27149e35, 0.310),
    (0.59282e15, 0.36752e35, 0.340),
    (0.64927e15, 0.48329e35, 0.370),
    (0.70686e15, 0.62008e35, 0.400),
    (0.76568e15, 0.77912e35, 0.430),
    (0.82583e15, 0.96151e35, 0.460),
    (0.88738e15, 0.11682e36, 0.490),
    (0.95043e15, 0.13999e36, 0.520),
    (0.10150e16, 0.16569e36, 0.550),
    (0.10813e16, 0.19389e36, 0.580),
    (0.11492e16, 0.22449e36, 0.610),
    (0.12189e16, 0.25733e36, 0.640),
    (0.12904e16, 0.29223e36, 0.670),
    (0.13636e16, 0.32903e36, 0.700),
    (0.14896e16, 0.39423e36, 0.750),
    (0.16207e16, 0.46399e36, 0.800),
    (0.17568e16, 0.53809e36, 0.850),
    (0.18978e16, 0.61645e36, 0.900),
    (0.20438e16, 0.69900e36, 0.950),
    (0.21948e16, 0.78573e36, 1.000),
    (0.25116e16, 0.97160e36, 1.100),
    (0.28480e16, 0.11739e37, 1.200),
    (0.32039e16, 0.13926e37, 1.300),
]


def load_code_eos(csv_path: str) -> list[tuple[float, float, float]]:
    """加载代码输出的 eos.csv → [(nB_fm3, eps_MeVfm3, P_MeVfm3)]"""
    data = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nB = float(row["nB_fm3"])
            eps = float(row["eps_MeVfm3"])
            P = float(row["P_MeVfm3"])
            data.append((nB, eps, P))
    return data


def interp_at(x_target: float, xs: list[float], ys: list[float]) -> float:
    """线性插值（xs 必须单调递增）"""
    if x_target <= xs[0]:
        return ys[0]
    if x_target >= xs[-1]:
        return ys[-1]
    # 二分查找
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x_target:
            lo = mid
        else:
            hi = mid
    # 线性插值
    t = (x_target - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


def main():
    # 定位 eos.csv
    script_dir = Path(__file__).parent
    eos_csv = script_dir / "output" / "2026-07-01_4L_scan" / "qmf18_L40" / "eos.csv"
    if not eos_csv.exists():
        print(f"ERROR: {eos_csv} not found", file=sys.stderr)
        sys.exit(1)

    # 加载代码 EOS
    code_data = load_code_eos(str(eos_csv))
    code_nB = [d[0] for d in code_data]
    code_eps_MeV = [d[1] for d in code_data]
    code_P_MeV = [d[2] for d in code_data]

    # 转换代码数据到 cgs（用于比对）
    code_eps_cgs = [e * MEVFM3_TO_ERGCM3 for e in code_eps_MeV]  # 注意：代码 eps 是 MeV/fm³，论文 ε 是 g/cm³
    # 实际上论文的 ε 是质量密度 g/cm³，不是能量密度 erg/cm³
    # 需要区分：论文 Table 3 列名是 ε (g cm⁻³) = 质量密度
    # 代码 eps_MeVfm3 是能量密度 MeV/fm³
    # 转换：质量密度 = 能量密度 / c²
    # 1 MeV/fm³ = 1.7827e12 g/cm³ (质量密度)
    code_eps_gcm3 = [e * MEVFM3_TO_GCM3 for e in code_eps_MeV]
    code_P_ergcm3 = [p * MEVFM3_TO_ERGCM3 for p in code_P_MeV]

    # 逐点比对
    results = []
    for eps_paper, P_paper, rhoN_paper in TABLE3_PAPER:
        # 插值代码 EOS 到论文的 rhoN 点
        eps_code_gcm3 = interp_at(rhoN_paper, code_nB, code_eps_gcm3)
        P_code_ergcm3 = interp_at(rhoN_paper, code_nB, code_P_ergcm3)

        # 偏差
        delta_eps = abs(eps_code_gcm3 - eps_paper) / abs(eps_paper) * 100
        delta_P = abs(P_code_ergcm3 - P_paper) / abs(P_paper) * 100

        results.append({
            "rhoN": rhoN_paper,
            "eps_paper": eps_paper,
            "eps_code": eps_code_gcm3,
            "delta_eps_pct": delta_eps,
            "P_paper": P_paper,
            "P_code": P_code_ergcm3,
            "delta_P_pct": delta_P,
        })

    # 输出表格
    print(f"{'ρ_N':>6} {'ε_paper':>12} {'ε_code':>12} {'Δε%':>7} {'P_paper':>12} {'P_code':>12} {'ΔP%':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['rhoN']:6.3f} {r['eps_paper']:12.4e} {r['eps_code']:12.4e} {r['delta_eps_pct']:7.2f} "
              f"{r['P_paper']:12.4e} {r['P_code']:12.4e} {r['delta_P_pct']:7.2f}")

    # 统计
    eps_deltas = [r["delta_eps_pct"] for r in results]
    P_deltas = [r["delta_P_pct"] for r in results]

    import math
    rms_eps = math.sqrt(sum(d**2 for d in eps_deltas) / len(eps_deltas))
    rms_P = math.sqrt(sum(d**2 for d in P_deltas) / len(P_deltas))
    max_eps = max(eps_deltas)
    max_P = max(P_deltas)
    mean_eps = sum(eps_deltas) / len(eps_deltas)
    mean_P = sum(P_deltas) / len(P_deltas)

    print("\n" + "=" * 80)
    print("统计汇总")
    print("=" * 80)
    print(f"  比对点数:     {len(results)}")
    print(f"  ε (质量密度):")
    print(f"    平均偏差:   {mean_eps:.2f}%")
    print(f"    RMS 偏差:   {rms_eps:.2f}%")
    print(f"    最大偏差:   {max_eps:.2f}% (ρ_N = {results[eps_deltas.index(max_eps)]['rhoN']:.3f})")
    print(f"  P (压强):")
    print(f"    平均偏差:   {mean_P:.2f}%")
    print(f"    RMS 偏差:   {rms_P:.2f}%")
    print(f"    最大偏差:   {max_P:.2f}% (ρ_N = {results[P_deltas.index(max_P)]['rhoN']:.3f})")

    # Tier 1 判定
    tier1_eps = 1.0  # ±1%
    tier1_P = 1.0
    n_pass_eps = sum(1 for d in eps_deltas if d <= tier1_eps)
    n_pass_P = sum(1 for d in P_deltas if d <= tier1_P)
    print(f"\n  Tier 1 判定 (±{tier1_eps}%):")
    print(f"    ε: {n_pass_eps}/{len(results)} PASS")
    print(f"    P: {n_pass_P}/{len(results)} PASS")

    if rms_eps <= tier1_eps and rms_P <= tier1_P:
        verdict = "GO"
    elif rms_eps <= 3.0 and rms_P <= 3.0:
        verdict = "CONDITIONAL GO"
    else:
        verdict = "FAIL"
    print(f"  Verdict: {verdict}")

    # 保存 JSON
    output = {
        "paper": "Zhu, Zhou & Li (2018), ApJ 862, 98",
        "model": "QMF18 (L=40)",
        "comparison": "Table 3 EOS point-by-point",
        "n_points": len(results),
        "statistics": {
            "eps_mean_pct": round(mean_eps, 3),
            "eps_rms_pct": round(rms_eps, 3),
            "eps_max_pct": round(max_eps, 3),
            "P_mean_pct": round(mean_P, 3),
            "P_rms_pct": round(rms_P, 3),
            "P_max_pct": round(max_P, 3),
        },
        "points": results,
    }
    json_path = script_dir / "output" / "2026-07-01_4L_scan" / "qmf18_L40" / "table3_comparison.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  详细数据已保存: {json_path}")


if __name__ == "__main__":
    main()
