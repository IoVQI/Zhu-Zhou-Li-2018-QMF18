# QMF18 EOS 复现

> **论文**: Zhu, Zhou & Li (2018), ApJ 862, 98 — "Neutron Star Equation of State from the Quark Level in Light of GW170817"
> **arXiv**: [1802.05510](https://arxiv.org/abs/1802.05510)
> **状态**: SNM 饱和 ✅ | TOV + 潮汐 ✅ | 4L 扫描 + 绘图 ✅ | Table 3 逐点比对 ✅ | Λ̃ 双星潮汐 ✅ | Λ(1.4) 偏差 ~4% (根因明确) | ⚠️ CONDITIONAL GO

## 目录结构

```
001_Zhu_Zhou_Li_2018_QMF18/
├── README.md              # 本文件
├── pyproject.toml         # 项目配置
├── REFERENCE.md           # 论文元信息
├── src/
│   └── qmf18_repro/       # 源代码包
│       ├── constants.py   # 物理常数、Table 1-2 参数、单位转换
│       ├── quark.py       # M_N*(σ) — Δ₁=0.509 线性近似 (快速路径)
│       ├── quark_barik_full.py  # ★ M_N*(σ) — 完整三项修正 (权威实现)
│       ├── fermi.py       # Fermi 积分 (解析闭式)
│       ├── fields.py      # σ/ω/ρ 介子场方程
│       ├── eos.py         # β 平衡迭代 + EOS 生成
│       ├── crust.py       # 外壳 EOS 加载 (BPS + N&V)
│       ├── main.py        # 旧主入口 (保留兼容)
│       ├── generate_eos.py    # ★ Step 1: EOS 生成
│       ├── compute_tov.py     # ★ Step 2: TOV + 潮汐
│       ├── plot_eos.py        # ★ 绘图
│       ├── compare_table3.py  # ★ Table 3 逐点比对
│       └── compute_tilde_lambda.py  # ★ 双星潮汐 Λ̃
├── data/                  # 输入数据
│   ├── bps_1971_outer_crust.dat
│   └── Negele_Vautherin_1973_inner_crust.dat
├── tests/                 # 基准对比测试
└── output/                # 生成产物
```

## 快速开始

```bash
cd 001_Zhu_Zhou_Li_2018_QMF18
pip install -e .

# Step 1: 生成 EOS
python src/qmf18_repro/generate_eos.py -L 20,40,60,80

# Step 2: TOV + 潮汐
python src/qmf18_repro/compute_tov.py output/<run_dir>/qmf18_L*

# Step 3: 绘图
python src/qmf18_repro/plot_eos.py output/<run_dir>

# Step 4: Table 3 逐点比对
python src/qmf18_repro/compare_table3.py
```

## 物理模块

### quark_barik_full.py — M*(σ) 关系 (★ 权威实现)

基于 Zhu & Li (2018) PRC 97, 035805 Eqs.(5)-(9) 的完整三项修正：

| 修正项 | 公式 | 特点 |
|------|------|------|
| 质心修正 ε_cm | Eq.(5) | 直接代数 |
| π 云修正 δM_N^π | Eq.(6) | 含高斯积分 |
| 胶子修正 (ΔE_N)_g | Eq.(7) | 直接代数 |

### fields.py — 介子场方程

自洽求解 σ, ω, ρ 三个介子场，使用 `scipy.optimize.least_squares` (TRF) 有界求解。

### eos.py — β 平衡 + EOS

对每个重子密度 ρ_B 迭代求解 β 平衡条件。

## 已验证的物理量

### SNM 饱和性质 (L=40)

| 物理量 | QMF18 Table 1 | 本代码 | 状态 |
|------|:---:|:---:|:---:|
| ρ₀ [fm⁻³] | 0.160 | 0.160 | ✅ |
| E/A(ρ₀) [MeV] | −16.0 | −16.57 | ⚠️ (−3.6%) |
| K(ρ₀) [MeV] | 240 | 238.6 | ✅ |
| M*/M(ρ₀) | 0.77 | 0.770 | ✅ |

### TOV + 潮汐 (L=40)

| 物理量 | QMF18 Table 4 | 本代码 | 偏差 |
|------|:---:|:---:|:---:|
| M_TOV [M⊙] | 2.0805 | 2.0992 | +0.9% |
| R(1.4) [km] | 11.77 | 11.89 | +1.0% |
| Λ(1.4) | 331 | 344 | +3.9% |

## 残余偏差分析

Λ(1.4) 偏差 ~4% 的主要来源：

1. **耦合参数拟合** (5-10%): Table 2 参数是论文通过未公开的 χ² 拟合得到
2. **N&V 内壳数据** (3-8%): 从原始 E/A 表二次推导的稀疏数据
3. **芯-壳拼接** (1-2%): ρ=0.08 处未做平滑匹配

详见源代码 `reproduction_code/README.md` 中的完整偏差分析。

## 外部依赖

| 依赖 | 用途 | 说明 |
|------|------|------|
| numpy, scipy | 数值计算 | pip install |
| matplotlib | 绘图 | pip install |
| eos_lab/tov_core | TOV C++ solver | 需单独编译 |

## 许可证

MIT License
