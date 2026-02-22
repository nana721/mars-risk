# 🚀 MARS: High-Performance Risk Modeling Framework

[![PyPI version](https://img.shields.io/pypi/v/mars-risk?color=blue&style=flat-square)](https://pypi.org/project/mars-risk/) [![Python Versions](https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square)](https://pypi.org/project/mars-risk/) [![License](https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square)](LICENSE) [![Tests](https://github.com/leeesq/mars-risk/actions/workflows/test.yml/badge.svg)](https://github.com/leeesq/mars-risk/actions) [![Downloads Month](https://img.shields.io/pypi/dm/mars-risk?style=flat-square&color=orange&label=downloads/mo)](https://pypi.org/project/mars-risk/) [![Downloads Total](https://static.pepy.tech/badge/mars-risk?style=flat-square)](https://pepy.tech/project/mars-risk)[![Author](https://img.shields.io/badge/Author-Christian-blue.svg)](https://github.com/leeesq)

**MARS** (Modeling Analysis Risk Score) 是一个面向信贷风控建模场景的 Python 工具库。它基于 **Polars** 构建数据处理逻辑以提升数据处理效率，并遵循 **Scikit-learn** 的 API 设计规范，旨在为信贷风控大规模宽表场景下的**数据画像、特征分析、自动建模、模型评估**和**模型监控**提供更高效的解决方案。 

## ✨ 已实现特性

### 1. 📊 高性能数据画像
提供数据质量诊断与可视化报告，性能比传统 Pandas 方案快数倍。
* **全量指标概览**: 一次性计算 `Missing, Zero, Unique, Top1` 等基础 DQ 指标。
* **Unicode Sparklines**: 在终端或 Notebook 中直接生成迷你分布图 (如 ` ▂▅▇█`)，快速洞察数据分布。
* **多维趋势分析**: 支持按时间 (Month/Vintage) 或客群进行分组分析，自动计算各种统计指标（`mean、psi`等）。
* **Excel 自动化报告**: 导出带有热力图、数据条和条件格式的精美 Excel 报表（**待优化功能**）。

### 2.  🚀 快速分箱引擎
针对风控场景深度优化的分箱器。
* **MarsNativeBinner**: 基于 Polars 表达式实现的快速分箱。
    * 支持 **Quantile** (等频), **Uniform** (等宽), **CART** (决策树) 三种模式；
    *  **Quantile** 和 **Uniform **基于 Polars 原生表达式实现，处理了各种极端情况，**速度极快**；
    * **并行加速**: 决策树分箱利用 `joblib` 实现多核并行，内存占用低。
* **MarsOptimalBinner**: 混合动力最优分箱。
    * **Hybrid Engine**: 结合 Polars 的快速预分箱 与 `optbinning` 的数学规划 (MIP/CP) 求解 。
    * 支持**单调性约束** (Monotonic Trend) 和**特殊值/缺失值**的独立分层处理。

### 3. 🧮 多功能特征评估

打破传统 Pandas 循环计算的性能瓶颈，专为海量宽表设计的 Map-Reduce 评估引擎。

- **Map-Reduce 流式架构**: 彻底告别 OOM 焦虑。采用单次扫描 (Single-Pass I/O) 将宽表逆透视为长表，底层全部采用 Polars 窗口函数与 SIMD 向量化计算，实现 `AUC` (梯形法则)、`KS`、`IV`、`PSI` 等指标的 $O(N)$ 极速运算。
- **多目标对齐评估 (Multi-Target)**: 独创的主副目标协同机制。输入多个 Target（如 `FPD30`, `FPD60`），系统会自动使用主目标训练分箱规则，副目标完全复用该边界进行数据落盘与指标计算，确保跨标签效能对比的绝对数学对齐。

### 4. 🎛️ 漏斗式特征筛选矩阵 (迭代ing)

在大规模风控建模中寻找“计算效率”与“召回精度”的最佳平衡点。设计为三大核心选择器，可像 Pipeline 一样自由组装：

- **MarsStatsSelector (统计前置滤网 - ✅ 已实现，待测试)**:
  - **“快慢指针”级联策略**: 面对 5 万+ 超大宽表，先使用低算力的 `NativeBinner` 进行粗筛（融合 Global IV 与 局部高 Lift 的联合召回），降维后再使用高算力的 `OptimalBinner` 追求严苛的单调性与 IV 截断，兼顾极速与不漏杀。
  - **时序动态侦测**: 内置高级时序策略，自动召回“全局 IV 低但近期爆发”的潜力股，并剔除表现波动的“神经刀”特征。
  - **业务规则干预**: 支持强制黑白名单 (Whitelist/Blacklist) 注入。
- **MarsLinearSelector (共线性滤网 - 🚧 规划中)**:
  - 专为逻辑回归 (LR) 场景设计。包含基于 Polars 的极速 Spearman 相关性聚类去重、VIF 共线性检验，以及包裹式的 Stepwise 逐步回归。
- **MarsImportanceSelector (模型后置精选 - 🚧 规划中)**:
  - 对接强基模型（LightGBM / XGBoost）。利用特征分裂增益 (Gain)、SHAP 或 RFE (递归特征消除)，输出精简、强悍的最终入模名单。

### 5. 🛠️ 工程化设计
* **Auto Polars**: 智能装饰器**支持 Pandas DataFrame** 无缝输入，内部自动转换为 Polars 计算，结果按需回退。
* **Pipeline Ready**: 所有组件均继承自 `MarsBaseEstimator` 和 `MarsTransformer`，兼容 Sklearn Pipeline。

---

## 📦 安装 (Installation)

```python
# 推荐使用 pip 安装
pip install mars-risk==0.0.10

# 或者从源码安装
git clone [https://github.com/leeesq/mars-risk.git](https://github.com/leeesq/mars-risk.git)
cd mars-risk
pip install -e .
```

依赖项: `polars`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `xlsxwriter`, `colorlog`, `optbinning`

## ⚡️ 快速上手 (Quick Start)

MARS 提供了从极简的高层 API 到硬核的底层组件的全面支持。以下是四个最典型的风控建模场景：

### 场景 1：一键风险全景画像 (End-to-End Risk Profiling)

这是 MARS 最强大的高层 API。只需一行核心代码，即可自动完成 **“最优分箱 -> 多目标评估 -> 跨期稳定性(PSI)计算 -> Excel 报表导出”** 全流程。

```python
import polars as pl
from mars.analysis import profile_risk

# 1. 加载数据 (无缝兼容 Pandas / Polars)
df = pl.read_parquet("credit_risk_data.parquet")

# 2. 执行全景评估 (支持多目标协同评估)
report = profile_risk(
    df=df,
    target=["bad_30d", "bad_90d"], # 首个为主目标(用于分箱)，后续为副目标
    features=["age", "income", "td_score"], 
    profile_by="month",            # 按月进行 OOT 趋势演变分析
    binning_type="opt",            # 启用最优分箱引擎
    monotonic_trend="auto",        # 自动探索单调性
    special_values=[-999],         # 特殊值独立成箱
    plot=True                      # 自动绘制特征效能趋势图
)

# 3. 结果展示与高保真导出
report.show_summary()              # Jupyter 内查看四维审计结果 (包含自动决策建议)
report.write_excel("risk_audit_report.xlsx") # 导出带红绿灯与数据条的精美 Excel
```

### 场景 2：高性能数据质量画像 (High-Performance DQ Profiling)

在建模初期快速摸底数据，计算速度比 Pandas Profiling 类工具快数倍。

```python
from mars.analysis.profiler import MarsDataProfiler

# 1. 初始化引擎 (安全隔离自定义缺失值，不污染正常均值/方差计算)
profiler = MarsDataProfiler(
    df, 
    custom_missing_values=[-999, "unknown", "\\N"]
)

# 2. 极速生成画像 (支持自动按时间粒度截断聚合)
report = profiler.generate_profile(
    profile_by="month", 
    dt_col="apply_date" # 若 profile_by 为 month/week, 自动转换该列
)

# 3. 查看交互式报告
report.show_overview()  # 查看包含 Unicode Sparklines (▂▅▇█) 的分布图
report.show_trend("missing") # 追踪缺失率随时间的衰变趋势
```

### 场景 3：精细化底层分箱控制 (Fine-grained Binning)

为模型用户提供可独立调用的分箱组件，无缝接入 Scikit-learn Pipeline。

```python
from mars.feature.binner import MarsNativeBinner, MarsOptimalBinner

# --- 方案 A: 极速衍生预分箱 ---
native_binner = MarsNativeBinner(
    method="quantile", n_bins=10, special_values=[-1]
)
X_binned_fast = native_binner.fit_transform(X_train, y_train)

# --- 方案 B: 评分卡单调性约束分箱 ---
opt_binner = MarsOptimalBinner(
    n_bins=5, 
    monotonic_trend="ascending", # 强制坏率严格递增
    min_bin_size=0.05
)
opt_binner.fit(X_train, y_train)

# 提取 WOE 映射字典或切点数组，用于部署规则引擎
woe_mapping = opt_binner.bin_woes_
bin_cuts = opt_binner.bin_cuts_
```

## 📂 项目结构 (Project Structure)

```Plaintext
mars/
├── analysis/              # 📊 数据画像与效能评估模块 (Core Analytics)
│   ├── profiler.py        # MarsDataProfiler: 高性能多维数据画像引擎
│   ├── evaluator.py       # MarsBinEvaluator: Map-Reduce 架构的特征评估与 profile_risk 入口
│   ├── report.py          # 报告容器: 封装 Jupyter 交互式渲染与双引擎 Excel 导出
│   └── config.py          # 画像与评估的全局配置对象
│
├── feature/               # 🚀 特征工程模块 (Feature Engineering)
│   ├── binner.py          # 分箱引擎: 包含 NativeBinner (极速) 与 OptimalBinner (单调最优)
│   ├── selector.py        # 特征筛选: MarsStatsSelector (快慢指针级联漏斗筛选)
│   ├── encoding.py        # 特征编码 (🚧 迭代ing)
│   └── imputer.py         # 缺失值填补 (🚧 迭代ing)
│
├── modeling/              # 🤖 自动建模流水线 (Modeling Pipeline)
│   ├── base.py            # 模型架构基类
│   ├── strategies.py      # 算法策略封装 (LGBM/XGB 等风控常用算法)
│   └── tuner.py           # 超参数自动搜索与调优
│
├── core/                  # 🏗️ 核心基类 (Core Foundations)
│   ├── base.py            # MarsBaseEstimator: 确保所有组件 100% 兼容 Scikit-learn API
│   └── exceptions.py      # 风控场景专属的自定义异常处理
│
└── utils/                 # 🛠️ 工程化工具箱 (Utilities)
    ├── logger.py          # 统一的彩色日志管理器
    ├── decorators.py      # 智能装饰器: 如 @time_it, 自动 Polars/Pandas 类型转换
    ├── date.py            # MarsDate: 极速时间序列截断与聚合计算
    └── plotter.py         # 可视化引擎: 负责渲染高质量的特征趋势图与热力图
```

## 📄 许可证 (License)
MIT License