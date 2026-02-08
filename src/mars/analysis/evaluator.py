from typing import List, Optional, Dict, Union, Any, Tuple, Literal
from collections import defaultdict
import inspect

import polars as pl
import numpy as np
import pandas as pd

from mars.core.base import MarsBaseEstimator
from mars.feature.binner import MarsBinnerBase, MarsNativeBinner, MarsOptimalBinner
from mars.analysis.report import MarsEvaluationReport 
from mars.utils.logger import logger
from mars.utils.decorators import time_it
from mars.utils.date import MarsDate
from mars.utils.plotter import MarsPlotter

class MarsBinEvaluator(MarsBaseEstimator):
    """
    [MarsBinEvaluator] 特征效能与稳定性评估引擎.

    该类实现了基于 **Map-Reduce 架构** 的大规模特征评估。
    它解决了传统 Python 风控库在处理宽表（Wide Table, 5000+ Cols）时的内存溢出和 I/O 瓶颈问题。

    Attributes
    ----------
    target_col : str
        目标变量列名 (0/1)。
    binner : MarsBinnerBase
        分箱器实例。如果未提供，evaluate 时会自动拟合。
    binner_kwargs : dict
        传递给自动分箱器的额外参数。
        
    核心架构 (Architecture)
    -----------------------
    1. **单次扫描 (Single-Pass I/O)**:
        全流程仅对原始大数据执行一次 `unpivot` 和 `agg` 操作（Map 阶段）。
        后续所有计算（WOE补全、基准对比、Total聚合）均在聚合后的“中间统计表”上完成（Reduce 阶段）。
    
    2. **向量化指标计算 (Vectorized Metrics)**:
        PSI, IV, KS, AUC, Lift 等指标均通过 Polars 表达式引擎在列式内存中计算，
        完全避免了 Python 循环，利用 SIMD 指令集加速。

    3. **内存/计算均衡**:
        利用 `streaming=True` 处理 Map 阶段的重型计算，利用内存计算处理 Reduce 阶段的逻辑，
        在速度和内存消耗之间取得最佳平衡。
        
    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import MarsBinEvaluator

    >>> # 1. 准备数据 (支持 Polars/Pandas)
    >>> df = pl.read_parquet("credit_risk_data.parquet")
    >>> target_col = "is_default"

    >>> # 2. 初始化评估器
    >>> evaluator = MarsBinEvaluator(target_col=target_col)

    >>> # 3. [最简模式] 一键评估 + 绘图
    >>> # 自动拟合分箱 -> 计算 IV/PSI -> 绘制 Top 10 特征趋势图
    >>> report = evaluator.evaluate_and_plot(df)

    >>> # 4. 查看结果
    >>> print(report.summary_table.head())  # 查看汇总表
    >>> report.write_excel("risk_report.xlsx") # 导出 Excel
    """
    
    # [新增] 定义统一的分组列名常量
    GROUP_COL_NAME = "mars_group"

    def __init__(
        self, 
        target_col: str = "target",
        binner: Optional[MarsBinnerBase] = None,
        bining_type: Literal["native", "opt"] = "native",
        **binner_kwargs
    ) -> None:
        """
        初始化评估引擎。

        Parameters
        ----------
        target_col : str, default "target"
            Label 列名。
        binner : MarsBinnerBase, optional
            预训练好的分箱器。若为 None，将在 evaluate 内部自动训练 MarsNativeBinner。
        bining_type : Literal["native", "opt"], default "native"
            分箱器类型选择。当 binner 为 None 时生效。
        **binner_kwargs : dict
            透传给自动分箱器的参数 (仅在 binner 为 None 时生效)，如 `n_bins`, `strategy` 等。
        """
        super().__init__()
        self.target_col = target_col
        self.binner = binner
        self.binner_kwargs = binner_kwargs
        self.bining_type = bining_type
        
    @time_it
    def evaluate(
        self,
        df: Union[pl.DataFrame, pd.DataFrame],  
        features: Optional[List[str]] = None,
        *,
        profile_by: Optional[str] = None,
        dt_col: Optional[str] = None,
        benchmark_df: Union[pl.DataFrame, pd.DataFrame, None] = None,
        # [新增参数] PSI 计算控制
        psi_include_missing: bool = False,
        psi_include_special: bool = False, 
        weights_col: Optional[str] = None,
        batch_size: int = 500
    ) -> "MarsEvaluationReport":
        """
        [Core] 执行特征评估的主入口 (Main Evaluation Engine).

        该方法编排了从原始数据到最终风控评估报告的全生命周期管理。它集成了
        自动分箱 (Auto-Binning)、宽表转长表 (Wide-to-Long Unpivoting)、
        流式聚合 (Streaming Aggregation) 以及向量化指标计算。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            待评估的数据集 (Evaluation Set)，支持 Pandas DataFrame（内部自动转换为 Polars）。
        features : Optional[List[str]]
            指定需要评估的特征列名列表。
            若为 None，系统将自动扫描除 `target`, `weights`, `group` 之外的所有列。
        profile_by : Optional[str]
            趋势分析的分组维度 (Group Key)。
            - 可以是具体的分类列 (e.g., 'city', 'vintage').
            - 也可以是时间粒度指令 ('day', 'week', 'month')，需配合 `dt_col` 使用。
        dt_col : Optional[str]
            日期列名 (Date Column)。
            - 用于辅助 `profile_by` 生成时间切片。
            - **智能默认**: 若提供了 `dt_col` 但未指定 `profile_by`，默认按 **'month'** 聚合。
        benchmark_df : Union[pl.DataFrame, pd.DataFrame, None]
            PSI 计算的基准数据集 (Expected Distribution Reference)。
            - **None (默认)**: 使用 `df` 中时间/分组顺序最早的一组作为基准 (Internal Baseline)。
            - **DataFrame**: 使用外部传入的数据集（如训练集）作为基准 (External Baseline)。
        weights_col : Optional[str]
            样本权重列名。
            若指定，所有的统计指标 (IV, AUC, KS, PSI, Lift) 均基于加权频数计算。
        batch_size : int, default=500
            **[性能调优]** Map 阶段的特征切片大小。
            - 较小的值 (e.g., 100) 降低内存峰值，适合超宽表 (5000+ features)。
            - 较大的值 (e.g., 1000) 提升 I/O 吞吐量，适合内存充足的场景。

        Returns
        -------
        MarsEvaluationReport
            评估报告容器对象。包含以下核心属性：
            - `summary_table`: 特征维度的审计汇总表 (IV, AUC, PSI_max, Efficiency_Score)。
            - `trend_tables`: 指标维度的趋势宽表字典 (Key: 'auc', 'psi'...)。
            - `detail_table`: 最细粒度的分箱明细表。

        Notes
        -----
        **执行流程 (Execution Pipeline):**
        1. **Context Prep**: 标准化输入格式 (Pandas -> Polars) 并解析分组逻辑。
        2. **Auto-Fit (Optional)**: 若未预置分箱器，基于当前数据自动训练 `MarsNativeBinner`。
        3. **Map Phase**: 将宽表切分为多个 Batch，流式执行 `unpivot` + `agg`，生成中间统计表。
        4. **Reduce Phase**: 在内存中汇总 Total 统计量，并计算 WOE/Expected 分布。
        5. **Vectorized Calc**: 并行计算所有特征的 PSI, IV, KS, AUC, Lift。
        6. **Audit**: 执行单调性检查 (Spearman) 与逻辑一致性检查 (RiskCorr)。
        """
        
        # 1. 上下文准备
        working_df = self._ensure_polars_dataframe(df)
        if benchmark_df is not None:
            benchmark_df = self._ensure_polars_dataframe(benchmark_df)

        # 检查 Target 有效性，避免后续 AUC/KS 计算崩溃
        n_unique = working_df.select(pl.col(self.target_col).n_unique()).item()
        if n_unique < 2:
            logger.warning(f"⚠️ Target '{self.target_col}' has < 2 unique values. Metrics (AUC/KS) may be invalid.")

        # [修改] 统一分组列名逻辑
        # _prepare_context 现在返回处理后的 df，它一定包含一列叫 "mars_group"
        working_df = self._prepare_context(working_df, profile_by, dt_col)
        
        # 锁定分组列名为常量 "mars_group"
        group_col = self.GROUP_COL_NAME

        # [修改] 自动识别特征列
        # 排除 target, weights, 和刚刚生成的统一 mars_group 列
        exclude_cols = {self.target_col, group_col}
        if weights_col: 
            exclude_cols.add(weights_col)
        
        target_features = features if features else [
            c for c in working_df.columns if c not in exclude_cols
        ]

        if self.binner is None:
            fit_kwargs = self.binner_kwargs if self.binner_kwargs is not None else {}
            
            binner_factory = {
                "native": MarsNativeBinner,
                "opt": MarsOptimalBinner
            }
            
            # 确定分箱器类型
            binner_cls = binner_factory.get(self.bining_type)
            if binner_cls is None:
                logger.warning(f"⚠️ Unknown bining_type '{self.bining_type}'. Fallback to 'native'.")
                binner_cls = MarsNativeBinner
            
            # 获取目标类的构造函数签名
            # inspect.signature 会分析 __init__(self, n_bins, min_bin_size, ...) 到底有哪些参数
            sig = inspect.signature(binner_cls.__init__)
            valid_keys = set(sig.parameters.keys())
            
            # 过滤参数：只保留目标类支持的参数
            # 排除 'self' 和 'features' (因为 features 我们是显式传递的)
            valid_keys.discard("self") 
            valid_keys.discard("features")
            
            clean_kwargs = {k: v for k, v in fit_kwargs.items() if k in valid_keys}
            
            # 记录被丢弃的参数，方便调试
            ignored_keys = set(fit_kwargs.keys()) - set(clean_kwargs.keys())
            if ignored_keys:
                logger.debug(f"🧹 Auto-cleaned kwargs for {binner_cls.__name__}. Ignored: {ignored_keys}")
            
            logger.info(f"⚙️ No binner provided. Auto-fitting {binner_cls.__name__} internally with params: {clean_kwargs}...")
            
            # 实例化并拟合分箱器
            self.binner = binner_cls(features=target_features, **clean_kwargs)
            y_series = working_df.get_column(self.target_col)
            self.binner.fit(working_df, y_series)
        
        # [Transform] 数据转换：将原始连续值/离散值映射为分箱索引 (Int16)
        # 映射后的列名为 {feat}_bin
        logger.debug("🔄 Transforming features to bin indices...")
        df_binned = self.binner.transform(working_df, return_type="index")
        
        # 2. [Map Phase] 执行全量数据的流式扫描
        # 将宽表 unpivot 后聚合，得到最小粒度统计表 (Group, Feature, Bin, Count, Bad)
        logger.debug("📊 Step 1: Scanning raw data for stats (Single Pass Map)...")
        group_stats_raw = self._agg_basic_stats(
            df_binned, group_col, target_features, self.target_col, weights_col,
            batch_size=batch_size 
        )
        
        # 3. [Reduce Phase A] 补全 WOE 信息
        # 计算 KS/AUC 依赖 WOE 排序。若分箱器无 WOE，利用 group_stats_raw 内存计算，无需扫原表。
        self._ensure_woe_info(group_stats_raw)

        # 4. [Reduce Phase B] 获取 PSI 基准分布
        # 获取 Expected Distribution。若无外部基准，取 group_stats_raw 中最早的一组。
        expected_dist = self._get_benchmark_dist(
            group_stats_raw, benchmark_df, group_col, target_features, weights_col
        )

        # 5. [Reduce Phase C] 汇总 Total 统计量
        # 在内存中将不同 Group 的统计量累加，得到全量数据的分布情况。
        logger.debug("∑  Step 2: Rolling up stats for Total (Reduce)...")
        total_stats_raw = (
            group_stats_raw
            .group_by(["feature", "bin_index"])
            .agg([
                pl.col("count").sum(),
                pl.col("bad").sum()
            ])
            .with_columns(pl.lit("Total").alias(group_col)) # 显式标记为全量
        )

        # 6. 指标计算 
        logger.debug("🧮 Step 3: Calculating metrics (PSI/AUC/KS/IV)...")
        
        # 6.1 计算 Trend 数据
        metrics_groups = (
            self._calc_metrics_from_stats(
                group_stats_raw, expected_dist, group_col,
                # 传参
                include_missing=psi_include_missing,
                include_special=psi_include_special
            )
            .with_columns(pl.col(group_col).cast(pl.String))
        )
        
        # 6.2 计算 Total 数据
        metrics_total = self._calc_metrics_from_stats(
            total_stats_raw, expected_dist, group_col,
            # 传参
            include_missing=psi_include_missing,
            include_special=psi_include_special
        )

        # 6.3 合并分组与总体结果
        metrics_total = metrics_total.select(metrics_groups.columns)

        # [BugFix] 防止单点评估模式下出现双重 "Total"
        # 判断逻辑：如果分组结果中只有 1 个组，且名字就叫 "Total"，说明这是单点评估模式
        # 此时 metrics_groups 和 metrics_total 是完全一样的数学意义
        is_single_snapshot = (
            metrics_groups.select(pl.col(group_col).n_unique()).item() == 1 and 
            metrics_groups.select(pl.col(group_col).first()).item() == "Total"
        )

        if is_single_snapshot:
            logger.debug("ℹ️ Single snapshot detected. Skipping Total concatenation.")
            stats_long = metrics_groups
        else:
            # 正常模式（有多组，或者单组但不叫Total）：将 Total 作为汇总行追加到头部
            stats_long = pl.concat([metrics_total, metrics_groups])

        # 7. 单调性检查 (Monotonicity Check)
        # 使用斯皮尔曼相关系数判断分箱索引与坏率的关系是否单调
        logger.debug("📉 Step 4: Checking monotonicity...")
        monotonicity_df = (
            stats_long
            .filter((pl.col("bin_index") >= 0) & (pl.col(group_col) == "Total"))
            .group_by("feature")
            .agg(
                pl.corr("bin_index", "bad_rate", method="spearman").alias("Monotonicity")
            )
        )

        # 8. 格式化输出：重塑数据为最终 Report 容器
        report = self._format_report(
            stats_long, metrics_groups, metrics_total, group_col, monotonicity_df
        )

        logger.info(f"✅ Evaluation complete. [Features: {len(target_features)} | Groups: {stats_long[group_col].n_unique() - 1}]")
        return report

    def _agg_basic_stats(
        self,
        df_binned: pl.DataFrame,
        group_col: str,
        features: List[str],
        y_col: str,
        weights_col: Optional[str],
        batch_size: int = 500 
    ) -> pl.DataFrame:
        """
        [Map Phase] 全量数据扫描，计算最重要的统计量：样本数和坏样本数。

        采用 "分批-流式-聚合" (Batch-Stream-Agg) 策略：
        1. 将数千个特征切分为多个批次 (Chunk)。
        2. 对每个批次构建独立的 Lazy Query Plan。
        3. 利用 Streaming 引擎执行聚合，并立即释放中间结果。
        4. 最后纵向合并 (Vertical Concat) 所有批次的统计结果。

        Parameters
        ----------
        df_binned : pl.DataFrame
            已经过分箱索引转换的数据集。
        group_col : str
            分组列。
        features : List[str]
            特征名列表。
        y_col : str
            目标变量列。
        weights_col : Optional[str]
            权重列。
        batch_size : int
            每次聚合处理的特征数量。

        Returns
        -------
        pl.DataFrame
            长表格式的统计汇总表，包含 [group_col, feature, bin_index, count, bad]。
        """
        # 构造 bin 列名
        theoretical_bin_cols = [f"{f}_bin" for f in features]
        
        # 获取实际存在的列名
        # 使用 set 运算过滤，防止传入了未被分箱的特征导致报错
        existing_cols = set(df_binned.columns)
        actual_bin_cols = [c for c in theoretical_bin_cols if c in existing_cols]
        
        # 记录丢失的列
        missing_cols = set(theoretical_bin_cols) - set(actual_bin_cols)
        if missing_cols:
            logger.warning(f"⚠️ {len(missing_cols)} features were not binned and will be skipped in evaluation. All missing: {list(missing_cols)}")
            
        if not actual_bin_cols:
            raise ValueError("❌ No valid binned columns found in dataframe. Check your binner fit results.")

        # 使用实际存在的列进行后续操作
        bin_cols = actual_bin_cols
        
        # 确定必须要保留的索引列 (Group, Target, Weight)
        index_cols = [group_col, y_col] # 注意这里 y_col 被放到 index 是为了 unpivot 后不丢失信息
        if weights_col:
            index_cols.append(weights_col)

        # 预定义聚合表达式 (Lazy Expr)，避免在循环中重复构建
        # 统计样本数 (Count)
        expr_count = pl.col(weights_col).sum() if weights_col else pl.len()
        # 统计坏样本数 (Bad)
        expr_bad = (pl.col(y_col) * pl.col(weights_col)).sum() if weights_col else pl.col(y_col).sum()
        
        agg_exprs = [
            expr_count.alias("count"),
            expr_bad.alias("bad")
        ]

        result_frames: List[pl.DataFrame] = []

        # 分批处理特征
        for i in range(0, len(bin_cols), batch_size):
            # 切片：获取当前批次的特征列
            batch_bins = bin_cols[i : i + batch_size]
            
            # 构造查询计划 (Lazy Plan)
            # 这里的 .lazy() 它允许 Polars 优化器仅针对当前切片进行内存规划
            batch_res = (
                df_binned.lazy()
                .select([pl.col(c).cast(pl.Int16) for c in batch_bins] + [pl.col(c) for c in index_cols])
                .unpivot(
                    index=index_cols, 
                    on=batch_bins, 
                    variable_name="feature_bin", 
                    value_name="bin_index"
                )
                # 还原原始特征名 (去除 _bin 后缀)
                .with_columns(
                    pl.col("feature_bin").str.replace("_bin", "").alias("feature")
                )
                # 聚合至最小粒度：(Group x Feature x Bin)
                .group_by([group_col, "feature", "bin_index"])
                .agg(agg_exprs)
                # 执行并物化 (Streaming 模式防止大聚合 OOM)
                .collect(streaming=True)
            )
            
            result_frames.append(batch_res)

        if not result_frames:
            return pl.DataFrame()

        # 合并结果：将所有批次的小表 (Reduced Tables) 纵向合并
        return pl.concat(result_frames)

    def _ensure_woe_info(self, group_stats_raw: pl.DataFrame):
        """
        内存内 WOE 反向补全.

        当评估器检测到 `binner` 实例中缺失某些特征的 WOE 映射表时（例如：仅做了 transform 但未在
        当前 Label 上 fit，或者直接加载了无 WOE 的分箱规则），该方法利用 **已聚合的统计长表** 反向计算 WOE。

        此操作完全在内存中完成（基于聚合后的由少量行组成的小表），避免了再次扫描原始海量数据的 I/O 开销。

        Parameters
        ----------
        group_stats_raw : pl.DataFrame
            Map 阶段产出的统计长表。必须包含以下列：`['feature', 'bin_index', 'count', 'bad']`。

        Returns
        -------
        None
            该方法为 **In-place** 操作，计算结果将直接更新至 `self.binner.bin_woes_` 字典中。
        """
        features = group_stats_raw["feature"].unique().to_list()
        missing_woe_feats = [
            f for f in features 
            if f not in self.binner.bin_woes_ or not self.binner.bin_woes_[f]
        ]
        
        if not missing_woe_feats:
            return

        logger.debug(f"⚡ Calculating missing WOEs for {len(missing_woe_feats)} features (Memory Optimized)...")
        
        # 过滤出需要计算的特征
        target_stats = group_stats_raw.filter(pl.col("feature").is_in(missing_woe_feats))
        
        epsilon = 1e-9 # 平滑因子，防止除零或对0取对数

        # 计算 WOE
        woe_df = (
            target_stats
            .group_by(["feature", "bin_index"])
            .agg([
                pl.col("bad").sum().alias("bin_bad"),
                pl.col("count").sum().alias("bin_total")
            ])
            .with_columns([
                (pl.col("bin_total") - pl.col("bin_bad")).alias("bin_good")
            ])
            .with_columns([
                pl.col("bin_bad").sum().over("feature").alias("feature_total_bad"),
                pl.col("bin_good").sum().over("feature").alias("feature_total_good")
            ])
            .with_columns([
                (
                    ((pl.col("bin_bad") + epsilon) / (pl.col("feature_total_bad") + epsilon)) / 
                    ((pl.col("bin_good") + epsilon) / (pl.col("feature_total_good") + epsilon))
                ).log().cast(pl.Float32).alias("woe")
            ])
        )
        # 提取数据并更新到分箱器
        # 使用 to_dict(as_series=False) 避免 Python 对象开销
        woe_data = woe_df.select(["feature", "bin_index", "woe"]).to_dict(as_series=False)
        
        # 使用 defaultdict 简化映射构建，免去初始化判断
        temp_woe_map: Dict[str, Dict[int, float]] = defaultdict(dict)
        
        for feature, bin_index, woe in zip(woe_data["feature"], woe_data["bin_index"], woe_data["woe"]):
            # 过滤掉非法的 bin_index (如 Null 或 NaN)
            if bin_index is not None and not (isinstance(bin_index, float) and np.isnan(bin_index)):
                temp_woe_map[feature][int(bin_index)] = woe
        
        self.binner.bin_woes_.update(temp_woe_map)

    def _get_benchmark_dist(
        self, 
        group_stats_raw: pl.DataFrame, 
        bench_df: Optional[pl.DataFrame], 
        group_col: str, 
        features: List[str], 
        w_col: str
    ) -> pl.DataFrame:
        """
        获取用于 PSI 计算的基准分布 (Expected Distribution).

        该方法负责计算 PSI 公式 $\\sum (A - E) \\times \\ln(A/E)$ 中的 $E$ (Expected Distribution)。
        支持两种基准策略，自动根据 `bench_df` 是否传入进行切换。

        Parameters
        ----------
        group_stats_raw : pl.DataFrame
            当前数据集的统计长表 (Actual Data Stats)。
            仅在 `bench_df` 为 None 时使用，用于提取时间最早的分组作为基准。
        bench_df : Optional[pl.DataFrame]
            外部基准数据集 (OOT/Training Set)。
            若提供，将对其执行 `transform` -> `unpivot` -> `agg` 流程以获取基准分布。
        group_col : str
            分组列名 (如 'month')。用于在内部基准模式下定位 "Earliest Group"。
        features : List[str]
            需要计算的特征列表。
        w_col : str
            权重列名。若存在，基准分布将基于权重求和计算；否则基于样本计数。

        Returns
        -------
        pl.DataFrame
            包含基准分布占比的长表。
            - Schema: `['feature', 'bin_index', 'expected_dist']`
            - `expected_dist`: 该分箱在基准集中的占比 (Float32, Sum=1.0)。

        Notes
        -----
        **策略详情 (Strategy Details):**
        1. **External Mode (外部基准)**:
           当传入 `bench_df` 时，系统会复用当前的 Binner 对其进行分箱转换，
           并执行与 `_agg_basic_stats` 类似的 Unpivot-Agg 操作。
           适用于 "Train vs OOT" 或 "Train vs Test" 的 PSI 计算。

        2. **Internal Mode (内部基准)**:
           当 `bench_df` 为 None 时，系统默认假设 `group_stats_raw` 是含有时间维度的。
           它会自动筛选 `group_col` 最小的组 (e.g., '2023-01') 作为基准。
           适用于 "Month vs Baseline Month" 的跨期稳定性监控。
        """
        if bench_df is not None:
            # Case A: 处理外部基准集 
            bench_binned = self.binner.transform(bench_df, return_type="index")
            bin_cols = [f"{f}_bin" for f in features]
            agg_expr = pl.col(w_col).sum().alias("expected_count") if w_col else pl.len().alias("expected_count")
            idx_cols = [w_col] if w_col else []
            
            return (
                bench_binned.select(bin_cols + idx_cols)
                .unpivot(index=idx_cols, on=bin_cols, variable_name="feat_bin", value_name="bin_index")
                .with_columns(pl.col("feat_bin").str.replace("_bin", "").alias("feature"))
                .group_by(["feature", "bin_index"])
                .agg(agg_expr)
                .with_columns((pl.col("expected_count") / pl.col("expected_count").sum().over("feature")).alias("expected_dist"))
                .select(["feature", "bin_index", "expected_dist"])
            )
        else:
            # Case B: 内部基准
            min_group = group_stats_raw.select(pl.col(group_col).min()).item()
            logger.debug(f"📅 Using earliest group '{min_group}' as baseline (from stats cache).")
            
            return (
                group_stats_raw
                .filter(pl.col(group_col) == min_group)
                .group_by(["feature", "bin_index"])
                .agg(pl.col("count").sum().alias("expected_count"))
                .with_columns((pl.col("expected_count") / pl.col("expected_count").sum().over("feature")).alias("expected_dist"))
                .select(["feature", "bin_index", "expected_dist"])
            )

    def _calc_metrics_from_stats(
        self, 
        stats_df: pl.DataFrame, 
        expected_dist: pl.DataFrame, 
        group_col: str,
        # [PSI 控制参数]
        include_missing: bool = True,
        include_special: bool = True
    ) -> pl.DataFrame:
        """
        [Math Core] 基于聚合结果的向量化指标计算引擎 (Vectorized Metrics Engine).

        该方法是评估器的数学核心。它利用 Polars 的 **窗口函数 (Window Functions)** 和 **列式表达式**，
        在内存中（无需再次扫描原始数据）并行计算所有特征在所有分组下的核心风控指标。

        相比于传统的 Python `for` 循环计算（复杂度 $O(N \times M)$），该实现利用 SIMD 指令集，
        将复杂度降低至 $O(N)$（其中 N 为分箱后的总行数），极度高效。

        Parameters
        ----------
        stats_df : pl.DataFrame
            基础统计长表。
            必须包含列：`[group_col, 'feature', 'bin_index', 'count', 'bad']`。
        expected_dist : pl.DataFrame
            PSI 基准分布表。
            必须包含列：`['feature', 'bin_index', 'expected_dist']`。
        group_col : str
            分组维度列名 (如 'month')。
            计算累积指标 (KS/AUC) 时，会以此列和 'feature' 作为窗口分区 (Partition)。

        Returns
        -------
        pl.DataFrame
            包含分箱级指标详情的 DataFrame。
            - **数据粒度**: `[group_col, feature, bin_index]`
            - **关键列**: `psi_bin`, `iv_bin`, `ks_bin`, `auc_bin` (梯形面积贡献), `lift`。
            - **副作用**: 返回的数据已严格根据 `woe` 排序，以支持正确的累积分布计算。

        Notes
        -----
        **指标计算逻辑 (Metric Formulas):**

        1. **PSI (群体稳定性指标)**:
           .. math:: PSI = \\sum (Actual\\% - Expected\\%) \\times \\ln(\\frac{Actual\\%}{Expected\\%})

        2. **IV (信息值)**:
           使用预计算的 WOE 避免重复 Log 计算风险：
           .. math:: IV = \\sum (Good\\% - Bad\\%) \\times WOE

        3. **KS (柯尔莫哥洛夫-斯米尔诺夫统计量)**:
           .. math:: KS = \\max | CumBad\\% - CumGood\\% |

        4. **AUC (ROC 曲线下面积)**:
           使用 **梯形法则 (Trapezoidal Rule)** 进行数值积分，无需展开样本：
           .. math:: AUC = \\sum \\frac{(CumBad_i + CumBad_{i-1}) \\times (CumGood_i - CumGood_{i-1})}{2}

        5. **Lift (提升度)**:
           .. math:: Lift = \\frac{\\text{Bin Bad Rate}}{\\text{Total Bad Rate}}
        """
        # 构建 WOE 映射表
        woe_data = [
            {"feature": f, "bin_index": i, "woe": w}
            for f, m in self.binner.bin_woes_.items() for i, w in m.items()
        ]
        schema = {"feature": pl.String, "bin_index": pl.Int16, "woe": pl.Float64}
        woe_df = pl.DataFrame(woe_data, schema=schema) if woe_data else pl.DataFrame([], schema=schema)

        # 合并统计量、基准分布与 WOE
        base_df = (
            stats_df
            .join(expected_dist, on=["feature", "bin_index"], how="left")
            .join(woe_df, on=["feature", "bin_index"], how="left")
            .with_columns([
                (pl.col("count") - pl.col("bad")).alias("good"), 
                pl.col("expected_dist").fill_null(1e-9), 
                pl.col("woe").fill_null(0)               
            ])
        )

        epsilon = 1e-9
        
        # ======================================================================
        # [优化] 2. 构建 PSI 专用计算域 (Scope Definition)
        # ======================================================================
        # 定义哪些箱子参与 PSI 计算
        # 约定: Missing=-1, Special <= -3, Normal >= 0, Other=-2 (Other通常算正常)
        psi_valid_cond = pl.lit(True)
        
        if not include_missing:
            psi_valid_cond &= (pl.col("bin_index") != -1)
        
        if not include_special:
            psi_valid_cond &= (pl.col("bin_index") > -3)

        # ======================================================================
        # [优化] 3. 计算双套分布 (Dual Distribution Calculation)
        # ======================================================================
        
        # 3.1 全量分布 (用于 IV, BadRate, Lift) - 保持原有逻辑
        base_df = base_df.with_columns([
            pl.col("count").sum().over([group_col, "feature"]).alias("total_count"),
            pl.col("bad").sum().over([group_col, "feature"]).alias("total_bad"),
            pl.col("good").sum().over([group_col, "feature"]).alias("total_good"),
        ])
        
        # 3.2 PSI 专用分布 (Re-normalized Distribution)
        # 利用 filter().sum() 动态计算剔除后的分母
        base_df = base_df.with_columns([
            # 动态计算 Actual 的有效总数
            pl.col("count")
              .filter(psi_valid_cond)
              .sum()
              .over([group_col, "feature"])
              .alias("total_count_psi"),
            
            # 动态计算 Expected 的有效总占比 (因为 expected_dist 是比例，sum 可能等于 0.8)
            pl.col("expected_dist")
              .filter(psi_valid_cond)
              .sum()
              .over([group_col, "feature"])
              .alias("total_expected_dist_psi")
        ])

        # ======================================================================
        # 4. 指标计算
        # ======================================================================
        base_df = base_df.with_columns([
            # --- 常规指标 (基于全量) ---
            ((pl.col("count") + epsilon) / (pl.col("total_count") + epsilon)).alias("actual_dist"),
            (pl.col("bad") / (pl.col("total_bad") + epsilon)).alias("bad_dist"),    
            (pl.col("good") / (pl.col("total_good") + epsilon)).alias("good_dist"), 
            (pl.col("bad") / (pl.col("count") + epsilon)).alias("bad_rate"),        
            
            # --- PSI (基于动态归一化) ---
            # 1. 计算归一化后的 Actual% (只针对有效箱)
            (pl.col("count") / (pl.col("total_count_psi") + epsilon)).alias("act_prob_clean"),
            # 2. 计算归一化后的 Expected% (只针对有效箱)
            #    例如：如果剔除缺失值后，剩余 expected_dist 之和为 0.8，则每一项除以 0.8 放大
            (pl.col("expected_dist") / (pl.col("total_expected_dist_psi") + epsilon)).alias("exp_prob_clean")
        ])

        # 最终计算 PSI bin contribution
        base_df = base_df.with_columns([
            # 仅在有效箱上计算 PSI，无效箱置为 None
            pl.when(psi_valid_cond)
            .then(
                (pl.col("act_prob_clean") - pl.col("exp_prob_clean")) * (pl.col("act_prob_clean") / (pl.col("exp_prob_clean") + epsilon)).log()
            )
            .otherwise(None)
            .alias("psi_bin"),
            
            # Lift (通常基于全量)
            (pl.col("bad_rate") / ((pl.col("total_bad") + epsilon) / (pl.col("total_count") + epsilon))).alias("lift"),
            
            # IV 
            (
                (pl.col("bad_dist") - pl.col("good_dist")) * ((pl.col("bad_dist") + epsilon) / (pl.col("good_dist") + epsilon)).log()
            ).cast(pl.Float32).alias("iv_bin")
        ])

        # 计算有序指标 (AUC, KS, IV)：必须按 WOE 风险程度排序
        sorted_df = base_df.sort([group_col, "feature", "woe"])

        # 累积分布用于计算 KS 和 AUC
        sorted_df = sorted_df.with_columns([
            pl.col("bad_dist").cum_sum().over([group_col, "feature"]).alias("cum_bad_dist"),
            pl.col("good_dist").cum_sum().over([group_col, "feature"]).alias("cum_good_dist"),
        ])

        sorted_df = sorted_df.with_columns([

            ((pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs() * 100).alias("ks_bin"),
            
            # AUC 梯形法则计算面积
            (
                (pl.col("cum_good_dist") - pl.col("cum_good_dist").shift(1, fill_value=0).over([group_col, "feature"])) 
                * (pl.col("cum_bad_dist") + pl.col("cum_bad_dist").shift(1, fill_value=0).over([group_col, "feature"])) 
                / 2
            ).alias("auc_bin")
        ])

        sorted_df = sorted_df.with_columns([
            pl.when(pl.col("psi_bin").abs() < 1e-12)
              .then(0.0)
              .otherwise(pl.col("psi_bin"))
              .alias("psi_bin")
        ])

        return sorted_df

    def _prepare_context(self, df: pl.DataFrame, profile_by: Optional[str], dt_col: Optional[str]) -> pl.DataFrame:
        """
        [Helper] 上下文准备：标准化分组维度 (Context Preparation).

        该方法负责解析用户传入的 `profile_by` 和 `dt_col` 参数，确定最终用于趋势分析的分组列。
        如果需要基于时间切片（如按月、按周），它会自动调用 `MarsDate` 生成派生列。

        Parameters
        ----------
        df : pl.DataFrame
            输入的 Polars DataFrame。
        profile_by : Optional[str]
            分组指令。可以是具体的列名（如 'channel'），也可以是时间粒度指令（'day', 'week', 'month'）。
        dt_col : Optional[str]
            日期列名。仅当 `profile_by` 为时间粒度指令时必须提供。

        Returns
        -------
        Tuple[pl.DataFrame, str]
            - **pl.DataFrame**: 处理后的 DataFrame。如果涉及时间截断或兜底逻辑，会新增一列。
            - **str**: 最终确定的分组列名（可能是原始列，也可能是新增的临时列）。

        Notes
        -----
        **策略优先级 (Strategy Priority):**

        1. **智能默认 (Smart Default)**:
           若仅提供 `dt_col` 但未指定 `profile_by`，系统默认按 **'month'** (月度) 进行聚合。

        2. **自动时间聚合 (Auto Date Truncation)**:
           若 `profile_by` 为 `['day', 'week', 'month']` 且提供了 `dt_col`，
           系统会自动生成一个新的时间切片列 (e.g., `_mars_auto_month`) 并以此分组。

        3. **常规分组 (Explicit Grouping)**:
           若 `profile_by` 指定了具体的现有列 (e.g., 'city', 'vintage')，则直接使用该列。

        4. **全局兜底 (Global Fallback)**:
           若未提供任何分组信息，系统生成一个包含常量值 "Total" 的列 `_mars_auto_total`，
           将所有样本视为同一个分组（适用于单点评估）。
        """
        
        target_col_name = self.GROUP_COL_NAME
        
        # 1. 智能默认：有时间没分组 -> 默认按月
        if dt_col and not profile_by:
            logger.info(f"ℹ️ `dt_col` provided without `profile_by`. Defaulting trend to 'month'.")
            profile_by = "month"

        # 2. 处理时间切片 (Day/Week/Month)
        if dt_col and profile_by in ["day", "week", "month"]:
            if profile_by == "month":
                date_expr = MarsDate.dt2month(dt_col)
            elif profile_by == "week":
                date_expr = MarsDate.dt2week(dt_col)
            else:
                date_expr = MarsDate.dt2day(dt_col)
            
            # 直接 alias 为统一列名 "group"
            return df.with_columns(date_expr.alias(target_col_name))
        
        # 3. 常规分组 (按现有列)
        if profile_by:
            if profile_by in df.columns:
                # 如果该列已经存在，且名字不是 "group"，则复制一份并重命名
                # (使用 with_columns 而不是 rename，防止破坏原始列供其他用途)
                if profile_by != target_col_name:
                    return df.with_columns(pl.col(profile_by).cast(pl.String).alias(target_col_name))
                return df
            else:
                logger.warning(f"⚠️ Column '{profile_by}' not found. Falling back to Snapshot mode.")

        # 4. 兜底逻辑：单点评估 (Snapshot)
        # 生成一个全为 "Total" 的列，名为 "group"
        logger.info(f"ℹ️ Evaluating as a single snapshot ({target_col_name}='Total').")
        return df.with_columns(pl.lit("Total").alias(target_col_name))

    def _format_report(
        self, 
        stats_long: pl.DataFrame, 
        metrics_groups: pl.DataFrame, 
        metrics_total: pl.DataFrame, 
        group_col: str, 
        monotonicity_df: pl.DataFrame
    ) -> "MarsEvaluationReport":
        """
        [Helper] 报告构造与特征稳定性审计引擎 (Report Construction & Audit Engine).

        该方法负责将向量化计算的中间结果重塑为具备业务决策深度的三层报表体系：
        明细层 (Detail)、审计层 (Summary) 和趋势层 (Trend)。

        相较于传统评估，本引擎引入了 **"Strength-Recency-Stability-Logic" (强度-时效-稳定性-逻辑)** 四维审计体系，提供可解释的自动化决策建议。

        Parameters
        ----------
        stats_long : pl.DataFrame
            全量分箱统计长表。包含每个特征、每个分组、每个分箱的原始统计量及分箱级指标。
            Schema 包含: [feature, group_col, bin_index, iv_bin, ks_bin, psi_bin, ...]
        metrics_groups : pl.DataFrame
            仅包含分组数据（如 Monthly）的长表。用于计算跨期稳定性与时效性。
        metrics_total : pl.DataFrame
            仅包含全量（Total）统计的数据。用于获取特征全局区分度 (IV_total)。
        group_col : str
            分组维度列名（如 'month'）。
        monotonicity_df : pl.DataFrame
            单调性检查结果。包含特征分箱索引与坏率的 Spearman 相关系数。

        Returns
        -------
        MarsEvaluationReport
            报告容器实例。包含 Summary, Trend, Detail 三张重塑后的报表。

        Notes
        -----
        **1. 审计汇总表 (Summary Table) 核心指标逻辑**

        * **A. 强度审计 (Strength Audit)**:
            - `IV_total`: 全量样本下的信息值。衡量特征的硬实力。
            - `AUC_total`: 全量样本下的 ROC 面积。

        * **B. 时效审计 (Recency Audit)**:
            - `IV_recent`: 最近 N 期（默认 3 期）的平均 IV。防止“僵尸特征”（过去强，最近弱）。
            - `IV_change_pct`: 衰减率 $\\frac{IV_{recent} - IV_{avg}}{IV_{avg}}$。
              若 < -0.3，说明特征正在快速失效。

        * **C. 稳定性审计 (Stability Audit)**:
            - `PSI_max`: 跨分组中最大的 PSI 值。捕捉最坏情况下的分布偏移。
            - `PSI_alert_cnt`: PSI 超过警戒线 (0.1) 的期数。衡量漂移的频次。

        * **D. 逻辑审计 (Consistency Audit)**:
            - `RC_min` (RiskCorr Min): 风险一致性的最小值。
              衡量坏账逻辑是否发生过反转（如某个月高风险箱变成了低风险箱）。
            - `RC_neg_cnt`: 逻辑反转 (RC < 0) 的发生次数。

        * **E. 自动化决策 (Mars_Decision)**:
            基于规则树的评级体系 (优先级从高到低):
            1. `❌ Drop: Logic Broken`: 逻辑反转 (RC_min < 0.5)。
            2. `🗑️ Drop: Weak Signal`: 特征过弱 (IV_total < 0.02)。
            3. `⚠️ Watch: High Drift`: 剧烈漂移 (PSI_max > 0.25)。
            4. `📉 Review: Decaying`: 正在失效 (IV_change_pct < -0.3)。
            5. `✅ Keep: High Quality`: 通过所有测试的优质特征。

        **2. 风险一致性 (RiskCorr/RC) 计算逻辑**
        $$RC_{group\_i} = \\text{Pearson\_Corr}(\\vec{BR}_{baseline}, \\vec{BR}_{group\_i})$$
        其中 $\\vec{BR}$ 是各分箱坏率组成的向量。选取最早的一个分组作为基准。
        RC 接近 1 说明“好人始终是好人，坏人始终是坏人”。

        **3. 趋势透视表 (Trend Tables) 逻辑**
        将各项指标（PSI, AUC, KS, IV, BadRate, RiskCorr）以 `feature` 为行，
        `group_col` 为列进行 `Pivot` 转换，生成宽表，用于热力图渲染。
        """
        # ==============================================================================
        # Part 1: Detail Table (明细表构造)
        # ==============================================================================
        
        # 映射分箱 Label 
        map_rows = []
        feats = set(stats_long["feature"].unique().to_list())
        
        for f, m in self.binner.bin_mappings_.items():
            if f in feats:
                for i, l in m.items(): 
                    # [关键修复] 强制将字典 Key 转为 int，防止 JSON 里的字符串 Key 污染类型
                    try:
                        idx_val = int(i)
                        map_rows.append({"feature": f, "bin_index": idx_val, "bin_label": str(l)})
                    except (ValueError, TypeError):
                        continue 
        
        # [修复] 显式指定 Schema 为 Int16
        map_schema = {"feature": pl.String, "bin_index": pl.Int16, "bin_label": pl.String}
        
        if map_rows:
            map_df = pl.DataFrame(map_rows, schema=map_schema)
        else:
            map_df = pl.DataFrame([], schema=map_schema)

        # 1. 关联 Label
        # 此时 stats_long 和 map_df 都是 Int16，Join 安全，不会发生类型提升
        detail_base = (
            stats_long
            .join(map_df, on=["feature", "bin_index"], how="left")
            .with_columns(pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)))
        )

        # ==============================================================================
        # Part 1.5: Trend Shape Injection (计算并注入单调性形状)
        # ==============================================================================
        
        # 1. 提取 WOE 序列
        trend_source = (
            metrics_total  # 使用 Total 数据
            .lazy()
            .filter(pl.col("bin_index") >= 0)  # 这里现在是安全的，因为 bin_index 确认为 Int16
            .sort(["feature", "bin_index"])
            .select(["feature", "woe"])
        )
        
        # 2. 调用 Binner 中的静态方法进行判断
        from mars.feature.binner import MarsBinnerBase 

        trend_shape_df = (
            trend_source
            .group_by("feature")
            .agg(pl.col("woe"))
            .with_columns(
                pl.col("woe").map_elements(
                    MarsBinnerBase._detect_trend_scientific, 
                    return_dtype=pl.Utf8
                ).alias("trend")
            )
            .select(["feature", "trend"])
            .collect()
        )

        # 3. 将趋势列 Join 回明细基表
        detail_base = detail_base.join(trend_shape_df, on="feature", how="left")

        # 2. [Modified] 构建自定义排序键 (Sort Key)
        # 显式 cast(pl.Int32) 解决 SchemaError
        detail_table = (
            detail_base
            .with_columns([
                # 构建排序辅助列 (0:Normal, 1:Special/Missing, 2:Total)
                pl.when(pl.col("bin_index") >= 0).then(0).otherwise(1).cast(pl.Int32).alias("_sort_group"),
                
                # 针对非 Normal 箱的内部排序:
                # -1 (Missing) -> 10000
                # -2 (Other) -> 10001
                # < -2 (Special) -> 20000 + abs
                pl.when(pl.col("bin_index") >= 0).then(pl.col("bin_index").cast(pl.Int32)) # 显式转 Int32
                  .when(pl.col("bin_index") == -1).then(10000) 
                  .when(pl.col("bin_index") == -2).then(10001) 
                  .otherwise(20000 + pl.col("bin_index").abs().cast(pl.Int32)) 
                  .alias("_sort_idx")
            ])
            .sort(["feature", group_col, "_sort_group", "_sort_idx"]) # 执行物理排序
        )

        # 3. [新增] 计算累积指标 (Cumulative Metrics)
        detail_table = detail_table.with_columns([
            # 累积样本数
            pl.col("count").cum_sum().over(["feature", group_col]).alias("cum_count"),
            # 累积坏样本数
            pl.col("bad").cum_sum().over(["feature", group_col]).alias("cum_bad"),
            # 累积好样本数
            (pl.col("count") - pl.col("bad")).cum_sum().over(["feature", group_col]).alias("cum_good")
        ]).with_columns([
            # 累积坏账率 = 累积坏 / 累积总
            (pl.col("cum_bad") / (pl.col("cum_count") + 1e-9)).alias("cum_bad_rate"),
            
            # [新增] 计算箱占比 pct = count / total_count
            # total_count 已经在 _calc_metrics_from_stats 中计算并包含在 stats_long 中
            (pl.col("count") / (pl.col("total_count") + 1e-9)).alias("pct"),
            
            pl.col("bin_index").max().over(["feature", group_col]).alias("bin_index_max")
        ]).with_columns([
            pl.when(
                (pl.col("bin_index") == pl.col("bin_index_max")) | (pl.col("bin_index") == 0)
            )
            .then(pl.lit("首尾组"))
            .when(
                (pl.col("bin_index") == -1)
            )
            .then(pl.lit("空值组"))
            .when(
                (pl.col("bin_index") == -2)
            )
            .then(pl.lit("其他组"))
            .when(
                (pl.col("bin_index") <= -3)
            )
            .then(pl.lit("特殊组"))
            .otherwise(pl.lit("正常组"))
            .alias("bin_type")
        
        ])

        # ==============================================================================
        # 3.5 [新增] 构造 Total 汇总行 (Aggregation for Total Row)
        # ==============================================================================
        # 对每个 (feature, group) 生成一行汇总数据，包含计算后的综合指标
        total_rows = (
            stats_long
            .group_by(["feature", group_col])
            .agg([
                # 基础统计量汇总
                pl.col("count").sum().alias("count"),
                pl.col("bad").sum().alias("bad"),
                pl.col("iv_bin").sum().alias("iv_bin"),  
                pl.col("psi_bin").sum().alias("psi_bin"), 
                pl.col("auc_bin").sum().alias("auc_bin"), 
                pl.col("ks_bin").max().alias("ks_bin"),   
                pl.col("lift").max().alias("lift"), 
                pl.col("count").sum().alias("total_count")
            ])
            .with_columns([
                # 衍生列
                (pl.col("count") - pl.col("bad")).alias("good"),
                (pl.col("bad") / (pl.col("count") + 1e-9)).alias("bad_rate"),

                # [新增] Total 行的占比恒为 1.0
                pl.lit(1.0).alias("pct"),
                
                # 累积列 (对于 Total 行，累积值等于自身)
                pl.col("count").alias("cum_count"),
                pl.col("bad").alias("cum_bad"),
                (pl.col("bad") / (pl.col("count") + 1e-9)).alias("cum_bad_rate"),
                
                # AUC 方向修正
                pl.when(pl.col("auc_bin") < 0.5)
                  .then(pl.lit(1) - pl.col("auc_bin"))
                  .otherwise(pl.col("auc_bin"))
                  .alias("auc_bin"),
                
                # 标识列与排序键 (确保排在最后)
                pl.lit(9999).cast(pl.Int16).alias("bin_index"),
                pl.lit("Total").alias("bin_label"),
                
                pl.lit("汇总组").alias("bin_type"),
                
                pl.lit(2).cast(pl.Int32).alias("_sort_group"), # [Fix] Int32
                pl.lit(0).cast(pl.Int32).alias("_sort_idx")    # [Fix] Int32
            ])
        )

        # [修复] Total 行也要 join trend 列
        total_rows = total_rows.join(trend_shape_df, on="feature", how="left")

        target_cols = [
            "feature", group_col, "bin_index", "bin_label", "_sort_group", "_sort_idx",
            "count", "pct", "bad", "good", "bad_rate", "lift", "trend",
            "cum_count", "cum_bad", "cum_bad_rate",
            "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count",
            "bin_type"
        ]
        
        detail_table = (
            pl.concat([
                detail_table.select(target_cols),
                total_rows.select(target_cols)
            ])
            .sort(["feature", group_col, "_sort_group", "_sort_idx"])
        )

        # 4. 选择最终输出列
        detail_table = detail_table.select([
            pl.lit(self.target_col).alias("y"),
            "feature", "trend", group_col, "bin_index", "bin_label", 
            "count", "bad", "good", "pct", "bad_rate", "lift", 
            "cum_count", "cum_bad", "cum_bad_rate",
            "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count",
            "bin_type"
        ])

        # ==============================================================================
        # Part 2: Intermediate Calculations (中间指标计算)
        # ==============================================================================
        
        # 2.1 RiskCorr (RC) 跨期稳定性逻辑 
        # 确定基准序列 (选取时间最早的一组)
        first_group = metrics_groups.select(pl.col(group_col).min()).item()
        baseline_df = (
            metrics_groups
            .filter((pl.col(group_col) == first_group) & (pl.col("bin_index") >= 0))
            .select(["feature", "bin_index", "bad_rate"])
            .rename({"bad_rate": "base_br"})
        )

        # 构造用于计算相关性的全量数据流
        all_metrics_for_corr = pl.concat([
            metrics_groups.select(["feature", group_col, "bin_index", "bad_rate"]),
            metrics_total.select(["feature", group_col, "bin_index", "bad_rate"]) 
        ])

        # 计算 RiskCorr 长表: [feature, group_col, risk_corr]
        risk_corr_long = (
            all_metrics_for_corr
            .filter(pl.col("bin_index") >= 0)
            .join(baseline_df, on=["feature", "bin_index"], how="left")
            .group_by(["feature", group_col])
            .agg(
                pl.corr("bad_rate", "base_br", method="pearson").fill_null(0).alias("risk_corr")
            )
        )

        # 2.2 分组指标聚合 (Group Level Aggregation)
        # 将分箱粒度聚合为分组粒度 (如: Month Level)
        group_level_metrics = (
            metrics_groups
            .group_by(["feature", group_col])
            .agg([
                pl.col("iv_bin").sum().alias("iv"),
                pl.col("auc_bin").sum().alias("auc"),
                pl.col("psi_bin").sum().alias("psi"),
            ])
            # 确保 AUC 方向正确 (>= 0.5)
            .with_columns(
                pl.when(pl.col("auc") < 0.5).then(pl.lit(1) - pl.col("auc")).otherwise(pl.col("auc")).alias("auc")
            )
            .join(risk_corr_long, on=["feature", group_col], how="left")
        )

        # ==============================================================================
        # Part 3: Summary Table 
        # ==============================================================================
        
        # 3.1 识别时间窗口 (最近 N 期)
        sorted_groups = metrics_groups.select(pl.col(group_col).unique()).sort(group_col)[group_col].to_list()
        recent_n = 3
        # 如果分组少于3个，则全部视为最近
        recent_groups = set(sorted_groups[-recent_n:]) if len(sorted_groups) > 0 else set()
        
        # 标记是否为最近窗口
        group_stats_enriched = group_level_metrics.with_columns(
            pl.col(group_col).is_in(recent_groups).alias("is_recent")
        )

        # 3.2 维度聚合
        summary_audit = (
            group_stats_enriched
            .group_by("feature")
            .agg([
                # A. 强度 (Strength) - 均值
                pl.col("iv").mean().alias("IV_avg"),
                pl.col("auc").mean().alias("AUC_avg"),
                
                # B. 时效 (Recency) - 最近表现
                # 计算最近 N 期的平均 IV，如果缺失则填 0
                pl.col("iv").filter(pl.col("is_recent")).mean().fill_null(0).alias("IV_recent"),
                
                # C. 稳定性 (Stability) - 漂移监控
                pl.col("psi").max().alias("PSI_max"),
                pl.col("psi").mean().alias("PSI_avg"),
                # 统计有多少期 PSI 超过了 0.1 (轻微漂移警戒线)
                (pl.col("psi") > 0.1).sum().alias("PSI_alert_cnt"),
                
                # D. 逻辑性 (Consistency) - 坏账逻辑
                pl.col("risk_corr").min().alias("RC_min"),
                pl.col("risk_corr").mean().alias("RC_avg"),
                # 统计逻辑反转 (RC < 0) 的次数，这是致命伤
                (pl.col("risk_corr") < 0).sum().alias("RC_neg_cnt")
            ])
            .with_columns(
                # 计算衰减率: (最近IV - 平均IV) / 平均IV
                # < 0 代表特征正在衰退， > 0 代表特征在增强
                ((pl.col("IV_recent") - pl.col("IV_avg")) / (pl.col("IV_avg") + 1e-9)).alias("IV_change_pct")
            )
        )

        # 3.3 全量指标聚合 (Total Metrics)
        total_metrics_agg = (
            metrics_total.group_by("feature")
            .agg([
                pl.col("iv_bin").sum().alias("IV_total"),
                pl.col("ks_bin").max().alias("KS_total"),
                pl.col("auc_bin").sum().alias("AUC_total")
            ])
            .with_columns(
                pl.when(pl.col("AUC_total") < 0.5).then(pl.lit(1) - pl.col("AUC_total")).otherwise(pl.col("AUC_total")).alias("AUC_total")
            )
        )

        # 3.4 最终合并与决策生成
        summary_df = (
            summary_audit
            .join(total_metrics_agg, on="feature", how="left")
            .join(monotonicity_df, on="feature", how="left")
            
            # --- 核心决策逻辑 (Rule-Based Decision) ---
            .with_columns(
                pl.when(pl.col("RC_min") < 0.5).then(pl.lit("❌ Drop: Logic Broken"))      
                .when(pl.col("IV_total") < 0.02).then(pl.lit("🗑️ Drop: Weak Signal"))      
                .when(pl.col("PSI_max") > 0.25).then(pl.lit("⚠️ Watch: High Drift"))      
                .when(pl.col("IV_change_pct") < -0.3).then(pl.lit("📉 Review: Decaying"))  
                .otherwise(pl.lit("✅ Keep: High Quality"))                                
                .alias("Mars_Decision")
            )
            # 排序：优先看硬实力(IV_total)，其次看逻辑稳定性
            .sort(["IV_total", "RC_min"], descending=[True, True])
            .with_columns(pl.lit("Float64").alias("dtype"))
            # 调整下列顺序，把核心指标放前面
            .select([
                "feature", "Mars_Decision", "IV_total", "IV_recent", "IV_change_pct", 
                "PSI_max", "PSI_alert_cnt", "RC_min", "RC_neg_cnt", "Monotonicity",
                "KS_total", "AUC_total", "IV_avg", "AUC_avg"
            ])
        )

        # ==============================================================================
        # Part 4: Trend Tables (趋势透视表)
        # ==============================================================================
        trend_tables = {}
        target_metrics = ["psi", "auc", "ks", "iv", "bad_rate", "risk_corr"] 
        
        for metric in target_metrics:
            if metric == "risk_corr":
                pivot_src = risk_corr_long
            else:
                if metric == "bad_rate": 
                    agg_func = (pl.col("bad").sum() / (pl.col("count").sum() + 1e-9))
                elif metric == "ks": 
                    agg_func = pl.col(f"{metric}_bin").max()
                else: 
                    agg_func = pl.col(f"{metric}_bin").sum()

                pivot_src = stats_long.group_by([group_col, "feature"]).agg(agg_func.alias(metric))
                
                # Pivot 前的方向校正
                if metric == "auc":
                    pivot_src = pivot_src.with_columns(
                        pl.when(pl.col(metric) < 0.5).then(pl.lit(1) - pl.col(metric)).otherwise(pl.col(metric)).alias(metric)
                    )

            # 执行 Pivot
            pivot_df = pivot_src.pivot(
                index="feature", on=group_col, values=metric
            ).sort("feature").with_columns(pl.lit("Float64").alias("dtype"))
            
            # 排序列顺序，确保 Total 在最右侧
            cols = [c for c in pivot_df.columns if c not in ["feature", "dtype"]]
            sorted_cols = sorted([c for c in cols if c != "Total"]) + (["Total"] if "Total" in cols else [])
            
            trend_tables[metric] = self._format_output(pivot_df.select(["feature", "dtype"] + sorted_cols))

        return MarsEvaluationReport(
            summary_table=self._format_output(summary_df), 
            trend_tables=trend_tables, 
            detail_table=self._format_output(detail_table), 
            group_col=group_col
        )
    
    def plot_feature_binning_risk_trends(
        self,
        report: Optional["MarsEvaluationReport"] = None,
        df_detail: Union[pl.DataFrame, pd.DataFrame, None] = None, # Modified: 支持 Pandas 输入
        features: Union[str, List[str], None] = None,
        group_col: Optional[str] = None, 
        # [新增] 支持外部传入 Target 名称，用于多目标绘图时的标题显示
        target_name: Optional[str] = None,
        sort_by: str = "iv",
        ascending: bool = False,
        dpi: int = 150
    ):
        """
        [Visualization] 批量绘制特征分箱风险趋势图。

        该图表展示了特征分箱在不同时间切片下的样本占比和坏率表现，是特征筛选最直观的依据。

        Parameters
        ----------
        report : MarsEvaluationReport, optional
            由 evaluate 生成的报告对象。
        df_detail : Union[pl.DataFrame, pd.DataFrame], optional
            直接传入明细表（若未提供 report）。支持 Pandas DataFrame。
        features : str or List[str], optional
            要绘图的特征名。若为 None，绘制明细表中所有特征。
        group_col : str, optional
            分组列名。
        target_name : str, optional
            目标变量名称。用于图表标题显示。
            若不提供，默认使用 self.target_col。
        sort_by : str, default "iv"
            特征展示的排序指标。
        ascending : bool, default False
            是否升序排列。
        dpi : int, default 150
            图像分辨率。
        """
        target_df = None
        target_group_col = None
        
        # 1. 尝试从 Report 提取绘图所需数据
        if report is not None:
            target_df = report.detail_table.filter(pl.col("bin_index") != 9999) 
            target_group_col = report.group_col
        # 2. 尝试从 df_detail 提取数据
        elif df_detail is not None:
            if isinstance(df_detail, pd.DataFrame):
                target_df = pl.from_pandas(df_detail).filter(pl.col("bin_index") != 9999)
            else:
                target_df = df_detail.filter(pl.col("bin_index") != 9999)
                
            if group_col:
                target_group_col = group_col
            else:
                # 自动推断分组列
                # 排除已知列，剩下的通常就是分组列
                known = {"feature", "bin_index", "bin_label", "count", "bad", "bad_rate", "lift", "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count", "trend", "y"}
                candidates = [c for c in target_df.columns if c not in known]
                target_group_col = candidates[0] if candidates else "month"
                logger.debug(f"ℹ️ Auto-inferred group_col: '{target_group_col}'")
        else:
            raise ValueError("❌ Must provide either 'report' or 'df_detail' to plot.")

        if features is None:
            features = target_df["feature"].unique().to_list()
        elif isinstance(features, str):
            features = [features]
            
        # 确定最终显示的 Target Name
        # 优先使用传入的 target_name (多目标循环时传入)，否则使用实例绑定的 target_col
        final_target_name = target_name if target_name else self.target_col
            
        # 调用 MarsPlotter 绘图组件进行渲染
        MarsPlotter.plot_feature_binning_risk_trend_batch(
            df_detail=target_df,
            features=features,
            group_col=target_group_col,
            target_name=final_target_name, # 透传参数
            sort_by=sort_by,
            ascending=ascending,
            dpi=dpi
        )
        
    def evaluate_and_plot(
        self,
        # --- 1. Evaluate 阶段核心参数 ---
        df: Union[pl.DataFrame, pd.DataFrame],
        features: Optional[List[str]] = None,
        profile_by: Optional[str] = None,
        dt_col: Optional[str] = None,
        target_col: Optional[str] = None, 
        benchmark_df: Union[pl.DataFrame, pd.DataFrame, None] = None,
        # [新增参数] 
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        weights_col: Optional[str] = None,
        batch_size: int = 500,
        
        # --- 2. Binner 策略参数 ---
        bining_type: Optional[Literal["native", "opt"]] = None, 
        
        # --- 3. Plot 阶段核心参数 ---
        sort_by: str = "iv",       
        ascending: bool = False,   
        max_plots: int = 10,       
        dpi: int = 120,
        
        # --- 4. Binner 临时透传参数 ---
        **kwargs
    ) -> "MarsEvaluationReport":
        """
        [One-Stop Shop] 一站式评估与绘图工作流 (Evaluation & Visualization Pipeline)。
        
        该方法封装了 "拟合 -> 转换 -> 评估 -> 绘图" 的完整闭环。它采用了 **上下文管理器 (Context Manager)** 的设计思想，
        允许用户在不污染实例原始状态的前提下，临时覆盖参数进行快速实验。


        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            待评估的输入数据集 (Train/Test/OOT)。
        features : List[str], optional
            指定评估的特征列表。若为 None，自动识别所有可用特征。
        profile_by : str, optional
            趋势分析的分组列名 (如 'month', 'vintage')。
            用于生成时间切片下的 Risk Trend 图表。
        dt_col : str, optional
            日期列名。
            - 若配合 `profile_by='week'`，则按周聚合。
            - **[智能默认]** 若提供了 `dt_col` 但未提供 `profile_by`，默认为 **'month'** (按月聚合)。
        target_col : str, optional
            **[临时覆盖]** 目标变量列名。
            允许临时指定不同的 Label (如 'is_bad_7d' vs 'is_bad_30d') 进行评估，执行后会自动还原。
        bining_type : {'native', 'opt'}, optional
            **[临时覆盖]** 分箱算法策略。
            - 'native': 极速原生分箱 (Quantile/Uniform)。
            - 'opt': 最优分箱 (OptBinning)。
            指定此参数将强制触发重新拟合 (Re-fit)。
        max_plots : int, default 10
            **[可视化熔断]**。
            限制最终生成的图表数量。即使评估了 5000 个特征，也只绘制排序最靠前 (Top-N) 的 N 张图。
            防止因渲染过多 Canvas 导致 Jupyter Notebook 卡死或内存溢出。
        sort_by : str, default 'iv'
            **[绘图筛选器]** 特征排序依据 (Ranking Metric)。
            决定在生成的报告中优先展示哪些特征的图表。支持四维审计体系的**简写指令**：

            * **强度 (Strength)**:
                - `'iv'`: 按总信息值排序 (IV_total)。
                - `'auc'`: 按区分度排序 (AUC_total)。
                - `'ks'`: 按 KS 统计量排序 (KS_total)。
            * **逻辑 (Logic)**:
                - `'rc'` / `'logic'`: 按风险一致性 (RiskCorr) 排序。
                - *建议配合 `ascending=True` 使用，快速定位逻辑崩坏 (RC<0) 的特征。*
            * **稳定性 (Stability)**:
                - `'psi'` / `'drift'`: 按最大漂移幅度 (PSI_max) 排序。
            * **时效 (Recency)**:
                - `'decay'` / `'trend'`: 按 IV 衰减率 (IV_change_pct) 排序。
                - *建议配合 `ascending=True` 使用，快速找出正在失效的特征。*
            * **综合 (Decision)**:
                - `'rank'`: 按 Mars 自动化决策评级 (Mars_Decision) 排序。

            同时也支持传入 summary 表中的原始列名，如 `'IV_recent'`, `'PSI_alert_cnt'` 等。
        ascending : bool, default False
            排序方向。默认降序 (Descending)，即指标值大的排前面。
        **kwargs : dict
            **[分箱器透传参数]**。
            直接传递给底层的 `MarsNativeBinner` 或 `MarsOptimalBinner`。
            例如: `n_bins=10`, `min_bin_size=0.05`, `monotonic_trend='ascending'`。
            注意: 传入任何 kwargs 都会触发分箱器的重新拟合。

        Returns
        -------
        MarsEvaluationReport
            包含汇总表 (Summary)、趋势表 (Trend) 和详情表 (Detail) 的报告容器对象。
            
        Examples
        --------
        **场景 1: 快速基线评估 (Quick Baseline)**
        使用默认的 Native 分箱 (Quantile) 快速查看数据概貌。

        >>> evaluator = MarsBinEvaluator(target_col="bad_0")
        >>> report = evaluator.evaluate_and_plot(
        ...     df=train_df,
        ...     profile_by="month",   # 按月查看趋势
        ...     dt_col="apply_date",
        ...     max_plots=5           # 只画 IV 最高的前 5 个特征
        ... )

        **场景 2: 策略 A/B 测试 (精细化最优分箱)**
        觉得原生分箱不够精细？切换到最优分箱 (OptBinning) 并注入严格的**风控业务约束**。
        这里展示了如何通过 `kwargs` 透传参数来控制求解器行为。

        >>> # 尝试: 最优分箱 + 强约束
        >>> # bining_type="opt" 会自动触发重新拟合
        >>> report_opt = evaluator.evaluate_and_plot(
        ...     df, 
        ...     profile_by="month",
        ...     dt_col="apply_date",
        ...     bining_type="opt",               # 1. 切换算法
        ...     # --- 以下参数直接透传给 MarsOptimalBinner ---
        ...     n_bins=10,                        # 最大分箱数
        ...     min_bin_size=0.05,               # 最小箱占比 5% 
        ...     min_bin_n_event=5,               #  每箱至少 5 个坏人
        ...     prebinning_method="cart",        # 预分箱方法
        ...     n_prebins=50,                     # 预分箱数
        ...     min_prebin_size=0.01,            # 预分箱最小占比 1%
        ...     monotonic_trend="auto_asc_desc", 
        ...     time_limit=20                    # 单特征求解超时限制 (秒)
        ... )

        **场景 3: 不同 Label 的快速验证 (Label Shifting)**
        无需重新实例化，直接测试不同的 Y (如 7天逾期 vs 30天逾期)。

        >>> # 评估 7天逾期
        >>> evaluator.evaluate_and_plot(df, target_col="is_bad_7d")
        # -> 触发重置。根据 is_bad_7d 重新计算最优分箱切点，计算 IV。

        >>> # 评估 30天逾期
        >>> evaluator.evaluate_and_plot(df, target_col="is_bad_30d")
        # -> 再次触发重置。根据 is_bad_30d 重新计算最优分箱切点，计算 IV。

        **场景 4: 内存受限的大规模计算**
        处理 5000+ 维宽表时，减小 batch_size 防止 OOM。

        >>> evaluator.evaluate_and_plot(
        ...     df_huge_wide, 
        ...     batch_size=100,  # 降低 Map 阶段内存压力
        ...     max_plots=20     # 只关注 Top 20 核心特征
        ... )
        """
        
        # ==============================================================================
        # 0. Context Setup: 状态暂存与环境配置
        # ==============================================================================
        # 暂存原始状态，以便在 finally 块中还原，保证函数无副作用 (Side-effect free)
        original_target = self.target_col
        original_bining_type = self.bining_type 
        original_binner_kwargs = self.binner_kwargs.copy() if self.binner_kwargs else {}
        
        # 1. 应用临时覆盖: Target
        if target_col:
            self.target_col = target_col
            
        # 2. 应用临时覆盖: Bining Type (算法策略)
        if bining_type:
            self.bining_type = bining_type

        # 3. 处理动态参数 (kwargs) 与强制重置逻辑
        #  增加 target_col 的判断
        # 只要发生以下任一情况，都必须抛弃旧的分箱器，重新训练：
        # 1. 传入了新的分箱参数 (kwargs)
        # 2. 切换了算法类型 (bining_type)
        # 3. 切换了目标变量 (target_col) -> 关键！因为最优分箱依赖 Y
        should_reset_binner = (
            (kwargs is not None and len(kwargs) > 0) or 
            (bining_type is not None) or
            (target_col is not None)  # <--- 新增这行
        )

        if kwargs:
            if self.binner_kwargs is None: 
                self.binner_kwargs = {}
            self.binner_kwargs.update(kwargs)
            
        if should_reset_binner and self.binner is not None:
            # 增加更详细的日志，告诉用户为什么重置了
            reason = []
            if kwargs: 
                reason.append("params_changed")
            if bining_type: 
                reason.append("type_changed")
            if target_col: 
                reason.append("target_changed") 
            
            logger.info(f"⚡ Context changed ({'+'.join(reason)}). Resetting binner to trigger auto-refit.")
            self.binner = None

        try:
            # ==========================================================================
            # 1. Execution: 执行核心评估计算
            # ==========================================================================
            # 调用底层 evaluate 方法，它会处理：
            # - Auto-Fitting (如果 binner 为 None)
            # - Transform (分箱映射)
            # - Aggregation (Map-Reduce 计算指标)
            report = self.evaluate(
                df=df,
                features=features,
                profile_by=profile_by,
                dt_col=dt_col,
                # 传参
                psi_include_missing=psi_include_missing,
                psi_include_special=psi_include_special,
                benchmark_df=benchmark_df,
                weights_col=weights_col,
                batch_size=batch_size
            )
            
            # ==========================================================================
            # 2. Selection: 智能筛选绘图特征 (Top-N 逻辑)
            # ==========================================================================
            plot_features = features # 默认回退：全部特征
            
            # 获取 Summary 表用于排序
            summary = report.summary_table
            # 兼容性处理: 确保 summary 是 Polars DataFrame 以便使用高性能 sort
            if isinstance(summary, pd.DataFrame):
                summary_pl = pl.from_pandas(summary)
            else:
                summary_pl = summary
            
            # 映射简写指令到真实列名
            sort_map = {
                # --- 强度 (Strength) ---
                "iv": "IV_total", 
                "ks": "KS_total", 
                "auc": "AUC_total", 
                
                # --- 稳定性 (Stability) ---
                "psi": "PSI_max",        # 关注最坏漂移情况
                "drift": "PSI_max",      # psi 的别名
                
                # --- 逻辑性 (Logic) ---
                "rc": "RC_min",          # 关注逻辑反转风险
                "logic": "RC_min",       # rc 的别名
                
                # --- 时效性 (Recency) ---
                "decay": "IV_change_pct", # 关注特征衰减程度
                "trend": "IV_change_pct", # decay 的别名
                
                # --- 综合决策 ---
                "rank": "Mars_Decision"   # 按 Keep/Drop/Watch 排序
            }
            
            # get(key, default) -> 允许用户直接传 'IV_recent' 这种原始列名
            sort_key = sort_map.get(sort_by.lower(), sort_by)
            
            # [防御性检查] 再次确认 sort_key 是否真的在 summary 表里
            if sort_key not in report.summary_table.columns:
                logger.warning(f"⚠️ Sort column '{sort_key}' not found in summary report. Fallback to 'IV_total'.")
                sort_key = "IV_total"
            
            # 执行排序与截断
            if sort_key in summary_pl.columns:
                sorted_feats = (
                    summary_pl
                    .sort(sort_key, descending=not ascending)
                    .get_column("feature")
                    .to_list()
                )
                
                # [熔断机制] 如果特征数超过 max_plots，仅绘制 Top N
                if max_plots and max_plots > 0 and len(sorted_feats) > max_plots:
                    logger.info(f"📉 [Visual Protection] Plotting Top {max_plots} features sorted by '{sort_key}' (out of {len(sorted_feats)}).")
                    plot_features = sorted_feats[:max_plots]
                else:
                    plot_features = sorted_feats
            else:
                logger.warning(f"⚠️ Sort key '{sort_key}' not found in summary table. Plotting unsorted features.")

            # ==========================================================================
            # 3. Visualization: 批量绘图
            # ==========================================================================
            self.plot_feature_binning_risk_trends(
                report=report,
                features=plot_features, # 传入筛选后的列表
                group_col=report.group_col,
                sort_by=sort_by,
                ascending=ascending,
                dpi=dpi
            )
            
            return report
            
        finally:
            # ==========================================================================
            # 4. Teardown: 状态还原
            # ==========================================================================
            # 无论执行成功与否，必须还原实例属性，防止本次临时参数污染后续调用
            self.target_col = original_target
            self.bining_type = original_bining_type
            # [核心改进] 还原 kwargs，防止污染
            self.binner_kwargs = original_binner_kwargs
            

def profile_risk(
    # --- 数据输入 (Data Input) ---
    df: Union[pl.DataFrame, pd.DataFrame],
    target: Union[str, List[str]], 
    features: Optional[List[str]] = None,
    
    # --- 评估上下文 (Evaluation Context) ---
    profile_by: Optional[str] = None,
    dt_col: Optional[str] = None,
    benchmark_df: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    weights_col: Optional[str] = None,
    
    # --- 分箱策略参数 (Binning Strategy) ---
    binning_type: Literal["native", "opt"] = "native",
    n_bins: int = 5,
    min_bin_size: float = 0.02,
    monotonic_trend: str = "auto_asc_desc",
    special_values: Optional[List[Any]] = None,
    missing_values: Optional[List[Any]] = None,
    binner_kwargs: Optional[Dict[str, Any]] = None,
    
    # --- 绘图控制 (Plotting Control) ---
    plot: bool = True,
    plot_target: Union[str, List[str], None] = None,
    max_plots: int = 10,
    sort_by: str = "iv",
    ascending: bool = False,
    dpi: int = 300,
    
    # --- 性能参数 (Performance) ---
    n_jobs: int = -1,
    batch_size: int = 500
) -> MarsEvaluationReport:
    """
    [Mars Risk Profiler] 风险特征全景画像扫描。
    
    该函数提供一站式的特征评估服务，自动完成 "自动分箱 -> 指标计算 (IV/PSI/AUC) -> 趋势绘图" 全流程。
    适用于快速评估特征的风险区分能力、跨期稳定性以及业务逻辑合理性。

    支持 **多目标 (Multi-Target)** 评估模式：
    - 当传入 `target` 为列表时 (e.g., `['bad_30d', 'bad_90d']`)。
    - 首个 Target 作为 **主目标 (Primary Target)**，用于训练分箱规则 (Fit)。
    - 其余 Target 作为 **副目标 (Secondary Targets)**，复用主目标的分箱规则进行统计 (Transform)。
    - 最终生成包含所有 Target 表现的汇总报表，便于对比同一特征在不同定义下的表现。

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        待评估的数据集 (Train/Test/OOT)。
    target : Union[str, List[str]]
        目标变量列名 (0/1)。
        - 若为 `str`: 单目标模式。
        - 若为 `List[str]`: 多目标模式。target[0] 为主目标，其余为副目标。
    features : List[str], optional
        指定评估特征白名单。若为 None，自动扫描除 Target/Weights 之外的所有可用特征。
    
    profile_by : str, optional
        趋势分析的分组维度 (如 'month', 'vintage')。
    dt_col : str, optional
        日期列名。若提供且 `profile_by` 为时间粒度指令 ('day'/'week'/'month')，则自动生成聚合列。
    benchmark_df : DataFrame, optional
        PSI 计算的基准数据集。若不提供，默认以 `df` 中时间最早的分组作为基准。
    weights_col : str, optional
        样本权重列名。若提供，所有指标（IV/AUC/PSI）均基于加权计算。

    binning_type : {'native', 'opt'}, default 'native'
        分箱算法策略。
        - `'native'`: 极速原生分箱 (Quantile/Uniform/CART)。
        - `'opt'`: 最优分箱 (OptBinning)，支持单调性约束。
    n_bins : int, default 5
        最大分箱数。
    min_bin_size : float, default 0.02
        最小箱占比约束 (0.0 ~ 0.5)。
    monotonic_trend : str, default 'auto_asc_desc'
        单调性约束 (仅当 `binning_type='opt'` 时有效)。
        可选值: 'ascending', 'descending', 'peak', 'valley', 'auto', 'auto_asc_desc'。
    special_values : List[Any], optional
        特殊值列表 (如 -999)。将强制独立成箱，不参与切分。
    missing_values : List[Any], optional
        自定义缺失值列表。默认仅识别 Null/NaN。
    binner_kwargs : Dict, optional
        透传给分箱器的其他高级参数。

    plot : bool, default True
        是否自动绘制特征趋势图。
    plot_target : Union[str, List[str], None], optional
        多目标模式下，指定需要绘图的 Target。
        - None (默认) / 'all': 绘制所有 Target 的趋势图。
        - 'primary': 仅绘制主目标。
        - List: 指定绘制列表，如 `['bad_30d']`。
    max_plots : int, default 10
        [熔断机制] 最多绘制多少张图。根据 `sort_by` 排序取 Top N。
    sort_by : str, default 'iv'
        绘图排序依据。可选: `'iv'`, `'psi'`, `'auc'`, `'ks'`, `'risk_corr'`, `'rank'`。
    ascending : bool, default False
        排序方向。默认降序 (Descending)，即指标值高的排前面。
    dpi : int, default 300
        图像分辨率。

    n_jobs : int, default -1
        并行核心数。
    batch_size : int, default 500
        批处理大小。降低此值可减少内存峰值，适合处理超宽表。

    Returns
    -------
    MarsEvaluationReport
        评估报告对象。包含：
        - `summary_table`: 特征级汇总表 (IV/AUC/PSI/Decision)。
        - `detail_table`: 分箱级明细表 (Count/BadRate/Lift)。
        - `trend_tables`: 指标趋势宽表字典 (仅包含主目标数据)。
    """
    
    # --- 0. 参数标准化与模式识别 (Argument Sanitization) ---
    target_list = [target] if isinstance(target, str) else target
    if not target_list:
        raise ValueError("Target cannot be empty.")
    
    primary_target = target_list[0]
    is_multi_target = len(target_list) > 1
    
    # 1. 组装分箱参数 (Fit Params Assembly)
    fit_params = {
        "n_bins": n_bins,
        "min_bin_size": min_bin_size,
        "monotonic_trend": monotonic_trend,
        "special_values": special_values,
        "missing_values": missing_values,
        "n_jobs": n_jobs
    }
    # 合并用户传递的额外参数 (如 time_limit, cat_cutoff)
    if binner_kwargs:
        fit_params.update(binner_kwargs)
        
    logger.info(f"🚀 Starting Risk Profiling [Primary Target: {primary_target} | Strategy: {binning_type.upper()}]")
    
    # --- 2. 核心：拟合主目标分箱器 (Fit Primary Binner) ---
    # 我们先创建一个针对主 Target 的 Evaluator，让它去 Fit
    primary_evaluator = MarsBinEvaluator(
        target_col=primary_target,
        bining_type=binning_type,
        **fit_params
    )
    
    logger.info(f"👉 Evaluating Primary Target: {primary_target}")
    primary_report = primary_evaluator.evaluate(
        df=df,
        features=features,
        profile_by=profile_by,
        dt_col=dt_col,
        benchmark_df=benchmark_df,
        weights_col=weights_col,
        batch_size=batch_size
    )
    
    # [Fast Path] 如果只有单目标，无需合并，直接准备绘图
    if not is_multi_target:
        # 统一使用最后的绘图逻辑
        final_report = primary_report
        # 这里的 target_list 就是 [primary_target]
    
    else:
        # --- 3. 多目标循环评估 (Loop for Secondary Targets) ---
        logger.info(f"🔄 Multi-Target Mode: Extending evaluation to {len(target_list)-1} secondary targets...")
        
        # 获取已经训练好的分箱器 (Trained Binner)
        trained_binner = primary_evaluator.binner
        
        # 定义辅助转换函数：确保所有 DataFrame 统一为 Polars 格式以便处理
        def to_pl(d: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
            return pl.from_pandas(d) if isinstance(d, pd.DataFrame) else d

        # 处理主目标的表：添加 target_col 列以区分来源
        p_summary = to_pl(primary_report.summary_table).with_columns(pl.lit(primary_target).alias("target_col"))
        p_detail = to_pl(primary_report.detail_table)

        all_details: List[pl.DataFrame] = [p_detail]
        all_summaries: List[pl.DataFrame] = [p_summary]
        
        # 循环评估其余 Target
        for sec_target in target_list[1:]:
            logger.info(f"👉 Evaluating Secondary Target: {sec_target}")
            
            # 实例化一个新的 Evaluator，但传入**已训练好的 Binner**
            # MarsBinEvaluator 会检测到 binner 非空，从而跳过 fit 阶段，直接 transform
            sec_evaluator = MarsBinEvaluator(
                target_col=sec_target,
                binner=trained_binner # <--- 核心：复用分箱规则
            )
            
            sec_report = sec_evaluator.evaluate(
                df=df,
                features=features,
                profile_by=profile_by,
                dt_col=dt_col,
                benchmark_df=benchmark_df,
                weights_col=weights_col,
                batch_size=batch_size
            )
            
            # 同样转换并标记副目标的表
            s_detail = to_pl(sec_report.detail_table)
            s_summary = to_pl(sec_report.summary_table).with_columns(pl.lit(sec_target).alias("target_col"))
            
            all_details.append(s_detail)
            all_summaries.append(s_summary)

        # --- 4. 结果合并 (Merge Results) ---
        logger.info("∑ Merging multi-target reports...")
        
        # 纵向合并所有 Detail 表 (Detail 表本身已有 'y' 列区分 target，无需额外处理)
        final_detail = pl.concat(all_details)
        
        # 纵向合并 Summary 表
        final_summary = pl.concat(all_summaries)
        
        logger.info("ℹ️ Note: 'trend_tables' in the report contains data for Primary Target only.")
        
        final_report = MarsEvaluationReport(
            summary_table=final_summary,
            trend_tables=primary_report.trend_tables, # 仅保留主目标趋势
            detail_table=final_detail,
            group_col=primary_report.group_col
        )

    # --- 5. 智能绘图 (Visualization Dispatch) ---
    if plot:
        # 解析需要绘图的 Target 列表
        targets_to_plot = []
        if plot_target is None or plot_target == "all":
            targets_to_plot = target_list
        elif plot_target == "primary":
            targets_to_plot = [primary_target]
        elif isinstance(plot_target, str):
            targets_to_plot = [plot_target]
        elif isinstance(plot_target, list):
            targets_to_plot = plot_target
            
        # 过滤无效 Target (不在计算列表中的)
        targets_to_plot = [t for t in targets_to_plot if t in target_list]
        
        if not targets_to_plot:
            logger.warning("⚠️ No valid targets specified for plotting.")
        else:
            logger.info(f"🎨 Plotting trends for targets: {targets_to_plot}")
            _plot_report_helper(
                evaluator=primary_evaluator, # 复用这个实例的 plot 方法即可
                report=final_report,
                target_list=targets_to_plot, # [关键] 传入要画的列表
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
                dpi=dpi
            )
            
    return final_report

def _plot_report_helper(
    evaluator: MarsBinEvaluator, 
    report: MarsEvaluationReport, 
    target_list: List[str], 
    sort_by: str, 
    ascending: bool, 
    max_plots: int, 
    dpi: int
) -> None:
    """
    [Internal Helper] 辅助绘图函数，处理多 Target 循环与 Top-N 筛选逻辑。
    
    参数
    ----
    evaluator : MarsBinEvaluator
        用于调用底层 plot_feature_binning_risk_trends 方法的实例。
    report : MarsEvaluationReport
        包含汇总数据的报告对象。
    target_list : List[str]
        需要绘制的目标变量列表。
    """
    # 准备 Summary 表用于排序
    summary_all = report.summary_table
    if isinstance(summary_all, pd.DataFrame):
        summary_all = pl.from_pandas(summary_all)
        
    # 准备 Detail 表用于绘图数据源
    detail_all = report.detail_table
    if isinstance(detail_all, pd.DataFrame):
        detail_all = pl.from_pandas(detail_all)

    # 映射排序简码到真实列名
    sort_map = {
        "iv": "IV_total", "psi": "PSI_max", "ks": "KS_total", 
        "auc": "AUC_total", "rc": "RC_min", "rank": "Mars_Decision"
    }
    sort_key = sort_map.get(sort_by.lower(), "IV_total")

    # --- 循环绘制每个 Target ---
    for current_target in target_list:
        logger.info(f"📈 [Plotting Target] {current_target} ...")
        
        # 1. 筛选当前 Target 的 Summary 数据（用于排序）
        # 注意：不同 Target 下特征的 IV/AUC 是不一样的，所以 Top 特征可能不同
        # 单目标模式下 summary 可能没有 target_col 列，需要兼容
        if "target_col" in summary_all.columns:
            curr_summary = summary_all.filter(pl.col("target_col") == current_target)
        else:
            curr_summary = summary_all
        
        plot_features = None
        if sort_key in curr_summary.columns:
            sorted_feats = curr_summary.sort(sort_key, descending=not ascending)["feature"].to_list()
            
            if len(sorted_feats) > max_plots:
                logger.info(f"   📉 Top {max_plots} features by '{sort_key}'")
                plot_features = sorted_feats[:max_plots]
            else:
                plot_features = sorted_feats
        
        # 2. 筛选当前 Target 的 Detail 数据（用于绘图）
        # 利用 Evaluator 的 plot 方法支持传入 df_detail 的特性
        # 这里的 filter("y") 是关键，我们在 Evaluator._format_report 里加了这一列
        curr_detail = detail_all.filter(pl.col("y") == current_target)
        
        if curr_detail.is_empty():
            logger.warning(f"   ⚠️ No detail data found for target '{current_target}'. Skipping.")
            continue

        # 3. 调用底层绘图
        # 注意：这里我们手动传入 df_detail，从而绕过 report.detail_table (那是全量的)
        evaluator.plot_feature_binning_risk_trends(
            report=None, # 不传 report，使用 df_detail
            df_detail=curr_detail, # 传筛选后的 detail
            features=plot_features,
            group_col=report.group_col,
            target_name=current_target, # 标题显示当前 Target 名
            sort_by=sort_by,
            ascending=ascending,
            dpi=dpi
        )
        
        # 增加一个分隔线日志，方便区分不同 Target 的图
        if len(target_list) > 1:
            logger.info(f"{'-'*40}")