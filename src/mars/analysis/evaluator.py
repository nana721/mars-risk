import polars as pl
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union, Any, Tuple, Literal

from mars.core.base import MarsBaseEstimator
from mars.feature.binner import MarsBinnerBase, MarsNativeBinner, MarsOptimalBinner
from mars.analysis.report import MarsEvaluationReport 
from mars.utils.logger import logger
from mars.utils.decorators import time_it
from mars.utils.date import MarsDate
from mars.utils.plotter import MarsPlotter

class MarsBinEvaluator(MarsBaseEstimator):
    """
    [MarsBinEvaluator] 特征效能与稳定性评估引擎 (High-Performance Production Edition).

    该类实现了基于 **Map-Reduce 架构** 的大规模特征评估。
    它解决了传统 Python 风控库在处理宽表（Wide Table, 5000+ Cols）时的内存溢出和 I/O 瓶颈问题。

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

    Attributes
    ----------
    target_col : str
        目标变量列名 (0/1)。
    binner : MarsBinnerBase
        分箱器实例。如果未提供，evaluate 时会自动拟合。
    binner_kwargs : dict
        传递给自动分箱器的额外参数。
    """

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
            Label 列名，通常为 0（好人）和 1（坏人）。
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
        df: Union[pl.DataFrame, pd.DataFrame],  # Modified: 支持 Pandas 输入
        features: Optional[List[str]] = None,
        profile_by: Optional[str] = None,
        dt_col: Optional[str] = None,
        benchmark_df: Union[pl.DataFrame, pd.DataFrame, None] = None, # Modified: 支持 Pandas 输入
        weights_col: Optional[str] = None 
    ) -> "MarsEvaluationReport":
        """
        [Core] 执行特征评估的主入口。

        该方法涵盖了从原始数据到最终评估报告的全流程，包括自动分箱、指标计算和单调性检测。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            待评估的数据集（通常是训练集、测试集或 OOT 数据）。支持 Pandas DataFrame。
        features : List[str], optional
            指定评估的特征列表。若为 None，自动识别除 Target/Group/Weight 外的所有列。
        profile_by : str, optional
            分组维度名（如 'month', 'vintage'）。若提供，将生成基于该维度的 Trend 趋势报表。
        dt_col : str, optional
            日期列名。配合 `profile_by='month'` 等参数实现自动日期截断聚合。
        benchmark_df : Union[pl.DataFrame, pd.DataFrame], optional
            外部基准数据集。用于计算 PSI 的 Expected 分布。支持 Pandas DataFrame。
            若为 None，默认使用 `df` 中时间/分组顺序最早的一组作为基准。
        weights_col : str, optional
            样本权重列名。若指定，所有指标（BadRate, AUC, PSI等）均基于加权值计算。

        Returns
        -------
        MarsEvaluationReport
            包含 Summary (汇总), Trend (趋势), Detail (分箱详情) 三张核心表的报告容器。
        """
        
        # --- 1. 上下文准备 (Context Setup) ---
        
        # [Pandas 兼容] 核心修改：利用父类方法统一转换为 Polars 并设置输出标志位
        # 如果 df 是 Pandas，self._return_pandas 会被设为 True
        working_df = self._ensure_polars_dataframe(df)
        
        # [Pandas 兼容] 同样处理 benchmark_df，但不需要改变 _return_pandas 状态（仅做计算用）
        if benchmark_df is not None:
            benchmark_df = self._ensure_polars_dataframe(benchmark_df)

        # 检查 Target 有效性，避免后续 AUC/KS 计算崩溃
        n_unique = working_df.select(pl.col(self.target_col).n_unique()).item()
        if n_unique < 2:
            logger.warning(f"⚠️ Target '{self.target_col}' has < 2 unique values. Metrics (AUC/KS) may be invalid.")

        # 处理日期聚合逻辑：将日期列转化为 '2023-01' 等格式
        # 注意：此处 working_df 已经是 Polars 格式，可以安全调用 Polars API
        working_df, group_col = self._prepare_context(working_df, profile_by, dt_col)
        
        # 自动识别特征列：排除目标列、分组列和权重列
        exclude_cols = {self.target_col, group_col}
        if weights_col: exclude_cols.add(weights_col)
        
        target_features = features if features else [
            c for c in working_df.columns if c not in exclude_cols
        ]

        # [自动化] 拟合阶段：如果没有提供分箱器，现场初始化并训练一个
        if self.binner is None and self.binner_kwargs is not None:
            
            if self.bining_type == "native":
                logger.info("⚙️ No binner provided. Fitting MarsNativeBinner internally...")
                self.binner = MarsNativeBinner(features=target_features, **self.binner_kwargs)
            elif self.bining_type == "opt":
                logger.info("⚙️ No binner provided. Fitting MarsOptimalBinner internally...")
                self.binner = MarsOptimalBinner(features=target_features, **self.binner_kwargs)
            
            self.binner.fit(working_df, working_df.get_column(self.target_col))
        
        # [Transform] 数据转换：将原始连续值/离散值映射为分箱索引 (Int16)
        # 映射后的列名为 {feat}_bin
        logger.debug("🔄 Transforming features to bin indices...")
        df_binned = self.binner.transform(working_df, return_type="index")
        
        # ==============================================================================
        # 🚀 [核心优化]: 单次扫描架构 (Single-Pass Aggregation)
        # ==============================================================================
        
        # 2. [Map Phase] 执行全量数据的流式扫描
        # 将宽表 unpivot 后聚合，得到最小粒度统计表 (Group, Feature, Bin, Count, Bad)
        logger.debug("📊 Step 1: Scanning raw data for stats (Single Pass Map)...")
        group_stats_raw = self._agg_basic_stats(
            df_binned, group_col, target_features, self.target_col, weights_col
        )
        
        # 3. [Reduce Phase A] 补全 WOE 信息
        # 计算 KS/AUC 依赖 WOE 排序。若分箱器无 WOE，利用 group_stats_raw 内存计算，无需扫原表。
        self._ensure_woe_info_optimized(group_stats_raw)

        # 4. [Reduce Phase B] 获取 PSI 基准分布
        # 获取 Expected Distribution。若无外部基准，取 group_stats_raw 中最早的一组。
        expected_dist = self._get_benchmark_dist_optimized(
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

        # --- 6. 向量化指标计算 (Vectorized Metrics Calculation) ---
        logger.debug("🧮 Step 3: Calculating metrics (PSI/AUC/KS/IV)...")
        
        # 6.1 计算 Trend 数据：每个分组（如每月）的特征表现
        metrics_groups = self._calc_metrics_from_stats(
            group_stats_raw, expected_dist, group_col
        ).with_columns(pl.col(group_col).cast(pl.String))
        
        # 6.2 计算 Total 数据：特征在全量样本上的表现
        metrics_total = self._calc_metrics_from_stats(
            total_stats_raw, expected_dist, group_col
        )

        # 6.3 合并分组与总体结果
        metrics_total = metrics_total.select(metrics_groups.columns)
        stats_long = pl.concat([metrics_total, metrics_groups])

        # --- 7. 单调性检查 (Monotonicity Check) ---
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
        # 注意：_format_report 内部现在会调用 _format_output 来处理 Pandas 转换
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
        weights_col: Optional[str]
    ) -> pl.DataFrame:
        """
        [Map Phase] 全量数据扫描与核心聚合。

        利用 Polars 的 Lazy 模式和 Streaming 机制，将 5000+ 列的宽表高效转换为长表。
        这是整个流程中唯一一次对大数据集的 I/O 操作。

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

        Returns
        -------
        pl.DataFrame
            长表格式的统计汇总表，包含 [group_col, feature, bin_index, count, bad]。
        """
        bin_cols = [f"{f}_bin" for f in features]
        
        # 动态构建聚合表达式，兼容加权和非加权计算
        if weights_col:
            agg_exprs = [
                pl.col(weights_col).sum().alias("count"),                       # 加权总样本数
                (pl.col(y_col) * pl.col(weights_col)).sum().alias("bad")         # 加权坏样本数
            ]
            cols_to_select = [group_col, y_col, weights_col] + bin_cols
            index_cols = [group_col, y_col, weights_col]
        else:
            agg_exprs = [
                pl.len().alias("count"),                                        # 样本总数
                pl.col(y_col).sum().alias("bad")                                # 坏样本数
            ]
            cols_to_select = [group_col, y_col] + bin_cols
            index_cols = [group_col, y_col]

        # 核心 Pipeline：
        # unpivot 将宽表折叠，streaming=True 保证在内存不足时能处理 TB 级数据。
        return (
            df_binned.lazy()
            .select(cols_to_select)
            .unpivot(
                index=index_cols, 
                on=bin_cols, 
                variable_name="feature_bin", 
                value_name="bin_index"
            )
            .with_columns(pl.col("feature_bin").str.replace("_bin", "").alias("feature"))
            .group_by([group_col, "feature", "bin_index"])
            .agg(agg_exprs)
            .collect(streaming=True)
        )

    def _ensure_woe_info_optimized(self, group_stats_raw: pl.DataFrame):
        """
        [Optimization] 内存内 WOE 补全。

        如果评估器缺乏 WOE 信息（如直接传入评估而非训练），利用已聚合的小表计算 WOE。
        
        Formula:
        WOE = ln( (Bad_i / Total_Bad) / (Good_i / Total_Good) )
        """
        features = group_stats_raw["feature"].unique().to_list()
        missing_woe_feats = [
            f for f in features 
            if f not in self.binner.bin_woes_ or not self.binner.bin_woes_[f]
        ]
        
        if not missing_woe_feats:
            return

        logger.debug(f"⚡ Calculating missing WOEs for {len(missing_woe_feats)} features (Memory Optimized)...")
        
        # 仅对缺失特征进行聚合
        target_stats = group_stats_raw.filter(pl.col("feature").is_in(missing_woe_feats))
        
        global_bin_stats = (
            target_stats
            .group_by(["feature", "bin_index"])
            .agg([
                pl.col("count").sum().alias("n"),
                pl.col("bad").sum().alias("b")
            ])
        )
        
        feature_totals = (
            global_bin_stats.group_by("feature")
            .agg([
                pl.col("n").sum().alias("total_n"),
                pl.col("b").sum().alias("total_b")
            ])
        )
        
        epsilon = 1e-9 # 防止分母为 0
        woe_df = (
            global_bin_stats
            .join(feature_totals, on="feature", how="left")
            .with_columns([
                (pl.col("n") - pl.col("b")).alias("g"),
                (pl.col("total_n") - pl.col("total_b")).alias("total_g")
            ])
            .with_columns([
                (((pl.col("b") + epsilon) / (pl.col("total_b") + epsilon)) / 
                 ((pl.col("g") + epsilon) / (pl.col("total_g") + epsilon))).log().alias("woe")
            ])
        )
        
        # 将计算出的 WOE 回填至分箱器对象中以便后续使用
        for (feat,), sub_df in woe_df.partition_by("feature", as_dict=True).items():
            self.binner.bin_woes_[feat] = dict(zip(sub_df["bin_index"].to_list(), sub_df["woe"].to_list()))

    def _get_benchmark_dist_optimized(
        self, 
        group_stats_raw: pl.DataFrame, 
        bench_df: Optional[pl.DataFrame], 
        group_col: str, 
        features: List[str], 
        w_col: str
    ) -> pl.DataFrame:
        """
        [Optimization] 获取用于 PSI 计算的基准分布。

        策略：
        - 若传入外部 bench_df：需对其执行 transform 并聚合。
        - 若未传入：取当前数据中最早的时间切片作为基准。
        """
        if bench_df is not None:
            # Case A: 处理外部基准集 (涉及 I/O 与计算)
            bench_binned = self.binner.transform(bench_df, return_type="index")
            bin_cols = [f"{f}_bin" for f in features]
            agg_expr = pl.col(w_col).sum().alias("N_E") if w_col else pl.len().alias("N_E")
            idx_cols = [w_col] if w_col else []
            
            return (
                bench_binned.select(bin_cols + idx_cols)
                .unpivot(index=idx_cols, on=bin_cols, variable_name="feat_bin", value_name="bin_index")
                .with_columns(pl.col("feat_bin").str.replace("_bin", "").alias("feature"))
                .group_by(["feature", "bin_index"])
                .agg(agg_expr)
                .with_columns((pl.col("N_E") / pl.col("N_E").sum().over("feature")).alias("expected_dist"))
                .select(["feature", "bin_index", "expected_dist"])
            )
        else:
            # Case B: 内部基准 (零 I/O，直接切片)
            min_group = group_stats_raw.select(pl.col(group_col).min()).item()
            logger.debug(f"📅 Using earliest group '{min_group}' as baseline (from stats cache).")
            
            return (
                group_stats_raw
                .filter(pl.col(group_col) == min_group)
                .group_by(["feature", "bin_index"])
                .agg(pl.col("count").sum().alias("N_E"))
                .with_columns((pl.col("N_E") / pl.col("N_E").sum().over("feature")).alias("expected_dist"))
                .select(["feature", "bin_index", "expected_dist"])
            )

    def _calc_metrics_from_stats(self, stats_df: pl.DataFrame, expected_dist: pl.DataFrame, group_col: str) -> pl.DataFrame:
        """
        [Math Core] 基于聚合结果的向量化指标计算引擎。

        通过 Polars 的窗口函数和累加操作，在 O(N) 时间内完成所有统计指标计算。

        Metrics:
        - PSI: (Actual% - Expected%) * ln(Actual% / Expected%)
        - IV: (Good% - Bad%) * WOE
        - KS: Max(|CumBad% - CumGood%|)
        - AUC: 基于梯形法则的数值积分
        - Lift: 分箱坏率 / 总体坏率
        """
        # 1. 构建 WOE 映射表
        woe_data = [
            {"feature": f, "bin_index": i, "woe": w}
            for f, m in self.binner.bin_woes_.items() for i, w in m.items()
        ]
        schema = {"feature": pl.String, "bin_index": pl.Int16, "woe": pl.Float64}
        woe_df = pl.DataFrame(woe_data, schema=schema) if woe_data else pl.DataFrame([], schema=schema)

        # 2. 基础关联：合并统计量、基准分布与 WOE
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
        
        # 3. 计算分组汇总值 (Window Functions)
        base_df = base_df.with_columns([
            pl.col("count").sum().over([group_col, "feature"]).alias("total_count"),
            pl.col("bad").sum().over([group_col, "feature"]).alias("total_bad"),
            pl.col("good").sum().over([group_col, "feature"]).alias("total_good"),
        ])

        # 4. 计算占比指标
        base_df = base_df.with_columns([
            ((pl.col("count") + epsilon) / (pl.col("total_count") + epsilon)).alias("actual_dist"),
            (pl.col("bad") / (pl.col("total_bad") + epsilon)).alias("bad_dist"),    
            (pl.col("good") / (pl.col("total_good") + epsilon)).alias("good_dist"), 
            (pl.col("bad") / (pl.col("count") + epsilon)).alias("bad_rate"),        
        ])

        # 5. 计算 PSI 和 Lift (无序指标)
        base_df = base_df.with_columns([
            ((pl.col("actual_dist") - pl.col("expected_dist")) * (pl.col("actual_dist") / pl.col("expected_dist")).log()).alias("psi_bin"),
            (pl.col("bad_rate") / ((pl.col("total_bad") + epsilon) / (pl.col("total_count") + epsilon))).alias("lift")  
        ])

        # 6. 计算有序指标 (AUC, KS, IV)：关键在于必须按 WOE 风险程度排序
        sorted_df = base_df.sort([group_col, "feature", "woe"])

        # 累积分布用于计算 KS 和 AUC
        sorted_df = sorted_df.with_columns([
            pl.col("bad_dist").cum_sum().over([group_col, "feature"]).alias("cum_bad_dist"),
            pl.col("good_dist").cum_sum().over([group_col, "feature"]).alias("cum_good_dist"),
        ])

        sorted_df = sorted_df.with_columns([
            # KS = |CumBad - CumGood|
            (pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs().alias("ks_bin"),
            
            # AUC 梯形法则计算面积
            ((pl.col("cum_good_dist") - pl.col("cum_good_dist").shift(1).over([group_col, "feature"]).fill_null(0)) * (pl.col("cum_bad_dist") + pl.col("cum_bad_dist").shift(1).over([group_col, "feature"]).fill_null(0)) / 2
            ).alias("auc_bin"),

            # IV 公式：(Good% - Bad%) * ln(Good%/Bad%)
            ((pl.col("good_dist") - pl.col("bad_dist")) * ((pl.col("good_dist") + epsilon) / (pl.col("bad_dist") + epsilon)).log()).alias("iv_bin")
        ])

        return sorted_df

    def _prepare_context(self, df: pl.DataFrame, profile_by: str, dt_col: str) -> Tuple[pl.DataFrame, str]:
        """[Helper] 内部辅助：处理日期截断逻辑。"""
        if dt_col and profile_by in ["day", "week", "month"]:
            if profile_by == "month":
                date_expr = MarsDate.dt2month(dt_col)
            elif profile_by == "week":
                date_expr = MarsDate.dt2week(dt_col)
            else:
                date_expr = MarsDate.dt2day(dt_col)
            
            temp_group = f"_mars_auto_{profile_by}"
            return df.with_columns(date_expr.alias(temp_group)), temp_group
        
        return df, profile_by

    def _format_report(
        self, 
        stats_long: pl.DataFrame, 
        metrics_groups: pl.DataFrame, 
        metrics_total: pl.DataFrame, 
        group_col: str, 
        monotonicity_df: pl.DataFrame
    ) -> "MarsEvaluationReport":
        """
        [Helper] 报告构造与特征稳定性审计引擎。

        该方法负责将 `evaluate` 阶段产出的向量化计算长表重塑为具备业务决策深度的三层报表体系：
        明细层 (Detail)、审计层 (Summary) 和趋势层 (Trend)。

        Parameters
        ----------
        stats_long : pl.DataFrame
            全量分箱统计长表。包含每个特征、每个分组、每个分箱的原始统计量及分箱级指标。
            Schema 包含: [feature, group_col, bin_index, iv_bin, ks_bin, psi_bin, ...]
        metrics_groups : pl.DataFrame
            仅包含分组数据的长表。用于计算跨期稳定性。
        metrics_total : pl.DataFrame
            仅包含全量（Total）统计的数据。用于获取特征全局区分度。
        group_col : str
            分组维度列名（如 'month'）。
        monotonicity_df : pl.DataFrame
            单调性检查结果。包含特征分箱索引与坏率的斯皮尔曼相关系数。

        Returns
        -------
        MarsEvaluationReport
            报告容器实例。包含 Summary, Trend, Detail 三张重塑后的报表。
            **[Pandas 兼容]**：如果 evaluate 输入为 Pandas，则返回的 Report 内部属性也会自动转为 Pandas。

        Notes
        -----
        **1. 明细表 (Detail Table) 逻辑**
        将 `stats_long` 与分箱器中的业务标签映射表 (`bin_mappings_`) 关联，将物理索引（如 0, 1）转换为
        业务可读标签（如 '[20, 30)'）。该表用于下钻分析特征在特定时间点的具体分布。

        **2. 审计汇总表 (Summary Table) 核心指标逻辑**
        这是 Mars 库的核心决策依据，包含以下审计指标：
        
        * **稳定性审计 (Stability Audit)**:
            - `IV_cv` (IV 变异系数): $\frac{\sigma(IV_{period})}{\mu(IV_{period}) + \epsilon}$。
              反映特征区分能力的波动程度。CV > 0.4 通常意味着特征在不同样本段表现极其不稳定。
            - `PSI_max`: 跨分组中最大的 PSI 值。识别特征分布偏移的最坏情况。
            - `RC_min` (最小风险一致性): 跨期 `RiskCorr` 的最小值。
              若 `RC_min < 0.7`，说明特征逻辑在某些月份发生了反转或崩溃。

        * **效能审计 (Efficiency Audit)**:
            - `IV_total`: 特征在全量样本上的总信息值。
            - `Efficiency_Score` (效能得分): $\frac{IV_{avg} \times RC_{avg}}{1 + IV_{cv}}$。
              这是 Mars 独有的性价比指标。它奖励“高 IV、高逻辑稳定性、低波动”的特征。
            - `Monotonicity`: 特征在全局维度下的坏率单调性得分 (-1 ~ 1)。

        * **自动化建议 (Mars_Decision)**:
            - `❌ Drop: Logical Inversion`: 风险一致性触碰底线 (RC < 0.7)。
            - `⚠️ Watch: High Drift`: 样本分布发生剧烈偏移 (PSI > 0.25)。
            - `🗑️ Drop: Low IV`: 特征区分度过低 (IV < 0.02)。
            - `✅ Keep: Stable & Strong`: 满足所有稳定性与强度要求的优质特征。

        **3. 风险一致性相关性 (RiskCorr/RC) 计算逻辑**
        $$RC_{group\_i} = \text{Pearson\_Corr}(\vec{BR}_{baseline}, \vec{BR}_{group\_i})$$
        其中 $\vec{BR}$ 是各分箱坏率组成的向量。选取最早的一个分组作为基准。
        RC 衡量了特征的“风险排序”随时间变化的稳定性。RC 接近 1 说明“好人始终是好人，坏人始终是坏人”。

        **4. 趋势透视表 (Trend Tables) 逻辑**
        将各项指标（PSI, AUC, KS, IV, BadRate, RiskCorr）以 `feature` 为行，`group_col` 为列进行 `Pivot` 转换，
        生成宽表，专门用于在 Jupyter 中渲染颜色热力图。
        """
        # 1. 映射分箱 Label (从数字索引映射回可读的范围描述)
        map_rows = []
        feats = set(stats_long["feature"].unique().to_list())
        for f, m in self.binner.bin_mappings_.items():
            if f in feats:
                for i, l in m.items(): map_rows.append({"feature":f, "bin_index":i, "bin_label":l})
        
        map_schema = {"feature": pl.String, "bin_index": pl.Int16, "bin_label": pl.String}
        map_df = pl.DataFrame(map_rows, schema=map_schema) if map_rows else pl.DataFrame([], schema=map_schema)

        # 2. Detail Table: 详细分箱表关联
        detail_table = (
            stats_long
            .join(map_df, on=["feature", "bin_index"], how="left")
            .with_columns(pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)))
            .select(["feature", group_col, "bin_index", "bin_label", "count", "bad", "bad_rate", "lift", "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count"])
            .sort(["feature", group_col, "bin_index"])
        )

        # --- 3. [计算前置] 计算 RiskCorr (RC) 跨期稳定性逻辑 ---
        # 确定基准序列 (选取时间最早的一组)
        first_group = metrics_groups.select(pl.col(group_col).min()).item()
        baseline_df = (
            metrics_groups
            .filter((pl.col(group_col) == first_group) & (pl.col("bin_index") >= 0))
            .select(["feature", "bin_index", "bad_rate"])
            .rename({"bad_rate": "base_br"})
        )

        # 构造包含 Group 和 Total 的全量数据流用于计算 RC
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
                pl.corr("bad_rate", "base_br", method="pearson").alias("risk_corr")
            )
        )

        # --- 4. Summary Table: 稳定性与效能审计核心汇总 ---
        
        # 4.1 A. 分组指标聚合 [feature, group_col] -> 月度指标总和
        group_level_metrics = (
            metrics_groups
            .group_by(["feature", group_col])
            .agg([
                pl.col("iv_bin").sum().alias("iv"),
                pl.col("auc_bin").sum().alias("auc"),
                pl.col("psi_bin").sum().alias("psi"),
            ])
            # [修正] 确保分组级别的 AUC 始终 >= 0.5，否则会干扰后续 Summary 的平均值计算
            .with_columns(
                pl.when(pl.col("auc") < 0.5).then(pl.lit(1) - pl.col("auc")).otherwise(pl.col("auc")).alias("auc")
            )
            .join(risk_corr_long, on=["feature", group_col], how="left")
        )

        # 4.2 B. 跨期稳定性计算 [feature] -> 审计汇总
        summary_audit = (
            group_level_metrics
            .group_by("feature")
            .agg([
                pl.col("iv").mean().alias("IV_avg"),
                (pl.col("iv").std() / (pl.col("iv").mean() + 1e-9)).alias("IV_cv"),
                pl.col("auc").mean().alias("AUC_avg"),
                pl.col("auc").std().alias("AUC_std"),
                pl.col("psi").max().alias("PSI_max"), 
                pl.col("psi").mean().alias("PSI_avg"),
                pl.col("risk_corr").min().alias("RC_min"), 
                pl.col("risk_corr").mean().alias("RC_avg")
            ])
        )

        # 4.3 C. 关联全量指标与单调性，生成最终审计决策
        total_metrics_agg = (
            metrics_total.group_by("feature")
            .agg([
                pl.col("iv_bin").sum().alias("IV_total"),
                pl.col("ks_bin").max().alias("KS_total"),
                pl.col("auc_bin").sum().alias("AUC_total")
            ])
            # [修复点] 确保全量 AUC 也是方向修正后的
            .with_columns(
                pl.when(pl.col("AUC_total") < 0.5).then(pl.lit(1) - pl.col("AUC_total")).otherwise(pl.col("AUC_total")).alias("AUC_total")
            )
        )

        summary_df = (
            summary_audit
            .join(total_metrics_agg, on="feature", how="left")
            .join(monotonicity_df, on="feature", how="left")
            # 效能得分计算
            .with_columns(
                ((pl.col("IV_avg") * pl.col("RC_avg")) / (1 + pl.col("IV_cv"))).alias("Efficiency_Score")
            )
            # 执行专家建议规则
            .with_columns(
                pl.when(pl.col("RC_min") < 0.7).then(pl.lit("❌ Drop: Logical Inversion"))
                .when(pl.col("PSI_max") > 0.25).then(pl.lit("⚠️ Watch: High Drift"))
                .when(pl.col("IV_total") < 0.02).then(pl.lit("🗑️ Drop: Low IV"))
                .when(pl.col("IV_cv") > 0.4).then(pl.lit("📉 Review: High Volatility"))
                .otherwise(pl.lit("✅ Keep: Stable & Strong"))
                .alias("Mars_Decision")
            )
            .sort("Efficiency_Score", descending=True)
            .with_columns(pl.lit("Float64").alias("dtype"))
        )

        # --- 5. Trend Tables: 宽表热力图数据构造 ---
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
                
                # [修正] 确保在 Pivot 之前，所有指标（包括 Total 行和分组行）都进行了方向校正
                if metric == "auc":
         
                    pivot_src = pivot_src.with_columns(
                        pl.when(pl.col(metric) < 0.5).then(pl.lit(1) - pl.col(metric)).otherwise(pl.col(metric)).alias(metric)
                    )

            # 执行 Pivot 重塑
            pivot_df = pivot_src.pivot(
                index="feature", on=group_col, values=metric
            ).sort("feature").with_columns(pl.lit("Float64").alias("dtype"))
            
            # 排序列顺序，确保 Total 在最右
            cols = [c for c in pivot_df.columns if c not in ["feature", "dtype"]]
            sorted_cols = sorted([c for c in cols if c != "Total"]) + (["Total"] if "Total" in cols else [])
            
            # [Pandas 兼容] 在这里对 dict 中的每个 df 进行输出格式化
            trend_tables[metric] = self._format_output(pivot_df.select(["feature", "dtype"] + sorted_cols))

        # [Pandas 兼容] 利用 self._format_output 处理最终返回的 DataFrame
        # MarsEvaluationReport 接收转换后的 Pandas DF 或 Polars DF
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
            # 此时 report.detail_table 可能是 Pandas 或 Polars，取决于之前的 evaluate
            # 为了保险起见，绘图前可以统一转回 Polars 处理，或者 MarsPlotter 支持 Pandas
            # 假设 MarsPlotter 支持 Polars，我们这里确保一下
            target_df = report.detail_table
            target_group_col = report.group_col
        # 2. 尝试从 df_detail 提取数据
        elif df_detail is not None:
            # [Pandas 兼容] 如果传入的是 Pandas，先转为 Polars 方便后续处理
            # 注意：这里不需要设置 _return_pandas，因为绘图不返回数据
            if isinstance(df_detail, pd.DataFrame):
                target_df = pl.from_pandas(df_detail)
            else:
                target_df = df_detail
                
            if group_col:
                target_group_col = group_col
            else:
                # 自动推断分组列
                known = {"feature", "bin_index", "bin_label", "count", "bad", "bad_rate", "lift", "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count"}
                candidates = [c for c in target_df.columns if c not in known]
                target_group_col = candidates[0] if candidates else "month"
                logger.debug(f"ℹ️ Auto-inferred group_col: '{target_group_col}'")
        else:
            raise ValueError("❌ Must provide either 'report' or 'df_detail' to plot.")

        if features is None:
            features = target_df["feature"].unique().to_list()
        elif isinstance(features, str):
            features = [features]
            
        # 调用 MarsPlotter 绘图组件进行渲染
        MarsPlotter.plot_feature_binning_risk_trend_batch(
            df_detail=target_df,
            features=features,
            group_col=target_group_col,
            target_name=self.target_col,
            sort_by=sort_by,
            ascending=ascending,
            dpi=dpi
        )