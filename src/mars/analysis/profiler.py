import dataclasses
from typing import List, Union, Optional, Any, Dict, Literal
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import pandas as pd

from mars.core.base import MarsBaseEstimator
from mars.analysis.report import MarsProfileReport
from mars.analysis.config import MarsProfileConfig
from mars.feature.binner import MarsNativeBinner
from mars.utils.logger import logger
from mars.utils.decorators import time_it # , monitor_os_memory
from mars.utils.date import MarsDate

class MarsDataProfiler(MarsBaseEstimator):
    """
    [MarsDataProfiler] 基于 Polars 的高性能多维数据画像工具。

    专为大规模风控建模数据集设计。它作为分析流程的入口，封装了从
    数据质量诊断、统计值计算到可视化生成的全链路逻辑。

    主要功能 (Key Features)
    -----------------------
    1. **全量指标概览 (Overview)**:
       - 计算 Missing/Zero/Unique/Top1 等基础 DQ 指标。
       - 自动识别并计算数值列的统计分布 (Mean, Std, Quantiles)。
    
    2. **迷你分布图 (Sparklines)**:
       - 在报告中生成 Unicode 字符画 (如  ▂▅▇█)。
       - **可视化逻辑**: 自动采样 (默认20w行) -> 剔除异常值 -> 等宽分箱 -> 字符映射。
       - 支持通过 Config 调整分箱精度和采样率。

    3. **多维趋势分析 (Trend Analysis)**:
       - 支持按时间 (Month/Vintage) 或客群 (Segment) 进行分组分析。
       - 自动计算组间稳定性指标 (Variance/CV)。

    Attributes
    ----------
    df : pl.DataFrame
        内部存储的 Polars DataFrame (已转换为 Polars 格式)。
    features : List[str]
        最终确定的待分析特征列表 (经过 exclude_features 和 include_dtypes 筛选后)。
    config : MarsProfileConfig
        全局配置对象。控制计算哪些指标、是否画图等。
        详见 `mars.analysis.config.MarsProfileConfig`。
    custom_missing : List[Any]
        自定义缺失值列表 (如 -999, 'null')。在计算 missing_rate 时计入分子，
        在计算统计分布 (Mean/Std) 时被剔除。
    special_values : List[Any]
        自定义特殊值列表。这些值被视为有效值参与 DQ 统计 (如 Top1)，
        但在计算数值分布 (Sparkline/Mean/PSI) 时会被剔除。
    psi_cv_ignore_threshold : float
        PSI 稳定性计算的门控阈值 (默认 0.05)。
        用于解决 "小基数陷阱"：若某特征在所有分组下的 PSI 最大值仍小于该阈值，
        则认为其处于绝对稳定区，强制将 Group CV 置为 0，避免微小波动触发误报。

    Examples
    --------
    >>> # 1. 基础用法
    >>> profiler = MarsDataProfiler(df)
    >>> report = profiler.generate_profile()
    >>> report.show_overview()

    >>> # 2. 高级用法：自定义缺失值 + 按月分组 + 关闭画图(提速)
    >>> profiler = MarsDataProfiler(df, custom_missing_values=[-999, "unknown"])
    >>> report = profiler.generate_profile(
    ...     profile_by="month",
    ...     config_overrides={"enable_sparkline": False, "stat_metrics": ["psi", "mean", "min", "max"]}
    ... )
    """

    def __init__(
        self, 
        df: Union[pl.DataFrame, pd.DataFrame], 
        features: Optional[List[str]] = None,
        *,
        exclude_features: Optional[List[str]] = None,
        include_dtypes: Union[type, pl.DataType, List[Union[type, pl.DataType]], None] = None,
        
        custom_missing_values: Optional[List[Union[int, float, str]]] = None,
        custom_special_values: Optional[List[Any]] = None,
        
        overview_batch_size: int = 500,  # 新增：控制概览计算的批大小
        
        psi_batch_size: int = 50, 
        psi_n_bins: int = 10,           
        psi_bin_method: Literal["quantile", "uniform"] = "quantile", 
        psi_cv_ignore_threshold: float = 0.05,
        
        sample_frac: Optional[float] = None, 
        
        config: Optional[MarsProfileConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            输入数据集。会自动转换为 Polars 格式以利用其向量化计算优势。
        features : List[str], optional
            指定要分析的列名列表。如果为 None，则分析所有列。
        exclude_features : List[str], optional
            [黑名单] 指定要排除的列名列表。
            逻辑优先级: final_features = (features or all_cols) - exclude_features
        include_dtypes : List[pl.DataType], optional
            [类型白名单] 仅包含指定数据类型的列进行分析。
            支持 Python 原生类型和 Polars 类型。例如: [int, pl.Int64, pl.Float64] 只分析数值列。
        custom_missing_values : List[Union[int, float, str]], optional
            指定自定义缺失值列表。例如: [-999, "unknown", "\\N"]。
        custom_special_values : List[Any], optional
            指定自定义特殊值列表 (如极端值)。这些值在计算分布图时会被单独处理。
        psi_batch_size : int, optional
            计算 PSI 时的特征批处理大小。默认为 50。
        psi_n_bins : int, optional
            计算 PSI 时的分箱数量。默认为 10。
        psi_bin_method : str, optional
            计算 PSI 时的分箱方法。支持 "quantile" 或 "uniform"。默认为 "quantile"。
        psi_cv_ignore_threshold : float, optional
            PSI 稳定性计算的门控阈值。默认 0.01。
            当某特征的平均 PSI 小于该值时，强制将 group_cv 置为 0，避免"小基数陷阱"。
        sample_frac : float, optional
            如果指定，则对输入数据进行随机采样，采样比例为该值 (0~1之间)。
            数据量非常大时可用以提升分析速度，但会牺牲一定精度。
        config : MarsProfileConfig, optional
            配置对象。如果为 None，则使用默认配置。
        
        """
        super().__init__()
        # 1. 数据接入与采样
        self.df = self._ensure_polars_dataframe(df)
        if sample_frac is not None and 0 < sample_frac < 1.0:
            logger.warning(f"🎲 Data is sampled (frac={sample_frac}). Metrics are estimates.")
            self.df = self.df.sample(fraction=sample_frac, shuffle=True)

        self.config = config if config else MarsProfileConfig()
        
        # 2. 值处理配置
        self.custom_missing = custom_missing_values if custom_missing_values else []
        self.special_values = custom_special_values if custom_special_values else []
        
        # 3. PSI 配置
        self.psi_batch_size = psi_batch_size
        self.psi_n_bins = psi_n_bins
        self.psi_bin_method = psi_bin_method
        self.psi_cv_ignore_threshold = psi_cv_ignore_threshold

        # 4. 特征筛选逻辑 
        # Step A: 初始范围
        candidates = features if features else self.df.columns
            
        # Step B: 黑名单剔除
        if exclude_features:
            exclude_set = set(exclude_features)
            candidates = [c for c in candidates if c not in exclude_set]

        # Step C: 类型白名单 (支持 Python原生类型 + Polars类型)
        # ---------------------------------------------------------
        if include_dtypes:
            import polars.selectors as cs
            
            # 1. 归一化为列表
            if not isinstance(include_dtypes, list):
                raw_dtypes = [include_dtypes]
            else:
                raw_dtypes = include_dtypes
            
            # 2. 类型映射：Python Type -> Polars Abstract Type
            target_dtypes = []
            for t in raw_dtypes:
                # --- Python Native Mapping ---
                if t is int:
                    target_dtypes.append(pl.Integer) # 匹配所有整型 (Int8~64, UInt)
                elif t is float:
                    target_dtypes.append(pl.Float)   # 匹配所有浮点 (Float32/64)
                elif t is str:
                    target_dtypes.append(pl.String)  # 匹配 String/Utf8
                elif t is bool:
                    target_dtypes.append(pl.Boolean)
                elif t is list:
                    target_dtypes.append(pl.List)
                # --- Polars Type Pass-through ---
                else:
                    target_dtypes.append(t)
            
            # 3. 智能选择
            try:
                # 利用 Selectors 进行宽容匹配
                dtype_selector = cs.by_dtype(target_dtypes)
                # 只在 candidates 范围内筛选
                matched_cols = self.df.select(pl.col(candidates)).select(dtype_selector).columns
                candidates = matched_cols
                
            except Exception as e:
                logger.error(f"Type filtering failed: {e}")
                # 降级策略: 简单的包含判断 (仅对 Polars 类型有效)
                candidates = [c for c in candidates if self.df.schema[c] in target_dtypes]

        if not candidates:
            raise ValueError("No features selected after filtering.")
        
        self._dtype_map = self.df.schema
        self.overview_batch_size = overview_batch_size
            
        self.features = candidates
        
    @time_it
    def generate_profile(
        self, 
        profile_by: Optional[str] = None, 
        *,
        dt_col: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> MarsProfileReport:
        """
        [Core] 执行数据画像分析管道，生成完整分析报告。

        该方法会自动计算两类指标：
        1. **Overview (全量概览)**: 包含数据分布(Sparkline)、DQ指标、统计指标。不涉及分组。
        2. **Trends (分组趋势)**: 如果指定了 `profile_by`，会计算各项指标随该维度的变化。
        
        **日期聚合功能 (New)**:
        如果指定了 `dt_col`，`profile_by` 可直接传入 "day", "week", "month"。
        程序会自动基于 `dt_col` 生成对应的时间粒度列进行分组分析。

        Parameters
        ----------
        profile_by : str, optional
            分组维度。
            - 若提供 `dt_col`: 可选 "day", "week", "month"。
            - 若未提供 `dt_col`: 必须是数据中已存在的列名。
            - None: 仅生成 Overview 和 Total 趋势列。
            - 如果是自动聚合，Overview 表中不会包含这个临时的日期分组列，只会在 Trends 表中体现。
        
        dt_col : str, optional
            指定日期/时间列名。用于配合 `profile_by` 进行自动时间聚合。

        config_overrides : Dict[str, Any], optional
            临时覆盖 `MarsProfileConfig` 中的默认配置。支持以下配置项：

            **1. 计算范围 (Metrics)**
            
            * ``stat_metrics`` (List[str]): 需要计算的统计指标。
              可选值: "psi", "mean", "std", "min", "max", "p25", "median", "p75", "skew", "kurtosis"。
            * ``dq_metrics`` (List[str]): 需要计算的数据质量指标。
              可选值: "missing", "zeros", "unique", "top1"。

            **2. 可视化 (Visualization)**
            
            * ``enable_sparkline`` (bool): 是否计算字符画形式的迷你分布图 (默认 True)。
            * ``sparkline_sample_size`` (int): 计算分布图时的采样行数。
            * ``sparkline_bins`` (int): 分布图的分箱精度。

        Returns
        -------
        MarsProfileReport
            包含概览表和趋势表的报告对象容器。

        Examples
        --------
        >>> # 1. 基础用法：生成并查看报告
        >>> profiler = MarsDataProfiler(df)
        >>> report = profiler.generate_profile()
        
        >>> # 拿到 report 后怎么用？
        >>> report # 在 Jupyter 中显示报告的用法
        >>> report.show_overview()  # 在 Jupyter 中显示数据概览
        >>> report.write_excel("my_analysis.xlsx")  # 导出 Excel

        >>> # 2. 高级用法：按月分组分析
        >>> report = profiler.generate_profile(profile_by="month")
        >>> report.show_trend("mean") # 查看均值随月份的变化趋势
        
        >>> # 3. 获取底层数据 (可以用于自动化特征筛选)
        >>> # 返回值结构:
        >>> # overview: DataFrame (全量概览)
        >>> # dq_tables: Dict[str, DataFrame] (DQ 指标趋势表字典)
        >>> # stat_tables: Dict[str, DataFrame] (统计指标趋势表字典)
        >>> overview, dq_tables, stat_tables = report.get_profile_data()
        
        >>> # 示例: 筛选出缺失率 > 90% 的特征列表
        >>> high_missing_cols = overview.filter(pl.col("missing_rate") > 0.9)["feature"].to_list()
        """
        # 1. 动态配置合并
        run_config: MarsProfileConfig = self.config
        if config_overrides:
            run_config = dataclasses.replace(self.config, **config_overrides)

        # ---------------------------------------------------------------------
        # 2. 构建分析上下文 (Context Setup)
        #    决定本次运行使用的数据集 (working_df) 和 分组列 (group_col)
        # ---------------------------------------------------------------------
        working_df = self.df
        group_col = profile_by

        # 自动日期聚合
        if dt_col and profile_by in ["day", "week", "month"]:
            if dt_col not in self.df.columns:
                raise ValueError(f"dt_col '{dt_col}' not found in DataFrame.")
            
            # 调用 MarsDate 工具类生成 Polars 表达式
            # 无论 dt_col 是 "20230101"(str), 20230101(int) 还是 "2023/01/01"
            # MarsDate 都能通过 smart_parse_expr 自动处理并截断
            if profile_by == "day":
                date_expr = MarsDate.dt2day(dt_col)
            elif profile_by == "week":
                date_expr = MarsDate.dt2week(dt_col)
            elif profile_by == "month":
                date_expr = MarsDate.dt2month(dt_col)
            else:
                raise ValueError(f"Unsupported time grain: {profile_by}")

            # 生成临时分组列名, 避免与现有列冲突
            temp_group_col = f"_mars_auto_{profile_by}"
            
            # 生成包含临时列的 working_df (Zero-Copy 机制下开销很小)
            working_df = self.df.with_columns(date_expr.alias(temp_group_col))
            group_col = temp_group_col
            
            logger.info(f"📅 Auto-aggregating '{dt_col}' by '{profile_by}' using MarsDate -> '{group_col}'")

        elif dt_col is None and profile_by is not None:
            # 常规模式：profile_by 必须是现有列
            if profile_by not in self.df.columns:
                raise ValueError(f"Column '{profile_by}' not found. Did you forget to set `dt_col`?")
        
        # 3. 计算全量概览 (Overview) 
        #    Overview 总是基于原始 self.df (或 working_df，不影响结果)
        overview_df: pl.DataFrame = self._calculate_overview(run_config)

        # 4. 计算趋势表 (Trend Tables)
        #    必须传入 context_df=working_df，因为它包含了可能新生成的 group_col
        dq_tables: Dict[str, pl.DataFrame] = {}
        
        # 4.1 DQ Trends
        for m in run_config.dq_metrics:
            dq_tables[m] = self._generate_pivot_report(m, group_col, context_df=working_df)

        # 4.2 Stats Trends
        stat_tables: Dict[str, pl.DataFrame] = {}
        for m in run_config.stat_metrics:
            # a. Pivot
            pivot: pl.DataFrame = self._generate_pivot_report(m, group_col, context_df=working_df)
            # b. Stability (CV/Var) - 仅在有分组时计算
            if group_col:
                pivot = self._add_stability_metrics(pivot, exclude_cols=["feature", "dtype", "total"])
            stat_tables[m] = pivot
            
        # 4.3 PSI Trend
        if group_col and ("psi" in run_config.stat_metrics):
            try:
                # 传入 working_df 以支持临时时间列
                psi_df = self._get_psi_trend(group_col, context_df=working_df)
                if not psi_df.is_empty():
                    stat_tables["psi"] = psi_df
            except Exception as e:
                logger.warning(f"⚠️ PSI calculation skipped due to error: {e}")

        return MarsProfileReport(
            overview=self._format_output(overview_df),
            dq_tables=self._format_output(dq_tables),
            stats_tables=self._format_output(stat_tables)
        )
        
    # ==========================================================================
    # Overview Pipeline 
    # ==========================================================================
    def _calculate_overview(self, config: MarsProfileConfig) -> pl.DataFrame:
        """
        [Internal] 计算全量概览大宽表 (Overview Table)。

        该方法采用 **One-Pass (单次扫描)** 策略，通过构建向量化表达式一次性计算所有指标（DQ + Stats），
        并自动拼接元数据、分布图 (Sparklines)，最后对列顺序和数据类型进行标准化整形。

        Parameters
        ----------
        config : MarsProfileConfig
            配置对象。控制计算哪些统计指标 (stat_metrics) 以及是否生成分布图 (enable_sparkline)。

        Returns
        -------
        pl.DataFrame
            概览宽表。
            - Index: feature (特征名)
            - Columns: dtype, distribution, missing_rate, top1_value, mean, ...
        """
        cols = self.features
        
        # 1. 向量化计算所有基础指标 (One-Pass)
        stats: pl.DataFrame = self._analyze_cols_vectorized(cols, config)
        
        # 2. 拼接 dtype 信息
        dtype_df: pl.DataFrame = self._get_feature_dtypes()
        stats = stats.join(dtype_df, on="feature", how="left")
        
        # 3. [Feature] 计算迷你分布图 (Sparklines)
        if config.enable_sparkline:
            sparkline_df: pl.DataFrame = self._compute_all_sparklines(cols, config)
            if not sparkline_df.is_empty():
                stats = stats.join(sparkline_df, on="feature", how="left")
        
        # 4. 显式指定列顺序：Feature -> Dtype -> Distribution -> DQ -> Stats
        ideal_order: List[str] = [
            "feature", "dtype", 
            "distribution",  
            "missing_rate", "zeros_rate", "unique_rate", 
            "top1_ratio", "top1_value"
        ] + config.stat_metrics
        
        # 容错：只选择实际存在的列并保持 ideal_order 的顺序
        final_cols: List[str] = []
        seen = set()
        for c in ideal_order:
            if c in stats.columns and c not in seen:
                final_cols.append(c)
                seen.add(c)
        
        # 如果还有其他未定义的列，放到最后
        remaining_cols = [c for c in stats.columns if c not in seen]
        
        return stats.select(final_cols + remaining_cols).sort(["dtype", "feature"])
    
    def _analyze_cols_vectorized(self, cols: List[str], config: Optional[MarsProfileConfig] = None) -> pl.DataFrame:
        """
        [Internal] 全量指标向量化计算引擎 (Overview 核心)。
        
        该方法通过“分批次向量化 (Batch Vectorization)”策略，平衡了 Polars 的并行计算能力
        与查询优化器 (Query Planner) 的解析开销。

        Core Logic
        ---------------------
        1. **分批执行 (Batching)**: 
           针对高维数据（如 5000+ 列），如果一次性生成数万个表达式，Polars 的 Query Planner 
           会因为逻辑图过于复杂而导致解析时间呈指数级上升。通过 `overview_batch_size` 
           将特征分块处理，可以有效避免这种“逻辑图爆炸”。

        2. **One-Pass 聚合**:
           在每个批次内部，通过构建巨大的表达式列表，实现单次扫描 (Single Scan) 计算该批次
           所有列的所有指标（Missing, Mean, Max 等）。

        3. **整形重构 (Reshape)**:
           - **Unpivot (宽变长)**: 将聚合后的单行结果（极宽）展开为长表格式。
           - **Metadata Parsing**: 利用字符串分割解析出 `feature` 和 `metric` 元数据。
           - **Pivot (长变宽)**: 将指标重新透视为标准的画像格式。

        4. **结果合并**:
           将各批次的计算结果通过 `pl.concat` 进行垂直合并，并统一进行类型转换。

        Parameters
        ----------
        cols : List[str]
            待计算指标的特征名称列表。
        config : MarsProfileConfig, optional
            配置对象。控制具体的统计指标计算范围。

        Returns
        -------
        pl.DataFrame
            包含所有特征统计指标的画像宽表。
            结构: [feature, metric1, metric2, ...]
        """
        if not cols: 
            return pl.DataFrame()
            
        cfg = config if config else self.config
        all_batches: List[pl.DataFrame] = []

        # 获取批次大小配置
        batch_size = self.overview_batch_size

        # 1. 开启批次迭代
        for i in range(0, len(cols), batch_size):
            batch_cols = cols[i : i + batch_size]
            all_exprs = []
            
            # 2. 构建当前批次的表达式池
            for col in batch_cols:
                # 获取该列在 Config 下的所有基础指标表达式
                base_exprs = self._build_expressions(col, cfg)
                for expr in base_exprs:
                    # 获取表达式的原始名称 (如 "mean", "missing_rate")
                    metric_name = expr.meta.output_name()
                    # 关键步骤：使用 ::: 作为分隔符编码元数据，便于后续解析
                    all_exprs.append(expr.alias(f"{col}:::{metric_name}"))

            # 3. 执行批次聚合 (计算 1 行结果)
            batch_raw = self.df.select(all_exprs)
            
            # 4. 立即执行 Reshape 操作，释放内存并降维
            # unpivot: [1 row x (Batch * Metrics) cols] -> [(Batch * Metrics) rows x 2 cols]
            batch_long = batch_raw.unpivot(variable_name="temp_id", value_name="value")
            
            # 5. 解析编码在 temp_id 中的特征名和指标名
            batch_pivoted = (
                batch_long
                .with_columns(
                    # 快速字符串分割：从 "age:::mean" 提取 ["age", "mean"]
                    pl.col("temp_id").str.split_exact(":::", 1)
                    .struct.rename_fields(["feature", "metric"])
                    .alias("meta")
                )
                .unnest("meta")
                # 透视：将 metric 列的值转为结果表的列名
                .pivot(on="metric", index="feature", values="value", aggregate_function="first")
            )
            all_batches.append(batch_pivoted)

        # 6. 垂直合并所有批次的结果集 (Horizontal Partitioning Concatenation)
        pivoted = pl.concat(all_batches)
        
        # 7. 类型标准化：将指标列统一转为 Float64
        # 排除 feature (String) 和 top1_value (Mixed String)
        cols_to_cast = [c for c in pivoted.columns if c not in ["feature", "top1_value"]]
        
        if cols_to_cast:
            pivoted = pivoted.with_columns([
                pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_cast
            ])
            
        return pivoted
    
    

    def _compute_all_sparklines(self, cols: List[str], config: MarsProfileConfig) -> pl.DataFrame:
        """
        [Internal] 批量计算数值列的迷你分布图 (Polars Native V3 + 并行优化版)。

        [优化]
        1. **Single-Sample Strategy**: 改为单次整体采样，避免高维场景下数千次重复采样产生的 I/O 瓶颈。
        2. **Thread-Level Parallelism**: 引入 ThreadPoolExecutor。由于 Polars 底层释放 GIL，
        并行计算 5000+ 列的直方图可获得近乎线性的加速。

        使用 Polars 原生 API (`series.hist`) 进行直方图统计，并映射为 Unicode 字符画。
        相比 Numpy 方案，减少了数据拷贝，并在处理缺失值和边缘情况时更加鲁棒。

        **分布图符号说明 (Visual Representation)**:
        -----------------------------------------
        * **正常分布**: 使用 Unicode 方块字符表示频率高低 (如 ``_ ▂▅▇█``)。
        - **0 值**: 强制使用下划线 ``_`` 作为基准线，确保视觉占位。
        - **非 0 值**: 使用 2/8 到 8/8 高度的方块 (``▂`` 到 ``█``)，跳过 1/8 块以增强可视性。
        
        * **无有效数据**: 显示全下划线 ``________``。
        - 场景: 原始列全为 Null/NaN，或者所有值都在 `custom_missing` 列表中。
        
        * **逻辑无分布**: 显示全下划线 ``________`` (并记录 Debug 日志)。
        - 场景: 数据存在 (len>0) 但无法构建直方图 (如全为无穷大 Inf)。
        
        * **单一值 (Constant)**: 显示居中方块 ``____█____``。
        - 场景: 方差为 0，所有有效值都相等。
        
        * **计算异常**: 显示 ``ERR``。

        Parameters
        ----------
        cols : List[str]
            待计算的列名列表。方法内部会自动筛选出数值型列。
        config : MarsProfileConfig
            包含 `sparkline_sample_size` (采样数) 和 `sparkline_bins` (字符画长度/分箱数) 配置。

        Returns
        -------
        pl.DataFrame
            包含 [feature, distribution] 的两列 DataFrame。
        """
        # 1. 筛选数值列 (非数值列无法画分布图)
        num_cols: List[str] = [c for c in cols if self._is_numeric(c)]
        if not num_cols:
            return pl.DataFrame(
                {"feature": [], "distribution": []}, 
                schema={"feature": pl.String, "distribution": pl.String}
            )

        # 2. 采样 (Sampling) - [性能优化: 单次整体采样]
        limit_n = config.sparkline_sample_size
        
        # 策略：先 Select 目标列 -> 执行单次全列采样
        # 这样避免了在循环中反复调用 sample() 带来的内存洗牌开销
        df_subset = self.df.select(num_cols)
        if df_subset.height > limit_n:
            sample_df = df_subset.sample(n=limit_n, with_replacement=False)
        else:
            sample_df = df_subset

        # 预加载参数
        bars = ['_', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588']
        n_bins: int = config.sparkline_bins

        # 3. 定义单列处理函数 (用于并行映射)
        def _process_column(col: str) -> Dict[str, str]:
            dist_str: str = "-" 
            try:
                # 获取清洗逻辑（排除 -999 等自定义缺失值）
                exclude_vals = self._get_values_to_exclude(col)
                target_s: pl.Series = sample_df[col]
                
                # --- A. 数据清洗 ---
                if target_s.dtype in [pl.Float32, pl.Float64]:
                    target_s = target_s.filter(target_s.is_not_nan())
                
                if exclude_vals:
                    target_s = target_s.filter(~target_s.is_in(exclude_vals))
                
                s: pl.Series = target_s.drop_nulls()
                
                # --- B. 边界检查 ---
                if s.len() == 0:
                    dist_str = "_" * n_bins 
                elif s.len() == 1 or s.min() == s.max():
                    dist_str = "____█____" 
                else:
                    # --- C. 核心计算 (Polars Hist) ---
                    hist_df: pl.DataFrame = s.hist(bin_count=n_bins)
                    counts: List[int] = hist_df.get_column(hist_df.columns[-1]).to_list()
                    
                    # --- D. 字符映射算法 ---
                    max_count = max(counts)
                    if max_count == 0:
                        dist_str = "_" * n_bins
                    else:
                        chars = []
                        for c in counts:
                            if c == 0:
                                chars.append(bars[0])
                            else:
                                idx = int(c / max_count * (len(bars) - 2)) + 1
                                idx = min(idx, len(bars) - 1)
                                chars.append(bars[idx])
                        dist_str = "".join(chars)
                        
            except Exception as e:
                logger.error(f"Sparkline calculation failed for feature '{col}': {str(e)}")
                dist_str = "ERR"
                
            return {"feature": col, "distribution": dist_str}

        # 4. [性能优化: 多核并发执行]
        # pl.thread_pool_size() 会自动识别当前系统的逻辑核心数。
        with ThreadPoolExecutor(max_workers=pl.thread_pool_size()-1) as executor:
            results = list(executor.map(_process_column, num_cols))

        return pl.DataFrame(
            results, 
            schema={"feature": pl.String, "distribution": pl.String}
        )
        
    # ==========================================================================
    # Trends & Pivot Pipeline 
    # ==========================================================================
    def _generate_pivot_report(
        self, metric: str, 
        group_col: Optional[str], 
        context_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        [Internal] 生成指定指标的分组趋势透视表 (Pivot Table)。

        该方法负责计算单一指标（如 'mean'）在不同时间切片或客群下的变化趋势，
        并将结果整形为 "特征 x 分组" 的矩阵格式。

        Core Logic
        ---------------------
        由于 Polars 是列式存储 (Column-oriented)，统计计算通常产出 "1行 x N特征" 的结果。
        为了生成可读的报告，需要进行 **转置 (Transpose)** 操作：
        1. **Total 计算**: 对全量数据聚合 -> 转置 -> 得到 [feature, total] 列。
        2. **Group 计算**: 按 `group_col` 聚合 -> 转置 -> 得到 [feature, group_val_1, group_val_2...] 列。
        3. **合并**: 将 Total 列与 Group 列通过 feature JOIN，形成最终宽表。

        Parameters
        ----------
        metric : str
            待计算的指标名称 (例如 'missing', 'mean', 'max')。
            必须能够被 `_get_single_metric_expr` 解析。
        group_col : str, optional
            分组列名 (例如 'month', 'vintage')。
            - 如果为 None，则只计算并返回 Total 列。
            - 如果存在，结果表将包含该分组下列的所有取值作为列名。
        context_df : pl.DataFrame, optional
            上下文数据集。
            通常传入包含临时生成的自动聚合时间列（如 '_mars_auto_month'）的 DataFrame。
            如果为 None，则默认使用 `self.df`。

        Returns
        -------
        pl.DataFrame
            透视后的趋势宽表。
            - Index: feature (特征名)
            - Columns: feature, dtype, [group_val_1, ...], total
        """
        # 优先使用传入的上下文 DF，否则回退到 self.df
        target_df = context_df if context_df is not None else self.df
        
        target_cols = [c for c in self.features if c != group_col]
        if not target_cols: 
            return pl.DataFrame()

        # 1. 计算 Total 列 (全局聚合)
        total_exprs = [self._get_single_metric_expr(c, metric).alias(c) for c in target_cols]
        total_row = target_df.select(total_exprs)
        # Transpose: [1, n_feats] -> [n_feats, 1]
        total_df = total_row.transpose(include_header=True, header_name="feature", column_names=["total"])

        # 2. 准备基础表 (feature + dtype + total)
        dtype_df = self._get_feature_dtypes()
        base_df = total_df.join(dtype_df, on="feature", how="left")
        
        # Case A: 没有分组 -> 直接返回 Total 表
        if group_col is None:
            return base_df.select(["feature", "dtype", "total"]).sort(["dtype", "feature"])

        # Case B: 有分组 -> 计算 Pivot 并 Join
        agg_exprs = [self._get_single_metric_expr(c, metric).alias(c) for c in target_cols]
        # GroupBy -> Agg -> Sort
        #   结果形状: M个分组 x N个特征
        grouped =target_df.group_by(group_col).agg(agg_exprs).sort(group_col)
        grouped = grouped.with_columns(pl.col(group_col).cast(pl.String))
        
        # 再次转置
        #   输入: 
        #   month  | age | income
        #   202301 | 25  | 10000
        #   202302 | 26  | 12000
        #
        #   输出 (Transpose后):
        #   feature | 202301 | 202302
        #   age     | 25     | 26
        #   income  | 10000  | 12000
        pivot_df = grouped.transpose(include_header=True, header_name="feature", column_names=group_col)

        # 3. Join Together
        result = base_df.join(pivot_df, on="feature", how="left")
        
        # 4. 调整列顺序: feature, dtype, ...groups..., total
        fixed = {"feature", "dtype", "total"}
        groups = [c for c in result.columns if c not in fixed]
        final_order = ["feature", "dtype"] + groups + ["total"]
        
        return result.select(final_order).sort(["dtype", "feature"])
    
    def _add_stability_metrics(self, df: pl.DataFrame, exclude_cols: List[str]) -> pl.DataFrame:
        """
        [Internal] 计算行级稳定性指标：方差 (Var) 和 变异系数 (CV)。
        
        利用 Polars 的 list 算子进行水平聚合 (Horizontal Aggregation)。
        
        我们想计算每一行 (每个特征) 在不同 group 间的波动。
        Polars 主要是列式计算，行计算比较麻烦。
        这里用了一个技巧：`concat_list`。
        把所有月份列的值，合并成一列 list: [25, 26, ...]
        然后直接对这个 list 列算 std 和 mean。
        
        Parameters
        ----------
        df : pl.DataFrame
            包含分组数据的透视表。
        exclude_cols : List[str]
            需要排除的非数据列 (如 feature, dtype)。

        Returns
        -------
        pl.DataFrame
            增加了 group_var 和 group_cv 列的 DataFrame。
        """
        if df.is_empty(): return df
        
        # 锁定纯分组列 (排除 feature, dtype, total)
        calc_cols = [
            c for c in df.columns 
            if c not in exclude_cols and df[c].dtype in [pl.Float64, pl.Float32]
        ]
        if not calc_cols: return df

        epsilon = 1e-9 # 防止除以0
        
        return (
            df
            .with_columns(pl.concat_list(calc_cols).alias("_tmp")) # 将分组列压缩为 List
            .with_columns([
                pl.col("_tmp").list.mean().fill_null(0).alias("group_mean"),
                pl.col("_tmp").list.var().fill_null(0).alias("group_var"),
                (pl.col("_tmp").list.std() / (pl.col("_tmp").list.mean().abs() + epsilon)).fill_null(0).alias("group_cv")
            ])
            .drop("_tmp")
            # 调整列顺序: feature, dtype, groups..., total, var, cv
            .select(["feature", "dtype"] + calc_cols + ["total", "group_mean", "group_var", "group_cv"])
        )
        
    # ==========================================================================
    # PSI Pipeline
    # ==========================================================================
    @time_it
    def _get_psi_trend(
        self, 
        group_col: str, 
        features: Optional[List[str]] = None,
        context_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        [Internal] 计算特征在分组维度上的 PSI (群体稳定性指标) 趋势及稳定性统计。

        该方法利用 Polars 的 Streaming 引擎和 Lazy API，高效地计算数值型和类别型特征
        随时间或客群的变化趋势。

        Core Logic
        ---------------------
        1. **基准选择 (Baseline)**: 
           自动选取 `group_col` 中值最小的分组（例如最早的月份）作为基准分布 (Expected)，
           其他所有分组作为实际分布 (Actual) 进行对比计算。

        2. **分箱策略 (Binning)**:
           - 数值特征: 使用 `MarsNativeBinner` 进行分箱 (默认 Quantile)。
           - 类别特征: 直接按类别值进行分布统计。

        3. **稳定性指标与门控机制 (Stability & Gating)**:
           计算 PSI 序列的均值 (Mean) 和变异系数 (CV)。
           **注意**: 为了解决 "小基数陷阱" (即 PSI 数值极小时，微小的抖动导致 CV 虚高)，
           引入了 `psi_cv_ignore_threshold` (在 __init__ 中定义):
           - **逻辑**: 如果某特征在**所有分组**下的 PSI 最大值 (`group_max`) 都小于该阈值，
             则认为该特征处于"绝对稳定区/噪声区"，强制将其 `group_cv` 置为 0。
           - 只有当历史数据中至少有一次 PSI 突破阈值时，才计算并展示真实的 CV。

        Parameters
        ----------
        group_col : str
            分组列名。通常是时间列 (如 'month') 或 Vintage 列。
        features : List[str], optional
            指定需要计算 PSI 的特征子集。如果为 None，则计算 `self.features` 中的所有列。
        context_df : pl.DataFrame, optional
            上下文数据集。通常传入包含临时生成的自动聚合时间列的 DataFrame。
            如果为 None，则使用实例内部的 `self.df`。

        Returns
        -------
        pl.DataFrame
            PSI 趋势宽表 (Pivot Table)。
            结构: [feature, dtype, group_val_1, group_val_2, ..., total, group_mean, group_var, group_cv]
            
        Notes
        -----
        - 架构优化: 采用 "Lazy Batching" 策略。分批将特征推入 Lazy 管道，
          并在 `_calc_psi_from_stats` 中完成最终合并，减少了频繁 collect 带来的内存碎片。
          
        Benchmarks:
        ---------
        - 性能: 在 5000 列 x 20 万行规模下 (i7-14700, 64G RAM), 采用全 Lazy 流水线后
        实现约 20% 的加速 (150s -> 120s)。
        - 瓶颈: 计算瓶颈已从内存实例化转向纯逻辑运算, 有效利用了多核并行能力。
        """
        target_df: pl.DataFrame = context_df if context_df is not None else self.df
        
        # ==============================================================================
        # 🛡️ [新增] 内存保护熔断机制 (Sanity Check)
        # ==============================================================================
        # PSI 矩阵是通过 Cross Join 生成骨架的。
        # 如果 group_col 误传了高基数主键(如 user_id)，会导致 (N_feat * N_bins * N_users) 的内存爆炸。
        # 设定一个安全阈值，例如 1000 (足以覆盖几十年的月份或常用的 Segment)
        MAX_PSI_GROUPS = 1000  
        
        # 快速检查分组数量 (使用 approx_n_unique 极速估算，或者直接 count unique)
        n_groups = target_df.select(pl.col(group_col).n_unique()).item()
        
        if n_groups > MAX_PSI_GROUPS:
            logger.error(f"❌ PSI Calculation aborted: Column '{group_col}' has {n_groups} unique values.")
            logger.error(f"   Threshold is {MAX_PSI_GROUPS}. Did you accidentally group by an ID column?")
            # 返回空表，避免程序 Crash，让报告其他部分能正常生成
            return pl.DataFrame()
        # ==============================================================================
        
        # 1. 确定计算范围
        candidates = features if features else self.features
        candidates = [c for c in candidates if c != group_col]
        
        if not candidates:
            return pl.DataFrame()

        num_cols = [c for c in candidates if self._is_numeric(c)]
        cat_cols = [c for c in candidates if c not in num_cols]

        try:
            baseline_group = target_df.select(pl.col(group_col).min()).item()
        except Exception:
            return pl.DataFrame()

        psi_result_parts = []
        common_schema_order = [group_col, "feature", "total", "psi"]
        
        BATCH_SIZE = self.psi_batch_size 
        # ==============================================================================
        # 🟢 路一：数值特征 PSI 
        # ==============================================================================
        if num_cols:
            try:
                numeric_missing = [v for v in self.custom_missing if isinstance(v, (int, float)) and not isinstance(v, bool)]
                numeric_special = [v for v in self.special_values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                
                # 1. Fit Global
                binner = MarsNativeBinner(
                    features=num_cols,
                    method=self.psi_bin_method, 
                    n_bins=self.psi_n_bins,          
                    special_values=numeric_special,
                    missing_values=numeric_missing,
                    remove_empty_bins=False     
                )
                binner.fit(target_df)
                
                # 预构建骨架所需的 Bin IDs
                possible_bins = list(range(self.psi_n_bins)) + [-1]
                if numeric_special:
                    possible_bins.extend([-3 - i for i in range(len(numeric_special))])
                b_ids = pl.DataFrame({"bin_id": possible_bins}, schema={"bin_id": pl.Int16})

                # 2. 分批处理 Loop
                for i in range(0, len(num_cols), BATCH_SIZE):
                    batch_cols = num_cols[i : i + BATCH_SIZE]
                    
                    # --- Local Scope Start ---
                    
                    # A. Transform (Input: Eager, Output: Lazy)
                    cols_needed = batch_cols + [group_col]
                    
                    # 这里直接传入 Eager DataFrame Slice
                    # Polars 的 select 是零拷贝的，不会复制数据，所以这里很快且内存安全
                    df_batch_input = target_df.select(cols_needed)
                    
                    # 开启 lazy=True，让 transform 内部转为 lazy 模式执行逻辑，避免生成巨大的中间结果矩阵
                    lf_binned: pl.LazyFrame = binner.transform(df_batch_input, return_type='index', lazy=True)
                    
                    # B. 构建当前批次的 Rename Map
                    feat_map_batch = {idx: name for idx, name in enumerate(batch_cols)}
                    bin_cols_batch = [f"{c}_bin" for c in batch_cols]
                    rename_map = {old: str(idx) for idx, old in enumerate(bin_cols_batch)}
                    
                    # C. [修改] 去掉 .collect()，保持为 LazyFrame
                    lf_agg_stats_batch = (
                        lf_binned
                        .rename(rename_map)
                        .select([group_col] + list(rename_map.values()))
                        .unpivot(
                            index=[group_col],
                            on=list(rename_map.values()),
                            variable_name="feat_idx", 
                            value_name="bin_id"
                        )
                        .with_columns([
                            pl.col("feat_idx").cast(pl.Int16),
                            pl.col("bin_id").cast(pl.Int16) 
                        ])
                        .group_by([group_col, "feat_idx", "bin_id"])
                        .len()  # 此时返回的是 LazyFrame
                    )

                    # D. [修改] 骨架构建也改为 Lazy (去掉 Eager 的 DataFrame 创建)
                    lf_f_ids = pl.LazyFrame({"feat_idx": list(feat_map_batch.keys())}, schema={"feat_idx": pl.Int16})
                    lf_b_ids = b_ids.lazy() # 假设之前的 b_ids 是 eager 的，转为 lazy
                    lf_unique_bins_skel = lf_f_ids.join(lf_b_ids, how="cross")

                    # E. [调用] 传入 LazyFrame，得到 LazyFrame 结果
                    # 注意：这里我们不再传入 skeleton，让 _calc_psi_from_stats 内部自己生成完整的 [分组x特征x分箱] 骨架
                    lf_psi_num_raw = self._calc_psi_from_stats(
                        stats_df=lf_agg_stats_batch, 
                        unique_bins_skel=lf_unique_bins_skel, 
                        group_col=group_col, 
                        baseline_group=baseline_group
                    )

                    # F. [还原] 保持 Lazy 链条
                    mapping_df = pl.LazyFrame({
                        "feat_idx": list(feat_map_batch.keys()),
                        "feature": list(feat_map_batch.values())
                    }, schema={"feat_idx": pl.Int16, "feature": pl.String})

                    psi_num_final = (
                        lf_psi_num_raw
                        .join(mapping_df, on="feat_idx", how="left")
                        .select(common_schema_order)
                        .collect(streaming=True) # 只有在最后合并前才 collect
                    )
                    
                    psi_result_parts.append(psi_num_final)
                    # --- Local Scope End ---

            except Exception as e:
                logger.error(f"Numeric PSI failed: {e}")

        # ==============================================================================
        # 🟡 路二：类别特征 PSI 
        # ==============================================================================
        if cat_cols:
            try:
                # 1. 构建聚合统计 LazyFrame (不 collect)
                lf_long_cat: pl.LazyFrame = (
                    target_df.lazy()
                    .select(cat_cols + [group_col])
                    .unpivot(
                        index=[group_col],
                        on=cat_cols,
                        variable_name="feature",
                        value_name="bin_id_raw"
                    )
                    .with_columns(
                        pl.col("bin_id_raw").fill_null("Missing").cast(pl.Utf8).alias("bin_id")
                    )
                    .group_by([group_col, "feature", "bin_id"])
                    .len()
                )

                # 2. 生成 [特征 x 类别] 的唯一组合骨架 (Lazy)
                lf_unique_bins_cat = lf_long_cat.select(["feature", "bin_id"]).unique()

                # 3. 调用重构后的计算函数 (传入 LazyFrame)
                # 注意：由于我们优化了 _calc_psi_from_stats，不再需要手动传入 skeleton_cat
                lf_psi_cat_raw = self._calc_psi_from_stats(
                    stats_df=lf_long_cat, 
                    unique_bins_skel=lf_unique_bins_cat, 
                    group_col=group_col, 
                    baseline_group=baseline_group
                )

                # 4. 执行并存入结果集
                # 在这里进行 collect 是为了将结果存入 list，方便最后的 pl.concat
                psi_cat_final = (
                    lf_psi_cat_raw
                    .select(common_schema_order)
                    .collect(streaming=True)
                )
                psi_result_parts.append(psi_cat_final)

            except Exception as e:
                logger.error(f"Categorical PSI failed: {e}")

        # ==============================================================================
        # 🏁 合并与整形
        # ==============================================================================
        if not psi_result_parts:
            return pl.DataFrame()

        final_long_psi: pl.DataFrame = pl.concat(psi_result_parts)

        # Pivot
        pivot_df = (
            final_long_psi
            .pivot(on=group_col, index=["feature", "total"], values="psi")
        )

        dtype_df = self._get_feature_dtypes()
        result = pivot_df.join(dtype_df, on="feature", how="left")
        
        raw_group_cols = [c for c in result.columns if c not in ["feature", "dtype", "total"]]
        psi_data_cols = sorted(raw_group_cols)
        
        if psi_data_cols:
            epsilon_stat = 1e-9
            result = (
                result
                .with_columns(pl.concat_list(psi_data_cols).alias("_tmp_psi_list"))
                .with_columns([
                    pl.col("_tmp_psi_list").list.mean().alias("group_mean"),
                    pl.col("_tmp_psi_list").list.max().fill_null(0).alias("group_max"),
                    
                    pl.col("_tmp_psi_list").list.var().fill_null(0).alias("group_var"),
                    pl.col("_tmp_psi_list").list.std().alias("_tmp_std") 
                ])
                .with_columns([
                    # 只有当历史上出现过的最大 PSI 都小于阈值时，才忽略波动(CV=0)
                    # 只要有一次 PSI 超过阈值，就老老实实计算 CV
                    pl.when(pl.col("group_max") < self.psi_cv_ignore_threshold)
                    .then(pl.lit(0.0))
                    .otherwise(
                        (pl.col("_tmp_std") / (pl.col("group_mean") + epsilon_stat))
                    )
                    .fill_null(0)
                    .alias("group_cv")
                ])
                .drop(["_tmp_psi_list", "_tmp_std", "group_max"]) 
            )
            
            final_order = ["feature", "dtype"] + psi_data_cols + ["total", "group_mean", "group_var", "group_cv"]
            return result.select(final_order).sort("feature")
        else:
            return result.sort("feature")

    def _calc_psi_from_stats(
        self, 
        stats_df: pl.LazyFrame,  
        unique_bins_skel: pl.LazyFrame, 
        group_col: str, 
        baseline_group: Any
    ) -> pl.LazyFrame: # 返回 LazyFrame
        """
        [Math Core] 基于聚合后的频次表 (Count Table) 计算 PSI。

        此方法不接触原始明细数据，直接在聚合后的统计表上进行向量化运算，
        是 PSI 计算高性能的核心所在。

        Formula
        --------
        1. **Expected (E)**: 基准组 (Baseline) 中各箱的占比。
        2. **Actual (A)**: 当前组 (Group) 中各箱的占比。
        3. **PSI Contribution**: (A - E) * ln(A / E)
        4. **Sum**: 对所有箱求和得到最终 PSI。

        Parameters
        ----------
        stats_df : pl.LazyFrame
            聚合后的频次统计表。
            结构必须包含: ``[group_col, feature (or feat_idx), bin_id, len]``。
        skeleton : pl.LazyFrame
            (Group x Feature x Bin) 的全排列骨架表。
            用于 Left Join 以确保计数为 0 的空箱不会丢失（会被填充 epsilon）。
        unique_bins_skel : pl.LazyFrame
            (Feature x Bin) 的唯一组合骨架表。
            用于计算全量数据 (Total) 的 PSI。
        group_col : str
            分组列的名称 (例如 'month', 'vintage')。
        baseline_group : Any
            基准组的具体取值 (例如 '2023-01')。
            该组的数据分布将作为 Expected 分布。

        Returns
        -------
        pl.LazyFrame
            计算结果宽表，包含 feature, group_psi, total_psi 等信息。
            
        Implementation Details
        ----------------------
        - 骨架机制: 函数内部通过 cross join 动态生成 [Group x Feature x Bin] 的全量 Lazy 骨架，
          强制执行零频格填充 (Epsilon filling)，确保在数据漂移极端（某分箱完全消失）时
          公式仍具备数值稳定性。
        - 性能: 输入输出均为 pl.LazyFrame，允许 Polars 优化器进行谓词下推和并行加速。
        """
        feat_col = "feat_idx" if "feat_idx" in stats_df.collect_schema().names() else "feature"
        epsilon = 1e-6
        div_safe = 1e-9  # 防止分母为 0 的保险系数

        # 1. 自动生成完整的 [分组 x 特征 x 分箱] 骨架 (Lazy)
        # 这样可以确保每个分组里都有完整的特征分箱名单
        unique_groups = stats_df.select(group_col).unique()
        full_skeleton = unique_bins_skel.join(unique_groups, how="cross")

        # 2. 计算基准分布 (Expected)
        expected = (
            stats_df
            .filter(pl.col(group_col) == baseline_group)
            .with_columns(
                (pl.col("len") / (pl.col("len").sum().over(feat_col) + div_safe)).alias("E")
            )
            .select([feat_col, "bin_id", "E"])
        )

        # 3. 计算实际分布 (Actual)
        actual = (
            stats_df
            .with_columns(
                (pl.col("len") / (pl.col("len").sum().over([group_col, feat_col]) + div_safe)).alias("A")
            )
            .select([group_col, feat_col, "bin_id", "A"])
        )

        # 4. 计算全量分布 (Global Actual)
        global_actual = (
            stats_df
            .group_by([feat_col, "bin_id"])
            .agg(pl.col("len").sum().alias("total_len"))
            .with_columns(
                (pl.col("total_len") / (pl.col("total_len").sum().over(feat_col) + div_safe)).alias("A_global")
            )
            .select([feat_col, "bin_id", "A_global"])
        )

        # 5. 核心计算逻辑 (Left Join 骨架并填充 epsilon)
        # 计算分组 PSI
        psi_group = (
            full_skeleton
            .join(actual, on=[group_col, feat_col, "bin_id"], how="left")
            .join(expected, on=[feat_col, "bin_id"], how="left")
            .with_columns([
                pl.col("A").fill_null(epsilon),
                pl.col("E").fill_null(epsilon)
            ])
            .with_columns(
                ((pl.col("A") - pl.col("E")) * (pl.col("A") / pl.col("E")).log()).alias("psi_contrib")
            )
            .group_by([group_col, feat_col])
            .agg(pl.col("psi_contrib").sum().alias("psi"))
        )

        # 计算全量 PSI
        psi_total = (
            unique_bins_skel
            .join(global_actual, on=[feat_col, "bin_id"], how="left")
            .join(expected, on=[feat_col, "bin_id"], how="left")
            .with_columns([
                pl.col("A_global").fill_null(epsilon),
                pl.col("E").fill_null(epsilon)
            ])
            .with_columns(
                ((pl.col("A_global") - pl.col("E")) * (pl.col("A_global") / pl.col("E")).log()).alias("psi_contrib_total")
            )
            .group_by(feat_col)
            .agg(pl.col("psi_contrib_total").sum().alias("total"))
        )

        return psi_group.join(psi_total, on=feat_col, how="left")


    # =========================================================================
    # Expression Factories 
    # =========================================================================
    def _build_expressions(self, col: str, config: MarsProfileConfig) -> List[pl.Expr]:
        """[Factory] 为单个列生成所有 Overview 指标的计算表达式。"""
        return self._get_full_stats_exprs(col, config)
    
    def _get_full_stats_exprs(self, col: str, config: MarsProfileConfig) -> List[pl.Expr]:
        """
        [Factory] 为单个列构建全量统计指标的 Polars 表达式列表。

        该方法封装了 Overview 计算的核心细节，包含以下关键逻辑：
        1. **大基数优化 (High Cardinality Opt)**: 
           对于超过 100w 行的数据集，自动将 `n_unique` 切换为 `approx_n_unique` (HyperLogLog)，
           在保持精度的同时极大降低内存消耗。
        2. **众数提取 (Mode Extraction)**:
           预先计算 `value_counts` 并提取 Top1 的结构体，同时获取其 **Value** (转换类型为 String) 
           和 **Ratio** (占比)，确保数据质量可见性。
        3. **动态指标生成**:
           根据 Config 中的 `stat_metrics` 动态生成均值、方差等统计表达式，
           并自动应用 `_get_metric_expr` 中的缺失值剔除逻辑。

        Returns
        -------
        List[pl.Expr]
            包含该列所有待计算指标的表达式列表。
        """
        
        total_len = pl.len()
        is_num = self._is_numeric(col)
        exprs = []

        # --- 动态生成 DQ 指标 (根据 config.dq_metrics 过滤) ---
        dq_targets = config.dq_metrics
        
        if "missing" in dq_targets:
            # [核心修复] 构建 联合缺失条件: 原生 Null | (如果是数值则包含 NaN) | 自定义缺失值
            missing_cond = pl.col(col).is_null()
            if is_num:
                missing_cond |= pl.col(col).is_nan()
            
            valid_missing = self._get_valid_missing(col)
            if valid_missing:
                missing_cond |= pl.col(col).is_in(valid_missing)
                
            exprs.append((missing_cond.sum() / total_len).alias("missing_rate"))
            
        if "zeros" in dq_targets:
            zeros_c = (pl.col(col) == 0).sum() if is_num else pl.lit(0, dtype=pl.UInt32)
            exprs.append((zeros_c / total_len).alias("zeros_rate"))
            
        if "unique" in dq_targets:
            if self.df.height > 1_000_000:
                unique_count_expr = pl.col(col).approx_n_unique()
            else:
                unique_count_expr = pl.col(col).n_unique()
            exprs.append((unique_count_expr / total_len).alias("unique_rate"))
            
        if "top1" in dq_targets:
            # 预计算 Top1 结构体 (避免重复写 value_counts 逻辑)
            # value_counts 返回 struct: {col_name: value, count: int}
            top1_struct = pl.col(col).value_counts(sort=True).first()
            
            exprs.append(
                (
                    pl.col(col)                         # 1. 选中目标列 (假设列名叫 "city")
                    
                    .value_counts(sort=True)            # 2. 统计每个值出现的次数，并按次数从多到少排序
                                                        #    返回数据格式 (List[Struct]): 
                                                        #    [{"city": "北京", "count": 100},  <-- 第1行 (次数最多)
                                                        #     {"city": "上海", "count": 80},   <-- 第2行
                                                        #     ...]

                    .first()                            # 3. 只取排序后的第 1 行数据 (也就是众数的那一行)
                                                        #    返回数据格式 (Struct): 
                                                        #    {"city": "北京", "count": 100}

                    .struct.field("count")              # 4. 从这个结构体(Struct)中，只提取 "count" 这个字段的值
                                                        #    返回数据: 100

                    / total_len                         # 5. 除以总行数 (例如总共有 1000 行)
                                                        #    计算: 100 / 1000 = 0.1

                ).alias("top1_ratio")                    # 6. 给这个计算结果起个名字叫 "top1_ratio"
            )
            
            # 增加 Top1 Value (众数具体值)
            # 强制转为 String，防止与 Mean/Std 等数值指标在 pivot 时发生类型冲突
            exprs.append(top1_struct.struct.field(col).cast(pl.Utf8).alias("top1_value"))

        # 动态统计指标 (基于 Config)
        # 数值统计指标
        if is_num:
            # 遍历 config.stat_metrics 动态生成
            for metric in config.stat_metrics:
                if metric.lower() == "psi":
                    # PSI 需要特殊处理，跳过
                    continue
                expr = self._get_metric_expr(col, metric)
                if expr is not None:
                    exprs.append(expr.alias(metric))
        # 非数值列，直接填充 Null
        else:
            null_lit = pl.lit(None, dtype=pl.Float64)
            for metric in config.stat_metrics:
                if metric.lower() == "psi":
                    # PSI 需要特殊处理，跳过
                    continue
                exprs.append(null_lit.alias(metric))
        return exprs

    def _get_single_metric_expr(self, col: str, metric_type: str) -> pl.Expr:
        """[Factory] 为单个列生成指定指标的计算表达式 (用于 Pivot)。"""
        return self._get_metric_expr(col, metric_type)
    
    def _get_metric_expr(self, col: str, metric_type: str) -> pl.Expr:
        """
        [Factory] 生成单个指标的计算表达式。

        **特殊值处理逻辑 (Special Values Handling)**:
        1. **DQ 指标 (Missing/Unique/Top1)**: 基于全量数据计算。特殊值会被视为“值”参与 Unique/Top1 统计，或被归为 Missing。
        2. **统计指标 (Mean/Std/Quantile)**: 基于剔除特殊值后的“净数据”计算。防止 -999 拉低均值或扭曲分布。

        Parameters
        ----------
        col : str
            目标列名。
        metric_type : str
            指标名称。

        Returns
        -------
        pl.Expr
            Polars 表达式对象。
        """
        # 1. 获取该列对应的特殊值/缺失值列表
        valid_missing = self._get_valid_missing(col)
        
        # 2. 定义基础列对象 (Raw Data)
        raw_col = pl.col(col)
        is_num = self._is_numeric(col)
        col_dtype = self._dtype_map.get(col)

        # ---------------------------------------------------------
        # Group A: 数据质量指标 (基于 Raw Data 计算)
        # ---------------------------------------------------------
        if metric_type == "missing":
            # 缺失率 = (原生 Null + NaN + 自定义特殊值) / 总行数
            # [修改] 增加对 NaN 的判定，因为 np.nan 在 Polars 中被识别为 NaN
            missing_cond = raw_col.is_null()
            if is_num and col_dtype in [pl.Float32, pl.Float64]:
                missing_cond |= raw_col.is_nan()
            
            if valid_missing:
                missing_cond |= raw_col.is_in(valid_missing)
                
            return missing_cond.sum() / pl.len()
        
        elif metric_type == "zeros":
            # 零值率 (物理意义上的 0)
            return (raw_col == 0).sum() / pl.len() if is_num else pl.lit(0, dtype=pl.UInt32)
        
        elif metric_type == "unique":
            # 唯一值数量 (包含特殊值)
            return raw_col.n_unique() / pl.len()
        
        elif metric_type == "top1":
            # 众数占比 (特殊值也可能成为众数，需暴露风险)
            return raw_col.value_counts(sort=True).first().struct.field("count") / pl.len()

        # ---------------------------------------------------------
        # Group B: 数值统计指标 (基于 Clean Data 计算)
        # ---------------------------------------------------------
            
        if not is_num: 
            return pl.lit(None)

        # 构建一个统一的布尔掩码 (Keep Mask)，而不是分步 filter
        # 初始条件：全部保留
        keep_mask = pl.lit(True)

        # 显式踢走 NaN，因为在 Polars 中，mean/std 遇到 NaN 会返回 NaN
        # 1. 如果是浮点数，必须排除 NaN
        if col_dtype in [pl.Float32, pl.Float64]:
            keep_mask &= raw_col.is_not_nan()

        # 2. 如果有自定义剔除值 (如 -999)，也叠加到掩码中
        exclude_vals = self._get_values_to_exclude(col)
        if exclude_vals:
            keep_mask &= ~raw_col.is_in(exclude_vals)

        # 3. 一次性应用掩码，确保上下文长度一致
        clean_col = raw_col.filter(keep_mask)

        mapper = {
            # 集中度
            "mean": clean_col.mean(),
            "median": clean_col.median(),
            "sum": clean_col.sum(),
            
            # 离散度
            "std": clean_col.std(),
            
            # 极值 (最小值如果是 -999 就没意义了，所以要用 clean_col)
            "min": clean_col.min(),
            "max": clean_col.max(),
            
            # 分位数
            "p25": clean_col.quantile(0.25),
            "p75": clean_col.quantile(0.75),
            
            # 分布形态
            "skew": clean_col.skew(),
            "kurtosis": clean_col.kurtosis()
        }
        
        return mapper.get(metric_type, pl.lit(None))

    # ==========================================================================
    # Helpers & Utilities
    # ==========================================================================
    def _get_feature_dtypes(self) -> pl.DataFrame:
        """[Helper] 获取 Schema 信息表"""
        schema = {"feature": [], "dtype": []}
        for n, d in self.df.schema.items():
            schema["feature"].append(n)
            schema["dtype"].append(str(d))
        return pl.DataFrame(schema)

    def _is_numeric(self, col: str) -> bool:
        """[Helper] 判断列是否为数值类型"""
        # 兼容 Polars 这里的类型判断
        dtype = self._dtype_map.get(col)
        return dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
                        pl.Float32, pl.Float64]

    def _get_valid_missing(self, col: str) -> List[Any]:
        """[Helper] 类型安全的缺失值匹配 (防止类型不匹配报错)"""
        # Polars 很严格，如果拿字符串 "unknown" 去过滤整数列，会崩。
        # 这个函数会检查当前列的类型，只返回类型匹配的自定义缺失值。
        if not self.custom_missing: 
            return []
        is_num = self._is_numeric(col)
        is_str = self.df[col].dtype == pl.String
        return [v for v in self.custom_missing if (is_num and isinstance(v, (int, float))) or (is_str and isinstance(v, str))]
    
    def _get_values_to_exclude(self, col: str) -> List[Any]:
        """
        [Helper] 获取当前列需要剔除的所有特定值 (类型安全)。

        该方法合并了实例的 `custom_missing` (自定义缺失值) 和 `special_values` (特殊值)，
        并根据目标列的物理类型 (`dtype`) 对值进行严格过滤。

        Polars 的比较算子 (`is_in`, `eq`) 是强类型的。如果尝试将字符串类型的值（如 "unknown"）
        应用于数值类型的列（如 `Int64`），程序会抛出异常。本方法确保后续过滤操作的类型安全性。

        Parameters
        ----------
        col : str
            目标列的名称。

        Returns
        -------
        List[Any]
            当前列中应当被视为“非正常数值”的列表。
            列表中的元素类型保证与 `col` 的数据类型兼容 (例如数值列只返回数值，字符串列只返回字符串)。
        """
        # 1. 合并两个列表 
        # 如果 self.special_values 还没定义，就用空列表代替
        special_vals = getattr(self, "special_values", [])
        candidates = self.custom_missing + special_vals
        
        if not candidates: 
            return []

        # 2. 获取列类型
        is_num = self._is_numeric(col)
        is_str = self.df[col].dtype == pl.String

        # 3. 类型安全过滤
        valid_values = []
        for v in candidates:
            # 只有当 值类型 与 列类型 匹配时，才加入列表
            if is_num and isinstance(v, (int, float)) and not isinstance(v, bool):
                valid_values.append(v)
            elif is_str and isinstance(v, str):
                valid_values.append(v)
                
        return valid_values

