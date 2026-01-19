from joblib import Parallel, delayed
from typing import List, Dict, Optional, Union, Any, Literal, Tuple, Set
import multiprocessing
import gc

import numpy as np
import pandas as pd
import polars as pl
from optbinning import OptimalBinning
from sklearn.tree import DecisionTreeClassifier

from mars.core.base import MarsTransformer
from mars.utils.logger import logger
from mars.utils.decorators import time_it

class MarsBinnerBase(MarsTransformer):
    """
    [分箱器抽象基类] MarsBinnerBase

    这是 Mars 特征工程体系中所有分箱组件的底层核心。它不仅定义了分箱器的状态契约，
    还封装了高度优化的转换（Transform）与分析（Profiling）算子。

    该基类采用了“计算与路由分离”的设计：
    - 计算：由子类实现的 `fit` 策略负责填充切点。
    - 路由：基类负责处理复杂的缺失值、特殊值、高基数类别路由以及 Eager/Lazy 混合执行。

    Attributes
    ----------
    bin_cuts_ : Dict[str, List[float]]
        数值型特征的物理切点。每个列表均以 `[-inf, ..., inf]` 闭合，确保全值域覆盖。
    cat_cuts_ : Dict[str, List[List[Any]]]
        类别型特征的分组映射规则。将零散的字符串/分类标签聚类为逻辑组。
    bin_mappings_ : Dict[str, Dict[int, str]]
        分箱可视化地图。将物理索引（如 -1, 0, 1）映射为业务可读标签（如 "Missing", "01_[0, 10)"）。
    bin_woes_ : Dict[str, Dict[int, float]]
        分箱权重字典。存储每个分箱索引对应的 WOE 值。
    feature_names_in_ : List[str]
        拟合时输入的原始特征列名。

    Notes
    -----
    **1. 极致性能架构**
    底层完全基于 Polars 的表达式引擎（Expression Engine）。在转换数千个特征时，基类会自动
    构建一个平坦化的计算图，通过单次 IO 扫描实现并行转换，规避了 Pandas 逐列循环的性能瓶颈。

    **2. 索引协议 (Index Protocol)**
    系统强制执行统一的索引协议以支持下游的风险监控（PSI/IV）：
    - `Missing`: -1
    - `Other`: -2
    - `Special`: -3, -4, ...
    - `Normal`: 0, 1, 2, ...

    **3. 内存与稳定性**
    内置“延迟物化（Lazy Materialization）”与“分批执行（Batch Execution）”机制，
    确保在处理亿级数据或超宽表时，内存曲线保持平稳，防止因计算图深度溢出导致的系统崩溃。
    """

    # 类型常量：用于快速判定数值列
    NUMERIC_DTYPES: Set[pl.DataType] = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
        pl.Float32, pl.Float64
    }

    # 索引协议常量
    IDX_MISSING: int = -1
    IDX_OTHER: int = -2
    IDX_SPECIAL_START: int = -3

    def __init__(
        self,
        features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        n_bins: int = 5,
        special_values: Optional[List[Union[int, float, str]]] = None,
        missing_values: Optional[List[Union[int, float, str]]] = None,
        join_threshold: int = 100,
        n_jobs: int = -1
    ) -> None:
        """
        初始化分箱器基类，配置全局业务规则与并行策略。

        Parameters
        ----------
        features : List[str], optional
            数值型特征白名单。若为空，子类通常会自动识别输入数据中的数值列。
        cat_features : List[str], optional
            类别型特征白名单。明确指定哪些列应按字符串分组逻辑处理。
        n_bins : int, default=5
            期望的最大分箱数量。最终生成的箱数可能少于此值（受单调性约束或样本量影响）。
        special_values : List[Union[int, float, str]], optional
            特殊值列表。在部分场景中，某些特定取值（如 -999, -1）代表特定含义，
            会被强制分配到独立的负数索引分箱中，不参与正常区间的切分。
        missing_values : List[Union[int, float, str]], optional
            自定义缺失值列表。除了原生的 `null` 和 `NaN` 外，用户可指定其他代表缺失的值。
        join_threshold : int, default=100
            **性能调优开关**。在 `transform` 阶段：
            - 当类别特征的基数（Unique Values）低于此值时，使用内存级 `replace` 映射。
            - 当基数超过此值时，自动切换为 `Hash Join` 模式。
            *这能有效防止因构建过深的逻辑分支树（When-Then Tree）导致的计算图解析缓慢。*
        n_jobs : int, default=-1
            并行计算的核心数：
            - `-1`: 自动使用 `CPU核心数 - 1`，预留一个核心保证系统响应。
            - `1`: 强制单线程模式，便于调试。
            - `N`: 使用指定的核心数。

        Notes
        -----
        初始化阶段不执行任何重型计算。所有计算资源（进程池、线程池）均在 `fit` 阶段按需按需申请。
        """
        super().__init__()
        self.features = features if features is not None else []
        self.cat_features = cat_features if cat_features is not None else []
        self.n_bins = n_bins
        self.special_values = special_values if special_values is not None else []
        self.missing_values = missing_values if missing_values is not None else []
        self.join_threshold = join_threshold
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1) if n_jobs == -1 else n_jobs

        # 状态属性初始化
        self.bin_cuts_: Dict[str, List[float]] = {}
        self.cat_cuts_: Dict[str, List[List[Any]]] = {}
        self.bin_mappings_: Dict[str, Dict[int, str]] = {}
        self.bin_woes_: Dict[str, Dict[int, float]] = {}

        # 缓存引用
        self._cache_X: Optional[pl.DataFrame] = None
        self._cache_y: Optional[Any] = None


    def _get_safe_values(self, dtype: pl.DataType, values: List[Any]) -> List[Any]:
        """
        [Helper] 跨引擎类型安全清洗函数。

        在强类型引擎（如 Polars/Rust）中，类型不匹配是导致崩溃的主要原因。该方法通过预扫描 
        Schema，确保用户定义的业务逻辑（缺失值、特殊值）与数据的物理存储类型保持绝对兼容。

        Parameters
        ----------
        dtype : polars.DataType
            当前处理列的原始数据类型。
        values : List[Any]
            用户在配置中指定的数值列表（如 [-999, 'unknown', None]）。

        Returns
        -------
        List[Any]
            经过物理类型对齐后的清洗列表。

        Notes
        -----
        **1. 严格过滤机制 (Numeric Path)**
        若目标列为数值型，系统会剔除所有非数值项。特别地，由于 Python 中 `True == 1`，
        系统会显式排除布尔类型，防止逻辑误判导致的异常成箱。

        **2. 宽容转换机制 (String/Categorical Path)**
        若目标列为非数值型，系统会将所有配置项强制转换为字符串。这保证了在进行 
        `is_in` 操作或 `join` 操作时，比较操作发生在相同的物理类型之上。

        **3. 空值剥离**
        `None` 和 `np.nan` 会在此阶段被剥离，转由 `is_null()` 和 `is_nan()` 算子在 
        Polars 内核中进行更高效率的处理。
        """
        if not values:
            return []
            
        is_numeric = dtype in self.NUMERIC_DTYPES
        safe_vals = []
        
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            
            if is_numeric:
                # 数值列：严格保留数值，剔除 bool (True==1 歧义) 和字符串
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    safe_vals.append(v)
            else:
                # 非数值列：宽容处理，全部转为字符串以匹配 Categorical/String 列
                safe_vals.append(str(v))
                
        return safe_vals
    def get_bin_mapping(self, col: str) -> Dict[int, str]:
        """获取指定列的分箱映射字典。"""
        return self.bin_mappings_.get(col, {})

    def _is_numeric(self, series: pl.Series) -> bool:
        """Helper: 判断 Series 是否为数值类型。"""
        if series.dtype == pl.Null:
            return False
        return series.dtype in self.NUMERIC_DTYPES

    @time_it
    def _materialize_woe(self, batch_size: int = 200) -> None:
        """
        [WOE 物化计算引擎] - 内存与性能的平衡器。

        该方法负责将分箱后的统计分布转换为证据权重（WOE）。针对超宽表（>2000列），
        采用了“局部物化”策略，以防止计算图过载导致的内存溢出。

        Parameters
        ----------
        batch_size : int, default=200
            分批处理的特征数量。批次越小，内存占用越稳；批次越大，IO 吞吐越高。

        Notes
        -----
        **1. 避免计算图爆炸 (Lazy-to-Eager Transition)**
        如果直接在 2000 列上链式调用 `transform` 产生 WOE，Polars 会构建一个拥有数万个节点
        的计算图，导致解析开销超过计算开销。该方法通过分批 `collect`（物化），强制清空
        内存并切断计算图链条。

        **2. 向量化计算协议**
        - 使用 `group_by` 算子一次性获取每个箱子的坏人（Bad）和总数（Total）。
        - 引入平滑因子（1e-6）防止在纯好人/纯坏人箱子中出现 `log(0)` 或除以零异常。
        - 公式：$$WOE_i = \ln\left(\frac{Bad\_Dist_i}{Good\_Dist_i}\right)$$

        **3. 引用缓存与垃圾回收**
        在每一批次结束后，显式调用 `gc.collect()`。这在处理大数据集（如 500w+ 行）时，
        对于维持主进程的稳定性至关重要。
        """
        if self._cache_X is None or self._cache_y is None:
            logger.warning("No training data cached. WOE cannot be computed.")
            return

        logger.info("⚡ [Auto-Trigger] Materializing WOE (Eager Cross-Grouping Mode)...")
        y_name = "_y_tmp"
        
        y_series = pl.Series(name=y_name, values=self._cache_y)
        total_bads = y_series.sum()
        total_goods = len(y_series) - total_bads
        
        # 涵盖数值和类别特征
        bin_cols_orig = [c for c in self.bin_cuts_.keys()] + \
                        (list(self.cat_cuts_.keys()) if hasattr(self, 'cat_cuts_') else [])

        for i in range(0, len(bin_cols_orig), batch_size):
            batch_features = bin_cols_orig[i : i + batch_size]
            
            # Step A: 局部 Eager 转换 (获取 Index)
            X_batch_bin = self.transform(
                self._cache_X.select(batch_features), 
                return_type="index", 
                lazy=False
            )
            X_batch_bin = X_batch_bin.with_columns(y_series)
            
            # Step B: 逐列聚合计算 bad/good
            for c in batch_features:
                c_bin = f"{c}_bin"
                stats = (
                    X_batch_bin.group_by(c_bin)
                    .agg([
                        pl.col(y_name).sum().alias("b"),
                        pl.count().alias("n")
                    ])
                )
                
                # Step C: 向量化计算 WOE 并存入字典
                idxs = stats.get_column(c_bin)
                b = stats.get_column("b")
                n = stats.get_column("n")
                
                woe_vals = (((b + 1e-6) / (total_bads + 1e-6)) / 
                            (((n - b) + 1e-6) / (total_goods + 1e-6))).log()
                
                self.bin_woes_[c] = dict(zip(idxs.to_list(), woe_vals.to_list()))
            
            # Step D: 强制内存断层
            del X_batch_bin
            gc.collect()

    def _transform_impl(
        self, 
        X: Union[pl.DataFrame, pl.LazyFrame], 
        return_type: Literal["index", "label", "woe"] = "index",
        woe_batch_size: int = 200,
        lazy: bool = False
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        [混合动力分箱转换实现] 
        核心转换逻辑，兼容数值与类别特征，支持 Eager 与 Lazy 模式。

        该方法采用了“表达式瀑布流 (Expression Waterfall)”设计，通过 Polars 的原生算子实现
        了高效的向量化转换。针对高基数类别特征，采用了 Join 优化策略以规避深层逻辑树带来的性能损耗。

        Parameters
        ----------
        X : Union[pl.DataFrame, pl.LazyFrame]
            待转换的数据集。支持延迟计算流 (LazyFrame) 以优化长流水线性能。
        return_type : {'index', 'label', 'woe'}, default='index'
            转换后的输出格式：
            - 'index': 输出分箱索引（Int16 类型）。
            - 'label': 输出分箱的可读标签（Utf8 类型，如 "01_[10.5, 20.0)"）。
            - 'woe': 输出对应的证据权重 (Weight of Evidence) 值（Float64 类型）。
        woe_batch_size : int, default=200
            仅在 return_type='woe' 且未预计算 WOE 时有效。指定并行计算 WOE 的批大小。
        lazy : bool, default=False
            是否保持延迟执行状态。若为 True，则无论输入是 Eager 还是 Lazy，均返回 LazyFrame。

        Returns
        -------
        Union[pl.DataFrame, pl.LazyFrame]
            转换后的数据集。原列保持不变，新增以 `_bin` 或 `_woe` 为后缀的转换列。

        Notes
        -----
        **1. 分箱索引协议 (Index Protocol)**
        为了确保与下游 Profiler 和 PSI 计算算子对齐，系统采用以下固定索引：
        - `IDX_MISSING (-1)`: 缺失值及自定义缺失值。
        - `IDX_OTHER (-2)`: 类别型特征中的未见类别 (Unseen categories)。
        - `IDX_SPECIAL_START (-3)`: 特殊值分箱起始索引（向负无穷延伸）。
        - `[0, N]`: 正常数值区间或类别分组索引。

        **2. 数值型转换 (Numeric Pipeline)**
        采用层级覆盖逻辑：
        - 预处理：利用 `_get_safe_values` 确保缺失值/特殊值的类型与列 Schema 严格一致。
        - 核心：使用 `pl.cut` 进行向量化区间划分。
        - 组合：通过 `pl.when().then()` 瀑布流，按照 "缺失值 -> 特殊值 -> 正常区间" 的优先级进行合并。

        **3. 类别型转换算法 (Categorical Pipeline)**
        针对类别特征采用双路径优化：
        - **路径 A (低基数)**: 使用 `replace` 算子进行内存级映射，速度极快。
        - **路径 B (高基数)**: 当类别数超过 `join_threshold` 时，自动转为 `Join` 模式。
            这避免了构建数千个 `when-then` 分支导致的逻辑树深度爆炸（Stack Overflow 风险），
            将逻辑判断转化为哈希连接操作，极大提升了宽表转换效率。

        **4. 自动路由与路由安全**
        - 在进行 Utf8 类型操作（如类别分组）前，系统会自动创建临时 Utf8 缓存列。
        - 转换结束后，会自动清理所有产生的中间 Join 列和临时缓存列，保证输出 Schema 纯净。

        Example
        -------
        >>> binner = MarsOptimalBinner(...)
        >>> binner.fit(train_df, y)
        >>> # 返回带 WOE 值的 LazyFrame
        >>> woe_lazy = binner.transform(test_df, return_type="woe", lazy=True)
        """
        exprs = []
        temp_join_cols = []
        
        # 索引协议常量: 与下游 Profiler 对齐
        IDX_MISSING = -1
        IDX_OTHER   = -2
        IDX_SPECIAL_START = -3

        # 自动触发 WOE 计算
        if return_type == "woe" and not self.bin_woes_:
            self._materialize_woe(woe_batch_size)
    
        # 获取 Schema
        schema_map = X.collect_schema() if isinstance(X, pl.LazyFrame) else X.schema
        current_columns = schema_map.names()
        
        all_train_cols = list(set(
            list(self.bin_cuts_.keys()) + 
            (list(self.cat_cuts_.keys()) if hasattr(self, 'cat_cuts_') else [])
        ))

        for col in all_train_cols:
            if col not in current_columns: 
                continue
            
            # 计算类型安全值, 防止例如在 Int 列上查询 "unknown" 导致的崩溃
            col_dtype = schema_map[col]
            safe_missing_vals: List[int|float] = self._get_safe_values(col_dtype, self.missing_values)
            safe_special_vals: List[int|float] = self._get_safe_values(col_dtype, self.special_values)
            is_numeric_col = col_dtype in self.NUMERIC_DTYPES

            # =========================================================
            # Part A: 数值型分箱 (Numeric Binning)
            # =========================================================
            if col in self.bin_cuts_:
                cuts = self.bin_cuts_[col]
                
                # 1. 缺失值逻辑: Is Null OR Is Missing Val
                missing_cond = pl.col(col).is_null() 
                if is_numeric_col: 
                    missing_cond |= pl.col(col).cast(pl.Float64).is_nan()
                for v in safe_missing_vals: 
                    missing_cond |= (pl.col(col) == v)
                # ⭐构建缺失值分箱表达式
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 2. 正常分箱逻辑: Cut
                breaks = cuts[1:-1] if len(cuts) > 2 else []
                
                col_mapping: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"} # 分箱标签映射表 IDX -> Label
                
                # 2.1 处理无切点情况
                if not breaks:
                    col_mapping[0] = "00_[-inf, inf)"
                    layer_normal = pl.lit(0, dtype=pl.Int16)
                else:
                    for i in range(len(cuts) - 1):
                        low, high = cuts[i], cuts[i+1]
                        col_mapping[i] = f"{i:02d}_[{low:.3g}, {high:.3g})"
                    # 显式生成 labels 确保 cast(Int16) 成功，修复 PSI=0 Bug
                    bin_labels: List[str] = [str(i) for i in range(len(breaks) + 1)]
                    # ⭐ 构建正常分箱表达式
                    layer_normal = pl.col(col).cut(
                        breaks, labels=bin_labels, left_closed=True
                    ).cast(pl.Int16)
                
                # 3. 特殊值逻辑: 瀑布流覆盖
                current_branch = layer_normal
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i 
                        col_mapping[idx] = f"Special_{v}"
                        # 注意这里的覆盖顺序: 后定义的优先级更高
                        current_branch = pl.when(pl.col(col) == v).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)
                
                # 最终的分箱表达式: Missing -> Special -> Normal
                final_idx_expr = layer_missing.otherwise(current_branch)
                self.bin_mappings_[col] = col_mapping
                
            # =========================================================
            # Part B: 类别型分箱 (Categorical Binning)
            # =========================================================
            elif hasattr(self, 'cat_cuts_') and col in self.cat_cuts_:
                splits = self.cat_cuts_[col]
                cat_to_idx: Dict[str, int] = {}
                idx_to_label: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}
                
                # 【新增】默认路由索引，默认为 -2
                default_bin_idx = IDX_OTHER
                
                # 更新映射表
                if safe_special_vals:
                    for i, val in enumerate(safe_special_vals):
                        idx_to_label[IDX_SPECIAL_START - i] = f"Special_{val}"

                for i, group in enumerate(splits):
                    disp_grp = group[:3] if len(group) > 3 else group
                    suffix = ",..." if len(group) > 3 else ""
                    idx_to_label[i] = f"{i:02d}_[{','.join(str(g) for g in disp_grp) + suffix}]"
                    for val in group: 
                        val_str = str(val)
                        cat_to_idx[val_str] = i
                        # 【新增】如果训练时这一箱含有 "Other_Pre"，则将其设为默认箱
                        if val_str == "Other_Pre":
                            default_bin_idx = i
                
                self.bin_mappings_[col] = idx_to_label
                # 强转 String，确保类别匹配安全
                target_col = pl.col(col).cast(pl.Utf8)
                
                # 1. 缺失值
                missing_cond = target_col.is_null()
                for v in safe_missing_vals:
                    missing_cond |= (target_col == str(v))
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 2. 特殊值
                current_branch = pl.lit(IDX_OTHER, dtype=pl.Int16)
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        current_branch = (
                            pl.when(target_col == str(v))
                            .then(pl.lit(idx, dtype=pl.Int16))
                            .otherwise(current_branch)
                        )
                
                # 3. 路由: Join (高基数) vs Replace (低基数)
                # Join 模式避免了在表达式中构建巨大的 when-then 树，极大提升性能
                # if len(cat_to_idx) > self.join_threshold:
                #     map_df = pl.DataFrame({
                #         "_k": list(cat_to_idx.keys()), 
                #         f"_idx_{col}": list(cat_to_idx.values())
                #     }).with_columns([
                #         pl.col("_k").cast(pl.Utf8),
                #         pl.col(f"_idx_{col}").cast(pl.Int16)
                #     ])
                #     # 兼容 Lazy 模式的 Join
                #     join_tbl = map_df.lazy() if isinstance(X, pl.LazyFrame) else map_df
                #     X = X.join(join_tbl, left_on=target_col, right_on="_k", how="left")
                    
                #     temp_join_cols.append(f"_idx_{col}")
                target_col_name = col
                if col_dtype != pl.Utf8:
                    target_col_name = f"_{col}_utf8_tmp"
                    X = X.with_columns(pl.col(col).cast(pl.Utf8).alias(target_col_name))

                if len(cat_to_idx) > self.join_threshold:
                    map_df = pl.DataFrame({
                        "_k": list(cat_to_idx.keys()), 
                        f"_idx_{col}": list(cat_to_idx.values())
                    }).cast({"_k": pl.Utf8, f"_idx_{col}": pl.Int16})
                    
                    # left_on 必须传字符串
                    X = X.join(map_df, left_on=target_col_name, right_on="_k", how="left")
                    temp_join_cols.append(f"_idx_{col}")
                    if target_col_name != col: temp_join_cols.append(target_col_name)
                    
                    layer_normal = pl.col(f"_idx_{col}")
                else:
                    layer_normal = target_col.replace(cat_to_idx, default=None).cast(pl.Int16)
                
                # 组合: Missing -> Normal (Join Result) -> Special/Other
                final_idx_expr = layer_missing.otherwise(
                    pl.when(layer_normal.is_not_null()).then(layer_normal).otherwise(current_branch)
                )
            
            else:
                continue

            # 输出分发
            if return_type == "index":
                exprs.append(final_idx_expr.alias(f"{col}_bin"))
            elif return_type == "woe":
                woe_map = self.bin_woes_.get(col, {})
                exprs.append(
                    final_idx_expr.replace(woe_map).cast(pl.Float64).alias(f"{col}_woe") if woe_map 
                    else pl.lit(0.0).alias(f"{col}_woe")
                )
            else:
                str_map = {str(k): v for k, v in self.bin_mappings_.get(col, {}).items()}
                exprs.append(final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin"))

        # 清理 Join 产生的临时列
        return X.with_columns(exprs).drop(temp_join_cols).lazy() if lazy else X.with_columns(exprs).drop(temp_join_cols)

    @time_it
    def compute_bin_stats(self, X: pl.DataFrame, y: Any) -> pl.DataFrame:
        """
        [极速指标矩阵引擎] 产出全量分箱深度分析报告。

        这是 Mars 库中最具性能代表性的方法之一。它摒弃了传统的“逐列循环聚合”模式，
        转而采用“矩阵逆透视聚合（Matrix Unpivot Aggregation）”技术。

        Parameters
        ----------
        X : polars.DataFrame
            原始特征数据集。
        y : Any
            目标标签。

        Returns
        -------
        polars.DataFrame
            包含各特征、各分箱详细指标的报表，包含：
            - 基础计数：count, bad, good
            - 占比指标：count_dist, bad_rate, lift
            - 风险分析：woe, bin_iv, total_iv
            - 稳定性分析：bin_ks

        Notes
        -----
        **1. 矩阵化聚合原理 (Unpivot Logic)**
        该方法将 `(N_rows * M_features)` 的宽表动态转换为 `(N_rows * M_features, 2)` 
        的长表（Unpivot/Melt）。这使得 Polars 可以利用单一的 `group_by(["feature", "bin_index"])` 
        查询计划在单次数据扫描中并行计算出成千上万个特征的所有指标。
        
        **2. 窗口算子计算 KS (Window Function KS)**
        利用 Polars 的窗口函数 `cum_sum().over("feature")` 实现组内累计分布计算。
        相比 Python 循环计算累计值，这能在 Rust 层实现零拷贝的偏移量累加。

        **3. Streaming 物化**
        在最后的 `collect(streaming=True)` 中开启流式处理。这意味着对于超过物理内存
        的数据量，Polars 也能通过外部排序和分块聚合产出结果。

        Example
        -------
        >>> stats = binner.compute_bin_stats(df, y)
        >>> # 查看 IV 大于 0.02 的有效特征
        >>> stats.filter(pl.col("total_iv") > 0.02)
        """
        y_name = "_target_tmp"
        # 强制开启 Lazy 转换以合并查询计划
        X_bin_lazy = self.transform(X, return_type="index", lazy=True)
        X_bin_lazy = X_bin_lazy.with_columns(pl.lit(np.array(y)).alias(y_name))
        
        # 获取全局统计量 (T, B)
        meta = X_bin_lazy.select([
            pl.count().alias("total_counts"),
            pl.col(y_name).sum().alias("total_bads")
        ]).collect()
        
        total_counts = meta[0, "total_counts"]
        total_bads = meta[0, "total_bads"]
        total_goods = total_counts - total_bads
        global_bad_rate = (total_bads / total_counts) if total_counts > 0 else 0
        
        current_cols = X_bin_lazy.collect_schema().names()
        bin_cols = [c for c in current_cols if c.endswith("_bin")]

        # 利用 unpivot 实现矩阵化并行聚合计划
        # (rows * cols) -> (rows * features, 2)
        lf_stats = (
            X_bin_lazy.unpivot(
                index=[y_name],
                on=bin_cols,
                variable_name="feature",
                value_name="bin_index"
            )
            .group_by(["feature", "bin_index"])
            .agg([
                pl.count().alias("count"),
                pl.col(y_name).sum().alias("bad")
            ])
            .with_columns([
                (pl.col("count") - pl.col("bad")).alias("good")
            ])
        )

        # 向量化计算各项指标 (WOE, IV, Lift)
        lf_stats = lf_stats.with_columns([
            (pl.col("count") / total_counts).alias("count_dist"),
            (pl.col("bad") / pl.col("count")).alias("bad_rate"),
            (pl.col("bad") / (total_bads + 1e-6)).alias("bad_dist"),
            (pl.col("good") / (total_goods + 1e-6)).alias("good_dist")
        ]).with_columns([
            ((pl.col("bad_dist") + 1e-6) / (pl.col("good_dist") + 1e-6)).log().alias("woe")
        ]).with_columns([
            ((pl.col("bad_dist") - pl.col("good_dist")) * pl.col("woe")).alias("bin_iv"),
            (pl.col("bad_rate") / (global_bad_rate + 1e-6)).alias("lift")
        ])

        # Window Function 计算 KS (基于特征内累计)
        lf_stats = lf_stats.sort(["feature", "bin_index"]).with_columns([
            pl.col("bad_dist").cum_sum().over("feature").alias("cum_bad_dist"),
            pl.col("good_dist").cum_sum().over("feature").alias("cum_good_dist")
        ]).with_columns([
            (pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs().alias("bin_ks")
        ])

        # 最终物理物化
        stats_df: pl.DataFrame = lf_stats.collect(streaming=True)
        
        # 关联标签并计算总 IV
        final_list = []
        for feat_name in bin_cols:
            orig_name = feat_name.replace("_bin", "")
            mapping = self.get_bin_mapping(orig_name)
            
            feat_stats = stats_df.filter(pl.col("feature") == feat_name).with_columns([
                pl.col("bin_index").cast(pl.Utf8).replace({str(k): v for k, v in mapping.items()}).alias("bin_label")
            ])
            
            # 同时同步 WOE 字典，方便后续 transform 使用
            self.bin_woes_[orig_name] = dict(zip(feat_stats["bin_index"].to_list(), feat_stats["woe"].to_list()))
            final_list.append(feat_stats)

        result_df: pl.DataFrame = pl.concat(final_list)
        iv_sum = result_df.group_by("feature").agg(pl.col("bin_iv").sum().alias("total_iv"))
        return result_df.join(iv_sum, on="feature")


class MarsNativeBinner(MarsBinnerBase):
    """
    [极速原生分箱器] MarsNativeBinner
    
    基于 Polars (数据预处理) 与 Scikit-learn (计算内核) 构建的高性能特征分箱引擎。
    该类旨在解决大规模宽表（数千维特征）在传统 Python 分箱工具中运行缓慢的问题。
    
    支持三种分箱策略：Quantile、Uniform 和 CART 分箱，适用于数值型特征。

    Attributes
    ----------
    bin_cuts_ : Dict[str, List[float]]
        拟合后生成的数值型特征切点字典。格式：{特征名: [-inf, 切点1, ..., inf]}。
    fit_failures_ : Dict[str, str]
        记录训练过程中发生异常的特征及其错误原因。
    feature_names_in_ : List[str]
        训练时输入的特征名称列表。
    _is_fitted : bool
        标识分箱器是否已完成拟合。

    Notes
    -----
    1. 性能：在 20w 行 x 5000 列数据下，含有自定义缺失值和特殊值的情况下，单机i7 14700 24核：
        - Quantile 分箱：约 40 秒
        - Uniform 分箱：约 25 秒
        - CART 分箱：约 80 秒
    2. 鲁棒性：内置常量列识别、缺失值自动过滤及异常特征自动退化机制。
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        method: Literal["cart", "quantile", "uniform"] = "quantile",
        n_bins: int = 5,
        special_values: Optional[List[Union[int, float, str]]] = None,
        missing_values: Optional[List[Union[int, float, str]]] = None,
        min_samples: float = 0.05,
        min_bin_samples_multiplier: float = 10,
        cart_params: Optional[Dict[str, Any]] = None,
        remove_empty_bins: bool = False,
        join_threshold: int = 100,
        n_jobs: int = -1,
    ) -> None:
        """
        初始化 MarsNativeBinner。

        Parameters
        ----------
        features : List[str], optional
            指定需要进行分箱的数值特征列名。若为 None，则自动识别 X 中的所有数值列。
        method : {'cart', 'quantile', 'uniform'}, default='quantile'
            分箱策略：
            - 'cart': 基于决策树的最优分箱。
            - 'quantile': 等频分箱（推荐用于工业级预处理）。
            - 'uniform': 等宽分箱。
        n_bins : int, default=5
            目标最大分箱数。
        special_values : List[Union[int, float, str]], optional
            特殊值列表。这些值将被强制独立成箱（如：-999, -9999）。
        missing_values : List[Union[int, float, str]], optional
            自定义缺失值列表。默认 None, NaN 会自动识别并归为 Missing 箱。
        min_samples : float, default=0.05
            仅在 method='cart' 时有效。决策树叶子节点的最小样本占比。
        min_bin_samples_multiplier : float, default=10
            最小样本量安全水位系数。
            计算公式：`min_rows = n_bins * multiplier`。若特征有效样本量低于该值，则不切分。
        cart_params : Dict, optional
            透传给 sklearn.tree.DecisionTreeClassifier 的额外参数。
        remove_empty_bins : bool, default=False
            仅在 method='uniform' 时有效。是否自动剔除并合并样本量为 0 的空箱。
        join_threshold : int, default=100
            在 Transform 阶段，类别型特征使用 Join 替代 Replace 的基数阈值。
        n_jobs : int, default=-1
            并行计算的核心数。-1 表示使用所有可用核心。
        """
        super().__init__(
            features=features, n_bins=n_bins, 
            special_values=special_values, missing_values=missing_values,
            join_threshold=join_threshold, n_jobs=n_jobs
        )
        self.method = method
        self.min_samples = min_samples
        self.min_bin_samples_multiplier = min_bin_samples_multiplier
        self.remove_empty_bins = remove_empty_bins
        
        self.cart_params = {
            "class_weight": None, 
            "random_state": 42,
            "min_impurity_decrease": 0.0
        }
        if cart_params:
            self.cart_params.update(cart_params)

    @time_it
    def _fit_impl(self, X: pl.DataFrame, y: Optional[Any] = None, **kwargs) -> None:
        """
        执行分箱拟合的核心入口逻辑。

        负责任务分发前的三道防线：
        1. 自动识别并排除非数值列。
        2. 极速全表扫描获取 Min/Max，识别并排除常量列。
        3. 路由分发至不同的分箱策略方法。

        Parameters
        ----------
        X : polars.DataFrame
            经过基类归一化后的训练数据。
        y : polars.Series, optional
            目标变量。在使用 'cart' 方法时必填。
        **kwargs : dict
            透传的额外配置参数。
        """
        # 1. 缓存数据引用，仅用于 transform 阶段请求 return_type='woe' 时的延迟计算
        self._cache_X = X
        self._cache_y = y

        # 2. 确定目标列 (仅筛选数值列，忽略全空列)
        all_target_cols = self.features if self.features else X.columns
        target_cols: List[str] = []
        null_cols: List[str] = [] 

        for c in all_target_cols:
            if c not in X.columns: continue
            
            # 判定全空/Null类型列，记录下来以便直接注册为空箱
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 仅处理数值类型
            if self._is_numeric(X[c]):
                target_cols.append(c)

        # 注册全空列为空切点，防止 transform 时漏列
        for c in null_cols:
            self.bin_cuts_[c] = []

        if not target_cols:
            if not null_cols:
                logger.warning("No numeric columns found for binning.")
            return

        # ========================================================
        # 极速预过滤: 识别常量列并跳过
        # ========================================================
        valid_cols: List[str] = []
        
        # 构建聚合表达式，一次性扫描全表获取 Min/Max
        stats_exprs = []
        for c in target_cols:
            stats_exprs.append(pl.col(c).min().alias(f"{c}_min"))
            stats_exprs.append(pl.col(c).max().alias(f"{c}_max"))
            
        # 触发计算 (Eager 模式下立即执行，速度极快)
        stats_row = X.select(stats_exprs).row(0)
        
        for i, c in enumerate(target_cols):
            min_val = stats_row[i * 2]
            max_val = stats_row[i * 2 + 1]
            
            # 如果 Min == Max，说明是常量列，无需分箱，直接设为全区间
            if min_val == max_val:
                logger.warning(f"Feature '{c}' is constant. Skipped.")
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            
            valid_cols.append(c)

        if not valid_cols:
            return

        # 3. 检查 CART 方法的依赖
        if y is None and self.method == "cart":
            raise ValueError("Decision Tree Binning ('cart') requires target 'y'.")

        # 4. 策略分发
        if self.method == "quantile":
            self._fit_quantile(X, valid_cols)
        elif self.method == "uniform":
            self._fit_uniform(X, valid_cols)
        elif self.method == "cart":
            self._fit_cart_parallel(X, y, valid_cols)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_quantile(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行极速等频分箱 (One-Shot Quantile Query)。

        该方法摒弃了传统的“循环、筛选、计算”模式，转而利用 Polars 的延迟计算特性，
        将数千个特征的分位数计算合并为一个单一的原子查询计划（Atomic Query Plan）。

        Parameters
        ----------
        X : polars.DataFrame
            训练数据集。
        cols : List[str]
            需要执行等频分箱的数值型特征列名列表。

        Notes
        -----
        **核心优化：查询计划合并 (One-Shot Logic)**
        - 传统实现：针对 $N$ 个特征执行 $N$ 次 `quantile()` 调用，触发 $N$ 次内存扫描。
        - Mars 实现：构建一个扁平化的表达式列表 `[col1_q1, col1_q2, ..., colN_qM]`。
          通过 `X.select(q_exprs)` 将该列表一次性喂给 Rust 引擎。引擎会优化执行路径，
          在单次（或极少数次）内存扫描中并行完成所有特征的切点计算。

        **数据质量控制 (Data Quality)**
        - 源头隔离：在计算分位数前，利用 `pl.when().then(None)` 将 `special_values` 和 
          `missing_values` 临时替换为 `Null`，确保切点的分布仅由业务层面的“正常值”决定。
        - 自动去重：针对高偏态数据（如某些取值极度集中的分位数一致），会自动执行 `set()` 
          去重并重新排序，防止生成重复切点导致的 `Cut Error`。

        Algorithm
        ---------
        1. 根据 `n_bins` 生成分位点序列（如 [0.2, 0.4, 0.6, 0.8]）。
        2. 为每个特征构建类型安全的 `quantile` 表达式。
        3. 聚合所有表达式，执行单一的 `Eager` 模式查询。
        4. 解析结果矩阵，生成对应的分箱边界 $[-\infty, q_1, \dots, q_k, \infty]$。
        """
        # 1. 构建分位点
        if self.n_bins <= 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1].tolist()
        
        # 预处理排除值
        raw_exclude = self.special_values + self.missing_values
        
        # 2. 构建表达式列表
        q_exprs = []
        for c in cols:
            # 获取当前列安全的排除值
            safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)
            target_col = pl.col(c)
            # 如果有需要排除的值，在计算分位数前先置为 Null (不参与计算)
            if safe_exclude:
                target_col = pl.when(pl.col(c).is_in(safe_exclude)).then(None).otherwise(pl.col(c))
            
            for i, q in enumerate(quantiles):
                # 别名技巧: col:::idx，便于后续解析
                alias_name = f"{c}:::{i}"
                q_exprs.append(target_col.quantile(q).alias(alias_name))
        
        # 3. 触发计算 (One-Shot Query)
        stats = X.select(q_exprs)
        row = stats.row(0)
        
        # 4. 解析结果并去重排序
        temp_cuts: Dict[str, List[float]] = {c: [] for c in cols}
        
        for val, name in zip(row, stats.columns):
            c_name, _ = name.split(":::")
            if val is not None and not np.isnan(val):
                temp_cuts[c_name].append(val)

        for c in cols:
            cuts = sorted(list(set(temp_cuts[c]))) 
            self.bin_cuts_[c] = [float('-inf')] + cuts + [float('inf')]

    def _fit_uniform(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行极速等宽分箱 (Uniform/Step Binning)。

        该方法利用 Polars 的向量化算子，将所有特征的统计信息提取和切点生成分为两个物理阶段，
        在保证统计严谨性的同时，最大程度减少对原始数据的扫描次数。

        Parameters
        ----------
        X : polars.DataFrame
            训练数据集。
        cols : List[str]
            需要执行等宽分箱的数值型特征列名列表。

        Notes
        -----
        分箱逻辑分为以下两个核心阶段：

        **阶段 1：基础统计量聚合 (Global Scan)**
        - 构建一个全局查询计划，一次性计算所有目标列的 `min` (最小值)、`max` (最大值) 
          和 `n_unique` (唯一值个数)。
        - 排除逻辑：在计算极值前，会自动过滤用户定义的 `special_values` 和 `missing_values`，
          确保切点仅基于“正常”数值分布生成。
        - 低基数处理：若特征唯一值个数小于目标箱数 (`n_unique <= n_bins`)，则自动退化为
          基于唯一值中点的精确切分，防止生成重复切点。

        **阶段 2：空箱动态优化 (Empty Bin Refinement)**
        - 仅在 `remove_empty_bins=True` 时触发。
        - 机制：利用 Polars 的 `cut` 和 `value_counts` 算子，在主进程中并行嗅探初始等宽
          切点下的样本分布。
        - 压缩逻辑：识别样本量为 0 的区间，并将相邻的空箱进行物理合并。这在数据分布极端
          偏态（如长尾分布）时，能有效防止产生毫无意义的无效分箱。

        Algorithm
        ---------
        1. 针对每列特征 $x$，计算有效值范围 $[min, max]$。
        2. 计算步长 $\Delta = (max - min) / n\_bins$。
        3. 初始切点集 $C = \{min + i \cdot \Delta \mid i=1, \dots, n\_bins-1\}$。
        4. 若开启优化，则根据各区间真实频数 $N_i$ 重新调整切点集 $C'$。
        5. 最终输出格式：$[-\infty, \text{切点}_1, \dots, \text{切点}_k, \infty]$。

        Performance
        -----------
        由于采用了“计划合并 (Query Plan Fusion)”技术，无论处理 100 列还是 2000 列，
        对原始内存的扫描次数始终保持在极低水位（通常为 1-2 次全表扫描）。
        """
        raw_exclude = self.special_values + self.missing_values
        
        # 1. 基础统计量
        exprs = []
        col_safe_excludes = {} 

        for c in cols:
            safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)
            col_safe_excludes[c] = safe_exclude # 缓存供后续使用

            target_col = pl.col(c)
            if safe_exclude:
                target_col = target_col.filter(~pl.col(c).is_in(safe_exclude))
            
            exprs.append(target_col.min().alias(f"{c}_min"))
            exprs.append(target_col.max().alias(f"{c}_max"))
            exprs.append(target_col.n_unique().alias(f"{c}_n_unique"))

        stats = X.select(exprs)
        row = stats.row(0)
        
        initial_cuts_map = {}
        pending_optimization_cols = []

        # 解析统计量，生成等距切点
        for i, c in enumerate(cols):
            base_idx = i * 3
            min_val, max_val, n_unique = row[base_idx], row[base_idx + 1], row[base_idx + 2]
            safe_exclude = col_safe_excludes[c]

            if min_val is None or max_val is None:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            
            # 优化: 低基数检查 (Unique <= N_Bins)，直接取中点切分
            if n_unique <= self.n_bins:
                unique_vals = X.select(pl.col(c).unique().sort()).to_series().to_list()
                clean_vals = [v for v in unique_vals if v not in safe_exclude and v is not None]
                
                if len(clean_vals) <= 1:
                    self.bin_cuts_[c] = [float('-inf'), float('inf')]
                else:
                    mid_points = [(clean_vals[k] + clean_vals[k+1])/2 for k in range(len(clean_vals)-1)]
                    self.bin_cuts_[c] = [float('-inf')] + mid_points + [float('inf')]
                continue

            if min_val == max_val:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue

            # 生成等宽切点
            raw_cuts = np.linspace(min_val, max_val, self.n_bins + 1)[1:-1].tolist()
            full_cuts = [float('-inf')] + raw_cuts + [float('inf')]
            initial_cuts_map[c] = full_cuts
            
            if self.remove_empty_bins:
                pending_optimization_cols.append(c)
            else:
                self.bin_cuts_[c] = full_cuts

        # 2. 空箱优化 (可选)
        if pending_optimization_cols:
            batch_exprs = []
            for c in pending_optimization_cols:
                cuts = initial_cuts_map[c]
                breaks = cuts[1:-1]
                target_col = pl.col(c)
                safe_exclude = col_safe_excludes[c]
                
                if safe_exclude:
                    target_col = target_col.filter(~pl.col(c).is_in(safe_exclude))
                
                labels = [str(i) for i in range(len(breaks)+1)]
                
                # 批量计算直方图 (Value Counts)
                batch_exprs.append(
                    target_col.cut(breaks, labels=labels, left_closed=True)
                    .value_counts().implode().alias(f"{c}_counts")
                )

            batch_counts_df = X.select(batch_exprs)
            
            # 解析并剔除 Count=0 的箱
            for c in pending_optimization_cols:
                inner_series = batch_counts_df.get_column(f"{c}_counts")[0]
                keys = inner_series.struct.fields
                dist_list = inner_series.to_list()
                
                valid_indices = set()
                for row in dist_list:
                    # row 是 {'brk': '0', 'counts': 100} 格式
                    idx_val = row.get(keys[0])
                    cnt_val = row.get(keys[1])
                    if idx_val is not None and cnt_val > 0:
                        valid_indices.add(int(idx_val))
                
                cuts = initial_cuts_map[c]
                breaks = cuts[1:-1]
                new_cuts = [cuts[0]]
                for i in range(len(breaks) + 1):
                    if i in valid_indices: new_cuts.append(cuts[i+1])
                
                if new_cuts[-1] != float('inf'): new_cuts.append(float('inf'))
                self.bin_cuts_[c] = sorted(list(set(new_cuts)))

    def _fit_cart_parallel(self, X: pl.DataFrame, y: pl.Series, cols: List[str]) -> None:
        """
        执行并行的决策树最优分箱。

        该方法是 Mars 库的“动力心脏”，专门针对高 PCR (计算传输比) 任务设计。
        它通过“生产-消费”流水线模式，将 Polars 的预处理能力与 Sklearn 的拟合能力深度耦合。

        Parameters
        ----------
        X : polars.DataFrame
            特征数据集。
        y : polars.Series
            目标变量。要求已在基类中完成类型对齐（pl.Series）。
        cols : List[str]
            需要执行决策树分箱的特征列名列表。

        Notes
        -----
        **1. 计算重心前置 (Source Cleaning Pipeline)**
        - 在 `cart_task_gen` 生成器中，利用 Polars 的位运算内核极速完成空值和特殊值的过滤。
        - **异构对齐**：使用生成的 Numpy 掩码 (Mask) 同时对 $x$ 和 $y$ 进行物理切片，
          确保两端数据行索引在没有任何显式 Join 操作的情况下实现绝对对齐。

        **2. 混合并行调度 (Hybrid Parallel Strategy)**
        - 后端选择：采用 `threading` 后端配合 `n_jobs`。
        - 依据：由于 `x_clean` 和 `y_clean` 切片已在主进程内存中完成，使用多线程可实现
          **零拷贝 (Zero-Copy)** 传递给 Worker，规避了多进程频繁序列化大数据块的物流负担。
        - 锁优化：利用 Sklearn 底层在拟合过程中会释放 GIL 的物理特性，实现真正的多核利用。

        **3. 内存与统计防护机制**
        - 采样系数：通过 `min_bin_samples_multiplier` (10x) 强制检查有效样本量。
          若样本量不足以支撑统计可信的分箱（即有效样本 < n_bins * 10），则特征自动退化。
        - 异常追踪：引入 `fit_failures_` 属性。任何由于数据极端分布或内存溢出导致的
          单特征失败将被捕获并记录原因，而不会触发主任务的中断（Fail-Soft 机制）。

        Algorithm
        ---------
        1. 将标签 $y$ 转换为内存连续的 Numpy 数组，优化内存预取。
        2. 启动 `cart_task_gen`：逐列进行源头清洗，产出纯净切片对。
        3. 线程池调度：Worker 函数并行执行 `DecisionTreeClassifier.fit`。
        4. 汇总结果：提取树节点阈值，生成切点并记录异常。
        """
        
        # 1. [优化] Polars Series -> Numpy (使用 zero-copy)
        # y 已经是 pl.Series，直接 to_numpy 是最快的，并强制连续内存布局
        y_np = np.ascontiguousarray(y.to_numpy())
        
        if len(y_np) != X.height:
            raise ValueError(f"Target 'y' length mismatch: X({X.height}) vs y({len(y_np)})")

        def worker(col_name: str, x_clean_np: np.ndarray, y_clean_np: np.ndarray) -> Tuple[str, List[float]]:
            try:
                if len(x_clean_np) < self.n_bins * self.min_bin_samples_multiplier:
                    return col_name, [float('-inf'), float('inf')]
                
                cart = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=self.min_samples,
                    **self.cart_params
                )
                cart.fit(x_clean_np, y_clean_np)
                cuts = cart.tree_.threshold[cart.tree_.threshold != -2]
                cuts = np.sort(np.unique(cuts)).tolist()
                return col_name, [float('-inf')] + cuts + [float('inf')], None # 成功
            except Exception as e:
                # 捕获真实崩溃信息以便排查
                error_info = f"{type(e).__name__}: {str(e)}"
                return col_name, [float('-inf'), float('inf')], error_info

        raw_exclude = self.special_values + self.missing_values

        # 2. [优化] 任务生成器：移除生成器内部的冗余转换
        def cart_task_gen():
            for c in cols:
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

                series = X.get_column(c)

                # [优化] 使用更紧凑的位运算构建 Mask
                valid_mask = series.is_not_null()
                if col_dtype in self.NUMERIC_DTYPES:
                    valid_mask &= (~series.is_nan())
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))

                if not valid_mask.any():
                    continue

                # [优化] x 端利用 zero-copy 转换
                # valid_mask 在 Polars 中是 BitMap，filter 之后转 numpy 非常快
                x_clean = (
                    series
                    .filter(valid_mask)
                    # .cast(pl.Float32)
                    .to_numpy(writable=False)
                    .reshape(-1, 1)
                )
                if not x_clean.flags['C_CONTIGUOUS']:
                    x_clean = np.ascontiguousarray(x_clean)
                
                # y 端利用 Numpy 的视图切片
                y_clean = y_np[valid_mask.to_numpy()]

                yield c, x_clean, y_clean

        # 3. Backend 选型
        # 如果数据量极大，threading 会受限于 GIL。
        # 但因为 Sklearn 的树拟合大部分是在 C++ 层释放了 GIL 的，
        # 且任务分发开销（PCR）在第一阶段很低，所以 threading 是合理的。
        # 执行并行
        results = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
            delayed(worker)(name, x, y) for name, x, y in cart_task_gen()
        )
        
        # 状态存储
        self.fit_failures_ = {} # 用于记录失败原因
        
        for col_name, cuts, error_msg in results:
            self.bin_cuts_[col_name] = cuts
            if error_msg:
                self.fit_failures_[col_name] = error_msg

        # fit 结束后统一警告
        if self.fit_failures_:
            logger.warning(
                f"⚠️ {len(self.fit_failures_)} features failed during CART binning and fallbacked to single bin. "
                f"Check `self.fit_failures_` for details. Sample fails: {list(self.fit_failures_.items())[:3]}"
            )


class MarsOptimalBinner(MarsBinnerBase):
    """
    [混合动力最优分箱引擎] MarsOptimalBinner

    该类是 Mars 库的高级核心组件，将极速预分箱技术（Native Pre-binning）与基于数学规划的
    最优分箱算法（OptBinning）深度集成。它旨在为风控模型提供具备单调约束、最优 IV 分布和
    极强鲁棒性的特征切点。

    Attributes
    ----------
    bin_cuts_ : Dict[str, List[float]]
        数值型特征最终生成的切点字典。
    cat_cuts_ : Dict[str, List[List[Any]]]
        类别型特征的分组规则字典。
    fit_failures_ : Dict[str, str]
        记录求解器超时或计算失败的特征原因。

    Notes
    -----
    **核心架构：双阶段启发式求解 (Two-Stage Heuristic Solver)**
    1. **Stage 1: Native 粗切**：利用 `MarsNativeBinner` 快速将连续变量离散化为 50-100 个初始区间（Pre-bins）。这一步在主进程中通过 Polars 的 Rust 内核完成，实现了数据的极大压缩。
    2. **Stage 2: MIP/CP 精切**：将压缩后的统计量送入子进程，利用数学规划求解器在满足单调性、最小箱占比等约束下，寻找信息熵最大化的最优解。

    **混合并行策略 (Hybrid Parallel Strategy)**
    - **数值型处理**：采用 `loky` 后端。由于最优求解涉及复杂的 Python 胶水逻辑和外部求解器调用，通过多进程（Loky）彻底规避 GIL 锁，释放多核 CPU 算力。
    - **PCR 优化**：在任务生成阶段完成“源头清洗”，仅将“入参即干净”的高纯度 Numpy 数据传递给子进程，最大限度降低跨进程序列化开销。
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        n_bins: int = 5,
        n_prebins: int = 50,
        prebinning_method: Literal["quantile", "uniform", "cart"] = "cart",
        monotonic_trend: Literal["ascending", "descending", "auto", "auto_asc_desc"] = "auto_asc_desc",
        solver: Literal["cp", "mip"] = "cp",
        time_limit: int = 10,
        cat_cutoff: Optional[int] = 100,
        special_values: Optional[List[Any]] = None,
        missing_values: Optional[List[Any]] = None,
        join_threshold: int = 100,
        n_jobs: int = -1
    ) -> None:
        """
        初始化 MarsOptimalBinner。

        Parameters
        ----------
        features : List[str], optional
            数值型特征列名。
        cat_features : List[str], optional
            类别型特征列名。
        n_bins : int, default=5
            最终输出的最大分箱数。
        n_prebins : int, default=50
            预分箱阶段的区间数。预分箱越多，求解越慢但精度越高。
        prebinning_method : str, default='cart'
            预分箱策略。推荐 'cart'，因为它能更好地捕获局部非线性特征。
        monotonic_trend : str, default='auto_asc_desc'
            单调性约束：'ascending', 'descending', 'auto', 'auto_asc_desc'。
        solver : {'cp', 'mip'}, default='cp'
            OptBinning 的求解引擎。'cp' (Constraint Programming) 通常在复杂约束下更快。
        time_limit : int, default=10
            单特征求解的最大秒数限制，超时将自动回退至预分箱结果。
        cat_cutoff : int, optional
            类别型特征的 Top-K 截断阈值。高基数特征（如：手机型号）会被强制截断。
        special_values : List, optional
            特殊值列表（如 [-999, 9999]），将独立成箱。
        missing_values : List, optional
            缺失值定义列表，自动归为 Missing 箱。
        join_threshold : int, default=100
            Transform 阶段的性能开关：超过此基数将启用 Join 模式进行向量化转换。
        n_jobs : int, default=-1
            并行核心数。
        """
        super().__init__(
            features=features, cat_features=cat_features, n_bins=n_bins,
            special_values=special_values, missing_values=missing_values,
            join_threshold=join_threshold, n_jobs=n_jobs
        )
        self.n_prebins = n_prebins
        self.prebinning_method = prebinning_method
        self.monotonic_trend = monotonic_trend
        self.solver = solver
        self.time_limit = time_limit
        self.cat_cutoff = cat_cutoff
        
        # 尝试导入 optbinning
        try:
            import optbinning
        except ImportError:
            logger.warning("⚠️ 'optbinning' not installed. Fallback logic might be triggered.")

    def _fit_impl(self, X: pl.DataFrame, y: pl.Series = None, **kwargs) -> None:
        """
        [拟合引擎调度器] 自动执行特征识别与任务流分发。

        执行流程：
        1. 执行 `y` 的类型转换与内存连续化优化。
        2. 特征自动洗牌：将数值型特征分流至 `_fit_numerical_impl`，类别型分流至 
           `_fit_categorical_impl`。
        3. 状态管理：注册并初始化全空列的占位规则。

        Parameters
        ----------
        X : polars.DataFrame
            训练集特征数据。
        y : polars.Series
            目标变量。要求必须可转换为二分类的 int32 数组。
        """
        if y is None:
            raise ValueError("Optimal Binning requires target 'y' to calculate IV/WOE.")

        y_np = np.ascontiguousarray(y.to_numpy()).astype(np.int32)
        
        all_target_cols = self.features if self.features else X.columns
        cat_set = set(self.cat_features)
        
        num_cols = []
        cat_cols = []
        null_cols = [] 

        for c in all_target_cols:
            if c not in X.columns: continue
            
            # 1. 优先判定类别
            if c in cat_set:
                cat_cols.append(c)
                continue
            
            # 2. 判定全空
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 3. 判定数值
            if self._is_numeric(X[c]):
                num_cols.append(c)

        if not num_cols and not cat_cols and not null_cols:
            logger.warning("No valid numeric or categorical columns found.")
            return

        # 注册全空列
        for c in null_cols:
            self.bin_cuts_[c] = []

        if num_cols:
            self._fit_numerical_impl(X, y_np, num_cols)

        if cat_cols:
            self._fit_categorical_impl(X, y_np, cat_cols)

    @time_it
    def _fit_numerical_impl(self, X: pl.DataFrame, y_np: np.ndarray, num_cols: List[str]) -> None:
        """
        [Pipeline] 数值型特征的混合动力求解流水线。

        本方法体现了“分而治之”的设计哲学。

        Optimization
        ------------
        - **计算重心前置**：在 `num_task_gen` 内部利用 Polars Rust 引擎进行极速过滤，
          Worker 仅接收经过净化的 Numpy 视图。
        - **两阶段联动**：先调用 `MarsNativeBinner` 获取粗粒度切点，
          随后将其作为 `user_splits` 注入 `optbinning`，极大缩小了数学规划的搜索空间。
        - **并发控制**：使用 `loky` 后端。由于单个特征的最优求解耗时较长（$PCR \gg 0$），
          支付跨进程通讯成本以换取独立 CPU 核心的满载运行是非常合算的。

        Parameters
        ----------
        X : polars.DataFrame
            特征数据。
        y_np : numpy.ndarray
            已经过内存对齐和类型转换的标签数组。
        num_cols : List[str]
            待处理的数值列名。
        """
        # --- Stage 1: Pre-binning (Native) ---
        # 复用 Native 实现，速度极快
        pre_binner = MarsNativeBinner(
            features=num_cols,
            method=self.prebinning_method, 
            n_bins=self.n_prebins, 
            special_values=self.special_values,
            missing_values=self.missing_values,
            n_jobs=self.n_jobs,
            remove_empty_bins=False 
        )
        pre_binner.fit(X, y_np)
        pre_cuts_map = pre_binner.bin_cuts_

        # 筛选需要优化的列
        active_cols = []
        for col, cuts in pre_cuts_map.items():
            if len(cuts) > 2: 
                active_cols.append(col)
            else:
                self.bin_cuts_[col] = cuts 

        if not active_cols:
            return

        # --- Stage 2: Parallel Optimization ---
        def num_worker(col: str, pre_cuts: List[float], col_data: np.ndarray, y_data: np.ndarray) -> Tuple[str, List[float]]:
            fallback_res = (col, pre_cuts)
            try:
                if len(col_data) < 10 or np.var(col_data) < 1e-8:
                    return fallback_res

                # 提取预切点 (去除 inf)
                user_splits = np.array(pre_cuts[1:-1])
                if len(user_splits) == 0:
                    return fallback_res
                
                opt = OptimalBinning(
                    name=col, dtype="numerical", solver=self.solver,
                    monotonic_trend=self.monotonic_trend, user_splits=user_splits,  
                    max_n_bins=self.n_bins, time_limit=self.time_limit, min_bin_size=0.05,
                    verbose=False
                )
                opt.fit(col_data, y_data)
                
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    res_cuts = [float('-inf')] + list(opt.splits) + [float('inf')]
                    # 防止优化过度
                    if len(res_cuts) <= 2 and len(pre_cuts) > 2:
                        return fallback_res
                    return col, res_cuts
                
                return fallback_res 
            except Exception:
                return fallback_res

        raw_exclude = self.special_values + self.missing_values

        def num_task_gen():
            """
            通过 yield 纯净的 NumPy 数组，触发 joblib 的 mmap 共享内存优化。
            """
            for c in active_cols:
                # 1. 类型感知与安全过滤列表获取
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
                
                # 2. 获取 Series 指针 (不使用 select, 避免 DataFrame 物化开销)
                series = X.get_column(c)
                
                # 3. [高性能算子] 构建 Polars 过滤掩码 (Rust 内核级位运算)
                # 基础过滤：非 null
                valid_mask = series.is_not_null()
                
                # 针对数值特征增加：非 NaN 过滤
                if col_dtype in self.NUMERIC_DTYPES:
                    valid_mask &= (~series.is_nan())
                
                # 针对业务特殊值进行排除 (如 -999, -998)
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))
                
                # 4. 将位掩码转换为 NumPy 布尔数组 (用于 y 的快速切片)
                mask_np = valid_mask.to_numpy()
                
                # 如果过滤后样本量不足，直接跳过此列，减少并行开销
                if not mask_np.any():
                    continue

                # 5. [核心优化] 特征列 X 处理
                # 流程：Polars 过滤 -> 强转 Float32 -> 导出 NumPy 只读视图
                # Float32 相比 Float64 可减少一半的通讯带宽，且不影响分箱精度
                col_np = (
                    series.filter(valid_mask)
                    .cast(pl.Float32)
                    .to_numpy(writable=False)
                )

                # 确保内存连续性 (C_CONTIGUOUS)
                # 连续内存能让子进程在反序列化和读取时达到最高的 CPU 缓存命中率
                if not col_np.flags['C_CONTIGUOUS']:
                    col_np = np.ascontiguousarray(col_np)

                # 6. 标签列 Y 处理
                # 直接利用已经生成的 mask_np 在 NumPy 侧同步切片
                # y_np 已经在入口处执行过 ascontiguousarray，此处切片极快
                clean_y = y_np[mask_np]

                # 7. 产出任务数据包
                # 此时产出的全部是纯粹的物理内存块，joblib 会自动识别并优化传输
                yield c, pre_cuts_map[c], col_np, clean_y
        
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(num_worker)(c, cuts, data, y) for c, cuts, data, y in num_task_gen()
        )
        
        for col, cuts in results:
            self.bin_cuts_[col] = cuts

    def _fit_categorical_impl(self, X: pl.DataFrame, y_np: np.ndarray, cat_cols: List[str]) -> None:
        """
        [Pipeline] 类别型特征的处理流水线。

        特别针对大规模类别型数据进行了逻辑增强。

        Notes
        -----
        - **长尾截断路由 (Other_Pre)**：针对频数极低或基数极大的类别，自动执行 
          `Top-K` 截断，并将长尾数据归并为特殊的 `Other_Pre` 类别。
        - **数据源头净化**：在任务生成器中完成字符串映射和空值隔离，
          Worker 进程拿到的直接是满足 `optbinning` 输入要求的 `pl.Utf8` 映射数据。
        - **并行后端**：使用 `loky` 后端。

        Parameters
        ----------
        X : polars.DataFrame
            特征数据。
        y_np : numpy.ndarray
            标签数组。
        cat_cols : List[str]
            待处理的类别列名。
        """
        raw_exclude = self.special_values + self.missing_values

        # 1. 更加健壮的 Worker
        def cat_worker(col: str, clean_data: np.ndarray, clean_y: np.ndarray) -> Tuple[str, Optional[List[List[Any]]]]:
            try:
                # Top-K 预处理: 将长尾类别归为 "Other_Pre"
                if self.cat_cutoff is not None:
                    unique_vals, counts = np.unique(clean_data, return_counts=True)
                    if len(unique_vals) > self.cat_cutoff:
                        top_indices = np.argsort(-counts)[:self.cat_cutoff]
                        top_vals = set(unique_vals[top_indices])
                        mask_keep = np.isin(clean_data, list(top_vals))
                        clean_data = np.where(mask_keep, clean_data, "Other_Pre")

                opt = OptimalBinning(
                    name=col, dtype="categorical", solver=self.solver,
                    max_n_bins=self.n_bins, 
                    time_limit=self.time_limit,
                    cat_cutoff=0.05, 
                    verbose=False
                )
                opt.fit(clean_data, clean_y)
                
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    return col, opt.splits
                return col, None
            except Exception:
                return col, None

        # 2. 统一的源头清洗生成器
        def cat_task_gen():
            for c in cat_cols:
                series = X.get_column(c)
                col_dtype = series.dtype
                
                # 获取该列的安全排除列表
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
                
                # 过滤条件：非空 且 不在排除列表中
                valid_mask = series.is_not_null()
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))
                
                # 执行过滤
                clean_series = series.filter(valid_mask)
                if clean_series.len() == 0:
                    continue
                
                valid_mask_np = valid_mask.to_numpy() # 预转 Numpy 掩码
                col_data = clean_series.cast(pl.Utf8).to_numpy()
                clean_y = y_np[valid_mask_np] # 使用预转好的 mask

                yield c, col_data, clean_y

        # 执行并行
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(cat_worker)(c, data, y) for c, data, y in cat_task_gen()
        )
        
        for col, splits in results:
            if splits is not None:
                self.cat_cuts_[col] = splits