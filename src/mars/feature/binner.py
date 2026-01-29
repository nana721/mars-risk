from joblib import Parallel, delayed
from typing import List, Dict, Optional, Union, Any, Literal, Tuple, Set
import multiprocessing
import gc

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier

from mars.core.base import MarsTransformer
from mars.utils.logger import logger
from mars.utils.decorators import time_it

class MarsBinnerBase(MarsTransformer):
    """
    [分箱器抽象基类] MarsBinnerBase

    这是 Mars 特征工程体系中所有分箱组件的底层核心。它不仅定义了分箱器的状态契约, 
    还封装了高度优化的转换 (Transform) 与分析 (Profiling) 算子。

    该基类采用了“计算与路由分离”的设计: 
    - 计算: 由子类实现的 `fit` 策略负责填充切点。
    - 路由: 基类负责处理复杂的缺失值、特殊值、高基数类别路由以及 Eager/Lazy 混合执行。

    Attributes
    ----------
    bin_cuts_: Dict[str, List[float]]
        数值型特征的物理切点。每个列表均以 `[-inf, ..., inf]` 闭合, 确保全值域覆盖。
    cat_cuts_: Dict[str, List[List[Any]]]
        类别型特征的分组映射规则。将零散的字符串/分类标签聚类为逻辑组。
    bin_mappings_: Dict[str, Dict[int, str]]
        分箱可视化地图。将物理索引 (如 -1, 0, 1) 映射为业务可读标签 (如 "Missing", "01_[0, 10)")。
    bin_woes_: Dict[str, Dict[int, float]]
        分箱权重字典。存储每个分箱索引对应的 WOE 值。
    feature_names_in_: List[str]
        拟合时输入的原始特征列名。

    Notes
    -----
    **1. 极致性能架构**
    底层完全基于 Polars 的表达式引擎 (Expression Engine)。在转换数千个特征时, 基类会自动
    构建一个平坦化的计算图, 通过单次 IO 扫描实现并行转换, 规避了 Pandas 逐列循环的性能瓶颈。

    **2. 索引协议 (Index Protocol)**
    系统强制执行统一的索引协议以支持下游的风险监控 (PSI/IV): 
    - `Missing`: -1
    - `Other`: -2
    - `Special`: -3, -4, ...
    - `Normal`: 0, 1, 2, ...

    **3. 内存与稳定性**
    内置“延迟物化 (Lazy Materialization)”与“分批执行 (Batch Execution)”机制, 
    确保在处理千万级数据或超宽表时, 内存曲线保持平稳, 防止因计算图深度溢出导致的系统崩溃。
    """

    # 类型常量: 用于快速判定数值列
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
        初始化分箱器基类, 配置全局业务规则与并行策略。

        Parameters
        ----------
        features: List[str], optional
            数值型特征白名单。若为空, 子类通常会自动识别输入数据中的数值列。
        cat_features: List[str], optional
            类别型特征白名单。明确指定哪些列应按字符串分组逻辑处理。
        n_bins: int, default=5
            期望的最大分箱数量。最终生成的箱数可能少于此值 (受单调性约束或样本量影响)。
        special_values: List[Union[int, float, str]], optional
            特殊值列表。在部分场景中, 某些特定取值 (如 -999, -1)代表特定含义, 
            会被强制分配到独立的负数索引分箱中, 不参与正常区间的切分。
        missing_values: List[Union[int, float, str]], optional
            自定义缺失值列表。除了原生的 `null` 和 `NaN` 外, 用户可指定其他代表缺失的值。
        join_threshold: int, default=100
            **性能调优开关**。在 `transform` 阶段: 
            - 当类别特征的基数 (Unique Values)低于此值时, 使用内存级 `replace` 映射。
            - 当基数超过此值时, 自动切换为 `Hash Join` 模式。
            *这能有效防止因构建过深的逻辑分支树 (When-Then Tree)导致的计算图解析缓慢。*
        n_jobs: int, default=-1
            并行计算的核心数: 
            - `-1`: 自动使用 `CPU核心数 - 1`, 预留一个核心保证系统响应。
            - `1`: 强制单线程模式, 便于调试。
            - `N`: 使用指定的核心数。

        Notes
        -----
        初始化阶段不执行任何重型计算。所有计算资源 (进程池、线程池)均在 `fit` 阶段按需按需申请。
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
        
    def to_dict(self) -> Dict[str, Any]:
        """
        将分箱器状态序列化为 Python 字典。
        """
        return {
            "params": {
                "n_bins": self.n_bins,
                "special_values": self.special_values,
                "missing_values": self.missing_values,
                "join_threshold": self.join_threshold,
                # 注意: 子类可能还有额外的 params (如 solver), 
                # 如果要完美设计, 这里应该通过 self.__dict__ 过滤或让子类重写
            },
            "state": {
                "bin_cuts_": self.bin_cuts_,
                "cat_cuts_": getattr(self, "cat_cuts_", {}), # 兼容可能没有 cat_cuts_ 的情况
                "bin_mappings_": self.bin_mappings_,
                "bin_woes_": self.bin_woes_,
                # [新增] 保存失败记录, 使用 getattr 防止未 fit 时报错
                "fit_failures_": getattr(self, "fit_failures_", {}) 
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        从字典中恢复分箱器实例。
        """
        # 实例化一个空对象
        instance = cls(**data["params"])
        
        # 恢复训练后的状态
        state = data["state"]
        instance.bin_cuts_ = state.get("bin_cuts_", {})
        instance.cat_cuts_ = state.get("cat_cuts_", {})
        instance.bin_mappings_ = state.get("bin_mappings_", {})
        instance.bin_woes_ = state.get("bin_woes_", {})
        
        # [新增] 恢复失败记录
        instance.fit_failures_ = state.get("fit_failures_", {})
        
        instance._is_fitted = True
        return instance
    
    def __getstate__(self):
        """
        Pickle 序列化时的钩子。
        在保存模型时, 自动剔除巨大的训练数据缓存, 只保留配置和计算结果。
        """
        state = self.__dict__.copy()
        # 移除大数据缓存, 防止模型文件变成几百 MB
        state["_cache_X"] = None
        state["_cache_y"] = None
        return state

    def __setstate__(self, state):
        """
        Pickle 反序列化时的钩子。
        恢复模型状态, 并将缓存初始化为 None。
        """
        self.__dict__.update(state)
        # 确保属性存在, 防止 AttributeError
        if "_cache_X" not in self.__dict__:
            self._cache_X = None
        if "_cache_y" not in self.__dict__:
            self._cache_y = None


    def _get_safe_values(self, dtype: pl.DataType, values: List[Any]) -> List[Any]:
        """
        [Helper] 跨引擎类型安全清洗函数。

        在强类型引擎 (如 Polars/Rust)中, 类型不匹配是导致崩溃的主要原因。该方法通过预扫描 
        Schema, 确保用户定义的业务逻辑 (缺失值、特殊值)与数据的物理存储类型保持绝对兼容。

        Parameters
        ----------
        dtype: polars.DataType
            当前处理列的原始数据类型。
        values: List[Any]
            用户在配置中指定的数值列表 (如 [-999, 'unknown', None])。

        Returns
        -------
        List[Any]
            经过物理类型对齐后的清洗列表。

        Notes
        -----
        **1. 严格过滤机制 (Numeric Path)**
        若目标列为数值型, 系统会剔除所有非数值项。特别地, 由于 Python 中 `True == 1`, 
        系统会显式排除布尔类型, 防止逻辑误判导致的异常成箱。

        **2. 宽容转换机制 (String/Categorical Path)**
        若目标列为非数值型, 系统会将所有配置项强制转换为字符串。这保证了在进行 
        `is_in` 操作或 `join` 操作时, 比较操作发生在相同的物理类型之上。

        **3. 空值剥离**
        `None` 和 `np.nan` 会在此阶段被剥离, 转由 `is_null()` 和 `is_nan()` 算子在 
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
                # 数值列: 严格保留数值, 剔除 bool (True==1 歧义) 和字符串
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    safe_vals.append(v)
            else:
                # 非数值列: 宽容处理, 全部转为字符串以匹配 Categorical/String 列
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
        [WOE 物化计算引擎] 
        将分箱统计分布转化为证据权重 (WOE)的核心物化算子。

        针对超宽表 (>2000列)场景，弃用了前一版的“逐列循环聚合”模式，采用了 
        "Unpivot-Aggregate" 向量化策略，实现了计算效率与内存稳定性的平衡。

        Parameters
        ----------
        batch_size: int, default=200
            分批处理的特征数量。
            - **调优建议**: 
              该参数决定了 `unpivot` 操作产生的临时长表大小 (Rows * batch_size)。
              若遇到内存溢出 (OOM)，请将此值调小 (如 50)；若内存充足，调大此值可提升吞吐量。

        Notes
        -----
        **1. 向量化聚合架构 (Unpivot-Aggregate)**
        不再对每一列单独启动 Polars 引擎。而是通过 `unpivot` 将宽表 (Wide)转换为长表 (Long)，
        利用 `group_by(["feature", "bin_index"])` 在单次引擎调用中并行计算数百个特征的 WOE。
        这极大减少了 Python/Rust 上下文切换和查询计划解析的开销。

        **2. 避免计算图爆炸 (Graph Cutting)**
        虽然计算是向量化的，但对于 5000+ 列的宽表，构建单一的超大计算图仍会导致解析器卡死。
        因此保留了 `batch_size` 机制，每批次计算后通过 `to_list()` 强制物化并 `del` 中间变量，
        显式切断计算图血缘 (Lineage Breaking)。

        **3. 数值稳定性 (Numerical Stability)**
        - 公式: $$WOE_i = \ln\left(\frac{Bad\_Dist_i}{Good\_Dist_i}\right)$$
        - 平滑: 引入平滑因子 (1e-6)防止 `log(0)` 或除以零异常。
        - 结果: 计算结果存储于 `self.bin_woes_` 字典中，作为后续 `transform` 的静态查找表。
        """
        if self._cache_X is None or self._cache_y is None:
            logger.warning("No training data cached. WOE cannot be computed.")
            return

        n_cols = len(self.bin_cuts_)+ len(self.cat_cuts_)
        logger.info(f"⚡ [Auto-Trigger] Materializing WOE for {n_cols} features...")
        
        y_name = "_y_tmp"
        y_series = pl.Series(name=y_name, values=self._cache_y)
        total_bads = y_series.sum()
        total_goods = len(y_series) - total_bads
        
        # 涵盖数值和类别特征
        bin_cols_orig = [c for c in self.bin_cuts_.keys()] + \
                        (list(self.cat_cuts_.keys()) if hasattr(self, 'cat_cuts_') else [])

        for i in range(0, len(bin_cols_orig), batch_size):
            batch_features = bin_cols_orig[i: i + batch_size]
            
            # 1. 执行转换
            X_batch_bin: pl.DataFrame = self.transform(
                self._cache_X.select(batch_features), 
                return_type="index", 
                lazy=False
            )
            X_batch_bin = X_batch_bin.with_columns(y_series)

            # [关键修复]：构造带 _bin 后缀的列名列表
            target_bin_cols = [f"{c}_bin" for c in batch_features]

            # 2. 逆透视 (Unpivot): 确保作用于转换后的索引列
            long_df = X_batch_bin.unpivot(
                index=[y_name],
                on=target_bin_cols, # ✅ 修复：使用转换后的索引列名
                variable_name="feature_raw", # 临时名称
                value_name="bin_index"
            ).with_columns(
                # 去掉后缀，恢复原始特征名，保持与 bin_woes_ 的 Key 一致
                pl.col("feature_raw").str.replace("_bin", "").alias("feature")
            )

            # 一次性聚合所有特征的统计量
            stats_df = (
                long_df.group_by(["feature", "bin_index"])
                .agg([
                    pl.col(y_name).sum().alias("bin_bads"),
                    pl.len().alias("bin_total")
                ])
                # 计算 WOE
                .with_columns(
                    (
                        # 分子: 坏人占比
                        ((pl.col("bin_bads")+ 1e-6)/ (total_bads + 1e-6))/ 
                        # 分母: 好人占比
                        ((pl.col("bin_total")- pl.col("bin_bads")+ 1e-6)/ (total_goods + 1e-6))
                    )
                    .log()
                    .cast(pl.Float32)
                    .alias("woe")
                )
            )

            # 结果回填字典 
            # partition_by(as_dict=True)会返回 {('feature_A',): DataFrame, ...}
            # 小规模数据的 partition_by 性能远优于循环 filter
            stats_dict: dict[tuple[str, ...], pl.DataFrame] = stats_df.partition_by("feature", as_dict=True)

            for key, sub_df in stats_dict.items():
                # partition_by 返回的 key 是 tuple, 如 ('feature_name',)
                feat_name = key[0] if isinstance(key, tuple) else key
                
                # [优化] 强制过滤掉 bin_index 中的 NaN 和 Null，并转换回整型
                # 只有合法的索引 (-1, 0, 1...) 允许进入 WOE 映射表
                valid_stats = sub_df.filter(
                    pl.col("bin_index").is_not_null() & 
                    ~pl.col("bin_index").cast(pl.Float64).is_nan()
                ).with_columns(pl.col("bin_index").cast(pl.Int16))

                # 直接转换两列为字典，确保 Key 是 Python 的 int 类型
                self.bin_woes_[feat_name] = dict(zip(
                    valid_stats["bin_index"].to_list(),
                    valid_stats["woe"].to_list()
                ))

            # 强制内存断层
            del X_batch_bin, long_df, stats_df
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
        
        核心转换逻辑, 兼容数值与类别特征, 支持 Eager 与 Lazy 模式。

        该方法采用了“表达式瀑布流 (Expression Waterfall)”设计, 通过 Polars 的原生算子实现
        了高效的向量化转换。针对高基数类别特征, 采用了 Join 优化策略以规避深层逻辑树带来的性能损耗。

        Parameters 
        ----------
        X: Union[pl.DataFrame, pl.LazyFrame]
            待转换的数据集。支持延迟计算流 (LazyFrame) 以优化长流水线性能。
        return_type: {'index', 'label', 'woe'}, default='index'
            转换后的输出格式: 
            - 'index': 输出分箱索引 (Int16 类型)。
            - 'label': 输出分箱的可读标签 (Utf8 类型, 如 "01_[10.5, 20.0)")。
            - 'woe': 输出对应的证据权重 (Weight of Evidence) 值 (Float32 类型)。
        woe_batch_size: int, default=200
            仅在 return_type='woe' 且未预计算 WOE 时有效。指定并行计算 WOE 的批大小。
        lazy: bool, default=False
            是否保持延迟执行状态。若为 True, 则无论输入是 Eager 还是 Lazy, 均返回 LazyFrame。

        Returns
        -------
        Union[pl.DataFrame, pl.LazyFrame]
            转换后的数据集。原列保持不变, 新增以 `_bin` 或 `_woe` 为后缀的转换列。

        Notes
        -----
        **1. 分箱索引协议**
        为了确保与下游 Profiler 和 PSI 计算算子对齐, 系统采用以下固定索引: 
        - `IDX_MISSING (-1)`: 缺失值及自定义缺失值。
        - `IDX_OTHER (-2)`: 类别型特征中的未见类别 (Unseen categories)。
        - `IDX_SPECIAL_START (-3)`: 特殊值分箱起始索引 (向负无穷延伸)。
        - `[0, N]`: 正常数值区间或类别分组索引。

        **2. 数值型转换**
        采用层级覆盖逻辑: 
        - 预处理: 利用 `_get_safe_values` 确保缺失值/特殊值的类型与列 Schema 严格一致。
        - 核心: 使用 `pl.cut` 进行向量化区间划分。
        - 组合: 通过 `pl.when().then()` 瀑布流, 按照 "缺失值 -> 特殊值 -> 正常区间" 的优先级进行合并。

        **3. 类别型转换**
        针对类别特征采用双路径优化: 
        - **路径 A (低基数)**: 使用 `replace` 算子进行内存级映射, 速度极快。
        - **路径 B (高基数)**: 当类别数超过 `join_threshold` 时, 自动转为 `Join` 模式。
            这避免了构建数千个 `when-then` 分支导致的逻辑树深度爆炸 (Stack Overflow 风险), 
            将逻辑判断转化为哈希连接操作, 极大提升了宽表转换效率。

        **4. 自动路由与路由安全**
        - 在进行 Utf8 类型操作 (如类别分组)前, 系统会自动创建临时 Utf8 缓存列。
        - 转换结束后, 会自动清理所有产生的中间 Join 列和临时缓存列, 保证输出 Schema 纯净。

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
                
                # 缺失值逻辑: Is Null OR Is Missing Val
                missing_cond = pl.col(col).is_null() 
                if is_numeric_col: 
                    missing_cond |= pl.col(col).cast(pl.Float64).is_nan()
                for v in safe_missing_vals: 
                    missing_cond |= (pl.col(col) == v)
                # ⭐构建缺失值分箱表达式
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 正常分箱逻辑: Cut
                raw_breaks = cuts[1:-1] if len(cuts) > 2 else []
                # [优化] 增加 set去重 和 sorted排序, 防止 pl.cut 报错
                    # 1. 去重 (set): 应对高偏态数据 (如大量 0 值) 导致多个分位数计算结果相同 (q25=0, q50=0)。
                    # 2. 排序 (sorted): Polars.cut 要求切点严格单调递增，否则 Rust 内核会抛出 Panic。
                breaks = sorted(list(set(raw_breaks)))

                col_mapping: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"} # 分箱标签映射表 IDX -> Label
                
                # 无切点逻辑
                if not breaks:
                    col_mapping[0] = "00_[-inf, inf)"
                    layer_normal = pl.lit(0, dtype=pl.Int16) # 全部归为 0 号箱
                else:
                    for i in range(len(cuts) - 1):
                        low, high = cuts[i], cuts[i+1]
                        col_mapping[i] = f"{i:02d}_[{low:.3g}, {high:.3g})"
                    # [优化] 显式生成 labels 确保 cast(Int16) 成功, 修复 PSI=0 Bug
                    bin_labels: List[str] = [str(i) for i in range(len(breaks) + 1)]
                    # ⭐ 构建正常分箱表达式
                    layer_normal = (
                        pl.col(col)
                        # 核心分箱: 将连续数值切分为离散区间
                        # 返回类型为 Categorical (分类类型), 底层由"物理ID"和"逻辑标签"组成
                        .cut(breaks, labels=bin_labels, left_closed=True)
                        
                        # [优化] 强制逻辑解引用
                        # 目的: 防御 Polars 内部物理索引偏移 (Off-by-one) 问题。
                        # 风险: 直接 cast(Int) 会读取到底层的物理存储 ID (Physical Index), 
                        #       该 ID 可能受 Null 值或全局字典影响而发生偏移 (如标签"0"对应ID=1)。
                        # 动作: 先转 Utf8 强制 Polars 查表返回业务标签值 (如字符串 "0", "1")。
                        .cast(pl.Utf8)
                        
                        # 类型对齐: 将业务标签字符串解析为最终的数字索引
                        # 动作: 将字符串 "0" -> 数字 0, 确保与 bin_mappings_ 字典的 Key 完美对齐。
                        .cast(pl.Int16)
                   )
                
                # 特殊值逻辑: 瀑布流覆盖
                current_branch = layer_normal
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i 
                        col_mapping[idx] = f"Special_{v}"
                        # 注意这里的覆盖顺序: 后定义的优先级更高
                        current_branch = pl.when(pl.col(col) == v).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)
                
                # ⭐ 最终的分箱表达式: Missing -> Special -> Normal
                final_idx_expr = layer_missing.otherwise(current_branch)
                self.bin_mappings_[col] = col_mapping
                
            # =========================================================
            # Part B: 类别型分箱 (Categorical Binning)
            # =========================================================
            elif hasattr(self, 'cat_cuts_') and col in self.cat_cuts_:
                splits = self.cat_cuts_[col]
                cat_to_idx: Dict[str, int] = {}
                idx_to_label: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}
                
                # [新增] 默认路由索引, 默认为 -2
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
                        # [新增] 如果训练时这一箱含有 "__Mars_Other_Pre__", 则将其设为默认箱
                        if val_str == "__Mars_Other_Pre__":
                            default_bin_idx = i
                
                self.bin_mappings_[col] = idx_to_label
                # 强转 String, 确保类别匹配安全
                target_col = pl.col(col).cast(pl.Utf8)
                
                # 缺失值
                missing_cond = target_col.is_null() | (target_col == "nan") # Polars 中 NaN 的字符串表现形式
                for v in safe_missing_vals:
                    missing_cond |= (target_col == str(v))
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 特殊值
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
                
                target_col_name = col
                if col_dtype != pl.Utf8:
                    target_col_name = f"_{col}_utf8_tmp"
                    X = X.with_columns(pl.col(col).cast(pl.Utf8).alias(target_col_name))

                # 路由: Join (高基数) vs Replace (低基数)
                if len(cat_to_idx) > self.join_threshold:
                    map_df = pl.DataFrame({
                        "_k": list(cat_to_idx.keys()), 
                        f"_idx_{col}": list(cat_to_idx.values())
                    }).cast({"_k": pl.Utf8, f"_idx_{col}": pl.Int16})
                    
                    # [修复] 根据 X 的类型自适应转换 map_df
                    join_tbl = map_df.lazy() if isinstance(X, pl.LazyFrame) else map_df
                    X = X.join(join_tbl, left_on=target_col_name, right_on="_k", how="left")
                    temp_join_cols.append(f"_idx_{col}")
                    if target_col_name != col: temp_join_cols.append(target_col_name)
                    
                    layer_normal = pl.col(f"_idx_{col}")
                else:
                    layer_normal = target_col.replace(cat_to_idx, default=IDX_OTHER).cast(pl.Int16)
                
                # 最终的分箱表达式: Missing -> Normal (Join Result) -> Special/Other
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
                if woe_map:
                    # [优化] 在 Replace 执行前，确保字典 Key 只有整数，剔除任何 NaN 键
                    # 这能防御从旧版本或外部加载的模型中潜在的类型污染
                    clean_woe_map = {
                        int(k): float(v) for k, v in woe_map.items() 
                        if k is not None and not (isinstance(k, float) and np.isnan(k))
                    }
                    
                    # [优化] 使用 clean_woe_map，并增加了 default=0.0
                    # 确保即便映射表里没定义的索引 (如 -2), 也会被强制转为 0 权重, 而不是保留索引原值
                    expr = final_idx_expr.replace(clean_woe_map, default=0.0).cast(pl.Float32)
                else:
                    # 如果压根没映射表, 保持原样的全列 0.0
                    expr = pl.lit(0.0)
                    logger.warning(f"WOE mapping for column '{col}' not found. Defaulting to 0.0.")
                exprs.append(expr.alias(f"{col}_woe"))
            else:
                str_map = {str(k): v for k, v in self.bin_mappings_.get(col, {}).items()}
                exprs.append(final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin"))

        # 清理 Join 产生的临时列
        return X.with_columns(exprs).drop(temp_join_cols).lazy() if lazy else X.with_columns(exprs).drop(temp_join_cols)

    @time_it
    def profile_bin_performance(self, X: pl.DataFrame, y: Any, update_woe: bool = True) -> pl.DataFrame:
        """
        [极速指标矩阵引擎] 产出全量分箱深度分析报告 (IV/KS/AUC/Lift)。

        这是 Mars 库中最具工程深度的方法之一。针对高维特征 (2000+ 列)进行了多重性能与逻辑优化, 
        实现了从“数据聚合”到“风险评估指标生成”的全链路向量化处理。

        Parameters
        ----------
        X: polars.DataFrame
            原始特征数据集。支持 Eager/Lazy 输入。
        y: Any
            目标标签 (二分类)。
        update_woe: bool, default=True
            - 是否根据当前输入数据更新实例内部的 `bin_woes_` 字典。
                - 训练集(Fit): 应设为 True, 以捕获训练样本的 WOE 权重供后续转换。
                - 验证集/OOT: 应设为 False, 仅计算指标, 防止模型原始权重被异质样本篡改。
 
        Returns
        -------
        polars.DataFrame
            包含各特征、各分箱详细指标的报表。
            - 业务维度: feature, bin_label。
            - 样本分布: count, bad, good, count_dist。
            - 风险识别: bad_rate, lift, woe, bin_iv, bin_ks, IV, KS, AUC。

        Notes
        -----
        **1. 矩阵化聚合l逻辑**
            采用“逆透视”将 `(N_rows * M_features)` 的宽表动态转换为长表, Polars 优化器
            可利用单一的查询计划在单次 I/O 扫描中并行计算数千个特征的所有指标。

        **2. 自动风险排序与指标计算**
            计算前根据 `woe` 进行升序排序。这确保了: 
            - **KS**: 反映了特征在不同风险段上的累计分布差异。
            - **AUC**: 基于梯形法则 (Trapezoidal Rule)计算 ROC 曲线下面积, 真实反映分箱的排序能力。
              公式: $$AUC = \\sum (G_{curr} - G_{prev}) \\times \\frac{(B_{curr} + B_{prev})}{2}$$

        **3. 高维性能调优**
            - **Streaming 物化**: 支持流式处理, 防止 2000+ 列计算时内存溢出。
            - **Dictionary Lookup**: 使用 `partition_by` 将结果字典化 (O(1) 复杂度), 
              规避了传统 Python 循环 `.filter()` 产生的大规模内存扫描开销。

        **4. 业务契约对齐**
            - 自动关联 `bin_mappings_`, 将物理索引 (-1, 0, 1)映射为业务标签 (Missing, 区间等)。
            - 强制执行 `max(AUC, 1 - AUC)`, 确保输出的 AUC 始终代表特征的绝对区分强度。

        Example
        -------
        >>> # 1. 训练阶段: 计算指标并同步权重
        >>> train_report = binner.profile_bin_performance(train_df, y_train, update_woe=True)
        >>> 
        >>> # 2. 评估阶段: 查看 OOT 稳定性, 保持权重不变
        >>> oot_report = binner.profile_bin_performance(oot_df, y_oot, update_woe=False)
        """
        y_name = "_target_tmp"
        actual_y_name = getattr(y, "name", "_unknown_target_")
        # 强制开启 Lazy 转换以合并查询计划
        X_bin_lazy: pl.LazyFrame = self.transform(X, return_type="index", lazy=True)
        X_bin_lazy = X_bin_lazy.with_columns(pl.lit(np.array(y)).alias(y_name))
        
        # 获取全局统计量
        meta = X_bin_lazy.select([
            pl.len().alias("total_counts"),
            pl.col(y_name).sum().alias("total_bads")
        ]).collect()
        
        total_counts = meta[0, "total_counts"]
        total_bads = meta[0, "total_bads"]
        total_goods = total_counts - total_bads
        global_bad_rate = (total_bads / total_counts) if total_counts > 0 else 0
        
        current_cols = X_bin_lazy.collect_schema().names()
        bin_cols = [c for c in current_cols if c.endswith("_bin") and c != f"{actual_y_name}_bin"]

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
                pl.len().alias("count"),
                pl.col(y_name).sum().alias("bad")
            ])
            .with_columns([
                (pl.col("count") - pl.col("bad")).alias("good")
            ])
       )

        # 计算基础占比指标
        lf_stats = lf_stats.with_columns([
            (pl.col("count") / total_counts).cast(pl.Float32).alias("count_dist"),
            (pl.col("bad") / pl.col("count")).cast(pl.Float32).alias("bad_rate"),
            # 分布占比为了计算 KS/AUC，保持 Float32
            (pl.col("bad") / (total_bads + 1e-6)).cast(pl.Float32).alias("bad_dist"),
            (pl.col("good") / (total_goods + 1e-6)).cast(pl.Float32).alias("good_dist")
        ])

        # [优化] 计算 WOE：对齐 _materialize_woe 的公式和精度
        # 公式改为：ln( ((bad + 1e-6)/(total_bads + 1e-6)) / ((good + 1e-6)/(total_goods + 1e-6)) )
        lf_stats = lf_stats.with_columns([
            (
                ((pl.col("bad") + 1e-6) / (total_bads + 1e-6)) / 
                ((pl.col("good") + 1e-6) / (total_goods + 1e-6))
            ).log().cast(pl.Float32).alias("woe")
        ])

        # 计算 IV (也强制转为 Float32)
        lf_stats = lf_stats.with_columns([
            ((pl.col("bad_dist") - pl.col("good_dist")) * pl.col("woe")).cast(pl.Float32).alias("bin_iv")
        ])

        # 计算 KS 和 AUC
        # 注意: 计算 AUC 必须按照风险 (WOE)升序排序, 以形成正确的 ROC 曲线
        lf_stats = lf_stats.sort(["feature", "woe", "bin_index"]).with_columns([
            pl.col("bad_dist").cum_sum().over("feature").alias("cum_bad_dist"),
            pl.col("good_dist").cum_sum().over("feature").alias("cum_good_dist")
        ])
        # 使用梯形法则计算 AUC: Σ (ΔGood * (Bad_prev + Bad_curr) / 2)
        lf_stats = lf_stats.with_columns([
            (pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs().alias("bin_ks"),
            # AUC 贡献度计算
            ((pl.col("cum_good_dist") - pl.col("cum_good_dist").shift(1, fill_value=0).over("feature")) * (pl.col("cum_bad_dist") + pl.col("cum_bad_dist").shift(1).over("feature").fill_null(0)) / 2
           ).alias("bin_auc_contrib")
        ])
        # 计算 IV, KS, AUC
        lf_stats = lf_stats.with_columns([
            pl.col("bin_iv").sum().over("feature").alias("IV"),
            pl.col("bin_ks").max().over("feature").alias("KS"),
            pl.col("bin_auc_contrib").sum().over("feature").alias("AUC"),
            (pl.col("bad_rate") / (global_bad_rate + 1e-9)).alias("Lift")
        ]).with_columns([
            # 保证 AUC 始终 >= 0.5 (如果特征是反向相关的, AUC 会小于 0.5, 取 max(auc, 1-auc))
            pl.when(pl.col("AUC") < 0.5).then(1 - pl.col("AUC")).otherwise(pl.col("AUC")).alias("AUC")
        ])

        # 获取最终结果集
        stats_df: pl.DataFrame = lf_stats.drop("bin_auc_contrib").collect(streaming=True)
        stats_dict: Dict[Tuple[str, ...], pl.DataFrame] = stats_df.partition_by("feature", as_dict=True)
        
        final_list = []
        for feat_name in bin_cols:
            orig_name = feat_name.replace("_bin", "")
            
            # [优化] 兼容性处理
            # partition_by(as_dict=True) 返回的 Key 通常是 tuple, 但也可能因版本差异有变
            # 优先尝试 tuple key, 若失败则尝试直接 key
            dict_key = (feat_name,)
            
            if dict_key in stats_dict:
                feat_stats = stats_dict[dict_key]
            elif feat_name in stats_dict: # 万一 Polars 返回的不是 tuple
                feat_stats = stats_dict[feat_name]
            else:
                logger.warning(f"Feature '{orig_name}' not found in stats dictionary. Skipping.")
                continue # 如果都找不到, 跳过
                
            
            # 状态同步
            if update_woe:
                # [优化] 过滤与强制类型转换逻辑
                valid_feat_stats = feat_stats.filter(
                    pl.col("bin_index").is_not_null() & 
                    ~pl.col("bin_index").cast(pl.Float64).is_nan()
                ).with_columns(pl.col("bin_index").cast(pl.Int16))

                self.bin_woes_[orig_name] = dict(
                    zip(valid_feat_stats["bin_index"].to_list(), valid_feat_stats["woe"].to_list())
               )

            # 可视化映射与输出重整
            mapping = self.get_bin_mapping(orig_name)
            # 自定义排序逻辑 
            feat_stats = (
                feat_stats 
                .with_columns([
                    pl.col("bin_index").cast(pl.Utf8)
                    .replace({str(k): v for k, v in mapping.items()})
                    .alias("bin_label")
                ])
                # 排序协议: 
                # 1. (pl.col("bin_index") < 0) 产生布尔值: False(0) 代表正常箱, True(1) 代表特殊/缺失箱
                #    这样正常箱会排在 0 序列, 特殊箱排在 1 序列 (即排在后面)
                # 2. 在各自序列内部, 按 bin_index 升序排列 (0,1,2... 或 -3,-2,-1)
                .sort(by=[pl.col("bin_index") < 0, "bin_index"]) 
           )

            feat_stats = feat_stats.select([
                pl.col("feature"),
                pl.col("bin_label"),
                pl.all().exclude(["feature", "bin_index", "bin_label"])
            ])
            
            final_list.append(feat_stats)

        if not final_list:
            return pl.DataFrame()

        return pl.concat(final_list)

class MarsNativeBinner(MarsBinnerBase):
    """
    [极速原生分箱器] MarsNativeBinner
    
    基于 Polars (数据预处理) 与 Scikit-learn (计算内核) 构建的高性能特征分箱引擎。
    该类旨在解决大规模宽表 (数千维特征)在传统 Python 分箱工具中运行缓慢的问题。
    
    支持三种分箱策略: Quantile、Uniform 和 CART 分箱, 适用于数值型特征, 暂不支持分类特征。

    Attributes
    ----------
    bin_cuts_: Dict[str, List[float]]
        拟合后生成的数值型特征切点字典。格式: {特征名: [-inf, 切点1, ..., inf]}。
    fit_failures_: Dict[str, str]
        记录训练过程中发生异常的特征及其错误原因。
    feature_names_in_: List[str]
        训练时输入的特征名称列表。
    _is_fitted: bool
        标识分箱器是否已完成拟合。

    Notes
    -----
    1. 性能: 在 20w 行 x 5000 列数据下, 含有自定义缺失值和特殊值的情况下, 单机i7 14700 24核: 
        - Quantile 分箱: 约 40 秒
        - Uniform 分箱: 约 25 秒
        - CART 分箱: 约 80 秒
    2. 鲁棒性: 内置常量列识别、缺失值自动过滤及异常特征自动退化机制。
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        method: Literal["cart", "quantile", "uniform"] = "quantile",
        n_bins: int = 10,
        special_values: Optional[List[Union[int, float, str]]] = None,
        missing_values: Optional[List[Union[int, float, str]]] = None,
        min_samples: float = 0.05,
        cart_params: Optional[Dict[str, Any]] = None,
        remove_empty_bins: bool = False,
        # join_threshold: int = 100,
        n_jobs: int = -1,
   ) -> None:
        """
        初始化 MarsNativeBinner。

        Parameters
        ----------
        features: List[str], optional
            指定需要进行分箱的数值特征列名。若为 None, 则自动识别 X 中的所有数值列。
        method: {'cart', 'quantile', 'uniform'}, default='quantile'
            分箱策略: 
            - 'cart': 基于决策树的最优分箱。
            - 'quantile': 等频分箱 (推荐用于工业级预处理)。
            - 'uniform': 等宽分箱。
        n_bins: int, default=10
            目标最大分箱数。
        special_values: List[Union[int, float, str]], optional
            特殊值列表。这些值将被强制独立成箱 (如: -999, -9999)。
        missing_values: List[Union[int, float, str]], optional
            自定义缺失值列表。默认 None, NaN 会自动识别并归为 Missing 箱。
        min_samples: float, default=0.05
            仅在 method='cart' 时有效。决策树叶子节点的最小样本占比。
        cart_params: Dict, optional
            透传给 sklearn.tree.DecisionTreeClassifier 的额外参数。
        remove_empty_bins: bool, default=False
            仅在 method='uniform' 时有效。是否自动剔除并合并样本量为 0 的空箱。
        # join_threshold: int, default=100
        #     在 Transform 阶段, 类别型特征使用 Join 替代 Replace 的基数阈值。
        n_jobs: int, default=-1
            并行计算的核心数。-1 表示使用所有可用核心。
        """
        super().__init__(
            features=features, n_bins=n_bins, 
            special_values=special_values, missing_values=missing_values,
            # join_threshold=join_threshold, 
            n_jobs=n_jobs
       )
        self.method = method
        self.min_samples = min_samples
        self.remove_empty_bins = remove_empty_bins
        
        self.cart_params = {} if cart_params is None and isinstance(cart_params, dict) else cart_params

    @time_it
    def _fit_impl(self, X: pl.DataFrame, y: Optional[Any] = None, **kwargs) -> None:
        """
        执行分箱拟合的核心入口逻辑。

        负责任务分发前的三道防线: 
        1. 自动识别并排除非数值列。
        2. 极速全表扫描获取 Min/Max, 识别并排除常量列。
        3. 路由分发至不同的分箱策略方法。

        Parameters
        ----------
        X: polars.DataFrame
            经过基类归一化后的训练数据。
        y: polars.Series, optional
            目标变量。在使用 'cart' 方法时必填。
        **kwargs: dict
            透传的额外配置参数。
        """
        # 缓存数据引用, 仅用于 transform 阶段请求 return_type='woe' 时的延迟计算
        self._cache_X = X
        self._cache_y = y

        # 确定目标列 (仅筛选数值列, 忽略全空列)
        # 获取 y 的名称 (如果 y 是 Series)
        y_name = getattr(y, "name", None)
        
        # 确定目标列: 如果没有指定 features, 则获取 X 的所有列, 但必须排除掉 y 所在的列
        if not self.features:
            all_target_cols = [c for c in X.columns if c != y_name]
        else:
            all_target_cols = self.features
        target_cols: List[str] = []
        null_cols: List[str] = [] 

        for c in all_target_cols:
            if c not in X.columns: continue
            
            # 判定全空/Null类型列, 记录下来以便直接注册为空箱
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 仅处理数值类型
            if self._is_numeric(X[c]):
                target_cols.append(c)

        # 注册全空列为空切点, 防止 transform 时漏列
        for c in null_cols:
            self.bin_cuts_[c] = []

        if not target_cols:
            if not null_cols:
                logger.warning("No numeric columns found for binning.")
            return

        # 极速全表扫描获取 Min/Max, 识别常量列
        valid_cols: List[str] = []
        stats_exprs = []
        for c in target_cols:
            col_dtype = X.schema[c]
            target_expr = pl.col(c)
            
            # 【修复】只有浮点数才做 is_not_nan 检查
            if col_dtype in [pl.Float32, pl.Float64]:
                target_expr = target_expr.filter(target_expr.is_not_nan())
                
            stats_exprs.append(target_expr.min().alias(f"{c}_min"))
            stats_exprs.append(target_expr.max().alias(f"{c}_max"))
        stats_row = X.select(stats_exprs).row(0)
        
        for i, c in enumerate(target_cols):
            min_val = stats_row[i * 2]
            max_val = stats_row[i * 2 + 1]
            
            # 如果 Min == Max, 说明是常量列, 无需分箱, 直接设为全区间
            if min_val == max_val:
                logger.warning(f"Feature '{c}' is constant. Skipped.")
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            valid_cols.append(c)
            
        if not valid_cols:
            return

        # 检查 CART 方法的依赖
        if y is None and self.method == "cart":
            raise ValueError("Decision Tree Binning ('cart') requires target 'y'.")
        
        # 初始化失败记录器
        self.fit_failures_: Dict[str, str] = {}

        # 策略分发
        if self.method == "quantile":
            self._fit_quantile(X, valid_cols)
        elif self.method == "uniform":
            self._fit_uniform(X, valid_cols)
        elif self.method == "cart":
            self._fit_cart_parallel(X, y, valid_cols)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if hasattr(self, "fit_failures_") and self.fit_failures_:
            logger.warning(
                f"⚠️ {len(self.fit_failures_)} numeric features are degenerate (single bin). "
                f"Check `.fit_failures_` for details."
           )

    def _fit_quantile(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行等频分箱 (One-Shot Quantile Query)。

        该方法摒弃了传统的“循环、筛选、计算”模式, 转而利用 Polars 的延迟计算特性, 
        将数千个特征的分位数计算合并为一个单一的原子查询计划 (Atomic Query Plan)。

        Parameters
        ----------
        X: polars.DataFrame
            训练数据集。
        cols: List[str]
            需要执行等频分箱的数值型特征列名列表。

        Notes
        -----
        **核心优化: 查询计划合并**
        - 传统实现: 针对 $N$ 个特征执行 $N$ 次 `quantile()` 调用, 触发 $N$ 次内存扫描。
        - Mars 实现: 构建一个扁平化的表达式列表 `[col1_q1, col1_q2, ..., colN_qM]`。
          通过 `X.select(q_exprs)` 将该列表一次性喂给 Rust 引擎。引擎会优化执行路径, 
          在单次 (或极少数次)内存扫描中并行完成所有特征的切点计算。

        **数据质量控制**
        - 源头隔离: 在计算分位数前, 利用 `pl.when().then(None)` 将 `special_values` 和 
          `missing_values` 临时替换为 `Null`, 确保切点的分布仅由业务层面的“正常值”决定。
        - 自动去重: 针对高偏态数据 (如某些取值极度集中的分位数一致), 会自动执行 `set()` 
          去重并重新排序, 防止生成重复切点导致的 `Cut Error`。
        
        **低基数优化**
        - 针对二值/离散整数 (如 0/1), Quantile 往往会切出 [0.0, 1.0] 这种尴尬边界。
        - 优化逻辑: 若特征唯一值数量 <= n_bins, 自动降级为"中点切分", 例如 [0, 1] 会被切在 0.5。

        Algorithm
        ---------
        1. 根据 `n_bins` 生成分位点序列 (如 [0.2, 0.4, 0.6, 0.8])。
        2. 为每个特征构建类型安全的 `quantile` 表达式。
        3. 聚合所有表达式, 执行单一的 `Eager` 模式查询。
        4. 解析结果矩阵, 生成对应的分箱边界 $[-\infty, q_1, \dots, q_k, \infty]$。
        """
        # 构建分位点
        if self.n_bins <= 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1].tolist()
        
        # 预处理排除值
        raw_exclude = self.special_values + self.missing_values
        
        # -------------------------------------------------------------
        # Step 1: 批量计算 n_unique, 用于路由低基数逻辑
        # -------------------------------------------------------------
        # 这一步开销很小, Polars 针对数值列的 n_unique 有极速优化
        unique_exprs = []
        for c in cols:
            col_dtype = X.schema[c]
            safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
            
            # 【修复】构建联合过滤条件 (Avoid Chained Filter)
            # 1. 基础: 非 Null
            keep_mask = pl.col(c).is_not_null()
            
            # 2. 叠加: 非 NaN (仅浮点)
            if col_dtype in [pl.Float32, pl.Float64]:
                keep_mask &= ~pl.col(c).is_nan()
            
            # 3. 叠加: 非特殊值
            if safe_exclude:
                keep_mask &= ~pl.col(c).is_in(safe_exclude)
                
            # 一次性应用过滤
            target_col = pl.col(c).filter(keep_mask)
            unique_exprs.append(target_col.n_unique().alias(c))
            
        unique_counts = X.select(unique_exprs).row(0)
        col_unique_map = dict(zip(cols, unique_counts))
        
        # 分流: 哪些列走 Quantile, 哪些列走 Midpoint (中点)
        quantile_cols = []
        low_card_cols = []
        
        for c in cols:
            # 如果唯一值比箱数还少, 算分位数没有意义, 直接切中点
            if col_unique_map[c] <= self.n_bins:
                low_card_cols.append(c)
            else:
                quantile_cols.append(c)

        # -------------------------------------------------------------
        # Step 2: 处理高基数列 (标准 Quantile 逻辑)
        # -------------------------------------------------------------
        if quantile_cols:
            q_exprs = []
            for c in quantile_cols:
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
                
                # 【修复】构建联合过滤条件
                # 初始条件: 非 Null
                valid_cond = pl.col(c).is_not_null()
                
                # 叠加: 非 NaN (仅浮点)
                if col_dtype in [pl.Float32, pl.Float64]:
                    valid_cond &= ~pl.col(c).is_nan()
                
                # 叠加: 非 Special Values
                if safe_exclude:
                    valid_cond &= ~pl.col(c).is_in(safe_exclude)
                
                # 应用过滤
                target_col = pl.col(c).filter(valid_cond)
                
                for i, q in enumerate(quantiles):
                    # 别名技巧: col:::idx, 便于后续解析
                    alias_name = f"{c}:::{i}"
                    q_exprs.append(target_col.quantile(q).alias(alias_name))
            
            # 计算 (One-Shot Query)
            if q_exprs:
                stats = X.select(q_exprs)
                row = stats.row(0)
                
                # 解析结果并去重排序
                temp_cuts: Dict[str, List[float]] = {c: [] for c in quantile_cols}
                
                for val, name in zip(row, stats.columns):
                    c_name, _ = name.split(":::")
                    if val is not None and not np.isnan(val):
                        temp_cuts[c_name].append(val)

                for c in quantile_cols:
                    cuts = sorted(list(set(temp_cuts[c]))) 
                    
                    if len(cuts) < 1:
                        # 极端情况：所有分位数都一样（例如全是0）
                        # 强制退化为全区间，防止后续 cut 算子切出空箱或单箱
                        self.bin_cuts_[c] = [float('-inf'), float('inf')]
                        if not hasattr(self, "fit_failures_"): self.fit_failures_ = {}
                        self.fit_failures_[c] = "Degenerate feature: all quantiles are identical."
                    else:
                        self.bin_cuts_[c] = [float('-inf')] + cuts + [float('inf')]

        # -------------------------------------------------------------
        # Step 3: 处理低基数列 (中点切分优化)
        # -------------------------------------------------------------
        if low_card_cols:
            for c in low_card_cols:
                safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)
                
                # 获取唯一值并排序
                # 这里的 unique 已经是全量 unique 减去 null, 但还需要排除 safe_exclude
                unique_vals = (
                    X.select(pl.col(c).unique())
                    .to_series()
                    .sort()
                    .to_list()
               )
                
                # 清洗, 因为唯一值极少, 速度很快
                clean_vals = [v for v in unique_vals if v is not None and (not isinstance(v, float) or not np.isnan(v))]
                if safe_exclude:
                    clean_vals = [v for v in clean_vals if v not in safe_exclude]
                
                if len(clean_vals) <= 1:
                    # 只有一个值, 无法切分
                    self.bin_cuts_[c] = [float('-inf'), float('inf')]
                    if not hasattr(self, "fit_failures_"): 
                        self.fit_failures_ = {}
                    self.fit_failures_[c] = "Degenerate feature: single unique value."
                else:
                    # 计算中点: (a+b)/2
                    # 例如 [0, 1] -> 切点 0.5 -> [-inf, 0.5, inf]
                    mid_points = [(clean_vals[k] + clean_vals[k+1])/2 for k in range(len(clean_vals)-1)]
                    self.bin_cuts_[c] = [float('-inf')] + mid_points + [float('inf')]

    def _fit_uniform(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行等宽分箱 (Uniform/Step Binning)。

        该方法利用 Polars 的向量化算子, 将所有特征的统计信息提取和切点生成分为两个物理阶段, 
        在保证统计严谨性的同时, 最大程度减少对原始数据的扫描次数。

        Parameters
        ----------
        X: polars.DataFrame
            训练数据集。
        cols: List[str]
            需要执行等宽分箱的数值型特征列名列表。

        Notes
        -----
        分箱逻辑分为以下两个核心阶段: 

        **阶段 1: 基础统计量聚合**
        - 构建一个全局查询计划, 一次性计算所有目标列的 `min` (最小值)、`max` (最大值) 
          和 `n_unique` (唯一值个数)。
        - 排除逻辑: 在计算极值前, 会自动过滤用户定义的 `special_values` 和 `missing_values`, 
          确保切点仅基于“正常”数值分布生成。
        - 低基数处理: 若特征唯一值个数小于目标箱数 (`n_unique <= n_bins`), 则自动退化为
          基于唯一值中点的精确切分, 防止生成重复切点。

        **阶段 2: 空箱动态优化**
        - 仅在 `remove_empty_bins=True` 时触发。
        - 机制: 利用 Polars 的 `cut` 和 `value_counts` 算子, 在主进程中并行嗅探初始等宽
          切点下的样本分布。
        - 压缩逻辑: 识别样本量为 0 的区间, 并将相邻的空箱进行物理合并。这在数据分布极端
          偏态 (如长尾分布)时, 能有效防止产生毫无意义的无效分箱。

        Algorithm
        ---------
        1. 针对每列特征 $x$, 计算有效值范围 $[min, max]$。
        2. 计算步长 $\Delta = (max - min) / n\_bins$。
        3. 初始切点集 $C = \{min + i \cdot \Delta \mid i=1, \dots, n\_bins-1\}$。
        4. 若开启优化, 则根据各区间真实频数 $N_i$ 重新调整切点集 $C'$。
        5. 最终输出格式: $[-\infty, \text{切点}_1, \dots, \text{切点}_k, \infty]$。

        Performance
        -----------
        由于采用了“计划合并 (Query Plan Fusion)”技术, 无论处理 100 列还是 2000 列, 
        对原始内存的扫描次数始终保持在极低水位 (通常为 1-2 次全表扫描)。
        """
        raw_exclude = self.special_values + self.missing_values
        
        # 基础统计量
        exprs = []
        col_safe_excludes = {} 

        for c in cols:
            col_dtype = X.schema[c]
            safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
            col_safe_excludes[c] = safe_exclude

            # 【修复】构建联合条件
            keep_mask = pl.lit(True)
            if col_dtype in [pl.Float32, pl.Float64]:
                keep_mask &= ~pl.col(c).is_nan()
            if safe_exclude:
                keep_mask &= ~pl.col(c).is_in(safe_exclude)

            target_col = pl.col(c).filter(keep_mask)
            
            exprs.append(target_col.min().alias(f"{c}_min"))
            exprs.append(target_col.max().alias(f"{c}_max"))
            exprs.append(target_col.n_unique().alias(f"{c}_n_unique"))

        stats = X.select(exprs)
        row = stats.row(0)
        
        initial_cuts_map = {}
        pending_optimization_cols = []

        # 解析统计量, 生成等距切点
        for i, c in enumerate(cols):
            base_idx = i * 3
            min_val, max_val, n_unique = row[base_idx], row[base_idx + 1], row[base_idx + 2]
            safe_exclude = col_safe_excludes[c]

            if min_val is None or max_val is None:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            
            # 低基数检查 (Unique <= N_Bins), 直接取中点切分
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

        # 空箱优化
        if pending_optimization_cols:
            batch_exprs = []
            for c in pending_optimization_cols:
                cuts = initial_cuts_map[c]
                breaks = cuts[1:-1]
                target_col = pl.col(c).filter(~pl.col(c).is_nan())
                safe_exclude = col_safe_excludes[c]
                
                if safe_exclude:
                    target_col = target_col.filter(~pl.col(c).is_in(safe_exclude))
                
                labels = [str(i) for i in range(len(breaks)+1)]
                
                # 批量计算直方图
                batch_exprs.append(
                    target_col.cut(breaks, labels=labels, left_closed=True)
                    .value_counts().implode().alias(f"{c}_counts")
               )

            batch_counts_df = X.select(batch_exprs)
            
            # 解析并剔除 Count=0 的箱
            for c in pending_optimization_cols:
                inner_series: pl.Series = batch_counts_df.get_column(f"{c}_counts")[0]
                # [动态解析] value_counts 返回的 Struct 字段名取决于原始列名 (例如: {"age": 25, "count": 10})
                # 不能硬编码 keys["count"]，必须通过 struct.fields 动态获取第 0 个 (Value) 和第 1 个 (Count) 字段名
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
        执行并行的决策树分箱。

        该方法是 Mars 库的“动力心脏”, 专门针对高 PCR (计算传输比) 任务设计。
        它通过“生产-消费”流水线模式, 将 Polars 的预处理能力与 Sklearn 的拟合能力深度耦合。

        Parameters
        ----------
        X: polars.DataFrame
            特征数据集。
        y: polars.Series
            目标变量。要求已在基类中完成类型对齐 (pl.Series)。
        cols: List[str]
            需要执行决策树分箱的特征列名列表。

        Notes
        -----
        **1. 计算重心前置 **
        - 在 `cart_task_gen` 生成器中, 利用 Polars 的位运算内核极速完成空值和特殊值的过滤。
        - **异构对齐**: 使用生成的 Numpy 掩码 (Mask) 同时对 $x$ 和 $y$ 进行物理切片, 
          确保两端数据行索引在没有任何显式 Join 操作的情况下实现绝对对齐。

        **2. 混合并行调度**
        - 后端选择: 采用 `threading` 后端配合 `n_jobs`。
        - 依据: 由于 `x_clean` 和 `y_clean` 切片已在主进程内存中完成, 使用多线程可实现
          **零拷贝** 传递给 Worker, 规避了多进程频繁序列化大数据块的物流负担。
        - 锁优化: 利用 Sklearn 底层在拟合过程中会释放 GIL 的物理特性, 实现真正的多核利用。

        **3. 内存防护机制**
        - 异常追踪: 引入 `fit_failures_` 属性。任何由于数据极端分布或内存溢出导致的
          单特征失败将被捕获并记录原因, 而不会触发主任务的中断 (Fail-Soft 机制)。

        Algorithm
        ---------
        1. 将标签 $y$ 转换为内存连续的 Numpy 数组, 优化内存预取。
        2. 启动 `cart_task_gen`: 逐列进行源头清洗, 产出纯净切片对。
        3. 线程池调度: Worker 函数并行执行 `DecisionTreeClassifier.fit`。
        4. 汇总结果: 提取树节点阈值, 生成切点并记录异常。
        """
        
        # [优化] Polars Series -> Numpy (使用 zero-copy)
        # y 已经是 pl.Series, 直接 to_numpy 是最快的, 并强制连续内存布局
        y_np = np.ascontiguousarray(y.to_numpy())
        
        if len(y_np) != X.height:
            raise ValueError(f"Target 'y' length mismatch: X({X.height}) vs y({len(y_np)})")
        
        
        n_total_samples = X.height

        def worker(col_name: str, x_clean_np: np.ndarray, y_clean_np: np.ndarray) -> Tuple[str, List[float]]:
            try:
                # 如果 min_samples 是浮点数 (如 0.05), 则基于 总行数(n_total_samples) 计算
                # 而不是基于 过滤后的行数(len(x_clean_np)) 计算
                if isinstance(self.min_samples, float):
                    min_samples_abs = int(np.ceil(self.min_samples * n_total_samples))
                else:
                    min_samples_abs = self.min_samples

                # 安全检查: 如果清洗后的数据量甚至不足以支撑 2 个最小叶子节点
                # 说明该特征在有效值范围内过于稀疏, 不应强行分箱
                if len(x_clean_np) < 2 * min_samples_abs:
                     return col_name, [float('-inf'), float('inf')], "Insufficient clean samples to satisfy global min_samples."
                
                cart = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=min_samples_abs,
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

        # [优化] 任务生成器: 移除生成器内部的冗余转换
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
                # valid_mask 在 Polars 中是 BitMap, filter 之后转 numpy 非常快
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

        # Backend 选型:
        # 如果数据量极大, threading 会受限于 GIL。
        # 但因为 Sklearn 的树拟合大部分是在 C++ 层释放了 GIL 的, 
        # 且任务分发开销 (PCR)在第一阶段很低, 所以 threading 是合理的。
        results = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
            delayed(worker)(name, x, y) for name, x, y in cart_task_gen()
       )
        
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

    该类是 Mars 库的高级核心组件, 将极速预分箱技术 (Native Pre-binning)与基于数学规划的
    最优分箱算法 (OptBinning)深度集成。它旨在为风控模型提供具备单调约束、最优 IV 分布和
    极强鲁棒性的特征切点。

    Attributes
    ----------
    bin_cuts_: Dict[str, List[float]]
        数值型特征最终生成的切点字典。
    cat_cuts_: Dict[str, List[List[Any]]]
        类别型特征的分组规则字典。
    fit_failures_: Dict[str, str]
        记录求解器超时或计算失败的特征原因。

    Notes
    -----
    **核心架构: 双阶段启发式求解**
    1. **Stage 1: Native 粗切**: 利用 `MarsNativeBinner` 快速将连续变量离散化为 20-50 个初始区间 (Pre-bins)。这一步在主进程中通过 Polars 的 Rust 内核完成, 实现了数据的极大压缩。
    2. **Stage 2: MIP/CP 精切**: 将压缩后的统计量送入子进程, 利用数学规划求解器在满足单调性、最小箱占比等约束下, 寻找信息熵最大化的最优解。

    **混合并行策略**
    - **数值型处理**: 采用 `loky` 后端。由于最优求解涉及复杂的 Python 胶水逻辑和外部求解器调用, 通过多进程 (Loky)彻底规避 GIL 锁, 释放多核 CPU 算力。
    - **PCR 优化**: 在任务生成阶段完成“源头清洗”, 仅将“入参即干净”的高纯度 Numpy 数据传递给子进程, 最大限度降低跨进程序列化开销。
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        n_bins: int = 10,
        min_n_bins: int = 1,
        min_bin_size: float = 0.02,
        min_bin_n_event: int = 3,
        n_prebins: int = 50,
        prebinning_method: Literal["quantile", "uniform", "cart"] = "cart",
        min_prebin_size: float = 0.01,
        monotonic_trend: Literal["ascending", "descending", "auto", "auto_asc_desc"] = "auto_asc_desc",
        solver: Literal["cp", "mip"] = "cp",
        time_limit: int = 10,
        cat_cutoff: Optional[int] = 100,
        special_values: Optional[List[Any]] = None,
        missing_values: Optional[List[Any]] = None,
        cart_params: Optional[Dict[str, Any]] = None,
        join_threshold: int = 100,
        n_jobs: int = -1
   ) -> None:
        """
        初始化 MarsOptimalBinner。

        Parameters
        ----------
        features: List[str], optional
            数值型特征的列名白名单。若为 None, fit 时将自动识别所有数值列。
        cat_features: List[str], optional
            类别型特征的列名白名单。若为 None, fit 时将自动识别所有类别列。
        n_bins: int, default=5
            **最大分箱数**。最终生成的有效分箱数量不会超过此值。
        min_n_bins: int, default=1
            **最小分箱数**。强制求解器至少切分出多少个箱子。
            若数据量不足以支撑此约束 (触发水位熔断), 将自动回退到预分箱结果。
        min_bin_size: float, default=0.05
            **最小箱占比约束**。
            指定每个分箱 (不含缺失值和特殊值箱)包含的样本量占总样本量的最小比例 (0.0 ~ 0.5)。
            例如 0.05 表示每箱至少包含 5% 的样本。
        min_bin_n_event: int, default=5
            **最小箱事件数约束**。
            指定每个分箱 (不含缺失值和特殊值箱)包含的事件数 (正样本数) 的最小数量。
            例如 5 表示每箱至少包含 5 个事件。
        n_prebins: int, default=50
            **预分箱数量** (Stage 1)。
            在调用求解器前, 先将连续变量离散化为多少个初始区间。
            值越大, 最终分箱越精细, 但求解速度越慢。建议值 20~100。
        prebinning_method: {'cart', 'quantile', 'uniform'}, default='cart'
            **预分箱策略**。
            - 'cart': 使用决策树进行初始切分, 能较好地保留非线性趋势 (推荐)。
            - 'quantile': 等频切分。
            - 'uniform': 等宽切分。
        min_prebin_size: float, default=0.05
            **预分箱的最小叶子节点占比**。
            仅在 `prebinning_method='cart'` 时有效, 控制决策树生长的精细度。
        monotonic_trend: str, default='auto_asc_desc'
            **单调性约束**。控制分箱后 Event Rate 的趋势: 
            - 'ascending': 单调递增。
            - 'descending': 单调递减。
            - 'auto': 自动选择单调递增或递减 (基于相关性)。
            - 'auto_asc_desc': 尝试升序和降序, 选择 IV 更高的一个 (推荐)。
        solver: {'cp', 'mip'}, default='cp'
            **数学规划求解引擎**。
            - 'cp' (Constraint Programming): 约束编程, 通常处理复杂约束时速度更快 (推荐)。
            - 'mip' (Mixed-Integer Programming): 混合整数规划。
        time_limit: int, default=10
            **求解超时时间** (秒)。
            单个特征的最优分箱求解最大允许耗时。若超时, 将自动回退。
        cat_cutoff: int, optional, default=100
            **类别特征高基数截断阈值**。
            对于类别型特征, 仅保留出现频率最高的 Top-K 个类别, 其余类别将被归并为 
            `"__Mars_Other_Pre__"`, 以减少求解器的搜索空间。
        special_values: List[Any], optional
            **特殊值列表**。
            这些值 (如 -999, -1)将被强制剥离, 分配到独立的负数索引分箱中 (-3, -4...), 
            不参与最优分箱的切分计算。
        missing_values: List[Any], optional
            **自定义缺失值列表**。
            除了标准的 Null/NaN 外, 额外视作缺失值的内容。会被归入 Missing 箱 (索引 -1)。
        join_threshold: int, default=100
            **Transform 性能调优阈值**。
            在 `transform` 阶段, 若类别特征的基数超过此值, 将自动启用 `Hash Join` 模式
            替代 `Replace` 模式, 以显著提升宽表转换性能。
        cart_params: Dict[str, Any], optional
            传递给决策树分箱器 (`sklearn.tree.DecisionTreeClassifier`) 的额外参数字典。
            仅在 `prebinning_method='cart'` 时有效。
        n_jobs: int, default=-1
            **并行核心数**。
            - `-1`: 使用所有可用核心 (保留一个)。
            - `1`: 单线程运行。
            - `N`: 指定使用的核心数。
        """
        super().__init__(
            features=features, cat_features=cat_features, n_bins=n_bins,
            special_values=special_values, missing_values=missing_values,
            join_threshold=join_threshold, n_jobs=n_jobs
       )
        self.min_n_bins = min_n_bins
        self.min_bin_size = min_bin_size
        self.min_bin_n_event = min_bin_n_event
        self.n_prebins = n_prebins
        self.prebinning_method = prebinning_method
        
        if self.prebinning_method not in ["cart", "quantile", "uniform"]:
            raise ValueError("prebinning_method must be one of {'cart', 'quantile', 'uniform'}")
            
        self.min_prebin_size = min_prebin_size
        self.monotonic_trend = monotonic_trend
        self.solver = solver
        self.time_limit = time_limit
        self.cat_cutoff = cat_cutoff
        self.cart_params = cart_params if cart_params is not None else {}
        
        # 尝试导入 optbinning
        try:
            from optbinning import OptimalBinning
            self.OptimalBinning = OptimalBinning
        except ImportError:
            logger.warning("⚠️ 'optbinning' not installed. Fallback logic might be triggered.")

    def _fit_impl(self, X: pl.DataFrame, y: pl.Series = None, **kwargs) -> None:
        """
        [拟合引擎调度器] 自动执行特征识别与任务流分发。

        执行流程: 
        1. 执行 `y` 的类型转换与内存连续化优化。
        2. 特征自动洗牌: 将数值型特征分流至 `_fit_numerical_impl`, 类别型分流至 
           `_fit_categorical_impl`。
        3. 状态管理: 注册并初始化全空列的占位规则。

        Parameters
        ----------
        X: polars.DataFrame
            训练集特征数据。
        y: polars.Series
            目标变量。要求必须可转换为二分类的 int32 数组。
        """
        # 缓存数据引用, 仅用于 transform 阶段请求 return_type='woe' 时的延迟计算
        self._cache_X = X
        self._cache_y = y
        
        if y is None:
            raise ValueError("Optimal Binning requires target 'y' to calculate IV/WOE.")

        y_np = np.ascontiguousarray(y.to_numpy()).astype(np.int32)
        
        # 获取 y 的名称 (如果 y 是 Series)
        y_name = getattr(y, "name", None)
        
        # 确定目标列: 如果没有指定 features, 则获取 X 的所有列, 但必须排除掉 y 所在的列
        if not self.features:
            all_target_cols = [c for c in X.columns if c != y_name]
        else:
            all_target_cols = self.features
        cat_set = set(self.cat_features)
        
        num_cols = []
        cat_cols = []
        null_cols = [] 

        for c in all_target_cols:
            if c not in X.columns: continue
            
            # 优先判定类别
            if c in cat_set:
                cat_cols.append(c)
                continue
            
            # 判定全空
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 判定数值
            if self._is_numeric(X[c]):
                num_cols.append(c)

        if not num_cols and not cat_cols and not null_cols:
            logger.warning("No valid numeric or categorical columns found.")
            return
        
        self.fit_failures_: Dict[str, str] = {}

        for c in null_cols:
            self.bin_cuts_[c] = []

        if num_cols:
            self._fit_numerical_impl(X, y_np, num_cols)

        if cat_cols:
            self._fit_categorical_impl(X, y_np, cat_cols)
            
        if self.fit_failures_:
            num_fails = len([k for k in self.fit_failures_ if k in num_cols])
            cat_fails = len([k for k in self.fit_failures_ if k in cat_cols])
            logger.warning(
                f"⚠️ MarsOptimalBinner: {len(self.fit_failures_)} features encountered issues "
                f"({num_fails} num, {cat_fails} cat). Fallback applied. "
                f"Check `.fit_failures_` for details. Sample: {list(self.fit_failures_.items())[:2]}"
           )

    @time_it
    def _fit_numerical_impl(self, X: pl.DataFrame, y_np: np.ndarray, num_cols: List[str]) -> None:
        """
         数值型特征的混合动力求解流水线。

        Optimization
        ------------
        - **计算重心前置**: 在 `num_task_gen` 内部利用 Polars Rust 引擎进行极速过滤, 
          Worker 仅接收经过净化的 Numpy 视图。
        - **两阶段联动**: 先调用 `MarsNativeBinner` 获取粗粒度切点, 
          随后将其作为 `user_splits` 注入 `optbinning`, 极大缩小了数学规划的搜索空间。
        - **并发控制**: 使用 `loky` 后端。由于单个特征的最优求解耗时较长 ($PCR \gg 0$), 
          支付跨进程通讯成本以换取独立 CPU 核心的满载运行是非常合算的。

        Parameters
        ----------
        X: polars.DataFrame
            特征数据。
        y_np: numpy.ndarray
            已经过内存对齐和类型转换的标签数组。
        num_cols: List[str]
            待处理的数值列名。
        """
        
        pre_binner = MarsNativeBinner(
            features=num_cols,
            method=self.prebinning_method, 
            n_bins=self.n_prebins, 
            special_values=self.special_values,
            missing_values=self.missing_values,
            min_samples=self.min_prebin_size,
            cart_params=self.cart_params,
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
        
        # 获取全局样本总数
        n_total_samples = X.height

        def num_worker(col: str, pre_cuts: List[float], col_data: np.ndarray, y_data: np.ndarray) -> Tuple[str, List[float], Optional[str]]:
            fallback_res = (col, pre_cuts, None)
            try:
                # 计算基于"总体"的绝对 min_bin_size
                if isinstance(self.min_bin_size, float):
                    min_bin_size_abs = int(np.ceil(self.min_bin_size * n_total_samples))
                else:
                    min_bin_size_abs = self.min_bin_size # 如果用户初始化时就传了整数
                
                # 绝对值检查
                # 如果当前数据量 < 最小分箱数 * 最小单箱大小, 直接回退
                if len(col_data) < self.min_n_bins * min_bin_size_abs:
                     return fallback_res

                if len(col_data) < 10 or np.var(col_data) < 1e-8:
                    return col, pre_cuts, "Low variance or insufficient samples"

                # 将绝对值转换回当前数据的相对比例
                # OptBinning 源码限制 min_bin_size 必须在 (0, 0.5] 之间
                    # 如果占比 > 0.5，意味着无法分出 2 个箱子，求解器会无解报错。
                    # 强制截断为 0.5 是为了让求解器至少能尝试分出 2 个箱 (或证明不可分)。
                current_ratio = min_bin_size_abs / len(col_data)

                # 如果比例超过 0.5, 说明当前数据甚至无法切分出两个满足要求的箱子
                # (例如: 要求每箱至少500人, 但当前只有800人, 500/800 = 0.625 > 0.5)
                if current_ratio > 0.5:
                    return fallback_res
                
                # 为了防止浮点精度问题导致正好等于 0.50000001 报错, 稍微做个截断保护
                # 虽然前面的 if 已经拦截了, 但为了保险起见
                current_ratio = min(current_ratio, 0.5)

                raw_splits = np.array(pre_cuts[1:-1])
                if len(raw_splits) > 1:
                    diffs = np.diff(raw_splits)
                    # 剔除过于接近的切点, 防止求解器报错
                    mask = np.concatenate(([True], diffs > 1e-6))
                    user_splits = raw_splits[mask]
                else:
                    user_splits = raw_splits

                if len(user_splits) == 0:
                    return fallback_res
                
                opt = self.OptimalBinning(
                    name=col, 
                    dtype="numerical", 
                    solver=self.solver,
                    monotonic_trend=self.monotonic_trend, 
                    user_splits=user_splits, 
                    min_n_bins=self.min_n_bins, 
                    max_n_bins=self.n_bins, 
                    time_limit=self.time_limit, 
                    min_bin_size=current_ratio,
                    min_bin_n_event=self.min_bin_n_event,
                    verbose=False
               )
                opt.fit(col_data, y_data)
                
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    res_cuts = [float('-inf')] + list(opt.splits) + [float('inf')]
                    return col, res_cuts, None
            
                # 捕获求解器非最优状态 (如 TIMEOUT)
                return col, pre_cuts, f"Solver status: {opt.status}"
                
            except Exception as e:
                # 捕获代码级异常
                return col, pre_cuts, f"{type(e).__name__}: {str(e)}"
                
        # 预处理排除值
        raw_exclude = self.special_values + self.missing_values
        def num_task_gen():
            """
            通过 yield 纯净的 NumPy 数组, 触发 joblib 的 mmap 共享内存优化。
            """
            for c in active_cols:
                # 类型感知与安全过滤列表获取
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
                
                # 获取 Series 指针, 不使用 select, 避免 DataFrame 物化开销
                series = X.get_column(c)
                
                # 构建 Polars 过滤掩码
                # 基础过滤: 非 null
                valid_mask = series.is_not_null()
                
                # 针对数值特征增加: 非 NaN 过滤
                if col_dtype in [pl.Float32, pl.Float64]:  # 仅对浮点数检查 NaN
                    valid_mask &= (~series.is_nan())
                
                # 针对业务特殊值进行排除, 如 -999, -998
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))
                
                # 将位掩码转换为 NumPy 布尔数组, 用于 y 的快速切片
                mask_np = valid_mask.to_numpy()
                
                # 如果过滤后样本量不足, 直接跳过此列, 减少并行开销
                if not mask_np.any():
                    continue

                # 特征列 X 处理
                col_np = (
                    series.filter(valid_mask)
                    .cast(pl.Float32)
                    .to_numpy(writable=False)
               )

                # [性能优化] 
                    # 1. CPU 缓存友好: 连续内存能利用 CPU 预取指令。
                    # 2. 序列化优化: joblib/pickle 对连续数组有特殊优化、，
                    #    在 'loky' 后端传输大数据块时，非连续数组可能会触发昂贵的内存重排和拷贝。
                if not col_np.flags['C_CONTIGUOUS']:
                    col_np = np.ascontiguousarray(col_np)

                # 标签列 Y 处理
                clean_y = y_np[mask_np]

                # 产出任务数据包。此时产出的全部是纯粹的物理内存块, joblib 会自动识别并优化传输
                yield c, pre_cuts_map[c], col_np, clean_y
        
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(num_worker)(c, cuts, data, y) for c, cuts, data, y in num_task_gen()
       )
        
        for col, cuts, error_msg in results:
            self.bin_cuts_[col] = cuts
            if error_msg:
                self.fit_failures_[col] = error_msg

    def _fit_categorical_impl(self, X: pl.DataFrame, y_np: np.ndarray, cat_cols: List[str]) -> None:
        """
        [Pipeline] 类别型特征的处理流水线。

        特别针对大规模类别型数据进行了逻辑增强。

        Notes
        -----
        - **长尾截断路由 (__Mars_Other_Pre__)**: 针对频数极低或基数极大的类别, 自动执行 
          `Top-K` 截断, 并将长尾数据归并为特殊的 `__Mars_Other_Pre__` 类别。
        - **数据源头净化**: 在任务生成器中完成字符串映射和空值隔离, 
          Worker 进程拿到的直接是满足 `optbinning` 输入要求的 `pl.Utf8` 映射数据。
        - **并行后端**: 使用 `loky` 后端。

        Parameters
        ----------
        X: polars.DataFrame
            特征数据。
        y_np: numpy.ndarray
            标签数组。
        cat_cols: List[str]
            待处理的类别列名。
        """
        raw_exclude = self.special_values + self.missing_values

        
        def cat_worker(col: str, clean_data: np.ndarray, clean_y: np.ndarray) -> Tuple[str, Optional[List[List[Any]]], Optional[str]]:
            try:
                # Top-K 预处理: 将长尾类别归为 "__Mars_Other_Pre__"
                if self.cat_cutoff is not None:
                    unique_vals, counts = np.unique(clean_data, return_counts=True)
                    if len(unique_vals) > self.cat_cutoff:
                        top_indices = np.argsort(-counts)[:self.cat_cutoff]
                        top_vals = set(unique_vals[top_indices])
                        mask_keep = np.isin(clean_data, list(top_vals))
                        # [隐式契约] 
                        # "__Mars_Other_Pre__" 这个特殊字符串必须与 MarsBinnerBase.transform 中的默认路由逻辑保持一致。
                        # 这里的归并操作对应 Transform 阶段的 "Other" (-2) 索引。
                        clean_data = np.where(mask_keep, clean_data, "__Mars_Other_Pre__")

                opt = self.OptimalBinning(
                    name=col, dtype="categorical", solver=self.solver,
                    max_n_bins=self.n_bins, 
                    time_limit=self.time_limit,
                    cat_cutoff=0.05, 
                    verbose=False
               )
                opt.fit(clean_data, clean_y)
                
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    return col, opt.splits, None
                return col, None, f"Solver status: {opt.status}"
            except Exception as e:
                return col, None, f"{type(e).__name__}: {str(e)}"

        
        def cat_task_gen():
            for c in cat_cols:
                series = X.get_column(c)
                col_dtype = series.dtype
                
                if self.cat_cutoff is not None:
                    # 1. 极速计算 Value Counts
                    # top_k 算子在 Polars 中有专门优化
                    top_k_df = series.value_counts(sort=True).head(self.cat_cutoff)
                    top_vals = top_k_df.get_column(c)
                    
                    # 2. 只有在 top_k 里的才保留，否则置为 Other
                    # 注意: 这里使用 is_in (Hash Set) 过滤
                    series = pl.when(series.is_in(top_vals))\
                               .then(series)\
                               .otherwise(pl.lit("__Mars_Other_Pre__"))
                
                # 获取该列的安全排除列表
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
                
                # 过滤条件: 非空 且 不在排除列表中
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

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(cat_worker)(c, data, y) for c, data, y in cat_task_gen()
       )
        
        for col, splits, error_msg in results:
            if splits is not None:
                self.cat_cuts_[col] = splits
            if error_msg:
                self.fit_failures_[col] = error_msg