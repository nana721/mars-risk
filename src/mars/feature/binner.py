from joblib import Parallel, delayed
from typing import List, Dict, Optional, Union, Any, Literal, Tuple, Set
import multiprocessing
import gc

import numpy as np
import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier

from mars.core.base import MarsTransformer
from mars.utils.logger import logger
from mars.utils.decorators import time_it

class MarsBinnerBase(MarsTransformer):
    """
    [分箱器基类] MarsBinnerBase

    所有分箱器的抽象基类，封装了通用的 Transform 逻辑、指标计算逻辑和状态管理。
    子类只需实现具体的 `fit` 策略来填充 `bin_cuts_` 或 `cat_cuts_`。

    Attributes
    ----------
    bin_cuts_ : Dict[str, List[float]]
        数值型特征的分箱切点字典。
        格式: ``{col_name: [-inf, split1, split2, ..., inf]}``
    cat_cuts_ : Dict[str, List[List[Any]]]
        类别型特征的分箱组合规则。
        格式: ``{col_name: [['A', 'B'], ['C'], ...]}``
    bin_mappings_ : Dict[str, Dict[int, str]]
        分箱索引到可视化标签的映射字典。
    bin_woes_ : Dict[str, Dict[int, float]]
        分箱索引到 WOE 值的映射字典。
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
        初始化分箱器基类。

        Parameters
        ----------
        features : List[str], optional
            数值特征列表。
        cat_features : List[str], optional
            类别特征列表。
        n_bins : int, default=5
            目标分箱数。
        special_values : List[Union[int, float, str]], optional
            特殊值列表 (独立成箱)。
        missing_values : List[Union[int, float, str]], optional
            自定义缺失值列表。
        join_threshold : int, default=100
            Transform 时使用 Join 替代 Replace 的类别基数阈值。
        n_jobs : int, default=-1
            并行线程数。
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
        [Helper] 类型安全清洗函数。
        
        **为什么这样做？**
        Polars 是强类型的。如果列是 Int64，但 `values` 列表中包含字符串 "unknown"，
        直接调用 `pl.col(c).is_in(values)` 会导致 Schema Error 或崩溃。
        
        **这行代码运行后有啥用？**
        根据列的物理类型，自动剔除不兼容的值。例如 Int 列只保留 Int 配置项，
        String 列则将所有配置项转为 String。

        Parameters
        ----------
        dtype : pl.DataType
            当前处理列的 Polars 数据类型。
        values : List[Any]
            用户配置的缺失值或特殊值列表。

        Returns
        -------
        List[Any]
            清洗后的类型安全列表。
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
    def _materialize_woe(self) -> None:
        """
        [WOE 物化引擎]
        
        **为什么这样做？**
        Transform 阶段如果每次都实时计算 WOE，对于 2000+ 特征的宽表，
        Polars 可能会构建过大的计算图导致内存飙升。
        
        **做了什么？**
        1. 将 WOE 计算拆分为 batch (200列一组)。
        2. 使用 Eager 模式立即计算并回收内存 (`gc.collect`)。
        3. 使用 group_by 极速聚合而不是构建复杂的 when-then 逻辑。
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

        batch_size = 200 
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

    # def transform(
    #     self, 
    #     X: Any, 
    #     return_type: Literal["index", "label", "woe"] = "index", 
    #     lazy: bool = False
    # ) -> Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]:
    #     """
    #     对数据应用分箱转换。

    #     Parameters
    #     ----------
    #     X : Any
    #         输入数据 (Pandas/Polars DataFrame)。
    #     return_type : Literal["index", "label", "woe"], default="index"
    #         返回类型：
    #         - 'index': 返回 Int16 的箱索引 (-1=Missing, 0, 1...)。最快。
    #         - 'label': 返回字符串标签 (如 "01_[0.5, 1.2)")。
    #         - 'woe': 返回 Float64 的 WOE 编码值。
    #     lazy : bool, default=False
    #         是否返回 LazyFrame。如果为 True，不会触发计算，适合构建计算图。

    #     Returns
    #     -------
    #     Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]
    #         转换后的数据框。
    #     """
    #     # 1. 智能输入处理：确保是 Polars 对象
    #     if isinstance(X, pl.LazyFrame):
    #         X_pl = X
    #     else:
    #         X_pl = self._ensure_polars(X)
        
    #     # 2. 模式切换：如果需要 Lazy，转为 LazyFrame
    #     if lazy and isinstance(X_pl, pl.DataFrame):
    #         X_pl = X_pl.lazy()
        
    #     # 3. 执行核心逻辑
    #     res = self._transform_impl(X_pl, return_type=return_type)
        
    #     # 4. 输出格式控制
    #     if not lazy:
    #         if isinstance(res, pl.LazyFrame): res = res.collect()
    #         if isinstance(X, pd.DataFrame): return res.to_pandas()
    #     return res

    def _transform_impl(
        self, 
        X: Union[pl.DataFrame, pl.LazyFrame], 
        return_type: Literal["index", "label", "woe"] = "index",
        lazy: bool = False
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        [混合动力分箱转换实现] 
        核心转换逻辑，兼容数值与类别特征，支持 Eager/Lazy。

        Parameters
        ----------
        X : Union[pl.DataFrame, pl.LazyFrame]
            待转换数据。
        return_type : Literal["index", "label", "woe"]
            输出格式。

        Returns
        -------
        Union[pl.DataFrame, pl.LazyFrame]
            转换后的数据。
        """
        exprs = []
        temp_join_cols = []
        
        # 索引协议常量: 与下游 Profiler 对齐
        IDX_MISSING = -1
        IDX_OTHER   = -2
        IDX_SPECIAL_START = -3

        # 自动触发 WOE 计算
        if return_type == "woe" and not self.bin_woes_:
            self._materialize_woe()

        # 获取 Schema (Lazy/Eager 兼容写法)
        schema_map = X.collect_schema() if isinstance(X, pl.LazyFrame) else X.schema
        current_columns = schema_map.names()
        
        all_train_cols = list(set(
            list(self.bin_cuts_.keys()) + 
            (list(self.cat_cuts_.keys()) if hasattr(self, 'cat_cuts_') else [])
        ))

        for col in all_train_cols:
            if col not in current_columns: continue
            
            # --- [关键] 计算类型安全值 ---
            # 这一步至关重要，防止例如在 Int 列上查询 "unknown" 导致的崩溃
            col_dtype = schema_map[col]
            safe_missing_vals = self._get_safe_values(col_dtype, self.missing_values)
            safe_special_vals = self._get_safe_values(col_dtype, self.special_values)
            is_numeric_col = col_dtype in self.NUMERIC_DTYPES

            # =========================================================
            # Part A: 数值型分箱 (Numeric Binning)
            # =========================================================
            if col in self.bin_cuts_:
                cuts = self.bin_cuts_[col]
                
                # 1. 缺失值逻辑: Is Null OR Is Missing Val
                missing_cond = pl.col(col).is_null() 
                if is_numeric_col: missing_cond |= pl.col(col).cast(pl.Float64).is_nan()
                for v in safe_missing_vals: missing_cond |= (pl.col(col) == v)
                
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 2. 正常分箱逻辑: Cut
                breaks = cuts[1:-1] if len(cuts) > 2 else []
                col_mapping = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}
                
                if not breaks:
                    col_mapping[0] = "00_[-inf, inf)"
                    layer_normal = pl.lit(0, dtype=pl.Int16)
                else:
                    for i in range(len(cuts) - 1):
                        low, high = cuts[i], cuts[i+1]
                        col_mapping[i] = f"{i:02d}_[{low:.3g}, {high:.3g})"
                    # 显式生成 labels 确保 cast(Int16) 成功，修复 PSI=0 Bug
                    bin_labels = [str(i) for i in range(len(breaks) + 1)]
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
                        current_branch = pl.when(pl.col(col) == v).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)
                
                # 组合: Missing -> Special -> Normal
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
                exprs.append(final_idx_expr.replace(woe_map).cast(pl.Float64).alias(f"{col}_woe") if woe_map else pl.lit(0.0).alias(f"{col}_woe"))
            else:
                str_map = {str(k): v for k, v in self.bin_mappings_.get(col, {}).items()}
                exprs.append(final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin"))

        # 清理 Join 产生的临时列
        return X.with_columns(exprs).drop(temp_join_cols).lazy() if lazy else X.with_columns(exprs).drop(temp_join_cols)

    @time_it
    def compute_bin_stats(self, X: pl.DataFrame, y: Any) -> pl.DataFrame:
        """
        [极速指标引擎 v2.0] 计算全量分箱指标。
        
        **核心优化**:
        使用 `unpivot` + `group_by` 实现矩阵化聚合，而非循环逐列聚合。
        
        Parameters
        ----------
        X : pl.DataFrame
            特征数据。
        y : Any
            目标标签。

        Returns
        -------
        pl.DataFrame
            包含 feature, bin_index, count, bad_rate, woe, iv, ks 等指标的宽表。
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
    
    完全基于 Polars 和 Sklearn 实现的高性能分箱器，适用于大规模数据的快速特征工程。
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        method: Literal["cart", "quantile", "uniform"] = "quantile",
        n_bins: int = 5,
        special_values: Optional[List[Union[int, float, str]]] = None,
        missing_values: Optional[List[Union[int, float, str]]] = None,
        min_samples: float = 0.05,
        cart_params: Optional[Dict[str, Any]] = None,
        remove_empty_bins: bool = False,
        join_threshold: int = 100,
        n_jobs: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        method : {'cart', 'quantile', 'uniform'}, default='quantile'
            分箱策略。
        min_samples : float, default=0.05
            仅 method='cart' 有效，叶子节点最小样本比例。
        cart_params : Dict, optional
            传递给 DecisionTreeClassifier 的参数。
        remove_empty_bins : bool, default=False
            是否剔除空箱 (仅 method='uniform' 有效)。
        """
        super().__init__(
            features=features, n_bins=n_bins, 
            special_values=special_values, missing_values=missing_values,
            join_threshold=join_threshold, n_jobs=n_jobs
        )
        self.method = method
        self.min_samples = min_samples
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
        训练实现的入口函数。

        Parameters
        ----------
        X : pl.DataFrame
            训练数据。
        y : Optional[Any]
            目标变量 (仅 CART 分箱需要)。
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
        # [优化] 极速预过滤 (常量特征剔除)
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
        执行极速等频分箱 (Quantile Binning)。
        
        **核心优化**:
        不使用 Python 循环逐列计算，而是构建一个包含所有列分位数计算的
        巨大 Polars 表达式列表，发送给 Rust 引擎一次性执行。
        """
        # 1. 构建分位点
        if self.n_bins <= 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1].tolist()
            
        raw_exclude = self.special_values + self.missing_values
        
        # 2. 构建表达式列表 (Flattened)
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
        
        **核心优化**:
        分为两阶段。第一阶段批量计算 Min/Max/Unique。
        第二阶段（可选）批量计算 Histogram 以剔除空箱。
        """
        raw_exclude = self.special_values + self.missing_values
        
        # --- 阶段 1: 基础统计量 ---
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

        # --- 阶段 2: 空箱优化 (可选) ---
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

    def _fit_cart_parallel(self, X: pl.DataFrame, y: Any, cols: List[str]) -> None:
        """
        执行并行的决策树分箱 (Decision Tree Binning)。
        
        **核心优化**:
        1. 使用 Generator (task_generator) 惰性产出数据，避免一次性复制所有列数据到内存。
        2. Worker 函数只接收 Numpy 数组，减少序列化开销。
        3. 在 Generator 内部利用 Polars Rust 内核进行极速过滤和类型转换 (Float32)。
        """
        y_np = np.array(y)
        if len(y_np) != X.height:
            raise ValueError(f"Target 'y' length mismatch: X({X.height}) vs y({len(y_np)})")

        # 定义 Worker 逻辑：纯 Sklearn 拟合
        def worker(col_name: str, x_clean_np: np.ndarray, y_clean_np: np.ndarray) -> Tuple[str, List[float]]:
            try:
                if len(x_clean_np) < self.n_bins * 10: 
                    return col_name, [float('-inf'), float('inf')]
                
                cart = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=self.min_samples,
                    **self.cart_params
                )
                cart.fit(x_clean_np, y_clean_np)
                cuts = cart.tree_.threshold[cart.tree_.threshold != -2]
                cuts = np.sort(np.unique(cuts)).tolist()
                return col_name, [float('-inf')] + cuts + [float('inf')]
            except Exception:
                return col_name, [float('-inf'), float('inf')]
        
        raw_exclude = self.special_values + self.missing_values
        
        # 任务生成器：按需生成数据
        def cart_task_gen():
            # 提前将 y 转化为连续内存的 Numpy，切片更快
            y_np_local = np.ascontiguousarray(y_np) 

            for c in cols:
                # 1. 获取列 Schema 和排除值
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

                # 2. 直接获取 Series，不通过 select 创建 DataFrame
                series = X.get_column(c)

                # 3. 计算布尔掩码 (Mask) —— 这一步在 Rust 内核跑，极速
                # 基础条件：非空
                valid_mask = series.is_not_null()

                # 数值型额外追加：非 NaN
                if col_dtype in self.NUMERIC_DTYPES:
                    valid_mask &= (~series.is_nan())

                # 业务排除值追加
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))

                # 4. 检查是否有有效数据
                if not valid_mask.any():
                    continue

                # 5. [关键优化] 同步过滤
                # x 端：利用 Polars 过滤并转 Numpy
                x_clean = series.filter(valid_mask).cast(pl.Float32).to_numpy(writable=False).reshape(-1, 1)
                
                # y 端：直接用 Numpy 掩码切片（这是物理层面的内存移动，极快）
                # valid_mask.to_numpy() 不会复制数据，只是产生一个布尔视图
                y_clean = y_np_local[valid_mask.to_numpy()]

                yield c, x_clean, y_clean
        # threading
        results = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
            delayed(worker)(name, x, y) for name, x, y in cart_task_gen()
        )
        
        for col_name, cuts in results:
            self.bin_cuts_[col_name] = cuts


class MarsOptimalBinner(MarsBinnerBase):
    """
    [混合动力分箱引擎] MarsOptimalBinner

    结合了 Native 极速预分箱和 OptBinning 数学规划的最优分箱器。
    支持数值型 (MIP) 和类别型 (Categorical Binning) 特征。
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        n_bins: int = 5,
        n_prebins: int = 50,
        prebinning_method: Literal["quantile", "uniform", "cart"] = "cart",
        monotonic_trend: str = "auto_asc_desc",
        solver: str = "cp",
        time_limit: int = 10,
        cat_cutoff: Optional[int] = 100,
        special_values: Optional[List[Any]] = None,
        missing_values: Optional[List[Any]] = None,
        join_threshold: int = 100,
        n_jobs: int = -1
    ) -> None:
        """
        Parameters
        ----------
        n_prebins : int, default=50
            预分箱数量。
        prebinning_method : str, default='quantile'
            预分箱方法。
        solver : str, default='cp'
            OptBinning 求解器 ('cp' 或 'mip')。
        cat_cutoff : int, optional
            类别特征 Top-K 截断阈值。
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

    def _fit_impl(self, X: pl.DataFrame, y: Optional[Any] = None, **kwargs) -> None:
        """
        训练入口：分流数值型和类别型特征到不同的 Pipeline。
        """
        if y is None:
            raise ValueError("Optimal Binning requires target 'y' to calculate IV/WOE.")

        y_np = np.ascontiguousarray(np.array(y)).astype(np.int32)
        
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
        [Pipeline] 数值型特征混合动力处理流水线。
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
                from optbinning import OptimalBinning
                
                # 数据清洗 (Worker 内处理，避免主进程卡顿)
                # valid_mask = ~np.isnan(col_data)
                # valid_data = col_data[valid_mask]
                # 建议改为直接使用，更清爽：
                if len(col_data) < 10 or np.var(col_data) < 1e-8:
                    return fallback_res
                
                # OptBinning 需要对应的 y
                # valid_y = y_np[valid_mask]
                # valid_y = y_data[valid_mask]

                # 提取用户切点 (去除 inf)
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

        # # 优化生成器：减少 DataFrame 创建和内存复制
        # def task_gen():
        #     for c in active_cols:
        #         series = X.get_column(c) # 比 select 快
        #         # 显式转 float，确保 OptBinning 兼容且 Zero-Copy (如果可能)
        #         if series.dtype in [pl.Float32, pl.Float64]:
        #             col_np = series.to_numpy()
        #         else:
        #             col_np = series.cast(pl.Float32).to_numpy()
        #         yield c, pre_cuts_map[c], col_np
        raw_exclude = self.special_values + self.missing_values

        def num_task_gen():
            for c in active_cols:
                # 1. 获取该列对应的类型安全排除列表 (防止类型冲突)
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
                
                # 2. 利用 Polars 极速过滤
                # 我们只需要那些：非空、非 NaN 且不在排除列表中的值
                series = X.get_column(c)
                
                # 构建过滤表达式
                valid_mask = series.is_not_null()
                if col_dtype in self.NUMERIC_DTYPES:
                    valid_mask &= (~series.is_nan())
                
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))
                
                # 3. 过滤数据并转换
                # 注意：y_np 也需要同步过滤，所以这里用带 mask 的方式
                clean_series = series.filter(valid_mask)
                
                # 如果这一列被过滤干了，就跳过
                if clean_series.len() == 0:
                    continue

                # --- 定位到 task_gen 循环最后 ---
                # 【修改】预先将 Polars Mask 转为 Numpy 布尔数组
                valid_mask_np = valid_mask.to_numpy()

                # 显式转为 Float32 以优化传输
                col_np = clean_series.cast(pl.Float32).to_numpy()

                # 【修改】直接使用 Numpy Mask 切片，速度极快
                clean_y = y_np[valid_mask_np]

                yield c, pre_cuts_map[c], col_np, clean_y
        
        # [CRITICAL FIX] 使用 threading 后端以避免序列化开销
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(num_worker)(c, cuts, data, y) for c, cuts, data, y in num_task_gen()
        )
        
        for col, cuts in results:
            self.bin_cuts_[col] = cuts

    def _fit_categorical_impl(self, X: pl.DataFrame, y_np: np.ndarray, cat_cols: List[str]) -> None:
        """
        [Pipeline] 类别型特征处理流水线 (带 Top-K 优化与源头清洗)。
        """
        raw_exclude = self.special_values + self.missing_values

        # 1. 更加健壮的 Worker
        def cat_worker(col: str, clean_data: np.ndarray, clean_y: np.ndarray) -> Tuple[str, Optional[List[List[Any]]]]:
            try:
                from optbinning import OptimalBinning
                # 此时数据已经是清洗过的且转为字符串了
                
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