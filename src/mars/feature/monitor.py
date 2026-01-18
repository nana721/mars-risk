# mars/monitor/feature_monitor.py

from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union, Literal
import numpy as np
import polars as pl
import pandas as pd

from mars.core.base import MarsBaseEstimator
from mars.feature.binner import MarsBinnerBase
from mars.utils.logger import logger
from mars.utils.decorators import time_it

@dataclass
class MarsMonitorReport:
    """监控结果容器"""
    summary_table: pd.DataFrame  # L1: 趋势宽表
    detail_table: pd.DataFrame   # L2: 分箱明细长表
    
    def get_summary(self, metric: str = "PSI") -> pd.DataFrame:
        return self.summary_table

class MarsFeatureMonitor(MarsBaseEstimator):
    """
    [特征监控器] MarsFeatureMonitor (Fixed V2)
    """
    
    def __init__(self, binner: MarsBinnerBase):
        super().__init__()
        if not hasattr(binner, "bin_cuts_") or not binner.bin_cuts_:
            raise ValueError("Provided MarsBinner must be fitted.")
        self.binner = binner
        self.features = list(binner.bin_cuts_.keys()) + list(binner.cat_cuts_.keys())
        self._baseline_dist: Optional[pl.DataFrame] = None 
        self._is_fitted = False

    @time_it
    def fit(self, X_train: Any, y_train: Any = None) -> "MarsFeatureMonitor":
        """[模式 A/B] 显式设定基准"""
        logger.info("⚙️ [Monitor] Establishing baseline...")
        X_pl = self._ensure_polars(X_train)
        
        # 1. 转换得到箱号 (返回结果包含 raw cols + bin cols)
        X_trans = self.binner.transform(X_pl.select(self.features), return_type="index")
        
        # [FIX] 提取 _bin 列并重命名为 feature 原名
        # 这样 unpivot 出来的 bin_idx 才是真正的箱号
        bin_cols = [f"{c}_bin" for c in self.features]
        rename_map = {f"{c}_bin": c for c in self.features}
        
        X_binned = X_trans.select(bin_cols).rename(rename_map)
        
        # 2. 计算全局分布
        self._baseline_dist = (
            X_binned.lazy()
            .unpivot(variable_name="feature", value_name="bin_idx")
            .with_columns(pl.col("bin_idx").cast(pl.Int16)) # 强转类型
            .group_by(["feature", "bin_idx"])
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum().over("feature")).alias("pct_expected")
            )
            .select(["feature", "bin_idx", "pct_expected"])
            .collect()
        )
        self._is_fitted = True
        return self

    @time_it
    def calculate(
        self, 
        data: Any, 
        target: str = "target", 
        dt_col: Optional[str] = None, 
        period: str = "month",
        baseline: Literal["fixed", "auto"] = "fixed"
    ) -> MarsMonitorReport:
        """[核心] 计算分组监控指标"""
        X_pl = self._ensure_polars(data)
        
        # 1. 时间处理
        group_col = f"_grp_{period}"
        if dt_col:
            if period == 'month':
                time_expr = pl.col(dt_col).cast(pl.Date).dt.truncate("1mo").cast(pl.String).str.slice(0, 7)
            elif period == 'day':
                time_expr = pl.col(dt_col).cast(pl.Date).cast(pl.String)
            elif period == 'week':
                time_expr = pl.col(dt_col).cast(pl.Date).dt.truncate("1w").cast(pl.String)
            else:
                time_expr = pl.col(dt_col).cast(pl.String)
            X_pl = X_pl.with_columns(time_expr.alias(group_col))
        else:
            raise ValueError("dt_col is required.")

        # 2. 分箱映射
        logger.info("🔄 [Monitor] Binning data...")
        X_trans = self.binner.transform(X_pl.select(self.features), return_type="index")
        
        # [FIX] 关键修复：只取 _bin 列并重命名
        bin_cols = [f"{c}_bin" for c in self.features]
        rename_map = {f"{c}_bin": c for c in self.features}
        
        # work_df 只包含: group, target, feature1(bin_idx), feature2(bin_idx)...
        work_df = pl.concat([
            X_pl.select([group_col, target]),
            X_trans.select(bin_cols).rename(rename_map)
        ], how="horizontal")

        # 3. 聚合统计
        logger.info("∑ [Monitor] Aggregating detailed stats...")
        stats_long = (
            work_df.lazy()
            .unpivot(
                index=[group_col, target],
                on=self.features, # 现在这里对应的是箱号列了
                variable_name="feature",
                value_name="bin_idx"
            )
            .with_columns(pl.col("bin_idx").cast(pl.Int16)) # 强转 Int16
            .group_by([group_col, "feature", "bin_idx"])
            .agg([
                pl.len().alias("n_bin"),
                pl.col(target).sum().alias("n_bad")
            ])
            .with_columns((pl.col("n_bin") - pl.col("n_bad")).alias("n_good"))
            .collect()
        )

        # 4. 确定 PSI 基准
        if baseline == "fixed":
            if not self._is_fitted:
                raise ValueError("Monitor not fitted.")
            baseline_df = self._baseline_dist
        else:
            min_grp = stats_long[group_col].min()
            logger.info(f"⚓ [Monitor] Using group '{min_grp}' as Auto-Baseline.")
            baseline_df = (
                stats_long.filter(pl.col(group_col) == min_grp)
                .with_columns(
                    (pl.col("n_bin") / pl.col("n_bin").sum().over("feature")).alias("pct_expected")
                )
                .select(["feature", "bin_idx", "pct_expected"])
            )

        # 5. 计算指标
        EPS = 1e-9
        
        detail_calc = (
            stats_long
            .join(baseline_df, on=["feature", "bin_idx"], how="left")
            .with_columns(pl.col("pct_expected").fill_null(EPS)) # 填充新出现的箱
            .with_columns([
                pl.col("n_bin").sum().over([group_col, "feature"]).alias("grp_total"),
                pl.col("n_bad").sum().over([group_col, "feature"]).alias("grp_bad"),
                pl.col("n_good").sum().over([group_col, "feature"]).alias("grp_good"),
            ])
            .with_columns([
                (pl.col("n_bin") / pl.col("grp_total")).alias("pct_actual"),
                (pl.col("n_bad") / (pl.col("grp_bad") + EPS)).alias("pct_bad_w"),
                (pl.col("n_good") / (pl.col("grp_good") + EPS)).alias("pct_good_w"),
                (pl.col("n_bad") / pl.col("n_bin")).alias("bin_bad_rate"),
                (pl.col("grp_bad") / pl.col("grp_total")).alias("grp_bad_rate")
            ])
            .sort([group_col, "feature", "bin_idx"])
            .with_columns([
                ((pl.col("pct_actual") - pl.col("pct_expected")) * (pl.col("pct_actual") / pl.col("pct_expected")).log()).alias("psi_i"),
                ((pl.col("pct_bad_w") - pl.col("pct_good_w")) * (pl.col("pct_bad_w") / (pl.col("pct_good_w") + EPS)).log()).alias("iv_i"),
                (pl.col("bin_bad_rate") / (pl.col("grp_bad_rate") + EPS)).alias("lift"),
                pl.col("pct_bad_w").cum_sum().over([group_col, "feature"]).alias("cum_bad"),
                pl.col("pct_good_w").cum_sum().over([group_col, "feature"]).alias("cum_good"),
            ])
        )

        # 6. 生成 L1 Summary
        summary_pl = (
            detail_calc
            .group_by([group_col, "feature"])
            .agg([
                pl.col("grp_total").max().alias("count"),
                pl.col("grp_bad_rate").max().alias("bad_rate"),
                pl.col("psi_i").sum().alias("PSI"),
                pl.col("iv_i").sum().alias("IV"),
                (pl.col("cum_bad") - pl.col("cum_good")).abs().max().alias("KS")
            ])
            .sort([group_col, "feature"])
        )

        # 7. 生成 L2 Detail (Mapping)
        map_rows = []
        for feat, mapping in self.binner.bin_mappings_.items():
            for idx, label in mapping.items():
                map_rows.append({"feature": feat, "bin_idx": int(idx), "bin_label": str(label)})
        
        map_df = pl.DataFrame(map_rows, schema={"feature": pl.String, "bin_idx": pl.Int16, "bin_label": pl.String})
        
        detail_pl = (
            detail_calc
            .join(map_df, on=["feature", "bin_idx"], how="left")
            # [FIX] 防止 Join 不上导致可视化报错，填充默认值
            .with_columns(pl.col("bin_label").fill_null(pl.col("bin_idx").cast(pl.String)))
            .join(summary_pl.select([group_col, "feature", "IV", "KS", "PSI"]), 
                  on=[group_col, "feature"], how="left")
            .select([
                "feature", group_col, "bin_idx", "bin_label", 
                "n_bin", "pct_actual", "pct_expected", 
                "bin_bad_rate", "lift", 
                "IV", "KS", "PSI", "grp_bad_rate",
            ])
            .sort(["feature", group_col, "bin_idx"])
        )

        return MarsMonitorReport(
            summary_table=summary_pl.to_pandas(),
            detail_table=detail_pl.to_pandas()
        )