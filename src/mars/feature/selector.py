from __future__ import annotations
import os
import json
from typing import List, Optional, Union, Any, Literal, Dict, Tuple, TYPE_CHECKING
import polars as pl
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from mars.core.base import MarsBaseSelector
from mars.feature.binner import MarsNativeBinner, MarsOptimalBinner
from mars.analysis.report import MarsEvaluationReport
from mars.utils.logger import logger
from mars.utils.decorators import time_it

if TYPE_CHECKING:
    from mars.analysis.evaluator import MarsBinEvaluator

class MarsStatsSelector(MarsBaseSelector):
    """
    [Mars 统计特征筛选] 
    
    全流程自动化特征筛选工具。

    本组件采用 “漏斗式” 大逃杀架构，通过六个阶段对特征进行多维度、高压力的筛选测试，
    旨在从数千维原始特征中提取出具备高区分度、高稳定性、且符合业务逻辑的优质入模特征。

    Funnel Stages
    -------------
    - Stage 1. 数据质量 (Data Quality): 剔除高缺失、高零值、极高众数的 “脏” 特征。
    - Stage 2. 快速粗筛 (Rough Scan): 利用轻量级分箱快速计算 IV 和单箱 Lift，实现大规模剪枝。
        - 数值特征走快速 Native Binner 粗筛，类别特征直接跳过进入精筛。
        - 建模时特征数量不多时，建议开启 skip_rough_scan 跳过此阶段，直接进入更精准的精筛。 
    - Stage 3. 精准精选 (Fine Scan): 利用最优分箱计算全量 IV、近期 IV 及 Lift 召回。
    - Stage 4. 分布稳定性 (PSI): 跨期分布稳定性校验，拦截分布剧变的特征。
    - Stage 5. 逻辑稳定性 (RiskCorr): 校验分箱坏率逻辑是否随时间翻转，防止倒挂。
    - Stage 6. 相关性去重 (Correlation): 基于 WOE 矩阵计算共线性，在冗余组中择优录取。

    Attributes
    ----------
    selected_features_ : List[str]
        拟合完成后，最终幸存的特征名单。
    report_records_ : List[Dict]
        详细的特征“尸检报告”，记录每个特征的生存/淘汰状态及具体原因。
    """
    def __init__(
        self,
        *,
        target: str,
        features: Optional[List[str]] = None,         
        time_col: Optional[str] = None,
        profile_by: Optional[str] = None, 
        
        missing_thr: float = 0.90,
        zeros_thr: float = 0.90,  
        mode_thr: float = 0.90,
        
        iv_thr: float = 0.02, 
        lift_thr: Optional[float] = 1.2,      
        min_sample_rate: float = 0.05,               
        
        psi_thr: float = 0.25,      
        rc_thr: float = 0.5,           
        corr_thr: float = 0.8,    

        skip_rough_scan: bool = True,
        rough_iv_thr: float = 0.01,
        rough_lift_thr: float = 1.2,
        rough_min_sample_rate: float = 0.05,  
        
        white_list: Optional[List[str]] = None,
        black_list: Optional[List[str]] = None,

        missing_values: Optional[List[Any]] = None,   
        special_values: Optional[List[Any]] = None,  
        
        binning_params: Optional[Dict[str, Any]] = None,    
        rough_binning_params: Optional[Dict[str, Any]] = None,  

        max_samples: Optional[int] = None,
        batch_size: Optional[int] = 100,    
        n_jobs: int = -1
    ):
        """
        Parameters
        ----------
        target : str
            目标变量名（通常为 0/1 标签）。
        features : List[str], optional
            待筛选的候选特征池。若为 None，则默认使用 X 中的所有非目标列/时间列。
        time_col : str, optional
            时间戳列名，用于计算近期 IV。
        profile_by : str, optional
            分组维度（如 'month'），用于计算跨期 PSI 和 RiskCorr。
            
        missing_thr : float, default 0.90
            最大允许缺失率。
        zeros_thr : float, default 0.90
            0值率阈值。数值型特征 0 占比超过此值则被剔除。
        mode_thr : float, default 0.90
            最大允许众数占比。
            
        iv_thr : float, default 0.02
            精筛阶段的总体 IV 门槛。
        lift_thr : float, optional, default 1.2
            精筛阶段的 Lift 召回门槛。若特征总体 IV 低于门槛但在某箱 Lift 极高，将被救活。
        min_sample_rate : float, default 0.05
            精筛阶段的最小样本率。用于配合 Lift 召回使用，确保高 Lift 箱有足够的代表性。
            
        psi_thr : float, default 0.25
            跨期分布稳定性阈值（PSI）。
        rc_thr : float, default 0.5
            风险逻辑一致性相关系数阈值，低于此值视为逻辑反转。
        corr_thr : float, default 0.8
            WOE 相关系数阈值，超过此值视为高度相关，保留 IV 更高的特征。    
            
        skip_rough_scan : bool, default True
            是否跳过粗筛阶段，直接进入精筛。默认开启跳过，适用于非海量特征规模的常规筛选。
        rough_iv_thr : float, default 0.01
            粗筛阶段的 IV 门槛（仅在 skip_rough_scan=False 时生效）。
        rough_lift_thr : float, default 1.2
            粗筛阶段的单箱 Lift 召回阈值。
        rough_min_sample_rate : float, default 0.05
            粗筛阶段的最小样本率。
            
        white_list : List[str], optional
            白名单。名单内的特征将无视中间筛选，直接保送至相关性阶段。
        black_list : List[str], optional
            黑名单。名单内的特征将在初始化阶段被直接拦截。

        missing_values : List[Any], optional
            自定义缺失值列表（如 [-999, 'unknown']），将统一计入缺失率。
        special_values : List[Any], optional
            业务特殊值（如 [-998, -997]），分箱时会将其隔离为独立箱处理。
            
        binning_params : Dict[str, Any], optional
            精筛阶段使用的最优分箱参数（支持 kwargs 透传）。
        rough_binning_params : Dict[str, Any], optional
            粗筛阶段使用的快速分箱参数。

        max_samples : int, optional
            性能优化参数。若数据量过大，可指定全局最大采样行数。
        n_jobs : int, default -1
            并行计算使用的核心数。
        """
        super().__init__(target=target)
        
        # 基础与名单
        self.features = features
        self.time_col = time_col
        self.profile_by = profile_by
        self.white_list = white_list if white_list else []
        self.black_list = black_list if black_list else []
        
        # 全局特殊值
        self.missing_values = missing_values if missing_values else []
        self.special_values = special_values if special_values else []
        
        # Stage 1
        self.missing_thr = missing_thr
        self.mode_thr = mode_thr
        self.zeros_thr = zeros_thr  
        
        # Stage 2
        self.skip_rough_scan = skip_rough_scan
        self.rough_binning_params = rough_binning_params or {"method": "cart", "n_bins": 20, "min_bin_size": 0.02}
        self.rough_iv_thr = rough_iv_thr
        self.rough_lift_thr = rough_lift_thr
        self.rough_min_sample_rate = rough_min_sample_rate
        
        # Stage 3
        self.binning_params = binning_params or {"prebinning_method": "cart", "n_bins": 10, "min_bin_size": 0.05}
        self.iv_thr = iv_thr
        self.lift_thr = lift_thr
        self.min_sample_rate = min_sample_rate
        
        # Stage 4 & 5 & 6
        self.psi_thr = psi_thr
        self.rc_thr = rc_thr
        self.corr_thr = corr_thr
        
        # 性能与执行
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        # 内部状态存储
        self._stage3_binner: Optional[MarsOptimalBinner] = None
        self._feature_iv_dict: Dict[str, float] = {}  # 记录最终特征的 IV, 用于相关性过滤优选

    @time_it
    def fit(self, X: pl.DataFrame, y: Optional[Any] = None) -> "MarsStatsSelector":
        """
        [流水线执行引擎] 执行全流程特征筛选。
        """
        X = self._ensure_polars_dataframe(X)
        self.n_features_in_ = len(X.columns) - (1 if self.target in X.columns else 0)
        
        
        # 数据采样与初始化
        if self.max_samples and X.height > self.max_samples:
            logger.info(f"⚡ Downsampling data from {X.height} to {self.max_samples} rows for performance.")
            X = X.sample(n=self.max_samples, seed=42)

        # 确定需要排除的保留列
        exclude_cols = {self.target}
        if self.time_col: exclude_cols.add(self.time_col)
        if self.profile_by and self.profile_by in X.columns: exclude_cols.add(self.profile_by)
        
        # 确定初始特征池 (Candidate Features)
        if self.features is not None:
            # 如果用户指定了特征池，只在指定的列表且真实存在的列中筛选，并剔除排除列
            candidate_features = [c for c in self.features if c in X.columns and c not in exclude_cols]
        else:
            # 否则扫描除保留列以外的所有列
            candidate_features = [c for c in X.columns if c not in exclude_cols]
        
        # 黑名单拦截
        current_features = [c for c in candidate_features if c not in self.black_list]
        for f in [c for c in candidate_features if c in self.black_list]:
            self._register_decision(f, "Dropped", "Init", "Black List")
            
        logger.info(f"🚀 Starting MarsStatsSelector with {len(current_features)} features.")

        # Stage 1: 质量清洗
        if current_features:
            current_features = self._filter_quality(X, current_features)
        
        # Stage 2: 粗筛 (Rough Scan)
        if current_features and not self.skip_rough_scan:
            current_features = self._filter_rough(X, current_features)
        elif self.skip_rough_scan:
            logger.info("⏩ Skipping Stage 2 (Rough Scan) as per configuration.")
        
        # Stage 3: 精选 (Fine Scan)
        if current_features:
            current_features = self._filter_fine(X, current_features)
            
        # Stage 4: 稳定性检查 (PSI)
        if current_features and (self.time_col or self.profile_by):
            current_features = self._filter_psi(X, current_features)

        # Stage 5: 逻辑稳定性检查 (RiskCorr)
        # 逻辑：只要提供了分组维度 profile_by 且 rc_thr 没被设为 None，就运行
        if current_features and (self.time_col or self.profile_by) and self.rc_thr is not None:
            current_features = self._filter_rc(X, current_features)

        # Stage 6: 相关性过滤 (WOE Correlation)
        # 逻辑：只要 corr_thr 没被设为 None，就运行
        if current_features and self.corr_thr is not None:
            current_features = self._filter_corr(X, current_features)

        # 处理白名单保送逻辑
        final_features = []
        for f in set(current_features + self.white_list):
            if f in X.columns:
                final_features.append(f)
                if f in self.white_list and f not in current_features:
                    self._register_decision(f, "Selected", "Final", "White List Forcing")

        self.selected_features_ = final_features
        self._is_fitted = True
        
        self.clear_cache()
        
        logger.info(f"✅ Selection Complete. Kept {len(self.selected_features_)} features out of {len(candidate_features)} scanned.")
        return self

    def _should_bypass_filter(self, feat: str) -> bool:
        """检查特征是否在白名单中，白名单特征免死"""
        return feat in self.white_list
    
    def clear_cache(self) -> None:
        """
        释放内部分箱器持有的原始数据缓存。
        
        调用此方法后，分箱器将无法再进行 '延迟 WOE 计算'，
        但依然可以进行 transform(index/label/woe) 操作。
        """
        if self._stage3_binner is not None:
            # 调用分箱器自身的清理方法
            self._stage3_binner.clear_cache()
            
        import gc
        gc.collect()
        logger.debug("Selector cache cleared.")

    def _filter_quality(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 1] 数据质量过滤"""
        
        from mars.analysis.profiler import MarsDataProfiler
        profiler = MarsDataProfiler(
            df, 
            features=features,
            missing_values=self.missing_values,
            special_values=self.special_values
        )
        report = profiler.generate_profile(
            config_overrides={
                "dq_metrics": ["missing", "zeros", "top1"],
                "stat_metrics": [],
                "enable_sparkline": False
            }
        )
        
        stats_records = report.overview_table.select([
            "feature", "missing_rate", "top1_ratio", "zeros_rate"
        ]).to_dicts()
        kept_features = []
        
        # 加入进度条显示
        for row in tqdm(stats_records, desc="Quality Check", leave=False):
            feat = row["feature"]
            missing = row["missing_rate"]
            mode_rate = row["top1_ratio"]
            zeros_rate = row["zeros_rate"]
            
            if missing > self.missing_thr:
                self._register_decision(feat, "Dropped", "Quality", "High Missing", missing)
            elif zeros_rate > self.zeros_thr:
                self._register_decision(feat, "Dropped", "Quality", "High Zero Rate", zeros_rate)
            elif mode_rate > self.mode_thr:
                self._register_decision(feat, "Dropped", "Quality", "Single Value (Mode)", mode_rate)
            else:
                self._register_decision(feat, "Selected", "Quality", "Pass", missing)
                kept_features.append(feat)
                
        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features

    def _filter_rough(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 2] 粗筛: 快指针 + Lift召回"""
        
        # 类别特征区分: MarsNativeBinner不支持类别，直通精筛
        cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
        cat_features = [c for c in features if df.schema[c] in cat_types]
        num_features = [c for c in features if c not in cat_features]
        
        kept_features = cat_features.copy()  # 类别特征直通
        if cat_features:
            logger.info(f"  -> Bypassing {len(cat_features)} categorical features to Fine Scan.")
        
        if not num_features:
            return kept_features

        # 针对数值特征进行快速 Native Binner 粗筛，透传特殊值
        binner = MarsNativeBinner(
            features=num_features,  
            missing_values=self.missing_values,
            special_values=self.special_values,
            n_jobs=self.n_jobs,
            **self.rough_binning_params
        )
        target = df.get_column(self.target)
        binner.fit(df, target)
        
        stats_df = binner.profile_bin_performance(df, target, update_woe=False)
        
        lift_recall_cond = (pl.col("Lift") > self.rough_lift_thr) & (pl.col("count_dist") > self.rough_min_sample_rate)
        feat_stats = (
            stats_df.group_by("feature")
            .agg([
                pl.col("IV").max().alias("IV_total"),
                lift_recall_cond.any().alias("has_high_lift"),
                pl.col("Lift").max().alias("max_lift") 
            ])
        ).to_dicts()
        
        for row in tqdm(feat_stats, desc="Rough Scan", leave=False):
            feat, iv, has_high_lift, max_lift = row["feature"], row["IV_total"], row["has_high_lift"], row["max_lift"]
            
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue
                
            if iv > self.rough_iv_thr or has_high_lift:
                reason = f"Pass (IV={iv:.3f})" if iv > self.rough_iv_thr else f"Pass (Lift={max_lift:.2f})"
                self._register_decision(feat, "Selected", "Rough_Scan", reason, iv)
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "Rough_Scan", "Low IV & Low Lift", iv)

        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features

    def _filter_fine(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 3] 精选: 利用 Evaluator 计算全量及分时 IV + 可选 Lift 召回"""
        from mars.analysis.evaluator import MarsBinEvaluator

        # 识别特征类型
        cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
        cat_features = [c for c in features if df.schema[c] in cat_types]

        # 执行分箱评估
        evaluator = MarsBinEvaluator(
            target=self.target, 
            bining_type="opt", 
            cat_features=cat_features,
            missing_values=self.missing_values, 
            special_values=self.special_values,
            **self.binning_params
        )
        
        report = evaluator.evaluate(
            df, 
            features=features, 
            dt_col=self.time_col, 
            profile_by=self.profile_by,
            batch_size=self.batch_size,
        )
        
        # 缓存 Binner 给后续阶段使用
        self._stage3_binner = evaluator.binner

        # 计算 Lift 召回名单 (可选)
        lift_recall_set = set()
        if self.lift_thr is not None:
            # 只在 'Total' 汇总组中筛选满足 Lift 和 占比条件的特征
            group_col = report.group_col
            lift_cond = (pl.col("lift") > self.lift_thr) & \
                        (pl.col("pct") > self.min_sample_rate)
            
            lift_passed = report.detail_table.filter(
                (pl.col(group_col) == "Total") & lift_cond
            )
            lift_recall_set = set(lift_passed["feature"].unique().to_list())

        # 遍历汇总表进行筛选决策
        kept_features = []
        for row in report.summary_table.to_dicts():
            feat = row["feature"]
            iv_total = row.get("iv", 0.0) 

            # A. 白名单直接放行
            if self._should_bypass_filter(feat):
                self._feature_iv_dict[feat] = iv_total
                kept_features.append(feat)
                continue

            # B. 检查各项准则 (删除了 is_recent_ok)
            is_iv_ok = iv_total >= self.iv_thr
            is_lift_recall = feat in lift_recall_set

            # C. 决策逻辑：(IV 达标) or (Lift 召回达标)
            # 删除了原来的 is_recent_ok 判断
            if is_iv_ok or is_lift_recall:
                decision_reason = "Pass (IV)" if is_iv_ok else "Pass (Lift Recall)"
                self._register_decision(feat, "Selected", "Fine_Scan", decision_reason, iv_total)
                
                self._feature_iv_dict[feat] = iv_total
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "Fine_Scan", "Low IV & No Lift Recall", iv_total)

        return kept_features
    def _filter_psi(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 4] 稳定性过滤: 复用 Stage 3 binner 进行零开销分箱"""
        
        from mars.analysis.evaluator import MarsBinEvaluator
        evaluator = MarsBinEvaluator(
            target=self.target,
            binner=self._stage3_binner 
        )
        
        report = evaluator.evaluate(df, features=features, dt_col=self.time_col, profile_by=self.profile_by)
        psi_map = {r["feature"]: r["psi_max"] for r in report.summary_table.select(["feature", "psi_max"]).to_dicts()}
        
        kept_features = []
        for feat in tqdm(features, desc="PSI Check", leave=False):
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue
                
            psi_val = psi_map.get(feat, 0.0)
            if psi_val < self.psi_thr:
                self._register_decision(feat, "Selected", "Stability", "Stable PSI", psi_val)
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "Stability", f"High PSI ({psi_val:.2f})", psi_val)

        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features

    def _filter_rc(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 5] 风险一致性过滤 (RiskCorr): 校验跨期坏账逻辑是否发生翻转"""
        
        from mars.analysis.evaluator import MarsBinEvaluator
        evaluator = MarsBinEvaluator(
            target=self.target,
            binner=self._stage3_binner 
        )
        
        report = evaluator.evaluate(df, features=features, dt_col=self.time_col, profile_by=self.profile_by)
        
        # 从 Summary 表中提取 rc_min
        if "rc_min" in report.summary_table.columns:
            rc_map = {r["feature"]: r["rc_min"] for r in report.summary_table.select(["feature", "rc_min"]).to_dicts()}
        else:
            rc_map = {}
            logger.warning("rc_min metric not found in report. Skipping RC check.")
        
        kept_features = []
        for feat in tqdm(features, desc="RC Check", leave=False):
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue
                
            # RC 默认值为 1.0
            rc_val = rc_map.get(feat, 1.0)
            
            # 若 rc_val 为 None (比如该特征在所有分组只有 1 个箱)，跳过该校验
            if rc_val is None or rc_val >= self.rc_thr:
                self._register_decision(feat, "Selected", "RiskCorr", "Stable Logic", rc_val if rc_val is not None else 1.0)
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "RiskCorr", f"Logic Broken (RC={rc_val:.2f})", rc_val)

        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features

    def _filter_corr(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 6] 相关性过滤: 纯 Polars 实现版本"""
        
        if len(features) < 2:
            return features

        woe_cols = [f"{c}_woe" for c in features]
        df_woe = self._stage3_binner.transform(df.select(features), return_type="woe")
        corr_matrix_df = df_woe.select(woe_cols).fill_null(0.0).corr()

        # 为了方便后续的“贪心逻辑”检索，将其转换为带有特征名列的结构,把特征名加回去，方便后面做 filter
        corr_matrix_with_names = corr_matrix_df.with_columns(
            feature_name=pl.Series(woe_cols)
        )

        sorted_feats = sorted(
            features, 
            key=lambda f: (-self._feature_iv_dict.get(f, 0.0), f) # 先按 IV 从高到低排序，IV 相同则按字母顺序排序，确保决策的稳定性和可复现性
        )

        kept_features_set = set()
        dropped_features = set()

        for feat in tqdm(sorted_feats, desc="Corr Check", leave=False):
            if self._should_bypass_filter(feat):
                kept_features_set.add(feat)
                continue
                
            if feat in dropped_features:
                continue
            
            kept_features_set.add(feat)
            self._register_decision(feat, "Selected", "Corr_Filter", "Independent", self._feature_iv_dict.get(feat, 0))
            
            target_woe_name = f"{feat}_woe"
            # 找出与当前特征相关性大于阈值的列名
            high_corr_row = corr_matrix_with_names.filter(pl.col("feature_name") == target_woe_name)
            
            for other_feat_woe in woe_cols:
                if other_feat_woe == target_woe_name:
                    continue
                    
                corr_val = abs(high_corr_row.get_column(other_feat_woe)[0])
                if corr_val > self.corr_thr:
                    orig_f = other_feat_woe[:-4]
                    if orig_f not in dropped_features and not self._should_bypass_filter(orig_f):
                        dropped_features.add(orig_f)
                        self._register_decision(orig_f, "Dropped", "Corr_Filter", f"Correlated with '{feat}'", corr_val)

        return [f for f in features if f in kept_features_set]

    def get_eval_report(self, X: Union[pl.DataFrame, pd.DataFrame]) -> Tuple["MarsEvaluationReport", "MarsBinEvaluator"]:
        """
        获取最终入选特征的详细评估报告。
        
        该方法将复用 Stage 3 生成的分箱规则，快速计算并返回 `selected_features_` 的详细报告。
        
        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            需要计算评估指标的数据集。
            
        Returns
        -------
        MarsEvaluationReport, MarsBinEvaluator
            - MarsEvaluationReport 包含汇总表、趋势表和分箱明细表的报告对象。可以调用其 .show_summary() 或 .write_excel();
            - MarsBinEvaluator 是评估实例，包含了分箱器和评估方法，可以用于后续的 transform 或者其他定制化评估分析。
        """
        self._check_is_fitted()
        
        if not self.selected_features_:
            raise ValueError("No selected features found. Cannot generate report.")
            
        X_pl = self._ensure_polars_dataframe(X)
        
        from mars.analysis.evaluator import MarsBinEvaluator
        
        # 优先复用现成的 Binner
        if self._stage3_binner is not None:
            evaluator = MarsBinEvaluator(
                target=self.target,
                binner=self._stage3_binner
            )
        else:
            # 兜底逻辑: 如果强制 skip 了前面的流程且丢失了 binner 缓存
            logger.warning("Cached binner not found. Re-fitting OptBinner for the selected features...")
            cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
            cat_features = [c for c in self.selected_features_ if X_pl.schema[c] in cat_types]
            
            evaluator = MarsBinEvaluator(
                target=self.target,
                bining_type="opt",
                cat_features=cat_features,
                missing_values=self.missing_values,
                special_values=self.special_values,
                **self.binning_params
            )
            
        if self._return_pandas:
            evaluator.set_output("pandas")

        logger.info(f"📊 Generating final evaluation report for {len(self.selected_features_)} selected features...")
        
        report: MarsEvaluationReport = evaluator.evaluate(
            X_pl, 
            features=self.selected_features_, 
            dt_col=self.time_col, 
            profile_by=self.profile_by
        )
        
        return report, evaluator

    def export_selector_report(self, path: str = "mars_selector_report.xlsx") -> None:
        """
        输出尸检详细报告。支持 .xlsx 或 .csv 后缀
        """
        report_df = self.get_report()
        # 【修复】兼容 Pandas / Polars 两种判断空表的方法
        if isinstance(report_df, pd.DataFrame):
            if report_df.empty:
                logger.warning("No report to export.")
                return
            pd_df = report_df
        else:
            if report_df.height == 0:
                logger.warning("No report to export.")
                return
            pd_df = report_df.to_pandas()
            
        logger.info(f"💾 Exporting Selection Report to {path}...")
        
        pd_df = report_df.to_pandas()
        
        if path.endswith(".csv"):
            pd_df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            try:
                # 附带样式导出
                styler = pd_df.style.map(
                    lambda v: 'color: green; font-weight: bold' if v == 'Selected' else 'color: red',
                    subset=['status']
                )
                styler.to_excel(path, index=False, engine="openpyxl")
            except Exception as e:
                logger.warning(f"Failed to export styled excel, falling back to basic export. Error: {e}")
                pd_df.to_excel(path, index=False)
                
        logger.info("✅ Export Complete.")

    def save_selector_lists(
        self, 
        path: str = "mars_lists.json", 
        blacklist_stages: Optional[List[str]] = None
    ):
        """
        将本次筛选的结果保存为黑白名单，用于下一次快速迭代。

        Parameters
        ----------
        path : str, default "mars_lists.json"
            保存 JSON 文件的路径。
        blacklist_stages : List[str], optional
            指定哪些阶段被淘汰的特征应进入黑名单。支持模糊匹配。
            例如: ['quality'] 会匹配 'Quality'
                 ['scan'] 会同时匹配 'Rough_Scan' 和 'Fine_Scan'
                 ['psi', 'rc'] 会匹配 'Stability' (PSI) 和 'RiskCorr'
        """
        self._check_is_fitted()
        
        # 预处理用户输入的关键词，统一转小写以实现不区分大小写匹配
        if blacklist_stages:
            patterns = [p.lower() for p in blacklist_stages]
        else:
            patterns = []

        def is_stage_matched(actual_stage: str) -> bool:
            # 如果没传 patterns，默认全选 (返回 True)
            if not patterns:
                return True
            # 只要 actual_stage 包含用户定义的任何一个关键词，即算匹配成功
            actual_stage_lower = actual_stage.lower()
            return any(p in actual_stage_lower for p in patterns)

        # 提取需要加黑的特征
        dropped_records = [
            r["feature"] for r in self.report_records_ 
            if r["status"] == "Dropped" and is_stage_matched(r["stage"])
        ]
        
        # 组装数据并去重
        data = {
            "white_list": self.selected_features_,
            "black_list": list(set(dropped_records + self.black_list))
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        match_msg = f"matching {blacklist_stages}" if blacklist_stages else "from all stages"
        logger.info(f"💾 Black/White lists saved to {path}. (Blacklisted features {match_msg})")

    @classmethod
    def load_lists_from_json(cls, path: str) -> Dict[str, List[str]]:
        """
        从 JSON 文件中加载名单。
        """
        if not os.path.exists(path):
            logger.warning(f"File {path} not found. Returning empty lists.")
            return {"white_list": [], "black_list": []}
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    
class MarsLinearSelector(MarsBaseSelector):
    def __init__(
        self,
        target: str,

        # --- Stage 1: 相关性去重 (Correlation Filter) ---
        # 作用: 快速去除高度相关的特征 (如 app_count_7d vs app_count_30d)
        enable_corr_filter: bool = True,
        corr_thr: float = 0.8,         # 相关性 > 0.8 视为冗余
        corr_method: str = "spearman",       # 推荐 spearman (分箱后是非线性的)
        
        # --- Stage 2: 多重共线性筛查 (VIF Filter) ---
        # 作用: 解决多重线性依赖 (A = B + C)
        # 注意: 计算矩阵逆非常慢，建议只对 < 1000 个特征开启
        enable_vif_filter: bool = False,
        vif_threshold: float = 5.0,          # 业界通常取 5.0 或 10.0
        
        # --- Stage 3: 逐步回归 (Stepwise Selection) ---
        # 作用: 基于 AIC/BIC 的包裹式筛选
        enable_stepwise: bool = False,
        stepwise_direction: str = "forward", # forward (快) / backward (慢但准)
        stepwise_criterion: str = "aic",     # 优化目标: aic / bic
        max_features: Optional[int] = None,  # 限制最终入模特征数
        
        n_jobs: int = -1
    ):
        """
        初始化线性筛选器。
        推荐作为 Pipeline 的第二步，接在 MarsStatsSelector 之后。
        """
        super().__init__()
        # ...
        

class MarsImportanceSelector(MarsBaseSelector):
    def __init__(
        self,
        target: str,
        
        # --- 模型配置 (Estimator) ---
        # 支持字符串简写，也支持传入实例化好的 sklearn/lgbm 对象
        estimator: Union[str, Any] = "lgbm", # lgbm, xgb, cat, rf, extra_trees
        estimator_params: Optional[dict] = None, # 比如 {'learning_rate': 0.05, 'n_estimators': 100}
        
        # --- 筛选策略 (Method) ---
        # 1. importance: 使用模型自带的 feature_importances_ (gain/split)
        # 2. shap: 训练后计算 SHAP mean values (更准但更慢)
        # 3. rfe: 递归特征消除 (反复训练剔除尾部，最慢但效果最好)
        # 4. sfm: SelectFromModel (基于阈值的一次性筛选)
        method: Literal["importance", "shap", "rfe", "sfm"] = "importance",
        
        # --- 阈值控制 (Criteria) ---
        # 决定保留多少特征
        selection_mode: Literal["top_k", "threshold", "percentile"] = "top_k",
        selection_threshold: Union[int, float, str] = 50, 
        # 解释:
        # - top_k: 保留前 50 个 (最常用)
        # - threshold: 重要性 > 0.01
        # - percentile: 保留前 20%
        
        # --- 交叉验证 (CV) ---
        # 是否使用 CV 后的平均重要性来减少随机性 (建议开启)
        cv: int = 3, 
        
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        初始化重要性筛选器。
        通常作为特征筛选的最后一步，产出最终入模名单。
        """
        super().__init__()
        # ...