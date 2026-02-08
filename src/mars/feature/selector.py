"""
TODO:
1. 支持按时间精细过滤（分时IV，Lift）和按整体粗略过滤（整体IV，Lift）。
2. 有时建模时，特征已经没有很多了，是否可以跳过 Stage 2 的粗筛，直接从 Stage 1 到 Stage 3？
3. 最终报告输出功能（可选）：输出每个阶段的详细报告，方便审计和复盘。这个报告和 MarsBinEvaluator 的报告可以结合起来，
或者直接在 MarsBinEvaluator 报告的新加加 sheet，专门记录特征筛选的过程和结果。
4. _filter_psi 方法中，复用 Stage 3 的 binner 来计算 PSI，而不是重新实例化一个 Evaluator 来跑一遍分箱。
5. 增加对类别特征的支持，目前主要针对数值特征。 -- 最低优先级实现的功能
6. 初始化入参阶段二和阶段三的分箱方法，考虑直接新建两个参数：rough_binning_params 和 fine_binning_params，分别传入 dict, 以支持更灵活的配置。
如果这两个参数为 None，则在初始化配置一个默认的分箱参数字典。
7. 特征黑名单/白名单支持, 即允许用户传入一个特征列表，在筛选过程中强制保留或剔除这些特征。
8. 新增可选第五步，相关性过滤 (Correlation Filter)，剔除与其他特征高度相关的冗余特征。
9. 新增可选单调性检验过滤 (Monotonicity Check)，剔除单调性差的特征。
10. 进度条显示: 对于较大的数据集和特征集，添加进度条显示当前处理进度，提升用户体验。
11. 数据行数过大时，支持采样计算，以提升性能。
"""

"""
TODO:

[板块一：核心逻辑与策略增强] (Logic & Strategy)
--------------------------------------------
1. [Time Dynamic Scan] 新增时序动态筛选逻辑 (建议在 Stage 3 后执行):
   不再仅依赖 Global IV，引入时间维度的细粒度评估：
   - 召回 (Recall): 针对 "潜力股"。若 Global IV < 阈值，但 Max_Monthly_IV > recall_threshold (如 0.05)，强制保留。
   - 剔除 (Drop Unstable): 针对 "神经刀"。若分时 IV 的变异系数 (CV) > max_iv_cv (如 0.5)，剔除。
   - 剔除 (Drop Decaying): 针对 "衰退股"。若 (最近N月均值 - 全局均值)/全局均值 < max_iv_decay (如 -0.3)，剔除。

2. [Correlation Filter] 新增 Stage 5 相关性过滤 (可选):
   - 在所有单变量筛选结束后，计算 Spearman 相关性矩阵。
   - 对高相关组 (Corr > 0.95)，保留 IV 最高的一个，剔除冗余特征。

3. [Monotonicity Check] 新增单调性校验 (可选):
   - 在 Stage 3 (Optimal Binning) 后，计算 bin_index 与 bad_rate 的 Spearman 相关系数。
   - 剔除严格不单调 (如 abs(corr) < 0.9) 的特征。

4. [Business Constraints] 黑白名单支持:
   - 入参增加 `force_include_features` (白名单): 跳过所有筛选，强制保留。
   - 入参增加 `force_exclude_features` (黑名单): 在 Stage 1 前直接剔除。

5. [Categorical Support] 类别特征支持 (低优先级):
   - 扩展 Binner 和 Evaluator，支持非数值型特征的自动分箱与评估。

[板块二：工程架构与性能优化] (Engineering & Performance)
-----------------------------------------------------
6. [Binner Reuse] 分箱器复用 (关键架构优化):
   - Stage 4 (PSI) 应直接复用 Stage 3 训练好的 `fine_binner_` 对象。
   - 避免重新实例化 Evaluator 跑一遍 Native 分箱，确保 PSI 计算口径与 IV 计算口径一致，且节省算力。

7. [Short-Circuit] 短路逻辑:
   - 当初始特征数较少 (如 < 1000) 时，自动跳过 Stage 2 (Rough Scan)。
   - 直接进入 Stage 3 (Fine Scan)，减少不必要的计算开销。

8. [Smart Sampling] 大数据采样优化:
   - 当数据行数极大 (如 > 1000w 行) 时，支持在 Stage 1 & 2 使用采样数据 (sample_frac)。
   - 在 Stage 3 (Fine Scan) 切回全量数据以保证切点精度。

[板块三：API 设计与交互体验] (API & UX)
------------------------------------
9. [Param Refactoring] 入参重构:
   - 将散落在 __init__ 中的 rough_* 和 fine_* 参数收敛为两个字典配置：
     `rough_binning_params: Dict` 和 `fine_binning_params: Dict`。
   - 若为 None，则加载默认配置。

10. [Reporting] 审计报告输出:
    - 提供 `get_audit_report()` 方法。
    - 输出每个特征的 "生死簿"：包含特征名、最终状态 (Selected/Dropped)、淘汰阶段、淘汰原因 (Reason)、关键指标值。
    - 可考虑与 MarsBinEvaluator 报告整合。

11. [Progress Bar] 进度条:
    - 集成 `tqdm`，在 fit 过程中显示当前 Stage 的进度，缓解用户等待焦虑。
"""

from typing import List, Optional, Union, Any, Literal
import polars as pl
import numpy as np

from mars.core.base import MarsBaseSelector
from mars.analysis.profiler import MarsDataProfiler
from mars.feature.binner import MarsNativeBinner, MarsOptimalBinner
from mars.analysis.evaluator import MarsBinEvaluator
from mars.utils.logger import logger
from mars.utils.decorators import time_it

class MarsStatsSelector(MarsBaseSelector):
    def __init__(
        self,
        target_col: str,
        # --- Stage 1 ---
        missing_threshold: float = 0.95,
        mode_threshold: float = 0.99,
        # --- Stage 2 ---
        rough_binning_method: str = "quantile",
        rough_n_bins: int = 20,
        rough_iv_threshold: float = 0.01,
        rough_lift_threshold: float = 1.5,
        rough_min_sample_rate: float = 0.005,
        # --- Stage 3 ---
        fine_binning_method: str = "opt",
        fine_n_bins: int = 5,
        fine_iv_threshold: float = 0.02,
        # --- Stage 4 ---
        psi_threshold: float = 0.25,
        time_col: Optional[str] = None,
        n_jobs: int = -1
    ):
        super().__init__(target_col)
        # 参数绑定
        self.missing_threshold = missing_threshold
        self.mode_threshold = mode_threshold
        self.rough_binning_method = rough_binning_method
        self.rough_n_bins = rough_n_bins
        self.rough_iv_threshold = rough_iv_threshold
        self.rough_lift_threshold = rough_lift_threshold
        self.rough_min_sample_rate = rough_min_sample_rate
        self.fine_binning_method = fine_binning_method
        self.fine_n_bins = fine_n_bins
        self.fine_iv_threshold = fine_iv_threshold
        self.psi_threshold = psi_threshold
        self.time_col = time_col
        self.n_jobs = n_jobs

    @time_it
    def fit(self, X: pl.DataFrame, y: Optional[Any] = None) -> "MarsStatsSelector":
        """
        [流水线执行引擎] 执行四阶段特征筛选。
        """
        # 0. 初始化
        X = self._ensure_polars_dataframe(X)
        self.n_features_in_ = len(X.columns) - (1 if self.target_col in X.columns else 0)
        
        # 确定初始特征池 (排除 Target 和 Time)
        exclude_cols = {self.target_col}
        if self.time_col: exclude_cols.add(self.time_col)
        current_features = [c for c in X.columns if c not in exclude_cols]
        
        logger.info(f"🚀 Starting MarsStatsSelector with {len(current_features)} features.")

        # =========================================================
        # Stage 1: 质量清洗 (Data Quality) - "扫地僧"
        # =========================================================
        # 策略: 向量化计算，秒级过滤
        current_features = self._filter_quality(X, current_features)
        
        # =========================================================
        # Stage 2: 粗筛 (Rough Scan) - "快指针"
        # =========================================================
        # 策略: Native Quantile 分箱，不漏杀高 Lift 特征
        if current_features:
            current_features = self._filter_rough(X, current_features)
        
        # =========================================================
        # Stage 3: 精选 (Fine Scan) - "慢指针"
        # =========================================================
        # 策略: Optimal 分箱，严格卡 IV，追求单调性
        if current_features:
            current_features = self._filter_fine(X, current_features)
            
        # =========================================================
        # Stage 4: 稳定性 (Stability) - "守门员" (可选)
        # =========================================================
        if current_features and self.time_col:
            current_features = self._filter_psi(X, current_features)

        # 结束
        self.selected_features_ = current_features
        self._is_fitted = True
        logger.info(f"✅ Selection Complete. Kept {len(self.selected_features_)} features.")
        return self

    # ==========================================================================
    # 核心逻辑实现
    # ==========================================================================

    def _filter_quality(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """
        [Stage 1] 数据质量过滤。
        
        Strategy:
        复用 MarsDataProfiler 计算 Missing Rate, Zero Rate, Top1 Ratio (Mode)。
        """
        logger.info(f"Step 1: Quality Check (via MarsDataProfiler) on {len(features)} features...")
        
        # 1. 实例化 Profiler
        # 我们只计算 DQ 指标，关闭 sparkline 和 stat metrics 以获得极致性能
        profiler = MarsDataProfiler(
            df, 
            features=features,
            # [Optimization] 如果数据量极大(如>1000万行)，这里可以开启 sample_frac=0.2 进行估算
            # sample_frac=None 
        )
        
        # 2. 生成轻量级报告
        # 我们需要: missing (缺失率), top1 (众数占比), zeros (0值率 - 仅作为参考或日志)
        report = profiler.generate_profile(
            config_overrides={
                "dq_metrics": ["missing", "top1", "zeros"], # 只算这三个
                "stat_metrics": [],       # 不算均值方差
                "enable_sparkline": False # 不画图
            }
        )
        
        # 3. 获取概览表
        overview = report.overview_table
        
        # 4. 执行过滤逻辑
        # 转为 Dict 列表遍历，方便 _register_decision 记录
        # Overview Schema: [feature, missing_rate, zeros_rate, top1_ratio, ...]
        stats_records = overview.select(
            ["feature", "missing_rate", "top1_ratio", "zeros_rate"]
        ).to_dicts()
        
        kept_features = []
        
        for row in stats_records:
            feat = row["feature"]
            missing = row["missing_rate"]
            mode_rate = row["top1_ratio"]
            # zeros = row["zeros_rate"] # 暂时不用作过滤条件，但可以打印 log
            
            # A. 缺失率检查
            if missing > self.missing_threshold:
                self._register_decision(feat, "Dropped", "Quality", "High Missing", missing)
                continue
                
            # B. 众数占比检查 (单一值)
            if mode_rate > self.mode_threshold:
                self._register_decision(feat, "Dropped", "Quality", "Single Value (Mode)", mode_rate)
                continue
            
            # C. 通过
            self._register_decision(feat, "Selected", "Quality", "Pass", missing)
            kept_features.append(feat)
            
        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features

    def _filter_rough(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 2] 粗筛: 快指针 + Lift召回"""
        logger.info(f"Step 2: Rough Scan (Native Binning) on {len(features)} features...")
        
        # 1. 实例化 Native Binner
        binner = MarsNativeBinner(
            features=features,
            n_bins=self.rough_n_bins,
            method=self.rough_binning_method, # quantile
            n_jobs=self.n_jobs
        )
        
        # 2. Fit & Profile
        target = df[self.target_col]
        binner.fit(df, target)
        
        # 获取分箱粒度指标 (Bin Level Stats)
        # Schema: [feature, bin_index, IV, Lift, count_dist, ...]
        # 注意: profile_bin_performance 返回的是包含所有指标的 DataFrame
        stats_df = binner.profile_bin_performance(df, target, update_woe=False)
        
        # 3. 聚合为特征粒度 (Feature Level Aggregation)
        # 核心逻辑:
        # - IV_max: 特征的总 IV (profile_bin_performance 里 IV 列已经是总 IV 了，取 max 即可)
        # - Lift_max_valid: 满足 min_sample_rate 条件下的最大 Lift
        
        # 标记是否满足 Lift 召回条件 (单箱 Lift > 阈值 且 样本足够)
        lift_recall_cond = (
            (pl.col("Lift") > self.rough_lift_threshold) & 
            (pl.col("count_dist") > self.rough_min_sample_rate)
        )
        
        feat_stats = (
            stats_df.group_by("feature")
            .agg([
                pl.col("IV").max().alias("IV_total"),
                # 只要有一箱满足 Lift 召回条件，has_high_lift 就为 True
                lift_recall_cond.any().alias("has_high_lift"),
                # 为了报告，记录一下最大 Lift
                pl.col("Lift").max().alias("max_lift") 
            ])
        )
        
        # 4. 执行过滤
        # 转为 Python 字典加速遍历
        records = feat_stats.to_dicts()
        kept_features = []
        
        for row in records:
            feat = row["feature"]
            iv = row["IV_total"]
            has_high_lift = row["has_high_lift"]
            max_lift = row["max_lift"]
            
            # OR 逻辑: IV 达标 或者 Lift 达标
            if (iv > self.rough_iv_threshold) or has_high_lift:
                reason = "Pass (IV)" if iv > self.rough_iv_threshold else f"Pass (Lift={max_lift:.2f})"
                self._register_decision(feat, "Selected", "Rough_Scan", reason, iv)
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "Rough_Scan", "Low IV & Low Lift", iv)

        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features

    def _filter_fine(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 3] 精选: 慢指针 + 严格IV"""
        logger.info(f"Step 3: Fine Scan (Optimal Binning) on {len(features)} features...")
        
        # 1. 实例化 Optimal Binner
        binner = MarsOptimalBinner(
            features=features,
            n_bins=self.fine_n_bins,
            method=self.fine_binning_method, # opt
            n_jobs=self.n_jobs
        )
        
        # 2. Fit & Profile
        target = df[self.target_col]
        # 注意: Optimal Binner 较慢，但此处特征已大幅减少
        binner.fit(df, target)
        stats_df = binner.profile_bin_performance(df, target, update_woe=False)
        
        # 3. 聚合 & 过滤
        feat_stats = stats_df.group_by("feature").agg(
            pl.col("IV").max().alias("IV_total")
        ).to_dicts()
        
        kept_features = []
        for row in feat_stats:
            feat = row["feature"]
            iv = row["IV_total"]
            
            if iv > self.fine_iv_threshold:
                self._register_decision(feat, "Selected", "Fine_Scan", "Pass", iv)
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "Fine_Scan", "Low IV (Fine)", iv)
                
        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features

    def _filter_psi(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 4] 稳定性过滤 (PSI)"""
        logger.info(f"Step 4: PSI Stability Check on {len(features)} features...")
        
        # 使用 MarsBinEvaluator 来计算 PSI
        # 这里的策略是：基于时间列，取最早的时间段作为 Benchmark
        
        evaluator = MarsBinEvaluator(
            target_col=self.target_col,
            # 这里不需要再分箱了，但为了复用 evaluator 的 PSI 计算逻辑，
            # 我们直接传 df，Evaluator 内部会处理
            # 这里的效率优化点：Evaluator 内部可能会重新分箱，
            # 如果想极致优化，应该传入 Stage 3 产生的 binner 给 Evaluator
            # 但为了代码解耦，这里先让 Evaluator 自己跑一遍 Native 粗分箱算 PSI (足够了)
            bining_type="native", 
            n_bins=10 
        )
        
        report = evaluator.evaluate(
            df, 
            features=features, 
            profile_by="month" if not self.time_col else None, # 只有当 dt_col 存在时生效
            dt_col=self.time_col
        )
        
        # 从 Summary 表中提取 PSI_max
        summary = report.summary_table
        # Schema: [feature, PSI_max, ...]
        
        psi_records = summary.select(["feature", "PSI_max"]).to_dicts()
        psi_map = {r["feature"]: r["PSI_max"] for r in psi_records}
        
        kept_features = []
        for feat in features:
            # 如果特征在报告里找不到(可能报错了)，默认保留或剔除? 建议保留并警告
            psi_val = psi_map.get(feat, 0.0)
            
            if psi_val < self.psi_threshold:
                self._register_decision(feat, "Selected", "Final", "Stable", psi_val)
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "Stability", f"High PSI ({psi_val:.2f})", psi_val)

        logger.info(f"  -> {len(features) - len(kept_features)} dropped. Remaining: {len(kept_features)}")
        return kept_features
    
    
class MarsLinearSelector(MarsBaseSelector):
    def __init__(
        self,
        target_col: str,

        # --- Stage 1: 相关性去重 (Correlation Filter) ---
        # 作用: 快速去除高度相关的特征 (如 app_count_7d vs app_count_30d)
        enable_corr_filter: bool = True,
        corr_threshold: float = 0.8,         # 相关性 > 0.8 视为冗余
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
        target_col: str,
        
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