from __future__ import annotations
import os
import json
from typing import List, Optional, Union, Any, Literal, Dict, Tuple, TYPE_CHECKING
import polars as pl
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from mars.core.base import MarsBaseSelector
from mars.feature.binner import MarsBinnerBase, MarsNativeBinner, MarsOptimalBinner
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
    - Stage 3. 精准精选 (Fine Scan): 利用最优分箱计算全量 IV、近期 IV 及 Lift 召回。
        - 💡 可通过 `skip_fine_scan=True` 跳过此阶段，全程使用极速的原生分箱规则进行后续稳定性测算。
    - Stage 4. 分布稳定性 (PSI): 跨期分布稳定性校验，拦截分布剧变的特征。
    - Stage 5. 逻辑稳定性 (RiskCorr): 校验分箱坏率逻辑是否随时间翻转，防止倒挂。
    - Stage 6. 相关性去重 (Correlation): 基于 WOE 矩阵计算共线性，在冗余组中择优录取。
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
        corr_thr: float = 0.95,    

        skip_rough_scan: bool = True,
        skip_fine_scan: bool = False, 
        rough_iv_thr: float = 0.01,
        rough_lift_thr: float = 1.2,
        rough_min_sample_rate: float = 0.02,  
        
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
        初始化大逃杀特征筛选漏斗。

        Parameters
        ----------
        target : str
            目标变量列名 (通常为 0/1 标签)。
        features : List[str], optional
            待筛选的特征白名单池。若为 None，系统将自动扫描输入数据中除 target/time_col 之外的所有列。
        time_col : str, optional
            时间戳列名。用于界定特征的近期表现与历史表现。
        profile_by : str, optional
            稳定性探查的分组维度 (如 'month', 'vintage')。若提供，将激活 Stage 4 & 5 的跨期稳定性校验。

        white_list : List[str], optional
            绝对白名单。名单内的特征将无视所有指标门槛，直接保送至最终结果。
        black_list : List[str], optional
            绝对黑名单。名单内的特征将在进入漏斗前被物理拦截。
        missing_values : List[Any], optional
            自定义缺失值定义 (如 [-999, "unknown"])，统一计入 Stage 1 的 missing_rate。
        special_values : List[Any], optional
            业务特殊值定义 (如 [-998, -997])，底层分箱器会将其强制隔离为独立箱。

        missing_thr : float, default 0.90
            最大允许缺失率阈值。
        zeros_thr : float, default 0.90
            最大允许零值率阈值 (仅对数值特征生效)。
        mode_thr : float, default 0.90
            最大允许众数占比阈值 (防御单一值常量特征)。

        skip_rough_scan : bool, default False
            是否跳过轻量级粗筛阶段。
        rough_iv_thr : float, default 0.01
            粗筛阶段的 IV 准入门槛。
        rough_lift_thr : float, default 1.2
            粗筛阶段的单箱 Lift 召回门槛。若某特征总体 IV 不达标但局部 Lift 极高，将被复活。
        rough_min_sample_rate : float, default 0.02
            触发 Lift 召回的最小箱占比前提 (防止微小噪音箱导致误召回)。
        rough_binning_params : Dict, optional
            传递给底层 `MarsNativeBinner` 的控制参数。
            默认采用 20 箱等频切分并开启碎片合并 (merge_small_bins=True)。

        skip_fine_scan : bool, default True
            是否跳过最优分箱精筛阶段。
            [架构注]: 默认 True，代表开启【极速模式】，全链路由 NativeBinner 驱动。适合海量宽表去水。
            若设为 False，将启用 `MarsOptimalBinner` 重新计算最优边界。
        iv_thr : float, default 0.02
            精筛阶段的最终 IV 准入门槛。
        lift_thr : float, default 1.2
            精筛阶段的单箱 Lift 召回门槛。
        min_sample_rate : float, default 0.05
            精筛阶段触发 Lift 召回的最小箱占比。
        binning_params : Dict, optional
            传递给底层最优分箱器 (`MarsOptimalBinner` 等) 的控制参数。

        psi_thr : float, default 0.25
            跨期分布稳定性阈值。超过该值的特征将被判定为 Data Drift 而被淘汰。
        rc_thr : float, default 0.5
            风险逻辑相关性 (RiskCorr) 阈值。低于 0.5 通常意味着某个月份特征的坏账单调性发生了翻转倒挂。
        corr_thr : float, default 0.95
            基于 WOE 编码的皮尔逊相关系数阈值。超过该值的特征组将被判定为高共线性，只保留组内 IV 最高者。

        max_samples : int, optional
            若数据量达到亿级，可开启全局最大采样行数限制，大幅缩短筛选时间。
        batch_size : int, default 100
            底层 Polars 并行计算表达式时的特征切割批次。大宽表场景下防止 Query Planner 解析爆炸。
        n_jobs : int, default -1
            Python 层多进程/多线程池并发度 (-1 代表使用全部可用逻辑核)。
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
        self.rough_binning_params = rough_binning_params or {
            "method": "quantile", 
            "n_bins": 20, 
            "min_bin_size": 0.01, 
            "merge_small_bins": True
        }
        self.rough_iv_thr = rough_iv_thr
        self.rough_lift_thr = rough_lift_thr
        self.rough_min_sample_rate = rough_min_sample_rate
        
        # Stage 3
        self.skip_fine_scan = skip_fine_scan # [新增]
        self.binning_params = binning_params or {
            "prebinning_method": "cart", 
            "n_bins": 10, 
            "min_bin_size": 0.05,
        }
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
        self._rough_binner: Optional[MarsNativeBinner] = None # 缓存粗筛分箱器
        self._stage3_binner: Optional[MarsBinnerBase] = None
        self._feature_iv_dict: Dict[str, float] = {}  # 记录最终特征的 IV, 用于相关性过滤优选
        
        self._funnel_stats = []

    @time_it
    def fit(self, X: pl.DataFrame, y: Optional[Any] = None) -> "MarsStatsSelector":
        
        # 防呆检查：不能同时跳过粗筛和精筛
        if self.skip_rough_scan and self.skip_fine_scan:
            raise ValueError("❌ Cannot skip both rough scan and fine scan. At least one binning stage is required.")

        X = self._ensure_polars_dataframe(X)
        self._funnel_stats = [] 
        self._feature_iv_dict = {} # 重置特征IV字典
        
        # 0. 初始状态
        exclude_cols = {self.target}
        if self.time_col: exclude_cols.add(self.time_col)
        if self.profile_by: exclude_cols.add(self.profile_by)
        
        candidate_features = [c for c in (self.features if self.features else X.columns) 
                              if c in X.columns and c not in exclude_cols]
        
        # 黑名单拦截
        current_features = [c for c in candidate_features if c not in self.black_list]
        self._record_funnel("Init", "Blacklist & Exclusions", 
                            {"black_list_len": len(self.black_list)}, 
                            len(candidate_features), len(current_features))

        # Stage 1: 质量清洗
        if current_features:
            prev_count = len(current_features)
            current_features = self._filter_quality(X, current_features)
            
            thr_msg = (
                f"miss < {self.missing_thr} & "
                f"zero < {self.zeros_thr} & "
                f"mode < {self.mode_thr}"
            )
            self._record_funnel(
                stage="Stage 1", 
                description="Data Quality Check", 
                thresholds=thr_msg, 
                count_before=prev_count, 
                count_after=len(current_features)
            )

        # Stage 2: 粗筛 
        if not self.skip_rough_scan and current_features:
            prev_count = len(current_features)
            current_features = self._filter_rough(X, current_features)
            thr_msg = f"iv >= {self.rough_iv_thr} | (lift >= {self.rough_lift_thr} & sample >= {self.rough_min_sample_rate})"
            self._record_funnel("Stage 2", "Rough Scan (Native)", thr_msg, prev_count, len(current_features))

        # Stage 3: 精选 
        if not self.skip_fine_scan and current_features:
            prev_count = len(current_features)
            current_features = self._filter_fine(X, current_features)
            thr_msg = f"iv >= {self.iv_thr} | (lift >= {self.lift_thr} & sample >= {self.min_sample_rate})"
            self._record_funnel("Stage 3", "Fine Scan (Optimal)", thr_msg, prev_count, len(current_features))
        
        # [新增] 晋升逻辑：如果跳过了精筛，直接把粗筛的 NativeBinner 提拔为全局工作分箱器
        elif self.skip_fine_scan:
            logger.info("⏩ Fine Scan skipped. Promoting Rough Scan Binner to main pipeline.")
            self._stage3_binner = self._rough_binner

        # Stage 4: 稳定性检查 (PSI)
        if current_features and (self.time_col or self.profile_by):
            prev_count = len(current_features)
            current_features = self._filter_psi(X, current_features)
            self._record_funnel(
                stage="Stage 4", 
                description="Stability Check (PSI)", 
                thresholds={"psi": self.psi_thr}, 
                count_before=prev_count, 
                count_after=len(current_features)
            )

        # Stage 5: 逻辑稳定性检查 (RiskCorr)
        if current_features and (self.time_col or self.profile_by) and self.rc_thr is not None:
            prev_count = len(current_features)
            current_features = self._filter_rc(X, current_features)
            self._record_funnel(
                stage="Stage 5", 
                description="Risk Consistency (RiskCorr)", 
                thresholds={"rc": self.rc_thr}, 
                count_before=prev_count, 
                count_after=len(current_features)
            )

        # Stage 6: 相关性
        if current_features and self.corr_thr is not None:
            prev_count = len(current_features)
            current_features = self._filter_corr(X, current_features)
            self._record_funnel("Stage 6", "Correlation Filter", 
                                {"corr": self.corr_thr}, 
                                prev_count, len(current_features))

        # 白名单保送
        self.selected_features_ = list(set(current_features + self.white_list))
        self._record_funnel("Final", "White List Forcing", 
                            {"white_list_len": len(self.white_list)}, 
                            len(current_features), len(self.selected_features_))

        # =====================================================================
        # [新增] 生命周期管理：模型瘦身 (Model Pruning)
        # =====================================================================
        if self._stage3_binner is not None:
            logger.info(f"🗜️ Pruning internal binner state to {len(self.selected_features_)} selected features...")
            self._stage3_binner.prune(self.selected_features_)
            
            # 释放 X 和 y 的 DataFrame 内存引用
            self.clear_cache()
        # =====================================================================

        self._is_fitted = True
        
        # 拟合结束自动展示精美报表
        self.show_summary()
        return self
    
    def _record_funnel(self, stage: str, description: str, thresholds: Union[Dict, str], count_before: int, count_after: int):
        """记录漏斗统计快照，增强逻辑表达"""
        if not hasattr(self, "_initial_feature_count") or self._initial_feature_count == 0:
            self._initial_feature_count = count_before

        # 如果传入的是字典，按逗号拼接；如果是字符串，直接使用
        if isinstance(thresholds, dict):
            thr_str = ", ".join([f"{k}={v}" for k, v in thresholds.items()])
        else:
            thr_str = thresholds

        self._funnel_stats.append({
            "Stage": stage,
            "Description": description,
            "Thresholds": thr_str,
            "Input": count_before,
            "Dropped": count_before - count_after,
            "Remaining": count_after,
            "Retention %": f"{(count_after / count_before * 100):.1f}%" if count_before > 0 else "0%",
            "Cumulative %": f"{(count_after / self._initial_feature_count * 100):.1f}%" if self._initial_feature_count > 0 else "0%"
        })
    
    def show_summary(self):
        """[Interactive] 展示特征筛选漏斗摘要报表"""
        if not self._funnel_stats:
            logger.warning("No funnel stats available. Run .fit() first.")
            return

        df_summary = pd.DataFrame(self._funnel_stats)
        
        def _color_retention(val):
            try:
                p = float(val.rstrip('%'))
                if p < 30: return 'color: #d32f2f; font-weight: bold;'
                if p < 70: return 'color: #ed6c02; font-weight: bold;'
                return 'color: #2e7d32; font-weight: bold;'
            except: return 'color: #757575;'

        # 定义逻辑符号高亮的正则表达式/判断逻辑
        def _style_logic_text(v):
            if not isinstance(v, str): return ''
            # 只要包含逻辑运算符，就给予深紫色加粗显示
            logic_ops = ['&', '|', '<', '>', '=']
            if any(op in v for op in logic_ops):
                return 'color: #7b1fa2; font-weight: bold; font-family: "Courier New", Courier, monospace;'
            return ''

        styler = (
            df_summary.style
            .set_caption("🛡️ Mars Stats Selector: Feature Selection Funnel")
            
            # 进度条渲染
            .bar(subset=['Dropped'], color='#ffdbdb', vmin=0)
            .bar(subset=['Remaining'], color='#e1f5fe', vmin=0)
            
            # 文本对齐
            .set_properties(**{'text-align': 'left'}, subset=['Stage', 'Description', 'Thresholds'])
            .set_properties(**{'text-align': 'center'}, 
                            subset=['Input', 'Dropped', 'Remaining', 'Retention %', 'Cumulative %'])
            
            # 数值与比例颜色
            .map(lambda v: 'color: #d32f2f; font-weight: bold;' if v > 0 else 'color: #999;', subset=['Dropped'])
            .map(lambda v: 'color: #1565c0; font-weight: bold;', subset=['Remaining'])
            .map(_color_retention, subset=['Retention %'])
            .map(lambda v: 'font-weight: bold; color: #1b5e20; background-color: #f1f8e9;', subset=['Cumulative %'])
            
            # 逻辑表达式高亮
            .map(_style_logic_text, subset=['Thresholds'])

            # 全局布局样式
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#f5f5f5'), ('color', '#333'), ('border-bottom', '2px solid #ddd'),
                    ('padding', '12px'), ('text-align', 'center')
                ]},
                {'selector': 'th.col0, th.col1, th.col2', 'props': [('text-align', 'left')]},
                {'selector': 'caption', 'props': [
                    ('font-size', '18px'), ('font-weight', 'bold'), ('padding', '12px'), ('color', '#1a237e'), ('text-align', 'left')
                ]}
            ])
        )

        try:
            from IPython.display import display
            display(styler)
        except ImportError:
            print(df_summary.to_string(index=False))

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
        for row in stats_records:
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
                
        return kept_features

    def _filter_rough(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 2] 粗筛: 快指针 + Lift召回 (现已全面支持数值与类别特征)"""
        
        if not features:
            return []

        cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
        cat_features = [c for c in features if df.schema[c] in cat_types]

        if cat_features:
            logger.info(f"  -> Native Binner will also evaluate {len(cat_features)} categorical features.")

        binner = MarsNativeBinner(
            features=features, 
            cat_features=cat_features,
            missing_values=self.missing_values,
            special_values=self.special_values,
            n_jobs=self.n_jobs,
            **self.rough_binning_params
        )
        target = df.get_column(self.target)
        binner.fit(df, target)
        
        # [新增] 缓存粗筛的分箱器，以备在 skip_fine_scan 时晋升使用
        self._rough_binner = binner
        
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
        
        kept_features = []
        
        for row in feat_stats:
            feat, iv, has_high_lift, max_lift = row["feature"], row["IV_total"], row["has_high_lift"], row["max_lift"]
            
            # [核心修改] 即使是在粗筛阶段，只要被选中/保送，就把 IV 存下来
            # 这是为了防止 skip_fine_scan 时，Stage 6 相关性过滤找不到排序依据
            
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                self._feature_iv_dict[feat] = iv 
                continue
                
            if iv > self.rough_iv_thr or has_high_lift:
                reason = f"Pass (IV={iv:.3f})" if iv > self.rough_iv_thr else f"Pass (Lift={max_lift:.2f})"
                self._register_decision(feat, "Selected", "Rough_Scan", reason, iv)
                kept_features.append(feat)
                self._feature_iv_dict[feat] = iv # 存储 IV
            else:
                self._register_decision(feat, "Dropped", "Rough_Scan", "Low IV & Low Lift", iv)

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
        for feat in features:
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue
                
            psi_val = psi_map.get(feat, 0.0)
            if psi_val < self.psi_thr:
                self._register_decision(feat, "Selected", "Stability", "Stable PSI", psi_val)
                kept_features.append(feat)
            else:
                self._register_decision(feat, "Dropped", "Stability", f"High PSI ({psi_val:.2f})", psi_val)

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
        for feat in features:
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

        return kept_features

    def _filter_corr(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """[Stage 6] 相关性过滤"""
        
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

        for feat in sorted_feats:
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

    def get_eval_report(self, df: Union[pl.DataFrame, pd.DataFrame]) -> Tuple["MarsEvaluationReport", "MarsBinEvaluator"]:
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
            
        X_pl = self._ensure_polars_dataframe(df)
        
        from mars.analysis.evaluator import MarsBinEvaluator
        
        # 优先复用现成的 Binner (如果开启了 skip_fine_scan，这里拿到的就是 NativeBinner)
        if self._stage3_binner is not None:
            evaluator = MarsBinEvaluator(
                target=self.target,
                binner=self._stage3_binner
            )
        else:
            # 兜底逻辑
            logger.warning("Cached binner not found. Re-fitting Binner for the selected features...")
            cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
            cat_features = [c for c in self.selected_features_ if X_pl.schema[c] in cat_types]
            
            # 根据是否跳过了精筛，智能选择兜底分箱类型
            bining_type = "native" if self.skip_fine_scan else "opt"
            binning_params = self.rough_binning_params if self.skip_fine_scan else self.binning_params
            
            evaluator = MarsBinEvaluator(
                target=self.target,
                bining_type=bining_type,
                cat_features=cat_features,
                missing_values=self.missing_values,
                special_values=self.special_values,
                **binning_params
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
                
        logger.info("Export Complete.")

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
        
    def print_stats(self, iv_thresholds: Optional[List[float]] = None) -> None:
        """
        [Interactive] 打印筛选结束后的特征质量统计战报。
        快速展示入模特征的全局 IV 分布、最高 IV 以及平均水平。

        Parameters
        ----------
        iv_thresholds : List[float], optional
            用于统计的 IV 阈值分档列表。如果不传，默认展示 [0.02, 0.05, 0.10]。
        """
        self._check_is_fitted()

        if not self.selected_features_:
            logger.warning("⚠️ No features survived the selection funnel.")
            return

        # 默认阈值与排序
        if iv_thresholds is None:
            iv_thresholds = [0.02, 0.05, 0.10]
        else:
            iv_thresholds = sorted(iv_thresholds) # 确保按从小到大排序展示

        # 提取最终入选特征的 IV 列表
        final_ivs = [self._feature_iv_dict.get(f, 0.0) for f in self.selected_features_]
        total = len(self.selected_features_)
        
        max_iv = max(final_ivs) if final_ivs else 0.0
        mean_iv = sum(final_ivs) / total if total > 0 else 0.0
        
        # 动态拼装打印信息
        stats_msg = [
            f"\n{'='*50}",
            f"🎯 Mars Feature Selection Final Stats",
            f"{'-'*50}",
            f"🔹 Survived Features : {total}",
            f"🔹 Maximum IV        : {max_iv:.4f}",
            f"🔹 Average IV        : {mean_iv:.4f}"
        ]
        
        # 动态遍历用户传入的阈值
        for thr in iv_thresholds:
            count = sum(1 for iv in final_ivs if iv >= thr)
            # 使用 < 占位符保证排版对齐
            stats_msg.append(f"🔹 IV >= {thr:<13} : {count} ({count/total:.1%})")
            
        stats_msg.append(f"{'='*50}")
        
        print("\n".join(stats_msg))
    
    
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