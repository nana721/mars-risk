import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Any, NamedTuple
from mars.utils.logger import logger

try:
    from IPython.display import display, HTML
except ImportError:
    display = None
    
class ProfileData(NamedTuple):
    overview: Union[pl.DataFrame, pd.DataFrame]
    dq_trends: Dict[str, Union[pl.DataFrame, pd.DataFrame]]
    stats_trends: Dict[str, Union[pl.DataFrame, pd.DataFrame]]

class MarsProfileReport:
    """
    [报告容器] MarsProfileReport - 统一管理数据画像结果的展示与导出。
    
    该类作为 MarsDataProfiler 的输出容器，负责将原始的统计数据 (DataFrame)
    转换为适合阅读分析的格式。它支持两种主要的输出渠道：
    1. **Jupyter Notebook**: 生成富文本 HTML，包含交互式表格、热力图和迷你分布图。
    2. **Excel 文件**: 导出带格式 (条件格式、数据条、百分比) 的 Excel 报表。

    Attributes
    ----------
    overview_table : Union[pl.DataFrame, pd.DataFrame]
        全量概览大宽表，包含所有特征的统计指标。
    dq_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
        数据质量 (DQ) 指标的分组趋势表字典，key 为指标名 (如 'missing')。
    stats_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
        统计指标的分组趋势表字典，key 为指标名 (如 'mean')。
    """

    def __init__(
        self, 
        overview: Union[pl.DataFrame, pd.DataFrame],
        dq_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        stats_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]]
    ) -> None:
        self.overview_table = overview
        self.dq_tables = dq_tables
        self.stats_tables = stats_tables
        
        # 建立索引：将所有指标名映射到对应的数据源类型 ('dq' 或 'stat')
        # 这允许我们在 show_trend 中快速定位
        self._metric_index: Dict[str, str] = {}
        for k in self.dq_tables.keys():
            self._metric_index[k] = "dq"
        for k in self.stats_tables.keys():
            self._metric_index[k] = "stat"

    def get_profile_data(self) -> ProfileData:
        """[API] 获取原始数据对象"""
        return ProfileData(
            overview=self.overview_table,
            dq_trends=self.dq_tables,
            stats_trends=self.stats_tables
        )

    def _repr_html_(self) -> str:
        """
        [Internal] Jupyter Notebook 控制面板 
        """
        df_ov = self.overview_table
        n_feats = len(df_ov) if hasattr(df_ov, "__len__") else df_ov.height
        
        dq_keys = list(self.dq_tables.keys())
        stat_keys = list(self.stats_tables.keys())

        # --- 样式定义 (Inline CSS for portability) ---
        # 胶囊样式，用于包裹指标名
        pill_style = (
            "background-color: #e8f4f8; color: #2980b9; border: 1px solid #bce0eb; "
            "padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; margin-right: 4px;"
        )
        # 代码块样式
        code_style = (
            "background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; "
            "font-family: monospace; color: #e74c3c; font-weight: bold;"
        )
        
        # --- 辅助函数：生成指标徽章列表 ---
        def _fmt_pills(keys):
            if not keys: return "<span style='color:#ccc'>None</span>"
            # 为了防止指标太多撑爆屏幕，限制显示数量 (例如只显示前 20 个，后面加 ...)
            display_keys = keys[:30] 
            pills = "".join([f"<span style='{pill_style}'>'{k}'</span>" for k in display_keys])
            if len(keys) > 30:
                pills += f"<span style='color:#999; font-size:0.8em'> (+{len(keys)-30} more)</span>"
            return pills

        # --- 组装 HTML ---
        return f"""
        <div style="border: 1px solid #e0e0e0; border-left: 5px solid #2980b9; border-radius: 4px; background: white; max-width: 900px; font-family: 'Segoe UI', sans-serif;">
            
            <div style="padding: 12px 15px; background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold; color: #2c3e50; font-size: 1.1em;">
                    📊 Mars Data Profile
                </div>
                <div style="font-size: 0.85em; color: #7f8c8d;">
                    <span style="margin-left: 15px;">🏷️ Features: <b>{n_feats}</b></span>
                    <span style="margin-left: 15px;">🔍 DQ Metrics: <b>{len(dq_keys)}</b></span>
                    <span style="margin-left: 15px;">📉 Stat Metrics: <b>{len(stat_keys)}</b></span>
                </div>
            </div>

            <div style="padding: 15px;">
                
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 0.8em; text-transform: uppercase; color: #95a5a6; font-weight: bold; margin-bottom: 5px;">Quick Actions</div>
                    <div style="display: flex; gap: 20px; font-size: 0.95em;">
                        <div>👉 <span style="{code_style}">.show_overview()</span> &nbsp;<span style="color:#555">View Full Report</span></div>
                        <div>💾 <span style="{code_style}">.write_excel()</span> &nbsp;<span style="color:#555">Export XLSX</span></div>
                        <div>📥 <span style="{code_style}">.get_profile_data()</span> &nbsp;<span style="color:#555">Get Raw Data</span></div>
                    </div>
                </div>

                <div style="border-top: 1px dashed #e0e0e0; padding-top: 12px;">
                    <div style="font-size: 0.8em; text-transform: uppercase; color: #95a5a6; font-weight: bold; margin-bottom: 8px;">
                        Trend Analysis <span style="font-weight:normal; text-transform:none; color:#bbb">(Use <code>.show_trend('metric_name')</code>)</span>
                    </div>
                    
                    <div style="display: flex; margin-bottom: 8px; align-items: baseline;">
                        <div style="width: 80px; font-weight: bold; color: #27ae60; font-size: 0.9em;">DQ:</div>
                        <div style="flex: 1; line-height: 1.6;">
                            {_fmt_pills(dq_keys)}
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: baseline;">
                        <div style="width: 80px; font-weight: bold; color: #2980b9; font-size: 0.9em;">Stats:</div>
                        <div style="flex: 1; line-height: 1.6;">
                            {_fmt_pills(stat_keys)}
                        </div>
                    </div>
                </div>

            </div>
            
            <div style="padding: 6px 15px; background-color: #fff8e1; border-top: 1px solid #fae5b0; font-size: 0.8em; color: #d35400;">
                💡 <b>Pro Tip:</b> Use <span style="{code_style}">.show_trend('psi')</span> to detect population stability drift.
            </div>
        </div>
        """

    def show_overview(self) -> "pd.io.formats.style.Styler":
        """展示全量概览大宽表"""
        return self._get_styler(
            self.overview_table, 
            title="Dataset Overview", 
            cmap="RdYlGn_r", 
            # 指定哪些列应用“红绿灯”配色 (高值=红)
            subset_cols=["missing_rate", "zeros_rate", "unique_rate", "top1_ratio"],
            fmt_as_pct=False # 概览表混合了多种类型，不强制全转百分比，由内部逻辑细分
        )

    def show_trend(self, metric: str, ascending: bool = True) -> "pd.io.formats.style.Styler":
        """
        [统一接口] 展示指定指标的分组趋势。

        该方法会自动根据指标类型（DQ 或 Stats）智能选择可视化模板：
        - **DQ指标 (missing, etc.)**: 自动使用百分比格式 + 红绿灯配色 (RdYlGn_r)。
        - **PSI**: 使用红绿灯配色 + 0.25 阈值锚定。
        - **Stability (group_cv)**: 自动附加数据条。
        - **常规统计 (mean, max)**: 使用蓝色热力图 (Blues)。

        Parameters
        ----------
        metric : str
            指标名称 (如 'missing', 'mean', 'psi')。
        ascending : bool, default True
            时间/分组列的排序方式。
        """
        # 1. 路由逻辑：查找指标属于哪个表
        source_type = self._metric_index.get(metric)
        if source_type is None:
             # 提供更友好的报错提示
            available = list(self._metric_index.keys())
            raise ValueError(f"❌ Metric '{metric}' not found. Available metrics: {available[:10]}...")

        # 2. 获取数据
        if source_type == "dq":
            df_raw = self.dq_tables[metric]
            # DQ 默认配置
            cmap = "RdYlGn_r"  # 红色代表高风险 (高缺失)
            fmt_pct = True     # DQ 指标通常是率 (Rate/Ratio)
            vmin, vmax = 0, 1  # 率通常在 0~1 之间
            
        else: # source_type == "stat"
            df_raw = self.stats_tables[metric]
            # Stats 默认配置
            cmap = "Blues"     # 蓝色代表数值高低 (中性)
            fmt_pct = False    # 统计值通常是绝对值
            vmin, vmax = None, None

        # 3. 特殊指标微调 (Override)
        if metric == "psi":
            cmap = "RdYlGn_r" # PSI 高了是坏事
            fmt_pct = False   # PSI 是数值不是百分比
            vmin, vmax = 0.0, 0.5 # 锚定阈值
        
        # 4. 转换与排序
        df = self._to_pd(df_raw)
        df = self._reorder_trend_cols(df, ascending)

        # 5. 生成样式
        return self._get_styler(
            df,
            title=f"Trend Analysis: {metric}",
            cmap=cmap,
            fmt_as_pct=fmt_pct,
            vmin=vmin, 
            vmax=vmax,
            add_bars=True # 所有趋势表都允许显示 CV 条
        )

    def _reorder_trend_cols(self, df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
        """[Internal Helper] 重新排列趋势表的列顺序。"""
        # 定义元数据列和末尾统计列
        meta_cols = ["feature", "dtype", "distribution", "top1_value"]
        stat_cols = ["total", "group_mean", "group_var", "group_cv"]
        
        # 识别中间的分组列（如时间列）
        all_cols = df.columns.tolist()
        group_cols = [c for c in all_cols if c not in meta_cols + stat_cols]
        
        # 排序分组列
        group_cols_sorted = sorted(group_cols, reverse=not ascending)
        
        # 组合最终顺序
        final_order = [c for c in meta_cols if c in all_cols] + \
                      group_cols_sorted + \
                      [c for c in stat_cols if c in all_cols]
        return df[final_order]

    def write_excel(self, path: str = "mars_report.xlsx", ascending: bool = True) -> None:
        
        logger.info(f"📊 Exporting report to: {path}...")
        
        # 1. 依赖检查
        try:
            import xlsxwriter
        except ImportError:
            logger.error("❌ 'xlsxwriter' is required for Excel export. Install it via: pip install xlsxwriter")
            return

        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                # -----------------------------------------------------------
                # 1. 导出概览页 (Overview)
                # -----------------------------------------------------------
                overview_styler = self.show_overview()
                if overview_styler is not None:
                    overview_styler.to_excel(writer, sheet_name="Overview", index=False)
                
                # -----------------------------------------------------------
                # 2. 统一导出所有趋势页 (Trend & DQ)
                # -----------------------------------------------------------
                # 将 DQ 和 Stats 的 key 合并处理
                dq_keys = list(self.dq_tables.keys())
                stat_keys = list(self.stats_tables.keys())
                all_metrics = dq_keys + stat_keys
                
                for metric in all_metrics:
                    # [关键修复] 全部统一调用 show_trend
                    styler = self.show_trend(metric, ascending=ascending)
                    
                    if styler is not None:
                        # 动态决定 Sheet 前缀
                        prefix = "DQ" if metric in self.dq_tables else "Trend"
                        # 生成安全名称 (截断到31字符)
                        sheet_name = f"{prefix}_{metric.capitalize()}"[:31]
                        
                        styler.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # 应用条件格式 (PSI 颜色 / Data Bars)
                        # 为了避免 write_excel 太长，我们将逻辑抽离到辅助函数
                        self._apply_excel_formatting(writer, sheet_name, metric, ascending)

                # 3. 自动列宽调整
                for sheet in writer.sheets.values():
                    sheet.autofit()
                    
            logger.info("✅ Report exported successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to export Excel: {e}", exc_info=True)

    def _apply_excel_formatting(self, writer, sheet_name: str, metric: str, ascending: bool):
        """
        [Helper] 抽离 Excel 条件格式逻辑，保持主流程清晰。
        """
        # 我们需要获取底层的 DataFrame 来确定列索引位置
        # 注意：需要重新获取对应的数据表并排序，以匹配 Excel 中的列顺序
        if metric in self.dq_tables:
            raw_df = self.dq_tables[metric]
        else:
            raw_df = self.stats_tables[metric]
            
        # 转换为 Pandas 并重排，确保与 Excel 内容一致
        df_pd = self._reorder_trend_cols(self._to_pd(raw_df), ascending)
        
        worksheet = writer.sheets[sheet_name]
        
        # 1. PSI 专用三色阶 (红绿灯)
        if metric == "psi":
            # 识别中间的数据列范围 (排除 feature, dtype 等元数据)
            # 简单的定位策略：从第3列开始(feature, dtype, distribution...) 到 倒数第4列结束
            # 更稳健的方法是排除掉已知的非数据列
            meta_cols = ["feature", "dtype", "distribution", "top1_value"]
            # 找到第一个不是 meta_col 的列索引
            start_col = 0
            for i, col in enumerate(df_pd.columns):
                if col not in meta_cols:
                    start_col = i
                    break
            
            # 排除末尾的聚合统计列
            stat_cols = ["total", "group_mean", "group_var", "group_cv"]
            end_col = len(df_pd.columns) - 1
            # 这里的逻辑：只对中间的分组列应用红绿灯
            # 如果你想对 total 列也应用，可以调整 end_col
            
            worksheet.conditional_format(1, start_col, len(df_pd), end_col, {
                'type': '3_color_scale',
                'min_type': 'num', 'min_value': 0.05, 'min_color': '#63BE7B', # Green
                'mid_type': 'num', 'mid_value': 0.15, 'mid_color': '#FFEB84', # Yellow
                'max_type': 'num', 'max_value': 0.25, 'max_color': '#F8696B'  # Red
            })

        # 2. 稳定性 Data Bars (针对 group_cv)
        if "group_cv" in df_pd.columns:
            col_idx = df_pd.columns.get_loc("group_cv")
            worksheet.conditional_format(1, col_idx, len(df_pd), col_idx, {
                'type': 'data_bar', 
                'bar_color': '#638EC6', 
                'bar_solid': True,
                'min_type': 'num', 'min_value': 0, 
                'max_type': 'num', 'max_value': 1
            })

    def _to_pd(self, df: Any) -> pd.DataFrame:
        """
        [辅助方法] 确保数据转换为 Pandas DataFrame 格式。
        """
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    def _get_styler(
        self, 
        df_input: Any, 
        title: str, 
        cmap: str, 
        subset_cols: Optional[List[str]] = None, 
        add_bars: bool = False, 
        fmt_as_pct: bool = False,
        vmin: Optional[float] = None, 
        vmax: Optional[float] = None
    ) -> Optional["pd.io.formats.style.Styler"]:
        """
        [Internal] 通用样式生成器。
        """
        if df_input is None:
            return None
        df: pd.DataFrame = self._to_pd(df_input)
        if df.empty:
            return None

        # 元数据排除列表
        exclude_meta: List[str] = [
            "feature", "dtype", 
            "group_mean", "group_var", "group_cv",
            "distribution",
            "top1_value"
            ]
        
        # 1. 确定色彩渐变范围
        if subset_cols:
            gradient_cols: List[str] = [c for c in subset_cols if c in df.columns]
        else:
            gradient_cols = [c for c in df.columns if c not in exclude_meta]

        styler = df.style.set_caption(f"<b>{title}</b>").hide(axis="index")
        
        # 2. 应用热力图
        if gradient_cols:
            styler = styler.background_gradient(
                cmap=cmap, 
                subset=gradient_cols, 
                axis=None,
                vmin=vmin,
                vmax=vmax
            )
        
        # 3. 应用数据条
        if add_bars and "group_cv" in df.columns:
            styler = styler.bar(subset=["group_cv"], color='#ff9999', vmin=0, vmax=1, width=90)
            styler = styler.format("{:.4f}", subset=["group_cv", "group_var"])

        # 4. 数值格式化逻辑
        num_cols: pd.Index = df.select_dtypes(include=['number']).columns
        data_cols: List[str] = [c for c in num_cols if c not in ["group_var", "group_cv", "distribution"]]

        pct_format: str = "{:.2%}"  
        float_format: str = "{:.2f}"

        if fmt_as_pct:
            if data_cols:
                styler = styler.format(pct_format, subset=data_cols)
        else:
            pct_cols: List[str] = [
                c for c in df.columns 
                if ("rate" in c or "ratio" in c) and (c in num_cols)
            ]
            
            if pct_cols:
                styler = styler.format(pct_format, subset=pct_cols)
            
            float_cols: List[str] = [c for c in data_cols if c not in pct_cols]
            if float_cols:
                styler = styler.format(float_format, subset=float_cols)
        
        # 5. 分布迷你图样式
        if "distribution" in df.columns:
            styler = styler.set_table_styles([
                {'selector': '.col_distribution', 'props': [
                    # 优先使用 Consolas (Win) 或 Menlo (Mac)，最后 fallback 到 monospace
                    ('font-family', '"Consolas", "Menlo", "Courier New", monospace'), 
                    ('color', '#1f77b4'),
                    ('white-space', 'pre'), # [关键] 防止 HTML 自动压缩连续空格
                    ('font-weight', 'bold'),
                    ('text-align', 'left')
                ]}
            ], overwrite=False)

        # 6. 全局表格外观
        styler = styler.set_table_styles([
            {
                'selector': 'th', 
                'props': [('text-align', 'left'), ('background-color', '#f0f2f5'), ('color', '#333')]
            },
            {
                'selector': 'caption', 
                'props': [('font-size', '1.2em'), ('padding', '10px 0'), ('color', '#2c3e50')]
            }
        ], overwrite=False)

        return styler
    
class MarsEvaluationReport:
    """
    [MarsEvaluationReport] 特征效能评估报告容器。

    该类负责存储、展示和导出特征评估结果。它支持 Polars 和 Pandas 输入，
    并确保返回的数据类型与输入时保持一致。

    核心功能：
    1. **交互展示**: 在 Jupyter Notebook 中提供带有热力图颜色的 Styler 对象。
    2. **监控报警**: 自动识别高 PSI 特征并在仪表盘中预警。
    3. **多维趋势**: 支持指标（PSI/AUC/IV/BadRate/RiskCorr）的时间序列热力图分析。
    4. **专业导出**: 生成符合金融业务标准的带条件格式的 Excel 监控周/月报。

    Attributes
    ----------
    summary_table : Union[pl.DataFrame, pd.DataFrame]
        特征级汇总统计表（包含 PSI 最大/均值、AUC 均值、IV 总计等）。
    trend_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
        按指标分类的时间趋势表字典（Key 为指标名，Value 为透视后的宽表）。
    detail_table : Union[pl.DataFrame, pd.DataFrame]
        分箱明细表（包含每个特征、每个时间切片、每个分箱的样本数、坏账率等）。
    group_col : str, optional
        分组列名（如 'month'），用于标识趋势分析的时间维度。
    """

    def __init__(
        self, 
        summary_table: Union[pl.DataFrame, pd.DataFrame],
        trend_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        detail_table: Union[pl.DataFrame, pd.DataFrame],
        group_col: Optional[str] = None
    ) -> None:
        """
        初始化报告容器。

        Parameters
        ----------
        summary_table : Union[pl.DataFrame, pd.DataFrame]
            特征级汇总表。
        trend_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            指标趋势表字典。
        detail_table : Union[pl.DataFrame, pd.DataFrame]
            最细粒度的分箱明细表。
        group_col : str, optional
            分组列名（例如：'month' 或 'vintage'）。
        """
        # 直接存储原始数据，不再强制命名为 _pl，以支持多种类型
        self._summary = summary_table
        self._trend_dict = trend_tables
        self._detail = detail_table
        self.group_col = group_col # [新增] 记录分组列名
        
    @property
    def summary_table(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """返回汇总统计表（类型与输入一致）。"""
        return self._summary

    @property
    def trend_tables(self) -> Dict[str, Union[pl.DataFrame, pd.DataFrame]]:
        """返回趋势宽表字典（类型与输入一致）。"""
        return self._trend_dict

    @property
    def detail_table(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """返回分箱明细表（类型与输入一致）。"""
        return self._detail

    def get_evaluation_data(self) -> Tuple[
        Union[pl.DataFrame, pd.DataFrame], 
        Dict[str, Union[pl.DataFrame, pd.DataFrame]], 
        Union[pl.DataFrame, pd.DataFrame]
    ]:
        """
        获取所有原始数据。
        
        Returns
        -------
        Tuple
            返回 (汇总表, 趋势表字典, 明细表)，类型与输入一致。
        """
        return self.summary_table, self.trend_tables, self.detail_table

    def _to_pd(self, df: Any) -> pd.DataFrame:
        """辅助函数：将输入对象转为 Pandas DataFrame（用于展示或导出）。"""
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    def _repr_html_(self) -> str:
        """
        [Dashboard] Jupyter Notebook 控制面板。
        """
        # 内部展示逻辑统一转为 Pandas 处理
        df_summary_pd = self._to_pd(self.summary_table)
        n_feats = len(df_summary_pd)
        
        # 简单统计报警数
        high_risk_psi = 0
        if "PSI_max" in df_summary_pd.columns:
            high_risk_psi = sum(df_summary_pd["PSI_max"] > 0.25)

        # 样式定义
        pill_style = (
            "background-color: #e8f4f8; color: #2980b9; border: 1px solid #bce0eb; "
            "padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; margin-right: 4px;"
        )
        
        # 动态生成 Trend 链接
        trend_keys = list(self.trend_tables.keys())
        trend_pills = "".join([f"<span style='{pill_style}'>'{k}'</span>" for k in trend_keys])

        lines = []
        # 查看类操作
        lines.append('👉 <code>.show_summary()</code> &nbsp;<span style="color:#7f8c8d">View Feature Ranking</span>')
        lines.append(f'👉 <code>.show_trend(metric)</code> <span style="color:#7f8c8d">metric: {trend_pills}</span>')
        
        # [新增] 获取数据类操作
        lines.append('<hr style="margin: 8px 0; border: 0; border-top: 1px dashed #ccc;">')
        lines.append('📥 <code>.get_evaluation_data()</code> &nbsp;<span style="color:#7f8c8d">Get Raw Data (summary, trends, detail)</span>')
        lines.append('💾 <code>.write_excel()</code> &nbsp;<span style="color:#7f8c8d">Export to Excel</span>')

        return f"""
        <div style="border-left: 5px solid #8e44ad; background-color: #f4f6f7; padding: 15px; border-radius: 0 5px 5px 0; font-family: 'Segoe UI', sans-serif;">
            <h3 style="margin:0 0 10px 0; color:#2c3e50;">📉 Mars Feature Evaluation</h3>
            
            <div style="display: flex; gap: 30px; margin-bottom: 12px; font-size: 0.95em;">
                <div><strong>🏷️ Features:</strong> {n_feats}</div>
                <div><strong>🚨 High PSI (>0.25):</strong> <span style="color: {'red' if high_risk_psi > 0 else 'green'}; font-weight:bold;">{high_risk_psi}</span></div>
                <div><strong>📅 Group By:</strong> {self.group_col if self.group_col else 'None (Total Only)'}</div>
            </div>
            
            <div style="font-size:0.9em; line-height:1.8; color:#2c3e50; background: white; padding: 10px; border: 1px solid #e0e0e0; border-radius: 4px;">
                { "<br>".join(lines) }
            </div>
        </div>
        """

    def show_summary(self) -> "pd.io.formats.style.Styler":
        """
        展示特征汇总评分表。
        """
        # 转换为 Pandas 以利用 Styler
        df = self._to_pd(self.summary_table)
        styler = df.style.set_caption("<b>Feature Performance Summary</b>").hide(axis="index")

        # 1. PSI: 低好高坏 (RdYlGn_r)
        if "PSI_max" in df.columns:
            styler = styler.background_gradient(
                cmap="RdYlGn_r", subset=["PSI_max", "PSI_avg"], vmin=0, vmax=0.5
            )
        
        # 2. AUC/KS: 高好低坏 (RdYlGn)
        good_metrics = [c for c in ["AUC_avg", "AUC_min", "KS_max"] if c in df.columns]
        if good_metrics:
            styler = styler.background_gradient(
                cmap="RdYlGn", subset=good_metrics, vmin=0.5, vmax=0.8
            )

        # 3. 稳定性: CV (低好高坏)
        if "CV_AUC" in df.columns:
            styler = styler.bar(subset=["CV_AUC"], color='#ff9999', vmin=0, vmax=0.2)

        # 4. 格式化
        return styler.format("{:.4f}", subset=df.select_dtypes("number").columns)

    def show_trend(self, metric: str, ascending: bool = False) -> "pd.io.formats.style.Styler":
        """
        展示指标的时间趋势热力图。
        """
        if metric not in self.trend_tables:
            raise ValueError(f"Unknown metric: {metric}. Options: {list(self.trend_tables.keys())}")
        
        # 转换为 Pandas 副本进行样式处理
        df = self._to_pd(self.trend_tables[metric]).copy()
        
        # 1. 识别列类型并排序
        meta_cols = ["feature", "dtype"]
        special_cols = ["Total"]
        time_cols = [c for c in df.columns if c not in meta_cols + special_cols]
        time_cols_sorted = sorted(time_cols, reverse=not ascending)

        # 2. 重排列顺序
        final_cols = [c for c in meta_cols if c in df.columns] + \
                     time_cols_sorted + \
                     [c for c in special_cols if c in df.columns]
        df = df[final_cols]

        # 3. 基础样式设置
        styler = df.style.set_caption(f"<b>Trend Analysis: {metric.upper()}</b>").hide(axis="index")
        styler = styler.set_properties(subset=["feature"], **{'text-align': 'left', 'font-weight': 'bold'})

        # 4. 根据指标类型选择配色 (关键修改点 👇)
        if metric == "psi":
            # PSI: 越小越绿 (RdYlGn_r)
            styler = styler.background_gradient(
                cmap="RdYlGn_r", subset=time_cols_sorted, vmin=0, vmax=0.25, axis=None
            )
        elif metric in ["auc", "ks", "iv"]:
            # 性能指标: 越大越绿 (RdYlGn)
            styler = styler.background_gradient(
                cmap="RdYlGn", subset=time_cols_sorted, axis=None
            )
        elif metric == "bad_rate":
            # 坏账率: 使用蓝色调 (Blues)
            styler = styler.background_gradient(
                cmap="Blues", subset=time_cols_sorted, axis=None
            )
        elif metric == "risk_corr":
            # [新增] 风险趋势相关性: 越接近 1 说明逻辑越稳定，越绿
            # 设置 vmin=0.5，因为相关性低于 0.7 通常就需要关注了，低于 0.5 逻辑可能已崩坏
            styler = styler.background_gradient(
                cmap="RdYlGn", subset=time_cols_sorted, vmin=0.5, vmax=1.0, axis=None
            )

        # 5. 格式化所有数值列（含 Total）
        format_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
        return styler.format("{:.4f}", subset=format_cols)

    def write_excel(self, path: str = "mars_evaluation_report.xlsx", ascending: bool = True) -> None:
        """
        导出为带有条件格式的 Excel 监控报表。

        Parameters
        ----------
        path : str
            输出路径。
        ascending : bool, default True
            趋势表中时间列的排序方式。True 为时间早的在前，False 为时间晚的在前。
        """
        logger.info(f"📊 Exporting evaluation report to: {path}...")
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                # 1. Summary Sheet
                summary_pd = self._to_pd(self.summary_table)
                summary_pd.to_excel(writer, sheet_name="Summary", index=False)
                
                # 2. Trend Sheets
                for metric, original_data in self.trend_tables.items():
                    df = self._to_pd(original_data).copy()
                    
                    # --- [新增] 应用与 show_trend 一致的排序逻辑 ---
                    meta_cols = ["feature", "dtype"]
                    special_cols = ["Total"]
                    time_cols = [c for c in df.columns if c not in meta_cols + special_cols]
                    
                    # 对时间列排序
                    time_cols_sorted = sorted(time_cols, reverse=not ascending)
                    
                    # 重新排列 Excel 中的列顺序
                    final_cols = [c for c in meta_cols if c in df.columns] + \
                                 time_cols_sorted + \
                                 [c for c in special_cols if c in df.columns]
                    df = df[final_cols]
                    # ---------------------------------------------

                    sheet_name = f"Trend_{metric.upper()}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    
                    # 计算条件格式的作用范围
                    first_row = 1
                    last_row = len(df)
                    start_col_idx = len([c for c in meta_cols if c in df.columns])
                    end_col_idx = start_col_idx + len(time_cols_sorted) - 1

                    # 3. 应用条件格式 (仅作用于时间列，不含 Total)
                    if metric == "psi":
                        worksheet.conditional_format(first_row, start_col_idx, last_row, end_col_idx, {
                            'type': '3_color_scale',
                            'min_type': 'num', 'min_value': 0,    'min_color': '#63BE7B',
                            'mid_type': 'num', 'mid_value': 0.1,  'mid_color': '#FFEB84',
                            'max_type': 'num', 'max_value': 0.25, 'max_color': '#F8696B'
                        })
                    elif metric in ["auc", "ks", "iv", "risk_corr"]: 
                        worksheet.conditional_format(first_row, start_col_idx, last_row, end_col_idx, {
                            'type': '2_color_scale',
                            'min_color': '#F8696B',
                            'max_color': '#63BE7B'
                        })
                    elif metric == "bad_rate":
                        worksheet.conditional_format(first_row, start_col_idx, last_row, end_col_idx, {
                            'type': '2_color_scale',
                            'min_color': '#FFFFFF',
                            'max_color': '#2E75B6'
                        })
                    
                    worksheet.autofit()
                    
            logger.info("✅ Export successful.")
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")