import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Any
from mars.utils.logger import logger

try:
    from IPython.display import display, HTML
except ImportError:
    display = None

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
        """
        初始化报告容器。

        Parameters
        ----------
        overview : Union[pl.DataFrame, pd.DataFrame]
            全量概览表，包含特征名、类型、分布图及各类统计指标。
        dq_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            数据质量 (DQ) 指标趋势表字典，包含缺失率、零值率等随分组维度的变化。
        stats_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            统计指标趋势表字典，包含均值、标准差等随分组维度的变化。
        """
        self.overview_table: Union[pl.DataFrame, pd.DataFrame] = overview
        self.dq_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]] = dq_tables
        self.stats_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]] = stats_tables

    def get_profile_data(self) -> Tuple[
        Union[pl.DataFrame, pd.DataFrame], 
        Dict[str, Union[pl.DataFrame, pd.DataFrame]], 
        Dict[str, Union[pl.DataFrame, pd.DataFrame]]
    ]:
        """
        [API] 获取纯净的原始数据对象。
        
        用于后续的特征筛选 (Selector)、自定义分析或将数据传入其他系统。

        Returns
        -------
        overview_df : Union[pl.DataFrame, pd.DataFrame]
            全量概览大宽表。
        dq_tables_dict : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            DQ 指标趋势字典。
        stats_tables_dict : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            统计指标趋势字典。
        """
        return self.overview_table, self.dq_tables, self.stats_tables

    def _repr_html_(self) -> str:
        """
        [Internal] Jupyter Notebook 的富文本展示接口。
        
        当在 Jupyter 环境中直接打印此对象时，生成一个交互式的 HTML 控制面板。

        Returns
        -------
        str
            包含概览统计信息和操作指南的 HTML 字符串。
        """
        df_ov: Union[pl.DataFrame, pd.DataFrame] = self.overview_table
        
        # 统计特征总数
        n_feats: int = len(df_ov) if hasattr(df_ov, "__len__") else df_ov.height
        
        # 推断分组数量
        sample_dq: Optional[Union[pl.DataFrame, pd.DataFrame]] = self.dq_tables.get('missing')
        n_groups: int = 0
        if sample_dq is not None:
            n_cols: int = len(sample_dq.columns)
            # 减去固定列: feature, dtype, total
            n_groups = max(0, n_cols - 3)

        # 构建控制面板内容
        lines: List[str] = []
        lines.append('<code>.show_overview()</code> 👈 <b>Full Overview (Recommended)</b>')
        
        dq_keys: List[str] = list(self.dq_tables.keys())
        dq_links: List[str] = [f"<code>.show_dq('{k}')</code>" for k in dq_keys]
        lines.append(f'DQ Trends: {", ".join(dq_links)}')
        
        stats_keys: List[str] = list(self.stats_tables.keys())
        if stats_keys:
            stat_links: List[str] = [f"<code>.show_trend('{k}')</code>" for k in stats_keys]
            lines.append(f'Stats Trends: {", ".join(stat_links)}')
        
        lines.append('<code>.write_excel()</code> Export formatted report')
        lines.append('<code>.get_profile_data()</code> Get raw data for feature selection')

        return f"""
        <div style="border-left: 5px solid #2980b9; background-color: #f4f6f7; padding: 15px; border-radius: 0 5px 5px 0;">
            <h3 style="margin:0 0 10px 0; color:#2c3e50;">📊 Mars Data Profile Report</h3>
            <div style="display: flex; gap: 20px; margin-bottom: 10px; color: #555;">
                <div><strong>🏷️ Features:</strong> {n_feats}</div>
                <div><strong>📅 Groups:</strong> {n_groups}</div>
            </div>
            <div style="font-size:0.9em; line-height:1.8; color:#7f8c8d; border-top: 1px solid #e0e0e0; padding-top: 8px;">
                { "<br>".join(lines) }
            </div>
        </div>
        """

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

    def show_overview(self) -> "pd.io.formats.style.Styler":
        """
        展示全量概览大宽表。
        
        采用 'RdYlGn_r' (红-黄-绿 反转) 色系展示数据质量指标：
        - 高缺失率/高单一值率 -> 红色 (警示风险)
        - 低缺失率 -> 绿色 (健康状态)

        Returns
        -------
        pd.io.formats.style.Styler
            配置了热力图、迷你图样式和数值格式化的 Pandas Styler 对象。
        """
        return self._get_styler(
            self.overview_table, 
            title="Dataset Overview", 
            cmap="RdYlGn_r", 
            subset_cols=["missing_rate", "zeros_rate", "unique_rate", "top1_ratio"],
            fmt_as_pct=False
        )

    def show_dq(self, metric: str, ascending: bool = True) -> "pd.io.formats.style.Styler":
        """
        展示指定数据质量 (DQ) 指标的趋势表。
        
        Parameters
        ----------
        metric : str
            DQ 指标名称，可选：'missing', 'zeros', 'unique', 'top1'。
        ascending : bool, default True
            趋势表中时间/分组列的排序方式。

        Returns
        -------
        pd.io.formats.style.Styler
            针对百分比指标优化的 Pandas Styler 对象。

        Raises
        ------
        ValueError
            当输入的指标名称不在 dq_tables 中时抛出。
        """
        if metric not in self.dq_tables:
            raise ValueError(f"Unknown DQ metric: {metric}")
        
        # 转换并排序
        df = self._to_pd(self.dq_tables[metric])
        df = self._reorder_trend_cols(df, ascending)
        
        return self._get_styler(
            df, 
            title=f"DQ Trends: {metric}", 
            cmap="RdYlGn_r",
            fmt_as_pct=True
        )

    def show_trend(self, metric: str, ascending: bool = True) -> "pd.io.formats.style.Styler":
        """
        展示指定统计指标的趋势表。
        
        针对稳定性指标 (group_cv) 会自动添加数据条 (Data Bars) 可视化。
        [新增] 针对 PSI 指标，使用红绿灯色系 (Red-Yellow-Green) 警示风险。

        Parameters
        ----------
        metric : str
            统计指标名称，例如：'mean', 'std', 'max', 'p50' 等。
        ascending : bool, default True
            趋势表中时间/分组列的排序方式。

        Returns
        -------
        pd.io.formats.style.Styler
            包含稳定性数据条展示的 Pandas Styler 对象。

        Raises
        ------
        ValueError
            当输入的指标名称不在 stats_tables 中时抛出。
        """
        if metric not in self.stats_tables:
            raise ValueError(f"Unknown stats metric: {metric}")
        
        # 转换并排序
        df = self._to_pd(self.stats_tables[metric])
        df = self._reorder_trend_cols(df, ascending)
        
        # [修改] 针对 PSI 的特殊可视化逻辑
        cmap = "Blues"
        vmin, vmax = None, None
        
        if metric == "psi":
            # 使用红黄绿反转色系: 低(绿) -> 中(黄) -> 高(红)
            cmap = "RdYlGn_r"
            # [关键] 锚定 PSI 的绝对值范围，确保 0.25 以上呈现明显的红色
            vmin = 0.0
            vmax = 0.5 
        
        return self._get_styler(
            df, 
            title=f"Stats Trend: {metric}", 
            cmap=cmap, 
            add_bars=True,
            fmt_as_pct=False,
            vmin=vmin,
            vmax=vmax
        )

    def write_excel(self, path: str = "mars_report.xlsx", ascending: bool = True) -> None:
        """
        将分析结果完整导出为带视觉格式的 Excel 文件。
        
        导出内容包括：
        1. Overview (概览页): 包含特征分布热力图。
        2. DQ_{Metric} (质量趋势页): 包含缺失率等趋势。
        3. Trend_{Metric} (分布趋势页): 包含稳定性分析及数据条展示。

        Excel 特性：
        - 百分比数字格式。
        - 自动列宽适配。
        - 冻结表头样式。

        Parameters
        ----------
        path : str, default "mars_report.xlsx"
            导出文件的目标路径。
        ascending : bool, default True
            趋势表中时间列的排序方式。True 为时间早的在前。
        """
        logger.info(f"📊 Exporting report to: {path}...")
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                # 1. 导出概览页
                overview_styler = self.show_overview()
                if overview_styler is not None:
                    overview_styler.to_excel(writer, sheet_name="Overview", index=False)
                
                # 2. 导出 DQ 指标页
                for name in self.dq_tables:
                    dq_styler = self.show_dq(name, ascending=ascending)
                    if dq_styler is not None:
                        dq_styler.to_excel(writer, sheet_name=f"DQ_{name}", index=False)
                
                # 3. 导出统计指标页 (特别处理 Data Bars 和 PSI Colors)
                for name in self.stats_tables:
                    trend_styler = self.show_trend(name, ascending=ascending)
                    if trend_styler is not None:
                        sheet_name: str = f"Trend_{name.capitalize()}"
                        trend_styler.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        df_pd: pd.DataFrame = self._reorder_trend_cols(self._to_pd(self.stats_tables[name]), ascending)
                        worksheet = writer.sheets[sheet_name]
                        
                        # [新增] 针对 PSI 的 Excel 条件格式 (3色阶)
                        if name == "psi":
                            # 识别数值列范围 (排除 feature, dtype)
                            meta_len = len([c for c in ["feature", "dtype"] if c in df_pd.columns])
                            start_col_idx = meta_len
                            # 排除最后的统计列
                            end_col_idx = len(df_pd.columns) - 1
                            
                            worksheet.conditional_format(1, start_col_idx, len(df_pd), end_col_idx, {
                                'type': '3_color_scale',
                                'min_type': 'num', 'min_value': 0,    'min_color': '#63BE7B',
                                'mid_type': 'num', 'mid_value': 0.1,  'mid_color': '#FFEB84',
                                'max_type': 'num', 'max_value': 0.25, 'max_color': '#F8696B'
                            })

                        # 导出 Data Bars
                        if "group_cv" in df_pd.columns:
                            col_idx: int = df_pd.columns.get_loc("group_cv")
                            worksheet.conditional_format(1, col_idx, len(df_pd), col_idx, {
                                'type': 'data_bar', 
                                'bar_color': '#FF9999', 
                                'bar_solid': True,
                                'min_type': 'num', 'min_value': 0, 
                                'max_type': 'num', 'max_value': 1
                            })
                            
                # 4. 自动列宽调整
                for sheet in writer.sheets.values():
                    sheet.autofit()
                    
            logger.info("✅ Report exported successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to export Excel: {e}")

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
                    ('font-family', 'monospace'), 
                    ('color', '#1f77b4'),
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
        """Jupyter Notebook 仪表盘视图"""
        # 内部展示逻辑统一转为 Pandas 处理
        df_summary_pd = self._to_pd(self.summary_table)
        n_feats = len(df_summary_pd)
        
        # 简单统计报警数
        high_risk_psi = 0
        if "PSI_max" in df_summary_pd.columns:
            high_risk_psi = sum(df_summary_pd["PSI_max"] > 0.25)

        lines = []
        lines.append('<code>.show_summary()</code> 👈 <b>Feature Ranking (PSI/AUC)</b>')
        
        # 动态生成链接
        trend_links = [f"<code>.show_trend('{k}')</code>" for k in self.trend_tables.keys()]
        lines.append(f'Trend Heatmaps: {", ".join(trend_links)}')
        lines.append('<code>.write_excel()</code> Export monitoring report')

        return f"""
        <div style="border-left: 5px solid #8e44ad; background-color: #f4f6f7; padding: 15px; border-radius: 0 5px 5px 0;">
            <h3 style="margin:0 0 10px 0; color:#2c3e50;">📉 Mars Feature Evaluation</h3>
            <div style="display: flex; gap: 30px; margin-bottom: 10px; color: #555;">
                <div><strong>🏷️ Features:</strong> {n_feats}</div>
                <div><strong>🚨 High PSI (>0.25):</strong> <span style="color: {'red' if high_risk_psi > 0 else 'green'}">{high_risk_psi}</span></div>
            </div>
            <div style="font-size:0.9em; line-height:1.8; color:#7f8c8d; border-top: 1px solid #e0e0e0; padding-top: 8px;">
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