from typing import List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import pandas as pd
import polars as pl
import numpy as np
from typing import Union, Optional
from IPython.display import display, HTML
import base64
from io import BytesIO
import uuid 
from mars.utils.logger import logger


class MarsPlotter:
    """
    [可视化组件] MarsPlotter - 专注于风控特征效能与稳定性分析的可视化引擎。

    该类提供了将特征分箱结果（Binning Details）转化为直观图表的能力。
    核心图表结合了：
    1. **柱状图 (Bars)**: 展示样本分布（Count Distribution），用于识别数据稀疏性。
    2. **折线图 (Lines)**: 展示风险趋势（Bad Rate），用于验证特征的逻辑单调性。
    3. **双轴设计**: 左轴代表分布占比，右轴代表坏率。
    4. **交互式容器**: 通过 HTML/JS 实现双击缩放，解决宽表展示不全的问题。
    """
    
    UNIT_WIDTH = 3  # 单个子图的基准宽度
    UNIT_HEIGHT = 2.75 # 单个子图的基准高度

    @staticmethod
    def _show_scrollable(fig: plt.Figure, dpi: int = 150):
        """
        [辅助函数] 将 Matplotlib 图表包装进可滚动、可点击放大的交互式 HTML 容器。

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            待显示的图表对象。
        dpi : int, default 150
            图像分辨率。
        """
        # 1. 将图像序列化为 Base64 字符串
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi) 
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # 及时关闭 figure 释放内存
        
        # 2. 生成唯一 ID 避免 HTML 元素冲突
        unique_id = str(uuid.uuid4())
        container_id = f"cont_{unique_id}"
        img_id = f"img_{unique_id}"
        hint_id = f"hint_{unique_id}"
        
        # 3. 构造 HTML 代码：包含缩放逻辑的 CSS 和 JS
        html_code = f"""
        <style>
            #{container_id} {{
                width: 100%;
                overflow-x: hidden;
                border: 1px solid #e0e0e0;
                padding: 5px;
                cursor: zoom-in;
                transition: all 0.2s ease;
                margin-bottom: 25px; 
            }}
            #{img_id} {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .mars-plotter-hint {{
                color: #888;
                font-size: 12px;
                text-align: left; 
                margin-bottom: 5px; 
                margin-left: 2px;
            }}
        </style>

        <div id="{container_id}" ondblclick="toggleZoom_{unique_id.replace('-', '_')}(this)">
            <img id="{img_id}" src="data:image/png;base64,{img_str}" title="双击图片：放大查看细节 / 缩小查看全貌" />
        </div>

        <script>
        (function() {{
            // 控制提示语仅在第一张图表上方显示
            if (typeof window.MARS_PLOTTER_HINT_SHOWN === 'undefined') {{
                document.getElementById('{hint_id}').style.display = 'block';
                window.MARS_PLOTTER_HINT_SHOWN = true;
            }}
        }})();

        // 双击切换缩放状态
        function toggleZoom_{unique_id.replace('-', '_')}(container) {{
            var img = container.querySelector('img');
            if (img.style.width === '100%' || img.style.width === '') {{
                img.style.width = 'auto';
                img.style.maxWidth = 'none';
                container.style.overflowX = 'auto';
                container.style.cursor = 'zoom-out';
            }} else {{
                img.style.width = '100%';
                img.style.maxWidth = '100%';
                container.style.overflowX = 'hidden';
                container.style.cursor = 'zoom-in';
            }}
        }}
        </script>
        """
        display(HTML(html_code))

    @staticmethod
    def plot_feature_binning_risk_trend(
        df_detail: Union[pd.DataFrame, pl.DataFrame], 
        feature: str, 
        group_col: str = "month",
        target_name: str = "Target",
        dpi: Optional[int] = 150
    ):
        """
        绘制特征分箱风险趋势图 (Feature Binning Risk Trend Plot)。

        该图表集成了特征的：
        - 样本分布 (Counts)
        - 坏率走势 (Bad Rate)
        - 跨期一致性 (RiskCorr)
        - 统计指标 (IV, KS, AUC, PSI)

        Parameters
        ----------
        df_detail : Union[pd.DataFrame, pl.DataFrame]
            评估明细数据表，需包含 'feature', 'bin_index', 'bad_rate', 'count' 等列。
        feature : str
            目标特征名。
        group_col : str, default "month"
            分组维度列名（如月份、客群）。
        target_name : str, default "Target"
            目标变量名称，用于标题显示。
        dpi : int, optional, default 150
            绘图分辨率。
        """
        # 数据类型标准化
        if isinstance(df_detail, pl.DataFrame):
            df_detail = df_detail.to_pandas()
            
        df_feat = df_detail[df_detail["feature"] == feature].copy()
        
        if df_feat.empty:
            print(f"❌ Feature '{feature}' not found.")
            return

        if group_col not in df_feat.columns:
             print(f"❌ Group column '{group_col}' not found.")
             return

        # 1. 提取全局汇总指标 (Total 维度)
        if "Total" in df_feat[group_col].values:
            df_total = df_feat[df_feat[group_col] == "Total"]
        else:
            df_total = df_feat
            
        total_count = df_total['count'].sum() if 'total_count' not in df_total.columns else df_total['total_count'].iloc[0]
        global_iv = df_total['iv_bin'].sum()
        global_ks = df_total['ks_bin'].max()
        global_auc = df_total['auc_bin'].sum()
        if global_auc < 0.5: global_auc = 1 - global_auc # 纠正 AUC 方向
        
        # [逻辑] 计算特征整体趋势 (Trend)：通过分箱序号与坏率的相关系数判断单调性
        df_trend_calc = df_total[df_total['bin_index'] >= 0].sort_values('bin_index')
        trend_str = "n.a."
        
        if len(df_trend_calc) > 1:
            x_arr = df_trend_calc['bin_index'].values
            y_arr = df_trend_calc['bad_rate'].values
            
            # 只有当坏率有波动时才计算相关系数，避免平直线导致的无效计算
            if np.std(y_arr) > 1e-9: 
                corr = np.corrcoef(x_arr, y_arr)[0, 1]
                if corr >= 0.5:
                    trend_str = f"asc({corr:.2f})" # 整体呈上升趋势
                elif corr <= -0.5:
                    trend_str = f"desc({corr:.2f})" # 整体呈下降趋势
                else:
                    trend_str = f"n.a.({corr:.2f})" # 无明显单调性
            else:
                trend_str = "flat" # 坏率是一条平直线
        
        # 计算缺失值占比
        missing_row = df_total[df_total['bin_index'] == -1]
        if not missing_row.empty and total_count > 0:
            miss_count = missing_row['count'].sum()
            miss_rate = miss_count / total_count
            miss_str = f"{miss_rate:.2%}" 
        else:
            miss_str = "nan%" 
        
        # 获取所有时间分组（排除 Total）
        groups = [g for g in df_feat[group_col].unique() if g != "Total"]
        groups = sorted(groups)
        time_range = f"[{groups[0]} ~ {groups[-1]}]" if groups else ""
        
        # [逻辑] 提取 RiskCorr (RC) 基准：使用最早的一个分组作为风险排序的标杆
        if groups:
            first_group = groups[0]
            base_vec = (
                df_feat[df_feat[group_col] == first_group]
                .sort_values("bin_index")
                .query("bin_index >= 0")["bad_rate"].values
            )
        else:
            base_vec = None

        # 2. 画布布局设置
        if "Total" in df_feat[group_col].values:
            all_groups = groups + ["Total"]
        else:
            all_groups = groups
        
        n_panels = len(all_groups)
        if n_panels == 0: return
        
        total_width = MarsPlotter.UNIT_WIDTH * n_panels
        total_height = MarsPlotter.UNIT_HEIGHT + 0.7
        
        fig = plt.figure(figsize=(total_width, total_height))
        
        # 动态计算字体大小，适配不同尺寸的子图
        base_h = 2.5
        fs_title = base_h * 1.8 + 2
        fs_label = base_h * 1.5 + 1.5
        fs_text  = base_h * 1.5 + 1
        
        gs = gridspec.GridSpec(
            1, n_panels, 
            figure=fig,
            wspace=0.09, 
            left=0.05, right=0.95, top=0.75, bottom=0.15 
        )
        
        # 3. 绘制顶部全局摘要信息栏
        summary_str_1 = f"{feature},  {target_name},  Total: {int(total_count)},  {time_range}"
        summary_str_2 = f"IV: {global_iv:.3f},  KS: {global_ks*100:.1f},  AUC: {global_auc:.2f},  Missing: {miss_str},  Trend: {trend_str}"
        
        fig.text(0.04, 0.94, summary_str_1 + "\n" + summary_str_2, 
                 fontsize=fs_title+0.85, va='top', ha='left', linespacing=1.6, 
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#cccccc", alpha=0.8))

        # [优化] 预计算全局最大值：确保所有子图的 Y 轴刻度一致，方便跨期对比
        global_max_count = 0.0
        global_max_bad = 0.0
        
        for group in all_groups:
            _df = df_feat[df_feat[group_col] == group]
            if _df.empty: continue
            _counts = _df["count"] / _df["count"].sum() if "count_dist" not in _df.columns else _df["count_dist"]
            _bads = _df["bad_rate"]
            if len(_counts) > 0: global_max_count = max(global_max_count, _counts.max())
            if len(_bads) > 0: global_max_bad = max(global_max_bad, _bads.max())
        
        # 4. 循环绘制每个分组的指标面板
        to_percent = FuncFormatter(lambda y, _: '{:.0%}'.format(y))

        for i, group in enumerate(all_groups):
            ax = plt.subplot(gs[i])
            
            # [核心计算] RiskCorr: 计算当前分组风险排序与首月相关性，评估模型稳定性
            rc_val = 1.0  
            if base_vec is not None:
                curr_df_g = df_feat[df_feat[group_col] == group].sort_values("bin_index")
                curr_vec = curr_df_g[curr_df_g["bin_index"] >= 0]["bad_rate"].values
                if len(curr_vec) == len(base_vec) and np.std(curr_vec) > 1e-9 and np.std(base_vec) > 1e-9:
                    rc_val = np.corrcoef(curr_vec, base_vec)[0, 1]
                elif group == "Total":
                    rc_val = np.nan 
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.2)
                # spine.set_edgecolor('#CCCCCC')
            
            df_g = df_feat[df_feat[group_col] == group].sort_values("bin_index")
            if df_g.empty: continue
            
            x = range(len(df_g))
            labels = df_g["bin_label"].tolist()
            indices = df_g["bin_index"].tolist()
            counts = df_g["count"] / df_g["count"].sum() if "count_dist" not in df_g.columns else df_g["count_dist"]
            bads = df_g["bad_rate"]
            
            # --- A. 柱状图：展示样本分布 (灰色) ---
            ax.bar(x, counts, color='grey', label='Count Dist', alpha=0.4)
            ax.set_ylim(0, global_max_count * 1.3) 
            
            if i == 0:
                ax.yaxis.set_major_formatter(to_percent)
                ax.tick_params(axis='y', labelsize=fs_label+1.5, colors='grey', length=0)
            else:
                ax.set_yticks([]) # 仅保留最左侧坐标轴
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=fs_label+1.5)
            ax.tick_params(axis='x', length=0)
            
            # --- B. 折线图：展示坏率趋势 (红色) ---
            ax2 = ax.twinx()
            for spine in ax2.spines.values():
                spine.set_linewidth(0.2)       # 保持与 ax 一致的宽度
                # spine.set_edgecolor('#CCCCCC') # 保持与 ax 一致的颜色
            mask_normal = np.array(indices) >= 0
            mask_special = ~mask_normal
            x_arr = np.array(x)
            bads_arr = np.array(bads)
            
            COLOR_RED = "#fc5853"   
            COLOR_BLUE = "#210fe8" 
            COLOR_GREY = '#555555' 
            
            if mask_normal.any():
                ax2.plot(x_arr[mask_normal], bads_arr[mask_normal], color=COLOR_RED, linewidth=1.2, zorder=1)
                ax2.scatter(x_arr[mask_normal], bads_arr[mask_normal], color=COLOR_RED, s=6.5, zorder=2)
            
            # 特殊箱（如缺失值、拒绝、异常值）用蓝色散点标记
            if mask_special.any():
                ax2.scatter(x_arr[mask_special], bads_arr[mask_special], color=COLOR_BLUE, s=6.5, zorder=2)
            
            y_max_limit = global_max_bad * 1.25 if global_max_bad > 0 else 1.0
            ax2.set_ylim(0, y_max_limit)
            
            if i == len(all_groups) - 1:
                ax2.yaxis.set_major_formatter(to_percent)
                ax2.tick_params(axis='y', labelsize=fs_label+1.5, colors="#a23633", length=0)
            else:
                ax2.set_yticks([]) # 仅保留最右侧坐标轴
            
            # --- C. 数据标注 (BadRate & Lift) ---
            for j, val in enumerate(bads):
                is_special = indices[j] < 0
                color_lift_text = COLOR_BLUE if is_special else 'black'
                
                # 标注 Lift 值（位于折线上方）
                if 'lift' in df_g.columns:
                    lift_val = df_g['lift'].iloc[j]
                    offset_up = y_max_limit * 0.02
                    ax2.text(j, val + offset_up, f"{lift_val:.1f}", color=color_lift_text, ha='center', va='bottom', fontweight='bold', fontsize=fs_text+2.6)

                # 标注坏率百分比（位于折线下方）
                offset_down = y_max_limit * 0.03
                ax2.text(j, val - offset_down, f"{val:.1%}", color=COLOR_GREY, ha='center', va='top', fontweight='bold', fontsize=fs_text+0.8)
                
                # 在柱状图内部底部标注样本分布占比
                ct_val = counts.iloc[j]
                ax.text(j, max(counts) * 0.05, f"{ct_val:.1%}", color='#333333', ha='center', va='bottom', fontsize=fs_text+0.5)
            # --- D. 子图顶部指标汇总 ---
            iv_val  = df_g['iv_bin'].sum()
            ks_val  = df_g['ks_bin'].max() * 100
            auc_val = df_g['auc_bin'].sum()
            auc_val = 1 - auc_val if auc_val < 0.5 else auc_val 
            psi_val = df_g['psi_bin'].sum() if 'psi_bin' in df_g.columns else 0.0

            total_bad   = df_g['bad'].sum()
            total_count = df_g['count'].sum() if 'count' in df_g.columns else df_g['total_count'].iloc[0]
            avg_bad_rate = total_bad / total_count if total_count > 0 else 0
            g_miss_row = df_g[df_g['bin_index'] == -1]
            g_miss_str = f"{g_miss_row['count'].sum() / total_count:.0%}" if not g_miss_row.empty and total_count > 0 else "0%"

            # 子图主标题
            ax.set_title(f"{group}   ({int(total_bad)}/{int(total_count)}, {avg_bad_rate:.1%})", fontsize=fs_title+0.85, y=1.05, ha='center')

            # 子图副标题指标栏 (RC, PSI, IV 等)
            rc_str   = f"RC:{rc_val:.2f}" if not np.isnan(rc_val) else "RC:n.a."
            rc_color = 'red' if (not np.isnan(rc_val) and rc_val < 0.7) else '#555555'

            perf_text = f"IV: {iv_val:.2f},  KS: {ks_val:.1f},  AUC: {auc_val:.2f},"
            ax.text(0.602, 1.015, perf_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=fs_title+0.85, color='black')
            ax.text(0.607, 1.015, f"  PSI: {psi_val:.2f},", transform=ax.transAxes, ha='left', va='bottom', fontsize=fs_title+0.85, color='red' if psi_val > 0.1 else 'black')
            ax.text(0.82, 0.945, f" {rc_str}", transform=ax.transAxes, ha='left', va='bottom', fontsize=fs_title+0.25, color=rc_color)
            ax.text(0.837, 1.015, f" Miss:{g_miss_str}", transform=ax.transAxes, ha='left', va='bottom', fontsize=fs_title+0.85, color='#555555')

            # 绘制整体平均坏率参考线
            ax2.axhline(avg_bad_rate, color='grey', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # 在图表左上角标注首箱(L)和尾箱(R)的详情，辅助判断头部和尾部风险集中度
            df_normal = df_g[df_g['bin_index'] >= 0].sort_values('bin_index')
            if not df_normal.empty:
                for suffix, idx in [("L", 0), ("R", -1)]:
                    row = df_normal.iloc[idx]
                    lft, bd = row.get('lift', 0), int(row.get('bad', 0))
                    rt = bd / total_bad if total_bad > 0 else 0
                    text = f"{suffix}: {lft:.2f}, {bd}, {rt:.1%}"
                    ax.text(0.02, 0.987 if suffix=="L" else 0.935, text, transform=ax.transAxes, color=COLOR_BLUE, fontsize=fs_text+1.8, ha='left', va='top')

        MarsPlotter._show_scrollable(fig, dpi=dpi)
        
    @staticmethod
    def plot_feature_binning_risk_trend_batch(
        df_detail: Union[pd.DataFrame, pl.DataFrame], 
        features: List[str], 
        group_col: str = "month",
        target_name: str = "Target",
        dpi: int = 150,
        sort_by: str = "iv", 
        ascending: bool = False
    ):
        """
        批量绘制多个特征的分箱风险趋势图。

        支持按指定指标（IV/KS/AUC）对特征进行排序展示。

        Parameters
        ----------
        df_detail : Union[pd.DataFrame, pl.DataFrame]
            评估明细数据表。
        features : List[str]
            待绘图的特征名称列表。
        group_col : str, default "month"
            分组维度列。
        target_name : str, default "Target"
            目标名。
        dpi : int, default 150
            图像分辨率。
        sort_by : str, default "iv"
            排序依据指标，可选 'iv', 'ks', 'auc'。
        ascending : bool, default False
            是否升序排列（默认降序，即最重要的特征排在最前面）。
        """
        if isinstance(df_detail, pl.DataFrame):
            df_detail = df_detail.to_pandas()

        # 重置交互式容器的显示标记
        display(HTML("<script>window.MARS_PLOTTER_HINT_SHOWN = undefined;</script>"))
            
        # 1. 计算全局排序得分
        if sort_by and sort_by.lower() in ['iv', 'ks', 'auc']:
            logger.info(f"📊 Calculating {sort_by.upper()} for sorting...")
            feature_stats = []
            for feat in features:
                df_feat = df_detail[df_detail["feature"] == feat]
                if df_feat.empty: continue
                df_calc = df_feat[df_feat[group_col] == "Total"] if "Total" in df_feat[group_col].values else df_feat
                
                val = 0
                if sort_by.lower() == 'iv': val = df_calc['iv_bin'].sum()
                elif sort_by.lower() == 'ks': val = df_calc['ks_bin'].max()*100
                elif sort_by.lower() == 'auc': 
                    val = df_calc['auc_bin'].sum()
                    if val < 0.5: val = 1 - val
                feature_stats.append({'feature': feat, 'score': val})
            
            df_stats = pd.DataFrame(feature_stats)
            if not df_stats.empty:
                df_stats = df_stats.sort_values(by='score', ascending=ascending)
                sorted_features = df_stats['feature'].tolist()
            else:
                sorted_features = features
        else:
            sorted_features = features

        logger.info(f"🚀 Starting batch plot for {len(sorted_features)} features...")
        
        # 2. 循环生成每个特征的图表
        for i, feat in enumerate(sorted_features):
            score_info = ""
            if sort_by and 'df_stats' in locals() and not df_stats[df_stats['feature'] == feat].empty:
                score = df_stats[df_stats['feature'] == feat]['score'].iloc[0]
                score_info = f" ({sort_by.upper()}={score:.4f})"
            
            logger.info(f"[{i+1}/{len(sorted_features)}] Plotting {feat}{score_info}...")
            
            MarsPlotter.plot_feature_binning_risk_trend(
                df_detail=df_detail, 
                feature=feat, 
                group_col=group_col, 
                target_name=target_name,
                dpi=dpi
            )
        logger.info("✅ Batch plotting completed.")