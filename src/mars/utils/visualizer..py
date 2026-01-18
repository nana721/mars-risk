import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
from typing import Optional, List, Tuple

class MarsVisualizer:
    """
    [Mars 可视化引擎]
    专注于生成复杂的风控图表。
    """

    @staticmethod
    def plot_binning_grid(
        detail_df: pd.DataFrame, 
        feature_col: str = "feature",
        group_col: str = "_grp_month", # 对应 Monitor 中的 group_col
        bin_col: str = "bin_label",
        features: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 5) # 单行高度基础值
    ):
        """
        绘制分箱详情网格图 (Grid Layout)。
        Rows: 特征 | Cols: 时间/分组
        
        Parameters
        ----------
        detail_df : pd.DataFrame
            由 MarsFeatureMonitor 计算出的 L2 detail_table。
        features : List[str], optional
            指定要画的特征列表。如果不传则画所有。
        groups : List[str], optional
            指定要画的时间组。如果不传则画所有。
        """
        # 1. 过滤数据
        if features:
            detail_df = detail_df[detail_df[feature_col].isin(features)]
        if groups:
            detail_df = detail_df[detail_df[group_col].isin(groups)]
            
        uniq_feats = detail_df[feature_col].unique()
        uniq_grps = detail_df[group_col].unique()
        
        n_rows = len(uniq_feats)
        n_cols = len(uniq_grps)
        
        if n_rows == 0 or n_cols == 0:
            print("No data to plot.")
            return

        # 动态调整高度
        total_height = figsize[1] * n_rows
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(figsize[0], total_height), 
            constrained_layout=True
        )
        
        # 降维处理：统一转为二维数组
        if n_rows == 1 and n_cols == 1: 
            axes = np.array([[axes]])
        elif n_rows == 1: 
            axes = axes.reshape(1, -1)
        elif n_cols == 1: 
            axes = axes.reshape(-1, 1)

        # 循环绘图
        for i, feat in enumerate(uniq_feats):
            for j, grp in enumerate(uniq_grps):
                ax = axes[i, j]
                
                # 数据切片
                data = detail_df[
                    (detail_df[feature_col] == feat) & 
                    (detail_df[group_col] == grp)
                ].copy()
                
                if data.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # 按 bin_idx 排序确保顺序正确
                if 'bin_idx' in data.columns:
                    data = data.sort_values('bin_idx')
                
                # --- 核心绘图逻辑 ---
                x = np.arange(len(data))
                
                # 1. 柱状图 (Count %) - 左轴
                #    使用浅灰色，避免抢夺视觉重点
                bars = ax.bar(x, data['pct_actual'], color='#e0e0e0', label='Count %', width=0.6)
                
                # 设置左轴
                y_max = data['pct_actual'].max()
                ax.set_ylim(0, y_max * 1.4) # 留出顶部空间给 L/R 文字
                ax.set_ylabel("Count %", fontsize=8, color='gray')
                ax.tick_params(axis='y', colors='gray')
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
                
                # 隐藏左轴的上边框和右边框，更极简
                ax.spines['top'].set_visible(False)
                
                # 柱内标注 (占比)
                for bar, pct in zip(bars, data['pct_actual']):
                    if pct > 0.01: # 太小的就不标了
                        ax.text(
                            bar.get_x() + bar.get_width()/2., 
                            bar.get_height() * 0.5, # 居中
                            f'{pct:.0%}', 
                            ha='center', va='center', fontsize=9, color='black'
                        )

                # 2. 折线图 (Bad Rate) - 右轴
                #    使用醒目的红色
                ax2 = ax.twinx()
                ax2.plot(x, data['bin_bad_rate'], color='#d62728', marker='o', linewidth=2, markersize=5, label='Bad Rate')
                
                # 设置右轴
                br_max = data['bin_bad_rate'].max()
                ax2.set_ylim(0, br_max * 1.25)
                ax2.set_ylabel("Bad Rate", color='#d62728', fontsize=9)
                ax2.tick_params(axis='y', labelcolor='#d62728')
                ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
                ax2.spines['top'].set_visible(False)

                # 折线点上方标注
                for _x, _y in zip(x, data['bin_bad_rate']):
                    ax2.text(_x, _y * 1.05, f'{_y:.1%}', ha='center', va='bottom', fontsize=9, color='#d62728', fontweight='bold')

                # 3. 辅助线 (Average Bad Rate)
                #    从 Summary 中取，或者简单计算 (bad sum / total sum)
                avg_br = data['bin_bad_rate'].mean() # 简化，实际可用加权
                ax2.axhline(avg_br, color='gray', linestyle='--', linewidth=1, alpha=0.5)

                # 4. 特殊指标标注 (Lift / PSI Contrib 等)
                #    模仿截图中的蓝色小字
                for _x, row in enumerate(data.itertuples()):
                    lift = getattr(row, 'lift', 0)
                    psi_i = getattr(row, 'psi_i', 0) # 也可以标 PSI贡献
                    
                    # 放在柱子上方，但在折线下方或附近
                    # 这里的 y 位置是左轴坐标系
                    text_y = bar.get_height() + (y_max * 0.05)
                    # 标注 Lift
                    ax.text(_x, ax.get_ylim()[1] * 0.9, f"L:{lift:.1f}", ha='center', va='top', fontsize=8, color='#1f77b4', fontweight='bold')

                # 5. X 轴标签
                ax.set_xticks(x)
                ax.set_xticklabels(data[bin_col], rotation=45, ha='right', fontsize=8)

                # 6. Title (KPIs)
                #    从第一行取 Summary 指标
                iv = data['IV'].iloc[0] if 'IV' in data.columns else 0
                ks = data['KS'].iloc[0] if 'KS' in data.columns else 0
                psi = data['PSI'].iloc[0] if 'PSI' in data.columns else 0
                
                title_str = f"[{grp}]  IV:{iv:.3f}  KS:{ks:.3f}  PSI:{psi:.3f}"
                
                # 根据 PSI 严重程度改变 Title 颜色
                t_color = 'black'
                if psi > 0.25: t_color = 'red'
                elif psi > 0.1: t_color = '#ff7f0e' # Orange
                
                ax.set_title(title_str, fontsize=11, loc='left', color=t_color, fontweight='bold')
                
                # 第一列显示 Feature Name
                if j == 0:
                    ax.set_ylabel(f"{feat}\n(Count %)", fontsize=10, fontweight='bold', color='gray')

        plt.show()