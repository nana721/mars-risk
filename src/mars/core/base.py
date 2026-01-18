from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Literal

import polars as pl
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mars.core.exceptions import NotFittedError, DataTypeError
from mars.utils.decorators import time_it
import warnings


class MarsBaseEstimator(BaseEstimator):
    """
    [MARS 基类] 负责输入数据的类型检测和输出数据的格式化。
    
    集成 Scikit-learn 的 BaseEstimator，支持 set_output API，
    允许用户在管道中灵活控制输出格式（Pandas 或 Polars）。
    """
    
    # Polars 的数值类型集合 (类属性，避免重复创建)
    _PL_NUMERIC_TYPES = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
        pl.Float32, pl.Float64
    }

    def __init__(self) -> None:
        # 内部标志位：是否返回 Pandas 格式
        # 默认 False (返回 Polars)，但在 _ensure_polars 中会根据输入自动调整
        self._return_pandas: bool = False
        
        # 用户配置：记录 set_output 的设定 ('default', 'pandas', 'polars')
        # 优先级高于输入类型的自动推断
        self._output_config: str = "default"

    def set_output(self, transform: Literal["default", "pandas", "polars"] = "default") -> "MarsBaseEstimator":
        """
        兼容 Sklearn 的 set_output API，允许用户强制指定输出格式。

        Parameters
        ----------
        transform : Literal["default", "pandas", "polars"]
            - "pandas": 强制输出 Pandas DataFrame。
            - "polars": 强制输出 Polars DataFrame。
            - "default": 保持默认行为 (通常跟随输入类型)。

        Returns
        -------
        MarsBaseEstimator
            返回实例本身以支持链式调用。
        """
        if transform not in ["default", "pandas", "polars"]:
            raise ValueError(f"Unknown output format: {transform}")
            
        self._output_config = transform
        
        # 立即应用配置 (虽然主要逻辑在 _ensure_polars 中，但这里设置好状态是个好习惯)
        if transform == "pandas":
            self._return_pandas = True
        elif transform == "polars":
            self._return_pandas = False
            
        return self

    def _determine_output_format(self, input_is_pandas: bool) -> None:
        """
        [决策逻辑] 根据用户配置(_output_config)和输入类型，决定最终输出格式。
        
        决策优先级: set_output > 输入类型
        """
        if self._output_config == "pandas":
            self._return_pandas = True
        elif self._output_config == "polars":
            self._return_pandas = False
        else: # default
            self._return_pandas = input_is_pandas

    def _ensure_polars(self, X: Any) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        [类型守卫] 确保输入数据转换为 Polars DataFrame/LazyFrame，并执行严格校验。
        同时根据输入类型设置默认的输出格式。
        """
        # Case 1: 已经是 Polars (Eager or Lazy)
        if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            self._determine_output_format(input_is_pandas=False)
            return X

        # Case 2: 是 Pandas (重点修改区域)
        elif isinstance(X, pd.DataFrame):
            # 1. 决定输出格式 (关键修复点：使用 _determine_output_format 替代直接赋值)
            self._determine_output_format(input_is_pandas=True)
            
            # 2. 执行转换 (尽可能 Zero-Copy)
            try:
                X_pl = pl.from_pandas(X)
            except Exception as e:
                raise DataTypeError(f"Failed to convert Pandas DataFrame to Polars: {e}")
            
            # 3. 🛡️【新增】执行转换后的类型一致性检查
            self._validate_conversion(X, X_pl)
            
            return X_pl
            
        elif isinstance(X, (pd.Series, pl.Series)):
            raise DataTypeError(f"Input must be a generic DataFrame (2D), got Series (1D): {type(X)}")
        else:
            raise DataTypeError(f"Mars expects Polars/Pandas DataFrame, got {type(X)}")

    def _validate_conversion(self, df_pd: pd.DataFrame, df_pl: pl.DataFrame):
        """
        [安全检查] 对比 Pandas 和 Polars 的 Schema，防止数值类型意外崩坏为字符串。
        """
        for col in df_pd.columns:
            pd_dtype = df_pd[col].dtype
            pl_dtype = df_pl[col].dtype
            
            # ------------------------------------------------------------------
            # 检查 1: 严格拦截 (Pandas 明确是数值 -> Polars 变成了非数值)
            # ------------------------------------------------------------------
            is_pd_numeric = pd.api.types.is_numeric_dtype(pd_dtype)
            is_pl_numeric = pl_dtype in self._PL_NUMERIC_TYPES
            
            if is_pd_numeric and not is_pl_numeric:
                # 允许例外: Pandas Int -> Polars Null (全空列可能发生)
                if pl_dtype == pl.Null:
                    continue
                    
                raise DataTypeError(
                    f"❌ Critical Type Mismatch for column '{col}'! \n"
                    f"   Pandas (Numeric): {pd_dtype} \n"
                    f"   Polars (Non-Numeric): {pl_dtype}\n"
                    "   This usually implies data corruption during conversion (e.g. overflow or encoding issues)."
                )

            # ------------------------------------------------------------------
            # 检查 2: 脏数据陷阱预警 (Pandas Object -> Polars Utf8)
            # ------------------------------------------------------------------
            if pd_dtype == "object" and pl_dtype == pl.Utf8:
                # 策略: 取前 10 个非空值进行嗅探
                # 这是一个极低开销的操作 (Zero-Copy Slice)
                sample_series = df_pl[col].drop_nulls().head(10)
                
                if sample_series.len() == 0:
                    continue
                
                # 获取样本数据
                samples = sample_series.to_list()
                
                # 启发式检查: 尝试看样本是否都能转为 float
                # 如果样本里全是数字字符串 (如 "1.5", "20", "NaN")，说明这很可能是被脏数据污染的数值列
                looks_like_numeric = True
                try:
                    for s in samples:
                        # 尝试转换，如果含有 "unknown" 等非数字字符，float() 会抛出 ValueError
                        float(s)
                except ValueError:
                    looks_like_numeric = False
                
                if looks_like_numeric:
                    warnings.warn(
                        f"\n⚠️  [Potential Dirty Data] Column '{col}' looks numeric but is treated as String.\n"
                        f"   - Input (Pandas): object (mixed types)\n"
                        f"   - Output (Polars): Utf8\n"
                        f"   - Sample Values: {samples[:5]}...\n"
                        f"   -> Risk: This column will be handled as Categorical. If it contains dirty strings "
                        f"(e.g. 'null', 'unknown'), please clean them upstream or add them to 'missing_values'.",
                        UserWarning,
                        stacklevel=2
                    )

    def _format_output(self, data: Any) -> Any:
        """
        [输出格式化] 根据 _return_pandas 标志位，决定是否将结果转回 Pandas。

        支持递归处理字典和列表结构。

        Parameters
        ----------
        data : Any
            待格式化的数据 (DataFrame, Dict, List 等)。

        Returns
        -------
        Any
            格式化后的数据。
        """
        # 如果不需要转 Pandas，或者数据本来就是 Polars，直接返回
        if not self._return_pandas:
            return data

        # 递归处理字典 (常见于 stats_reports)
        if isinstance(data, dict):
            return {k: self._format_output(v) for k, v in data.items()}
        
        # 递归处理列表
        if isinstance(data, list):
            return [self._format_output(v) for v in data]

        # 核心转换逻辑：Polars -> Pandas
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
            
        return data


class MarsTransformer(MarsBaseEstimator, TransformerMixin, ABC):
    """
    [转换器基类]
    集成了自动 Pandas 互操作性。
    """

    def __init__(self):
        super().__init__() # 初始化 _return_pandas
        self.feature_names_in_: List[str] = []
        self._is_fitted: bool = False

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_in_

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> "MarsTransformer":
        # 嗅探输入类型 + 转 Polars
        X_pl = self._ensure_polars(X)
        
        # 执行核心逻辑
        self._fit_impl(X_pl, y, **kwargs)
        
        # 更新状态
        self.feature_names_in_ = X_pl.columns
        self._is_fitted = True
        return self

    def transform(self, X: Any, **kwargs) -> Any:
        """
        模板方法：Transform。
        
        支持通过 **kwargs 向 _transform_impl 传递额外参数 (如 return_type)。
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")
        
        # 1. 类型转换 (Pandas -> Polars)
        X_pl = self._ensure_polars(X)
        
        # 2. 核心逻辑 (支持参数透传)
        X_new = self._transform_impl(X_pl, **kwargs)
        
        # 3. 输出格式化 (Polars -> Pandas/List/Dict)
        return self._format_output(X_new)

    @abstractmethod
    def _fit_impl(self, X: pl.DataFrame, y=None, **kwargs): 
        """
        [Abstract Core] 子类必须实现的核心拟合逻辑。
        """
        pass

    @abstractmethod
    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame: 
        """
        [Abstract Core] 子类必须实现的核心转换逻辑。
        必须返回 Polars DataFrame。
        """
        pass
