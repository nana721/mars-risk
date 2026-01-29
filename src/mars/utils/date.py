import polars as pl
from typing import Union

class MarsDate:
    """
    [MarsDate] 日期处理核心组件 (Pure Polars Edition).

    专为 Polars DataFrame 操作设计。
    所有方法均返回 ``pl.Expr`` 对象，可直接用于 Polars 的表达式系统中。

    Notes
    -----
    该类不直接处理数据，而是构建 Polars 表达式树。
    这意味着它的开销极低，且能完美融入 ``lazy()`` 执行计划中。
    """

    @staticmethod
    def _to_expr(col: Union[str, pl.Expr]) -> pl.Expr:
        """
        [Internal] 将输入归一化为 Polars 表达式。

        Parameters
        ----------
        col : Union[str, pl.Expr]
            如果是字符串，视为列名并转换为 ``pl.col(col)``。
            如果是表达式，原样返回。

        Returns
        -------
        pl.Expr
            Polars 表达式对象。
        """
        if isinstance(col, str):
            return pl.col(col)
        return col

    @staticmethod
    def smart_parse_expr(col: Union[str, pl.Expr]) -> pl.Expr:
        """
        [智能解析] 生成多路尝试的日期解析表达式。

        采用 "Coalesce" (多路合并) 策略，能够自动处理混合格式的脏数据。
        
        优化策略
        --------
        1. **类型优先保护**: 优先尝试直接 Cast。如果输入已经是 Date/Datetime，
           则跳过后续字符串解析，大幅提升处理规整数据时的性能。
        2. **强制转 String**: 对于无法直接 Cast 的类型，转换为 ``pl.Utf8`` 统一处理。
           这解决了整数日期 (如 20250101) 被误读为天数偏移的 bug。
        3. **多格式尝试**: 依次尝试解析常用的 ISO 格式、紧凑格式、斜杠和点号格式。

        Parameters
        ----------
        col : Union[str, pl.Expr]
            待解析的列名或表达式。支持 String, Int (如 20230101), Date, Datetime 类型。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Date`` 的表达式。无法解析的值将变为 Null。
        """
        expr = MarsDate._to_expr(col)
        
        # 预生成 String 表达式用于多格式解析尝试
        str_expr = expr.cast(pl.Utf8)

        # Coalesce: 从上到下尝试，返回第一个非 Null 的结果
        return pl.coalesce([
            # A. [优先] 尝试直接 Cast
            # 如果是原生 Date/Datetime 或标准 "YYYY-MM-DD" 字符串，此步最高效
            expr.cast(pl.Date, strict=False),
            
            # B. 标准 ISO 格式 (2025-01-01) 
            # 强化匹配：部分特殊 Object 转 Str 后可能符合此格式
            str_expr.str.to_date("%Y-%m-%d", strict=False),

            # C. 紧凑格式 (20250101) 
            # 解决 Int 类型转为 Str 后的情况（风控数仓常见格式）
            str_expr.str.to_date("%Y%m%d", strict=False),
            
            # D. 斜杠格式 (2025/01/01)
            str_expr.str.to_date("%Y/%m/%d", strict=False),
            
            # E. 点号格式 (2025.01.01)
            str_expr.str.to_date("%Y.%m.%d", strict=False),
        ])

    @staticmethod
    def dt2day(dt: Union[str, pl.Expr]) -> pl.Expr:
        """
        将日期转换为 'Day' 粒度 (即解析为标准 Date)。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Date`` 的表达式。
        """
        return MarsDate.smart_parse_expr(dt)

    @staticmethod
    def dt2week(dt: Union[str, pl.Expr]) -> pl.Expr:
        """
        将日期转换为 'Week' 粒度 (向下取整到周一)。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Date`` 的表达式，值为该日期所在周的周一。
        """
        return (
            MarsDate.smart_parse_expr(dt)
            .dt.truncate("1w")
            .cast(pl.Date) 
        )

    @staticmethod
    def dt2month(dt: Union[str, pl.Expr]) -> pl.Expr:
        """
        将日期转换为 'Month' 粒度 (向下取整到当月1号)。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Date`` 的表达式，值为该日期所在月的1号。
        """
        return (
            MarsDate.smart_parse_expr(dt)
            .dt.truncate("1mo")
            .cast(pl.Date)
        )
    
    @staticmethod
    def format_dt(dt: Union[str, pl.Expr], fmt: str = "%Y-%m-%d") -> pl.Expr:
        """
        [展示用] 将日期解析并格式化为指定字符串。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        fmt : str, optional
            输出的格式化字符串，默认 "%Y-%m-%d"。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8`` (String) 的表达式。
        """
        return MarsDate.smart_parse_expr(dt).dt.strftime(fmt)