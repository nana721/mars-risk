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
        1. **强制转 String**: 优先将输入转换为 ``pl.Utf8``。这解决了整数日期 (如 20250101) 
           直接转 Date 时被 Polars 误读为 "Unix Timestamp (天数)" 从而变成 5万年以后的 bug。
        2. **多格式尝试**: 依次尝试解析 '%Y%m%d', '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d'。
        3. **兜底机制**: 最后保留原始 Cast 逻辑，以兼容已经是 Date 类型的数据。

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
        
        # 1. 强制转为 String 以统一处理 
        #    (解决了 Int 20250101 被误读为 "5万年以后" 的 Bug)
        str_expr = expr.cast(pl.Utf8)

        # 2. Coalesce: 从上到下尝试，返回第一个非 Null 的结果
        return pl.coalesce([
            # A. 紧凑格式 (20250101) - 包含 Int 类型转为 Str 后的情况
            str_expr.str.to_date("%Y%m%d", strict=False),
            
            # B. 标准 ISO 格式 (2025-01-01) - 包含原生的 Date/Datetime 转为 Str 后的情况
            str_expr.str.to_date("%Y-%m-%d", strict=False),
            
            # C. 斜杠格式 (2025/01/01)
            str_expr.str.to_date("%Y/%m/%d", strict=False),
            
            # D. 点号格式 (2025.01.01)
            str_expr.str.to_date("%Y.%m.%d", strict=False),
            
            # E. [兜底] 尝试直接 Cast
            #    如果输入本身已经是 pl.Date 或 pl.Datetime，上面的转 Str 解析也能成功，
            #    但为了保险起见（或者处理某些带时区的特殊 Datetime），保留这个作为最后手段。
            #    注意：Int 类型在步骤 A 就会命中返回，不会走到这一步，所以安全了。
            expr.cast(pl.Date, strict=False),
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