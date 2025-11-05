import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, periods: int = 252):
    if returns is None or len(returns) == 0:
        return 0.0
    sr = (returns.mean() / (returns.std(ddof=1) + 1e-9)) * np.sqrt(periods)
    return float(sr)


def max_drawdown(equity: pd.Series):
    if equity is None or len(equity) == 0:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def sortino(returns: pd.Series, periods: int = 252):
    if returns is None or len(returns) == 0:
        return 0.0
    neg = returns[returns < 0]
    downside = (neg.std(ddof=1) + 1e-9) * np.sqrt(periods)
    ann_ret = returns.mean() * periods
    return float((ann_ret) / downside) if downside > 0 else 0.0


def calmar(returns: pd.Series, equity: pd.Series):
    md = abs(max_drawdown(equity))
    if md == 0:
        return 0.0
    ann_ret = returns.mean() * 252
    return float(ann_ret / md)


def composite_score(returns: pd.Series, equity: pd.Series):
    s = sharpe(returns)
    so = sortino(returns)
    cm = calmar(returns, equity)
    md = max_drawdown(equity)
    return {"sharpe": s, "sortino": so, "calmar": cm, "max_drawdown": md, "score": (s + so + cm) / (1 + abs(md))}
