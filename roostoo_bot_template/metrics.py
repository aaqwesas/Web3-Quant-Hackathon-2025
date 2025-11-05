import pandas as pd
import numpy as np

def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = equity_curve / (roll_max + 1e-12) - 1.0
    return float(dd.min())  # negative number

def sharpe(returns: pd.Series) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=0)
    if sigma == 0: return 0.0
    return float(mu / sigma)

def sortino(returns: pd.Series) -> float:
    mu = returns.mean()
    neg = returns[returns < 0]
    sigma_d = neg.std(ddof=0)
    if sigma_d == 0: return 0.0
    return float(mu / sigma_d)

def calmar(returns: pd.Series, equity_curve: pd.Series) -> float:
    mu = returns.mean()
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0: return 0.0
    return float(mu / mdd)

def composite_score(returns: pd.Series, equity_curve: pd.Series) -> dict:
    s = sharpe(returns)
    so = sortino(returns)
    mdd = max_drawdown(equity_curve)
    ca = calmar(returns, equity_curve)
    score = 0.4*so + 0.3*s + 0.3*ca
    return {"sharpe": s, "sortino": so, "max_drawdown": mdd, "calmar": ca, "score": score}
