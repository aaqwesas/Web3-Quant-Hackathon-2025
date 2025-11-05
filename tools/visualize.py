import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Simple visualization tool to plot metrics and trades CSV files.
Usage: python tools/visualize.py
It looks for `data/metrics.csv` and `data/trades.csv` in the package directory.
"""

ROOT = os.path.dirname(os.path.dirname(__file__))
# Prefer package data dir if present (roostoo_bot_template/data), else project-root/data
pkg_data = os.path.join(ROOT, 'roostoo_bot_template', 'data')
proj_data = os.path.join(ROOT, 'data')
if os.path.isdir(pkg_data):
    DATA_DIR = pkg_data
else:
    DATA_DIR = proj_data

PKG_DIR = os.path.join(ROOT, 'roostoo_bot_template') if os.path.isdir(os.path.join(ROOT, 'roostoo_bot_template')) else ROOT
METRICS_CSV = os.path.join(DATA_DIR, 'metrics.csv')
TRADES_CSV = os.path.join(DATA_DIR, 'trades.csv')


def read_csv_if_exists(path, cols=None):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    if cols:
        for c in cols:
            if c not in df.columns:
                print(f"Warning: {c} not in {path}")
    return df


def plot_metrics(metrics_df):
    # plot equity curve with drawdown shading and optional trade overlays
    ts = pd.to_datetime(metrics_df['ts']) if 'ts' in metrics_df.columns else pd.RangeIndex(len(metrics_df))
    eq = metrics_df['equity'].astype(float)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, eq, label='Equity', color='tab:blue')
    # drawdown shading
    running_max = eq.cummax()
    ax.fill_between(ts, eq, running_max, where=(eq < running_max), color='red', alpha=0.2, label='Drawdown')

    # overlay trades if provided
    def _overlay_trades(trades_df):
        if trades_df is None:
            return
        trades_df['ts'] = pd.to_datetime(trades_df['ts'])
        buys = trades_df[trades_df['side'] == 'BUY']
        sells = trades_df[trades_df['side'] == 'SELL']
        if not buys.empty:
            ax.scatter(buys['ts'], buys['price'], marker='^', color='g', s=50, zorder=5, label='BUY')
            for _, r in buys.iterrows():
                ax.annotate(f"{r.get('qty','')}", (r['ts'], r['price']), textcoords="offset points", xytext=(0,8), ha='center', color='g', fontsize=8)
        if not sells.empty:
            ax.scatter(sells['ts'], sells['price'], marker='v', color='r', s=50, zorder=5, label='SELL')
            for _, r in sells.iterrows():
                ax.annotate(f"{r.get('qty','')}", (r['ts'], r['price']), textcoords="offset points", xytext=(0,-12), ha='center', color='r', fontsize=8)

    # If caller passed a global trades_df into module scope, use it; main will pass it explicitly
    try:
        trades_df_global = globals().get('TRADES_DF', None)
    except Exception:
        trades_df_global = None
    _overlay_trades(trades_df_global)

    ax.set_title('Equity Curve with Drawdown and Trades')
    ax.set_xlabel('Time')
    ax.set_ylabel('Equity')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    out = os.path.join(PKG_DIR, 'equity_curve.png')
    fig.savefig(out)
    print('Saved', out)
    plt.close(fig)


def plot_metrics_with_trades(metrics_csv, trades_csv):
    metrics_df = pd.read_csv(metrics_csv)
    ts = pd.to_datetime(metrics_df['ts']) if 'ts' in metrics_df.columns else pd.RangeIndex(len(metrics_df))
    eq = metrics_df['equity'].astype(float) if 'equity' in metrics_df.columns else metrics_df['equity'].astype(float)
    plt.figure(figsize=(12, 6))
    plt.plot(ts, eq, label='Equity', color='tab:blue')
    running_max = eq.cummax()
    plt.fill_between(ts, eq, running_max, where=(eq < running_max), color='red', alpha=0.2, label='Drawdown')

    # overlay trades
    if os.path.exists(trades_csv):
        td = pd.read_csv(trades_csv)
        if 'ts' in td.columns:
            td['ts'] = pd.to_datetime(td['ts'])
        for _, r in td.iterrows():
            mark_color = 'g' if r['side'] == 'BUY' else 'r'
            # place marker at nearest equity timestamp (approx)
            if 'ts' in td.columns and 'ts' in metrics_df.columns:
                # find nearest index
                idx = metrics_df['ts'].astype('datetime64[ns]').sub(pd.to_datetime(r['ts'])).abs().idxmin()
                x = pd.to_datetime(metrics_df['ts'].iloc[idx])
                y = eq.iloc[idx]
                plt.scatter([x], [y], color=mark_color, marker='^' if r['side']=='BUY' else 'v', s=80)
            else:
                # fallback: plot at end
                plt.scatter([ts.iloc[-1]], [eq.iloc[-1]], color=mark_color, marker='^' if r['side']=='BUY' else 'v')

    plt.title('Equity Curve with Trades')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(PKG_DIR, 'equity_with_trades.png')
    plt.savefig(out)
    print('Saved', out)
    plt.close()


def plot_trades(trades_df):
    # trades: ts,symbol,side,qty,price,pnl
    trades_df['ts'] = pd.to_datetime(trades_df['ts'])
    fig, ax = plt.subplots(figsize=(12,6))
    for sym in trades_df['symbol'].unique():
        s = trades_df[trades_df['symbol'] == sym]
        buys = s[s['side'] == 'BUY']
        sells = s[s['side'] == 'SELL']
        if not buys.empty:
            ax.scatter(buys['ts'], buys['price'], marker='^', color='g', label=f'{sym} BUY')
            for _, r in buys.iterrows():
                ax.annotate(f"{r['qty']}", (r['ts'], r['price']), textcoords="offset points", xytext=(0,8), ha='center', color='g')
        if not sells.empty:
            ax.scatter(sells['ts'], sells['price'], marker='v', color='r', label=f'{sym} SELL')
            for _, r in sells.iterrows():
                ax.annotate(f"{r['qty']}", (r['ts'], r['price']), textcoords="offset points", xytext=(0,-12), ha='center', color='r')

    ax.set_title('Trades (markers)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    out = os.path.join(PKG_DIR, 'trades_markers.png')
    plt.savefig(out)
    print('Saved', out)
    plt.close()


def main():
    metrics_df = read_csv_if_exists(METRICS_CSV)
    trades_df = read_csv_if_exists(TRADES_CSV)
    if metrics_df is not None and 'equity' in metrics_df.columns:
        plot_metrics(metrics_df)
    if trades_df is not None and 'price' in trades_df.columns:
        plot_trades(trades_df)

if __name__ == '__main__':
    main()
