import os
import json
from datetime import datetime
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PKG_DIR, 'data')


def load_backtest_files():
    # Try a few likely data locations: repo/data or repo/roostoo_bot_template/data
    candidates = []
    repo_root = os.path.dirname(os.path.dirname(__file__))
    candidates.append(os.path.join(repo_root, 'data'))
    candidates.append(os.path.join(repo_root, 'roostoo_bot_template', 'data'))
    candidates.append(os.path.join(repo_root, 'roostoo_bot_template', 'roostoo_bot_template', 'data'))
    found = None
    for c in candidates:
        if os.path.exists(os.path.join(c, 'backtest_trades.csv')) and os.path.exists(os.path.join(c, 'backtest_metrics.csv')):
            found = c
            break
    if found is None:
        raise FileNotFoundError('Could not find backtest_trades.csv/backtest_metrics.csv in expected data dirs: ' + ','.join(candidates))
    trades_fp = os.path.join(found, 'backtest_trades.csv')
    metrics_fp = os.path.join(found, 'backtest_metrics.csv')
    trades = pd.read_csv(trades_fp)
    metrics = pd.read_csv(metrics_fp)
    # normalize DATA_DIR to the actual location found
    global DATA_DIR
    DATA_DIR = found
    return trades, metrics


def pair_trades(trades: pd.DataFrame):
    # Produce per-trade rows by pairing BUY -> SELL for long-only strategy.
    # Works per-symbol if `symbol` column exists, otherwise on the whole sequence.
    out_rows = []
    trades = trades.sort_values(by='ts').reset_index(drop=True)
    trades['ts'] = pd.to_datetime(trades['ts'])

    group_keys = ['symbol'] if 'symbol' in trades.columns else [None]
    if group_keys == [None]:
        groups = [(None, trades)]
    else:
        groups = list(trades.groupby('symbol'))

    for g_key, g_df in groups:
        open_pos = None
        for _, r in g_df.iterrows():
            side = str(r.get('side', '')).upper()
            price = float(r.get('price', 0.0))
            qty = float(r.get('qty', 0.0))
            fee = float(r.get('fee', 0.0)) if 'fee' in r.index else 0.0
            slippage = float(r.get('slippage', 0.0)) if 'slippage' in r.index else 0.0
            ts = r['ts']

            if side == 'BUY' and open_pos is None:
                open_pos = dict(entry_ts=ts, entry_price=price, entry_qty=qty, entry_fee=fee, entry_slippage=slippage, symbol=g_key)
            elif side == 'SELL' and open_pos is not None:
                # close
                exit_price = price
                exit_ts = ts
                exit_fee = fee
                exit_slippage = slippage
                qty_used = min(open_pos['entry_qty'], qty) if qty > 0 else open_pos['entry_qty']
                gross = (exit_price - open_pos['entry_price']) * qty_used
                net = gross - (open_pos.get('entry_fee', 0.0) + exit_fee) - (open_pos.get('entry_slippage', 0.0) + exit_slippage)
                hold_time = (exit_ts - open_pos['entry_ts']).total_seconds()
                # compute net pnl pct relative to initial notional (entry_price * qty)
                denom = open_pos['entry_price'] * qty_used if open_pos['entry_price'] * qty_used != 0 else 1.0
                net_pnl_pct = (net / denom) * 100.0

                out_rows.append({
                    'symbol': g_key if g_key is not None else '',
                    'entry_ts': open_pos['entry_ts'].isoformat(),
                    'exit_ts': exit_ts.isoformat(),
                    'entry_price': open_pos['entry_price'],
                    'exit_price': exit_price,
                    'qty': qty_used,
                    'gross_pnl': gross,
                    'net_pnl': net,
                    'net_pnl_pct': net_pnl_pct,
                    'entry_fee': open_pos.get('entry_fee', 0.0),
                    'exit_fee': exit_fee,
                    'entry_slippage': open_pos.get('entry_slippage', 0.0),
                    'exit_slippage': exit_slippage,
                    'hold_time_s': hold_time,
                })
                open_pos = None
            else:
                # ignore other cases (consecutive BUYs or SELLs when no open).
                continue
    return pd.DataFrame(out_rows)


def summarize(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {}
    wins = trades_df[trades_df['net_pnl'] > 0]
    losses = trades_df[trades_df['net_pnl'] <= 0]
    total = len(trades_df)
    win_rate = len(wins) / total
    avg_win = wins['net_pnl'].mean() if not wins.empty else 0.0
    avg_loss = losses['net_pnl'].mean() if not losses.empty else 0.0
    avg_pnl = trades_df['net_pnl'].mean()
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    total_pnl = trades_df['net_pnl'].sum()
    avg_hold_s = trades_df['hold_time_s'].mean()
    return {
        'total_trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_pnl': avg_pnl,
        'expectancy': expectancy,
        'total_net_pnl': total_pnl,
        'avg_hold_seconds': avg_hold_s,
    }


def make_interactive_html(metrics_df: pd.DataFrame, trades_df: pd.DataFrame, out_html_path: str, paired_trades: pd.DataFrame = None):
    # Build JSON arrays for Plotly and write an HTML that uses Plotly CDN
    metrics_df = metrics_df.copy()
    if 'ts' in metrics_df.columns:
        x = [str(t) for t in pd.to_datetime(metrics_df['ts']).tolist()]
    else:
        x = list(range(len(metrics_df)))
    y = list(metrics_df['equity'].astype(float))

    buys = []
    sells = []
    # prefer paired trades (contains entry/exit and net_pnl_pct); fallback to raw trades_df
    if paired_trades is not None and not paired_trades.empty:
        p = paired_trades.copy()
        # ensure datetimes
        p['entry_ts'] = pd.to_datetime(p['entry_ts'])
        p['exit_ts'] = pd.to_datetime(p['exit_ts'])
        # index metrics by ts for mapping equity to trade times
        metrics_idx = None
        if 'ts' in metrics_df.columns and 'equity' in metrics_df.columns:
            m = metrics_df.copy()
            m['ts'] = pd.to_datetime(m['ts'])
            metrics_idx = m.set_index('ts')['equity']

        for _, r in p.iterrows():
            entry_ts = r['entry_ts']
            exit_ts = r['exit_ts']
            entry_price = float(r.get('entry_price', 0.0))
            exit_price = float(r.get('exit_price', 0.0))
            qty = r.get('qty', '')
            net_pnl = float(r.get('net_pnl', 0.0))
            net_pnl_pct = float(r.get('net_pnl_pct', 0.0)) if 'net_pnl_pct' in r.index else None
            hold_s = float(r.get('hold_time_s', 0.0))

            # map to equity y-values if possible, otherwise fall back to price
            if metrics_idx is not None:
                entry_y = float(metrics_idx.asof(entry_ts)) if not metrics_idx.asof(entry_ts) is None else entry_price
                exit_y = float(metrics_idx.asof(exit_ts)) if not metrics_idx.asof(exit_ts) is None else exit_price
            else:
                entry_y = entry_price
                exit_y = exit_price

            buys.append({'x': str(entry_ts), 'y': entry_y, 'text': f"BUY qty={qty}, price={entry_price}, time={entry_ts}, net_pnl_pct={'{:.2f}%'.format(net_pnl_pct) if net_pnl_pct is not None else 'n/a'}"})
            sells.append({'x': str(exit_ts), 'y': exit_y, 'text': f"SELL qty={qty}, price={exit_price}, time={exit_ts}, net_pnl={net_pnl:.2f}, net_pnl_pct={'{:.2f}%'.format(net_pnl_pct) if net_pnl_pct is not None else 'n/a'}, hold={int(hold_s)}s"})
    elif trades_df is not None and not trades_df.empty:
        tdf = trades_df.copy()
        if 'ts' in tdf.columns:
            tdf['ts'] = pd.to_datetime(tdf['ts'])
        for _, r in tdf.iterrows():
            side = str(r.get('side','')).upper()
            price = float(r.get('price', r.get('exit_price', 0.0)))
            ts = str(r.get('ts'))
            txt = f"{side} {r.get('qty','')}, price={price}"
            if side == 'BUY':
                buys.append({'x': ts, 'y': price, 'text': txt})
            elif side == 'SELL':
                sells.append({'x': ts, 'y': price, 'text': txt})

    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <title>Backtest interactive</title>
  </head>
  <body>
    <div id="plot" style="width:100%;height:80vh;"></div>
    <script>
      const x = {json.dumps(x)};
      const y = {json.dumps(y)};
      const buys = {json.dumps(buys)};
      const sells = {json.dumps(sells)};

      const eq = {{ x: x, y: y, mode: 'lines', name: 'Equity' }};
      const buyTrace = {{ x: buys.map(d=>d.x), y: buys.map(d=>d.y), mode: 'markers', marker: {{color:'green',symbol:'triangle-up',size:10}}, name: 'BUY', text: buys.map(d=>d.text), hoverinfo:'text' }};
      const sellTrace = {{ x: sells.map(d=>d.x), y: sells.map(d=>d.y), mode: 'markers', marker: {{color:'red',symbol:'triangle-down',size:10}}, name: 'SELL', text: sells.map(d=>d.text), hoverinfo:'text' }};

      const data = [eq, buyTrace, sellTrace];
      const layout = {{ title: 'Equity curve with trades', xaxis: {{title: 'Time'}}, yaxis: {{title: 'Equity'}} }};
      Plotly.newPlot('plot', data, layout, {{responsive:true}});
    </script>
  </body>
</html>
"""
    with open(out_html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    trades, metrics = load_backtest_files()
    # Attempt to pair trades into per-trade summary. If trade rows are already paired (have entry/exit), handle gracefully.
    try:
        paired = pair_trades(trades)
    except Exception:
        paired = pd.DataFrame()

    # If pairing failed but trades already look paired (have entry_ts/exit_ts), try that
    if paired.empty and {'entry_ts', 'exit_ts', 'net_pnl'}.issubset(trades.columns):
        paired = trades[['symbol','entry_ts','exit_ts','entry_price','exit_price','qty','net_pnl']].copy()

    # Ensure net_pnl_pct exists in paired
    if 'net_pnl_pct' not in paired.columns and not paired.empty:
        paired['net_pnl_pct'] = paired.apply(lambda r: (r['net_pnl'] / (r['entry_price'] * r['qty']) * 100.0) if (r['entry_price'] * r['qty']) != 0 else 0.0, axis=1)

    report_fp = os.path.join(DATA_DIR, 'backtest_report.csv')
    paired.to_csv(report_fp, index=False)
    print('Saved per-trade report to', report_fp)

    summary = summarize(paired)
    # append summary into a small JSON alongside
    meta_fp = os.path.join(DATA_DIR, 'backtest_report_summary.json')
    with open(meta_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Saved summary to', meta_fp)

    # produce interactive HTML using paired trades for richer hover labels
    out_html = os.path.join(PKG_DIR, 'backtest_interactive.html')
    make_interactive_html(metrics, trades, out_html, paired_trades=paired)
    print('Saved interactive HTML to', out_html)


if __name__ == '__main__':
    main()
