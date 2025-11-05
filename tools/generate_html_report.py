import os
import base64
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG_DIR = os.path.join(ROOT, 'roostoo_bot_template')
DATA_DIR = os.path.join(PKG_DIR, 'data')


def _b64_image(path):
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode('ascii')


def load_files():
    # candidate locations
    candidates = [DATA_DIR, os.path.join(ROOT, 'data'), os.path.join(ROOT, 'roostoo_bot_template', 'data')]
    found = None
    for c in candidates:
        if os.path.exists(os.path.join(c, 'backtest_report.csv')):
            found = c
            break
    if found is None:
        found = DATA_DIR
    report_fp = os.path.join(found, 'backtest_report.csv')
    summary_fp = os.path.join(found, 'backtest_report_summary.json')
    metrics_fp = os.path.join(found, 'backtest_metrics.csv')
    trades_fp = os.path.join(found, 'backtest_trades.csv')
    eq_png = os.path.join(PKG_DIR, 'equity_curve.png')
    trades_png = os.path.join(PKG_DIR, 'trades_markers.png')
    interactive_html = os.path.join(ROOT, 'backtest_interactive.html')
    return {
        'report': report_fp if os.path.exists(report_fp) else None,
        'summary': summary_fp if os.path.exists(summary_fp) else None,
        'metrics': metrics_fp if os.path.exists(metrics_fp) else None,
        'trades': trades_fp if os.path.exists(trades_fp) else None,
        'eq_png': eq_png if os.path.exists(eq_png) else None,
        'trades_png': trades_png if os.path.exists(trades_png) else None,
        'interactive': interactive_html if os.path.exists(interactive_html) else None,
    }


def build_html(files, out_path):
    # load small bits
    summary = {}
    if files['summary']:
        with open(files['summary'], 'r', encoding='utf-8') as f:
            summary = json.load(f)
    report_df = pd.DataFrame()
    if files['report']:
        report_df = pd.read_csv(files['report'])
    metrics_df = pd.DataFrame()
    if files['metrics']:
        metrics_df = pd.read_csv(files['metrics'])

    eq_data = _b64_image(files['eq_png']) if files['eq_png'] else ''
    trades_data = _b64_image(files['trades_png']) if files['trades_png'] else ''

    # Build simple HTML
    html = ['<!doctype html>', '<html><head><meta charset="utf-8"><title>Backtest Report</title>',
            '<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px} table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:8px} th{background:#f4f4f4}</style>', '</head><body>']
    html.append('<h1>Backtest Report</h1>')
    html.append(f'<p>Generated: {pd.Timestamp.now().isoformat()}</p>')

    html.append('<h2>Summary</h2>')
    if summary:
        html.append('<table>')
        for k, v in summary.items():
            html.append(f'<tr><th>{k}</th><td>{v}</td></tr>')
        html.append('</table>')
    else:
        html.append('<p>No summary available.</p>')

    html.append('<h2>Metrics (sample)</h2>')
    if not metrics_df.empty:
        html.append(metrics_df.tail(5).to_html(index=False, classes='metrics'))
    else:
        html.append('<p>No metrics CSV found.</p>')

    html.append('<h2>Per-trade Sample (first 20)</h2>')
    if not report_df.empty:
        html.append(report_df.head(20).to_html(index=False, classes='trades'))
    else:
        html.append('<p>No per-trade report found.</p>')

    html.append('<h2>Plots</h2>')
    if eq_data:
        html.append(f'<h3>Equity Curve</h3><img src="{eq_data}" style="max-width:100%;height:auto;border:1px solid #ccc"/>')
    if trades_data:
        html.append(f'<h3>Trade Markers</h3><img src="{trades_data}" style="max-width:100%;height:auto;border:1px solid #ccc"/>')

    if files['interactive']:
        rel = os.path.relpath(files['interactive'], ROOT)
        html.append(f'<p><a href="{rel}" target="_blank">Open interactive HTML plot</a></p>')

    html.append('</body></html>')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    files = load_files()
    out = os.path.join(ROOT, 'backtest_human_report.html')
    build_html(files, out)
    print('Saved human-readable HTML to', out)


if __name__ == '__main__':
    main()
