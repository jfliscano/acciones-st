"""
Acciones SuperTrend Analyzer — Dash version
Adaptado de etf_dash_app.py para acciones con datos 1h y 1 año de historial.
Detecta cambios de dirección (BUY/Sell) diarios en la lista acciones.txt.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from fpdf import FPDF
import flask

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def _resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', None) or _app_dir()
    else:
        base = _app_dir()
    return os.path.join(base, relative_path)

SCRIPT_DIR         = _app_dir()
DATA_DIR           = os.path.join(SCRIPT_DIR, "acciones_data")
ACCIONES_FILE      = _resource_path("acciones.txt")
DEFAULT_INVESTMENT = 10_000
HISTORY_DAYS       = 365
INTERVAL           = '1h'
CACHE_TTL_SECONDS  = 3600  # 1h para datos horarios
MARKER_OFFSET      = 0.02
MAX_WORKERS        = 8

PREFIX_SUFFIX = {"asx:": ".AX", "twse:": ".TW"}

COUNTRY_SUFFIX = {
    "ks": (".KS", []),
    "is": (".IS", {"CCOLA","ANHYT","ANSGR","TRENJ","AEFES","TURSG","GEDIK",
                   "AKSGY","RYGYO","AGESA","ENJSA","ARDYZ","AYDEM","BASGZ",
                   "ATATP","ESCAR","ARASE","GRSEL","LIDER","AHGAZ"}),
    "mc": (".MC", {"ANA","ENG","NTGY","REP","ELE","BKT","SCYR","MEL",
                   "ITX","BBVA","SAN","ACS","MAP","SAB","ALM","CABK",
                   "ENO","LOG","AENA"}),
    "mi": (".MI", {"UCG","IRE","ISP","BRE","MB","DLG","HER","IP",
                   "ACE","BMPS","CE","BGN","PRY","ANIM","PST","BMED",
                   "ENAV","BAMI","PIRC"}),
    "to": (".TO", {"ALS","CCA","L","MX","WN","LUN","SVM","ORE",
                   "CVE","SIA","BTE","BDT","EXE","CJ","CRON","NWC","ATD"}),
    "l":  (".L",  {"RAT","NWG","VTY","LLOY","KLR","KIE",
                   "BARC","CMCX","VOD","AAL","SDLF","CURY","TBCG","BPCR","IHP",
                   "QLT","MNG","OSB","SHAW"}),
    "t":  (".T",  {"8522","8253","3405","8393","8309","8616","8609",
                   "8341","2914","9107","1820","1605","3231","8418",
                   "7744","8725","7181","7189","7327","5832"}),
    "jo": (".JO", {"NED","SBK","ABG","SOL","EXX","SLM","INL","MTM","RDF","OMU"}),
}

DOT_TO_HYPHEN = {"BBD.B": "BBD-B.TO", "CHE.UN": "CHE-UN.TO",
                 "TECK.B": "TECK-B.TO", "BT.A": "BT-A.L"}

KOREAN_KOSDAQ = {"A036830", "A086450"}

def load_acciones_list():
    if not os.path.exists(ACCIONES_FILE):
        log.error(f"ARCHIVO NO ENCONTRADO: {ACCIONES_FILE}")
        return [], {}, {}

    with open(ACCIONES_FILE, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    tickers, ticker_name = [], {}
    for i in range(0, len(lines), 3):
        ticker = lines[i].strip()   if i   < len(lines) else ''
        name   = lines[i+1].strip() if i+1 < len(lines) else ''
        if ticker:
            tickers.append(ticker)
            if name:
                ticker_name[ticker] = name

    log.info(f"Lista cargada: {len(tickers)} símbolos desde {ACCIONES_FILE}")
    return tickers, ticker_name

ACCIONES_LIST, TICKER_NAME = load_acciones_list()

def yahoo_symbol(ticker):
    for prefix, suffix in PREFIX_SUFFIX.items():
        if ticker.lower().startswith(prefix):
            core = ticker[len(prefix):]
            return f"{core.upper()}{suffix}"
    t = ticker.upper()
    if t in DOT_TO_HYPHEN:
        return DOT_TO_HYPHEN[t]
    if len(t) == 7 and t.startswith('A') and t[1:].isdigit():
        suffix = ".KQ" if t in KOREAN_KOSDAQ else ".KS"
        return f"{t[1:]}{suffix}"
    for _key, (suffix, tickers) in COUNTRY_SUFFIX.items():
        if t in tickers:
            return f"{t}{suffix}"
    return t

def supertrend(df, atr_period=10, multiplier=1.7):
    df    = df.copy()
    high  = df['High'].values.flatten()
    low   = df['Low'].values.flatten()
    close = df['Close'].values.flatten()

    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr  = np.maximum(high - low,
                     np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values

    hl_avg = (high + low) / 2
    upper  = hl_avg + multiplier * atr
    lower  = hl_avg - multiplier * atr

    n    = len(df)
    st   = np.full(n, np.nan)
    dir_ = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(atr[i]) or np.isnan(atr[i-1]):
            continue
        if np.isnan(st[i-1]):
            st[i]   = lower[i] if close[i] > upper[i] else upper[i]
            dir_[i] = 1 if close[i] > st[i] else -1
        else:
            if dir_[i-1] == 1:
                if close[i] > st[i-1]:
                    dir_[i] = 1
                    st[i]   = max(lower[i], st[i-1])
                else:
                    dir_[i] = -1
                    st[i]   = upper[i]
            else:
                if close[i] < st[i-1]:
                    dir_[i] = -1
                    st[i]   = min(upper[i], st[i-1])
                else:
                    dir_[i] = 1
                    st[i]   = lower[i]

    df['ST']       = st
    df['ST_Dir']   = dir_
    df['ST_Upper'] = upper
    df['ST_Lower'] = lower
    return df

def generate_signals(df, investment=DEFAULT_INVESTMENT):
    df   = df.copy()
    dirs = df['ST_Dir'].values
    sig  = np.zeros(len(dirs), dtype=int)
    for i in range(2, len(dirs)):
        if dirs[i-2] != 0 and dirs[i-1] != 0 and dirs[i-2] != dirs[i-1]:
            sig[i] = dirs[i-1]

    trades, pos_type, entry_price, entry_date = [], None, 0.0, None
    for i in range(len(df)):
        if sig[i] == 0:
            continue

        if pos_type is None:
            pos_type = 'LONG' if sig[i] == 1 else 'SHORT'
            entry_price = float(df['Open'].iloc[i])
            entry_date  = df.index[i]
            continue

        exit_price = float(df['Open'].iloc[i])
        if pos_type == 'LONG':
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price
        trades.append(dict(
            entry_date=str(entry_date.date()),
            exit_date=str(df.index[i].date()),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            return_pct=round(ret * 100, 2),
            pnl_usdt=round(investment * ret, 2),
            type=pos_type,
            signal='SELL' if sig[i] == -1 else 'BUY',
        ))

        pos_type = 'LONG' if sig[i] == 1 else 'SHORT'
        entry_price = exit_price
        entry_date  = df.index[i]

    if pos_type is not None:
        exit_price = float(df['Close'].iloc[-1])
        if pos_type == 'LONG':
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price
        trades.append(dict(
            entry_date=str(entry_date.date()),
            exit_date=str(df.index[-1].date()),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            return_pct=round(ret * 100, 2),
            pnl_usdt=round(investment * ret, 2),
            type=pos_type,
            signal='CLOSE',
        ))

    if trades:
        total_pnl = sum(t['pnl_usdt'] for t in trades)
        win_rate  = sum(1 for t in trades if t['pnl_usdt'] > 0) / len(trades) * 100
        equity, peak, max_dd = investment, investment, 0.0
        for t in trades:
            equity += t['pnl_usdt']
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
    else:
        total_pnl, win_rate, max_dd = 0.0, 0.0, 0.0

    return dict(total_trades=len(trades), win_rate=round(win_rate, 2),
                total_pnl_usdt=round(total_pnl, 2),
                final_value=round(investment + total_pnl, 2),
                max_drawdown_pct=round(max_dd, 2),
                trades=trades)

def build_pdf(title, col_names, col_widths, rows):
    pdf = FPDF(orientation='P', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(4)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    total = sum(col_widths)
    scale = usable / total if total > 0 else 1
    widths = [w * scale for w in col_widths]
    pdf.set_font('Helvetica', 'B', 8)
    for i, name in enumerate(col_names):
        pdf.cell(widths[i], 6, name, border=1, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 8)
    for row in rows:
        for i, val in enumerate(row):
            pdf.cell(widths[i], 5, str(val), border=1, align='C')
        pdf.ln()
    return bytes(pdf.output())

def export_trades(trades, symbol, atr_p, mult, investment=10000):
    cols = ['Type', 'Entry', 'Exit', 'Entry $', 'Exit $', 'Ret%', 'PnL']
    widths = [14, 22, 22, 22, 22, 18, 22]
    rows = [[t['type'], t['entry_date'], t['exit_date'],
             f"${t['entry_price']:.2f}", f"${t['exit_price']:.2f}",
             f"{t['return_pct']:.2f}%", f"${t['pnl_usdt']:.2f}"] for t in trades]
    title = f"Supertrend {symbol} 1h | ATR={atr_p} Mult={mult} | {len(trades)} trades"
    return build_pdf(title, cols, widths, rows)

def export_screener(results, errors, atr_p, mult):
    cols = ['Symbol', 'Close $', 'Prev ST', 'Curr ST', 'Change', 'Changed']
    widths = [24, 20, 24, 24, 28, 18]
    rows = [[r['symbol'], f"${r['close']:.2f}", r['prev_dir'],
             r['curr_dir'], r['change'], r['changed']] for r in results]
    title = f"Screener Acciones | ATR={atr_p} Mult={mult} | {len(results)} stocks, {len(errors)} errors"
    return build_pdf(title, cols, widths, rows)

def fetch_data(symbol, force_refresh=False, days=HISTORY_DAYS):
    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = f"_{days}d" if days != HISTORY_DAYS else ""
    filepath  = os.path.join(DATA_DIR, f"{symbol}{suffix}.csv")
    yahoo_sym = yahoo_symbol(symbol)

    if force_refresh and os.path.exists(filepath):
        os.remove(filepath)

    if os.path.exists(filepath):
        if (time.time() - os.path.getmtime(filepath)) < CACHE_TTL_SECONDS:
            try:
                df = pd.read_csv(filepath, index_col=0)
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = _strip_tz(df)
                df = df.dropna(subset=['Close'])
                if len(df) > 20 and df['Close'].dtype != object:
                    return df
            except Exception as e:
                log.warning(f"Cache corrupto {symbol}: {e}")
        os.remove(filepath)

    log.info(f"Descargando {symbol} ({yahoo_sym}) {days}d 1h...")
    try:
        df = yf.download(yahoo_sym,
                         period='1mo' if days <= 31 else '1y',
                         interval=INTERVAL,
                         progress=False, auto_adjust=False)
    except Exception as e:
        log.warning(f"Error descargando {symbol}: {e}")
        return None

    if df is None or df.empty:
        log.warning(f"Sin datos: {symbol}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = _strip_tz(df)
    df.to_csv(filepath)
    return df

def _strip_tz(df):
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def make_chart(df, symbol, atr_period, multiplier):
    df     = supertrend(df, atr_period=atr_period, multiplier=multiplier)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    stats  = generate_signals(df)

    dirs       = df['ST_Dir'].values
    sig_arr    = np.zeros(len(dirs), dtype=int)
    for j in range(2, len(dirs)):
        if dirs[j-2] != 0 and dirs[j-1] != 0 and dirs[j-2] != dirs[j-1]:
            sig_arr[j] = dirs[j-1]
    sig_series = pd.Series(sig_arr, index=df.index)
    buy_idx    = sig_series[sig_series == 1].index
    sell_idx   = sig_series[sig_series == -1].index

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='', showlegend=False,
        increasing_line_color='#089981', decreasing_line_color='#f23645',
        whiskerwidth=0))

    st_green = df['ST'].where(df['ST_Dir'] ==  1)
    st_red   = df['ST'].where(df['ST_Dir'] == -1)
    fig.add_trace(go.Scatter(x=df.index, y=st_green, mode='lines',
                             line=dict(color='#089981', width=1.8), name='ST Uptrend'))
    fig.add_trace(go.Scatter(x=df.index, y=st_red, mode='lines',
                             line=dict(color='#f23645', width=1.8), name='ST Downtrend'))

    fig.add_trace(go.Scatter(
        x=buy_idx, y=df.loc[buy_idx, 'Low'] * (1 - MARKER_OFFSET), mode='markers',
        marker=dict(symbol='triangle-up', size=20, color='#2962ff',
                    line=dict(width=1, color='white')), name='BUY'))
    fig.add_trace(go.Scatter(
        x=sell_idx, y=df.loc[sell_idx, 'High'] * (1 + MARKER_OFFSET), mode='markers',
        marker=dict(symbol='triangle-down', size=20, color='#f23645',
                    line=dict(width=1, color='white')), name='SELL'))

    fig.update_layout(
        template='plotly_dark',
        title=dict(text=f'{symbol} — SuperTrend 1h 1y | ATR={atr_period} Mult={multiplier}',
                   font=dict(size=14)),
        autosize=True, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation='h', y=1.02, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False, hovermode='x unified',
        font=dict(size=10))
    fig.update_yaxes(title_text='')
    last = df.index[-1]
    fig.update_xaxes(range=[last - pd.Timedelta(days=15), last])
    return fig, stats

SCREENER_DAYS = 30

def screener_single(symbol, atr_p, mult, force_refresh=False):
    df = fetch_data(symbol, force_refresh=force_refresh, days=SCREENER_DAYS)
    if df is None or len(df) < 5:
        return None
    df   = supertrend(df, atr_period=atr_p, multiplier=mult)
    dirs = df['ST_Dir'].values
    if len(dirs) < 2:
        return None
    prev_dir    = int(dirs[-2])
    curr_dir    = int(dirs[-1])
    changed     = prev_dir != curr_dir
    change_type = ('UP2DOWN' if prev_dir == 1 else 'DOWN2UP') if changed else ''
    return dict(
        symbol   = symbol,
        name     = TICKER_NAME.get(symbol, ''),
        close    = round(float(df['Close'].iloc[-1]), 2),
        prev_dir = 'UPTREND'  if prev_dir == 1 else 'DOWNTREND',
        curr_dir = 'UPTREND'  if curr_dir == 1 else 'DOWNTREND',
        changed  = 'YES' if changed else 'NO',
        change   = change_type,
    )

app    = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                   title='Acciones SuperTrend 1h')
server = app.server

# ---- PWA ----
@app.server.route('/manifest.json')
def serve_manifest():
    return flask.send_file(os.path.join(SCRIPT_DIR, 'manifest.json'))

@app.server.route('/sw.js')
def serve_sw():
    return flask.send_file(os.path.join(SCRIPT_DIR, 'sw.js'))

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
  <link rel="manifest" href="/manifest.json">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="Acc ST">
  {%metas%}
  <title>{%title%}</title>
  {%favicon%}
  {%css%}
  <style>
    @media (max-width: 576px) {
      h2 { font-size: 1rem !important; margin-top: 0.5rem !important; margin-bottom: 0.25rem !important; }
      p.text-muted { display: none !important; }
      .card-body { padding: 0.3rem !important; }
      .card-body h4 { font-size: 0.85rem !important; }
      .card-body h6 { font-size: 0.6rem !important; }
      .dash-table-container { overflow-x: auto !important; }
      .dash-table-container table { font-size: 10px !important; }
      .dash-dropdown { font-size: 13px !important; }
      .nav-pills .nav-link { font-size: 12px !important; padding: 4px 10px !important; }
    }
  </style>
</head>
<body>
  {%app_entry%}
  <footer>
    {%config%}
    {%scripts%}
    {%renderer%}
  </footer>
  <script>
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js');
    }
  </script>
</body>
</html>
'''

file_warning = html.Div()
if not ACCIONES_LIST:
    file_warning = dbc.Alert(
        [html.B("acciones.txt no encontrado. "),
         f"Colócalo en: {SCRIPT_DIR} y reinicia la app."],
        color="danger", className="mb-2")

controls = dbc.Row([
    dbc.Col([
        html.Div([html.Span('ATR ', style={'fontWeight':700,'fontSize':13}),
                  dcc.Input(id='atr-input', type='number', value=10, min=2, max=50, step=1,
                            className='form-control',
                            style={'width':52,'height':26,'fontSize':13,'padding':'1px 4px'})],
                 style={'display':'flex','alignItems':'center','gap':3}),
    ], xs=4, sm=3),
    dbc.Col([
        html.Div([html.Span('Mult ', style={'fontWeight':700,'fontSize':13}),
                  dcc.Input(id='mult-input', type='number', value=1.7, min=0.5, max=5.0, step=0.1,
                            className='form-control',
                            style={'width':52,'height':26,'fontSize':13,'padding':'1px 4px'})],
                 style={'display':'flex','alignItems':'center','gap':3}),
    ], xs=4, sm=3),
    dbc.Col([
        html.Small(f'{len(ACCIONES_LIST)} stocks 1h 1y',
                   className='text-success fw-bold' if ACCIONES_LIST else 'text-danger',
                   style={'fontSize':11}),
    ], xs=4, sm=3),
], className='mb-1', style={'display':'flex','alignItems':'center'})

tab_chart = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[{'label': f'{t}  —  {TICKER_NAME.get(t,"")}', 'value': t}
                         for t in ACCIONES_LIST],
                value=(ACCIONES_LIST[0] if ACCIONES_LIST else None),
                clearable=False, style={'color': '#333'}),
        ], xs=12),
    ], className='mb-1'),
    html.Div(id='stats-cards', className='mb-1'),
    dcc.Store(id='chart-stats-store', data=None),
    dcc.Download(id='download-trades'),
    html.Div(dcc.Loading(type='circle',
                          children=dcc.Graph(id='main-chart',
                                             style={'height': '42vh'})),
             className='mb-1'),
    dbc.Button('Export PDF', id='export-trades-btn', color='info',
               size='sm', className='w-100 mb-1'),
    html.Div(id='trade-table', style={'maxHeight': '30vh', 'overflowY': 'auto'}),
], fluid=True)

tab_screener = dbc.Container([
    html.Div([
        html.Button('Run Screener', id='screener-btn',
                    className='btn btn-secondary btn-sm', style={'whiteSpace':'nowrap'}),
        html.Button('Export PDF', id='export-screener-btn',
                    className='btn btn-secondary btn-sm', style={'whiteSpace':'nowrap'}),
        html.Span(id='screener-status', className='text-muted', style={'fontSize':12}),
    ], style={'display':'flex','alignItems':'center','gap':'6px','flexWrap':'wrap'},
       className='mb-3'),
    dcc.Store(id='screener-store', data=None),
    dcc.Download(id='download-screener'),
    dcc.Loading(type='circle', children=html.Div(id='screener-output')),
], fluid=True)

app.layout = dbc.Container([
    html.H2('Acciones SuperTrend 1h Screener', className='text-center mt-3 mb-2'),
    html.P('Detecta cambios de dirección (BUY=SELL) en velas 1h · 1 año',
           className='text-center text-muted'),
    file_warning,
    controls,
    html.Div([
        dbc.Nav([
            dbc.NavItem(dbc.NavLink('Chart', id='tab-chart-link', active=True)),
            dbc.NavItem(dbc.NavLink('Screener', id='tab-screener-link')),
        ], pills=True),
        dbc.Button('Refresh Data', id='refresh-btn', color='secondary',
                   size='sm'),
    ], style={'display':'flex','alignItems':'center','justifyContent':'center','gap':'6px','flexWrap':'wrap'},
       className='mb-2'),
    html.Div(id='tab-content'),
], fluid=True)

@callback(
    Output('tab-content', 'children'),
    [Output('tab-chart-link', 'active'),
     Output('tab-screener-link', 'active')],
    [Input('tab-chart-link', 'n_clicks'),
     Input('tab-screener-link', 'n_clicks')],
)
def switch_tab(nc, ns):
    if ctx.triggered_id == 'tab-screener-link':
        return tab_screener, False, True
    return tab_chart, True, False

@callback(
    [Output('main-chart',  'figure'),
     Output('stats-cards', 'children'),
     Output('trade-table', 'children'),
     Output('chart-stats-store', 'data')],
    [Input('symbol-dropdown', 'value'),
     Input('atr-input',       'value'),
     Input('mult-input',      'value'),
     Input('refresh-btn',     'n_clicks')],
    prevent_initial_call=False,
)
def update_chart(symbol, atr_period, multiplier, _n):
    if not symbol:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    force = (ctx.triggered_id == 'refresh-btn')
    df    = fetch_data(symbol, force_refresh=force)

    atr_period = atr_period or 10
    multiplier = multiplier or 1.7

    if df is None:
        fig = go.Figure()
        fig.add_annotation(text='No data available', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template='plotly_dark', height=400)
        return fig, html.Div('No data', className='text-danger'), html.Div(), dash.no_update

    try:
        fig, stats = make_chart(df, symbol, atr_period, multiplier)
    except Exception as e:
        log.exception(f"Error en chart {symbol}")
        fig = go.Figure()
        fig.add_annotation(text=f'Error: {e}', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template='plotly_dark', height=400)
        return fig, html.Div(f'Error: {e}', className='text-danger'), html.Div(), dash.no_update

    pct      = ((stats['final_value'] / DEFAULT_INVESTMENT) - 1) * 100
    pnl_cls  = 'text-success' if stats['total_pnl_usdt'] >= 0 else 'text-danger'
    pct_cls  = 'text-success' if pct >= 0                      else 'text-danger'
    dd_cls   = 'text-danger'  if stats['max_drawdown_pct'] > 20 else 'text-warning'

    lb = {'fontSize':11,'marginBottom':2}
    vl = {'fontSize':17,'fontWeight':700}
    cw1 = {"xs": 3, "sm": 2, "md": 2}
    cw2 = {"xs": 6, "sm": 2, "md": 2}
    cards = dbc.Row([
        dbc.Col([html.Div('Trades', className='text-muted', style=lb), html.Div(stats['total_trades'], className='text-white', style=vl)], width=cw1),
        dbc.Col([html.Div('Win Rate', className='text-muted', style=lb), html.Div(f"{stats['win_rate']}%", className='text-white', style=vl)], width=cw1),
        dbc.Col([html.Div('Total PnL', className='text-muted', style=lb), html.Div(f"{stats['total_pnl_usdt']:.2f}", className=pnl_cls, style=vl)], width=cw1),
        dbc.Col([html.Div('Final Value', className='text-muted', style=lb), html.Div(f"{stats['final_value']:.2f}", className='text-white', style=vl)], width=cw1),
        dbc.Col([html.Div('Return', className='text-muted', style=lb), html.Div(f"{pct:.2f}%", className=pct_cls, style=vl)], width=cw2),
        dbc.Col([html.Div('Max Drawdown', className='text-muted', style=lb), html.Div(f"{stats['max_drawdown_pct']:.2f}%", className=dd_cls, style=vl)], width=cw2),
    ], className='g-0')

    col_map = {'type':'Type','entry_date':'Entry','exit_date':'Exit',
               'entry_price':'Entry $','exit_price':'Exit $',
               'return_pct':'Ret%','pnl_usdt':'PnL'}
    cols = list(col_map.keys())
    if stats['trades']:
        rows = []
        for t in reversed(stats['trades']):
            color = '#089981' if t['pnl_usdt'] >= 0 else '#f23645'
            cells = [html.Td(str(t[c]),
                              style={'textAlign':'center','padding':'3px 6px','color':color,
                                     'whiteSpace':'nowrap'})
                     for c in cols]
            rows.append(html.Tr(cells))
        trade_table = html.Div([
            html.H6(f'Trades ({stats["total_trades"]})',
                    className='text-white mb-2', style={'fontWeight':600}),
            html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(col_map[c], style={'padding':'4px 6px','textAlign':'center',
                                                   'whiteSpace':'nowrap'})
                        for c in cols])),
                    html.Tbody(rows),
                ], style={'borderCollapse':'collapse','fontSize':12}),
            ], style={'overflowX':'auto'}),
        ], style={'maxHeight':'40vh','overflowY':'auto'})
    else:
        trade_table = html.Div([
            html.H6('Trades (0)', className='text-white mb-2', style={'fontWeight':600}),
            html.Div('No trades', className='text-muted'),
        ])

    store_data = dict(symbol=symbol, atr_p=atr_period, mult=multiplier,
                      trades=stats.get('trades', []))
    return fig, cards, trade_table, store_data

@callback(
    Output('download-trades', 'data'),
    Input('export-trades-btn', 'n_clicks'),
    State('chart-stats-store', 'data'),
    prevent_initial_call=True,
)
def export_trades_pdf(n, store):
    if not store or not store.get('trades'):
        return dash.no_update
    pdf_bytes = export_trades(store['trades'], store['symbol'],
                              store['atr_p'], store['mult'])
    fname = f"Supertrend_{store['symbol']}_1h_{datetime.now().strftime('%Y%m%d')}.pdf"
    return dcc.send_bytes(pdf_bytes, fname)

@callback(
    [Output('screener-output', 'children'),
     Output('screener-status', 'children'),
     Output('screener-store', 'data')],
    Input('screener-btn', 'n_clicks'),
    [State('atr-input', 'value'),
     State('mult-input', 'value')],
    prevent_initial_call=True,
)
def run_screener(n_clicks, atr_period, multiplier):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update

    atr_period = atr_period or 10
    multiplier = multiplier or 1.7
    results, errors = [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(screener_single, sym, atr_period, multiplier, True): sym
                   for sym in ACCIONES_LIST}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                r = future.result()
                if r:   results.append(r)
                else:   errors.append(sym)
            except Exception as e:
                log.warning(f"Screener {sym}: {e}")
                errors.append(sym)

    results.sort(key=lambda x: (x['changed'] != 'YES', x['symbol']))

    changed       = [r for r in results if r['changed'] == 'YES']
    changed_count = len(changed)

    status = html.Span([
        f'Done: {len(results)}/{len(ACCIONES_LIST)} stocks | Cambiaron: ',
        html.Span(f'{changed_count}', style={'color':'#f0ad4e','fontWeight':'bold'}),
        f' | Sin datos: {len(errors)}',
    ])

    if not results:
        return html.Div('No data available', className='text-danger'), status, dash.no_update

    cols    = ['symbol', 'name', 'close', 'prev_dir', 'curr_dir', 'change', 'changed']
    col_map = {'symbol':'Symbol','name':'Name','close':'Close $',
               'prev_dir':'Prev ST','curr_dir':'Curr ST',
               'change':'Change','changed':'Changed'}

    table = dash_table.DataTable(
        columns=[{'name': col_map[c], 'id': c} for c in cols],
        data=results,
        style_as_list_view=True,
        style_header={'backgroundColor':'#1e1e1e','color':'#e0e0e0',
                      'fontWeight':'bold','fontSize':12},
        style_cell={'backgroundColor':'#2a2e39','color':'#e0e0e0',
                    'textAlign':'center','fontSize':12,'padding':'5px 10px'},
        style_cell_conditional=[
            {'if':{'column_id':'name'},'textAlign':'left','maxWidth':'200px',
             'overflow':'hidden','textOverflow':'ellipsis'},
        ],
        style_data_conditional=[
            {'if':{'filter_query':'{changed} = YES'},
             'backgroundColor':'#2a1f00','color':'#f0ad4e','fontWeight':'bold'},
            {'if':{'filter_query':'{curr_dir} = UPTREND'},
             'color':'#089981'},
            {'if':{'filter_query':'{curr_dir} = DOWNTREND'},
             'color':'#f23645'},
        ],
        page_size=100,
        sort_action='native',
    )

    summary = html.Div()
    if changed:
        summary = html.Div([
            html.H6(f'Cambios hoy ({changed_count})',
                    className='text-warning mt-2 mb-1', style={'fontWeight':600}),
            html.Div(', '.join(r['symbol'] for r in changed),
                     style={'color':'#f0ad4e','fontSize':13}),
        ])

    store_data = dict(atr_p=atr_period, mult=multiplier,
                      results=results, errors=errors)
    return html.Div([summary, html.Hr(), table]), status, store_data

    store_data = dict(atr_p=atr_period, mult=multiplier,
                      results=results, errors=errors)
    return html.Div([summary, html.Hr(), table]), status, store_data

@callback(
    Output('download-screener', 'data'),
    Input('export-screener-btn', 'n_clicks'),
    State('screener-store', 'data'),
    prevent_initial_call=True,
)
def export_screener_pdf(n, store):
    if not store:
        return dash.no_update
    pdf_bytes = export_screener(store['results'], store['errors'],
                                store['atr_p'], store['mult'])
    fname = f"Screener_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return dcc.send_bytes(pdf_bytes, fname)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
