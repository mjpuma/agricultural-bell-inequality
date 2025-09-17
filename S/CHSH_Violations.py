import numpy as np
import pandas as pd
import yfinance as yf
import itertools
import requests
from io import StringIO
from tqdm import tqdm
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

#========================================================================================================
#Data
#========================================================================================================

disruptions_df = pd.read_csv("Data/disruptions.csv")

def yf_stocks():
    df = pd.read_csv("Data/yfinance_tickers.csv")
    tickers = df['ticker'].tolist()
    return tickers

def yf_futures():
    tickers = {
        'ZC=F': 'Corn',               
        'ZS=F': 'Soybeans',             
        'ZW=F': 'Wheat',                 
        'ZL=F': 'Soybean_Oil',             
        'ZM=F': 'Soybean_Meal',           
        'CC=F': 'Cocoa',                  
        'CT=F': 'Cotton',           
        'KC=F': 'Coffee',             
        'SB=F': 'Sugar',               
        'OJ=F': 'Orange_Juice',          
        'LE=F': 'Live_Cattle',             
        'HE=F': 'Lean_Hogs',            
        'CL=F': 'Crude_Oil',              
        'NG=F': 'Natural_Gas'            
    }
    tickers = list(tickers.keys())
    return tickers

def download_yf_data(tickers, START_DATE = "2000-01-01", END_DATE   = date.today().strftime("%Y-%m-%d")):
    print(f"Downloading data for {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                    group_by="ticker", auto_adjust=True, threads=True, progress=False)

    adj_close = pd.DataFrame()
    for t in tickers:
        try:
            adj_close[t] = raw[t]['Close']
        except Exception:
            continue

    adj_close = adj_close.dropna(axis=1, how="all")
    returns = adj_close.pct_change().dropna(how="all").sort_index()
    print(f"Final usable tickers: {len(returns.columns)}")
    print(f"Returns shape: {returns.shape}")
    return returns

#========================================================================================================
#Computation
#========================================================================================================

def compute_s1_sliding_pair(x, y, window_size=20, q=0.95):
    n = x.shape[0]
    m = n - window_size
    if m <= 0:
        return np.array([])

    shape = (m, window_size)
    stride = x.strides[0]
    x_win = np.lib.stride_tricks.as_strided(x, shape=shape, strides=(stride, stride)).copy()
    y_win = np.lib.stride_tricks.as_strided(y, shape=shape, strides=(stride, stride)).copy()

    a_sgn, b_sgn = np.sign(x_win), np.sign(y_win)
    abs_x, abs_y = np.abs(x_win), np.abs(y_win)

    thr_x = np.quantile(abs_x, q, axis=1)
    thr_y = np.quantile(abs_y, q, axis=1)

    mask_x0 = abs_x >= thr_x[:, None]
    mask_y0 = abs_y >= thr_y[:, None]

    def E(mask):
        term = (a_sgn * b_sgn) * mask
        s = term.sum(axis=1)
        cnt = mask.sum(axis=1)
        e = np.zeros_like(s, dtype=float)
        nz = cnt > 0
        e[nz] = s[nz] / cnt[nz]
        return e

    return E(mask_x0 & ~mask_y0) + E(mask_x0 & mask_y0) + \
           E(~mask_x0 & mask_y0) - E(~mask_x0 & ~mask_y0)

def run_computation(returns, OUTPUT_PCT_CSV = "chsh_agro_violation.csv", WINDOW_SIZE = 20, THRESHOLD_Q = 0.95, BOUND = 2):
    dates = returns.index[WINDOW_SIZE:]
    violation_counts = np.zeros(len(dates), dtype=int)
    total_counts = np.zeros(len(dates), dtype=int)

    pairs = itertools.combinations(returns.columns, 2)
    for A, B in tqdm(pairs, total=(len(returns.columns)*(len(returns.columns)-1))//2,
                    desc="Pairs"):
        ts = returns[[A, B]].dropna()
        if ts.shape[0] <= WINDOW_SIZE:
            continue

        x, y = ts[A].values, ts[B].values
        S1 = compute_s1_sliding_pair(x, y, window_size=WINDOW_SIZE, q=THRESHOLD_Q)
        if S1.size == 0:
            continue

        pair_dates = ts.index[WINDOW_SIZE:]
        pos = np.searchsorted(dates, pair_dates)
        valid = (pos >= 0) & (pos < len(dates))
        pos, S1 = pos[valid], S1[valid]

        np.add.at(total_counts, pos, 1)
        np.add.at(violation_counts, pos, (np.abs(S1) > BOUND).astype(int))

    pct = np.where(total_counts > 0, 100 * violation_counts / total_counts, np.nan)
    out_df = pd.DataFrame({"Date": dates, "ViolationPct": pct,
                        "TotalPairs": total_counts, "ViolationCounts": violation_counts})
    out_df.to_csv(OUTPUT_PCT_CSV, index=False)
    print(f"Saved percent violation to {OUTPUT_PCT_CSV}")
    return out_df
#========================================================================================================
#Plots
#========================================================================================================

def plot(out_df, output_path):
    derivative = np.gradient(out_df['ViolationPct'], edge_order=2)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("S1 Violations Over Time", "Derivative of S1 Violations")
    )

    fig.add_trace(
        go.Scatter(
            x=out_df['Date'],
            y=out_df['ViolationPct'],
            mode='lines',
            name=None,
            line=dict(color='lightskyblue'),
            showlegend=False
        ),
        row=1, col=1
    )
    disruptions_df["start_date"] = pd.to_datetime(disruptions_df["start_date"])
    disruptions_df["end_date"] = pd.to_datetime(disruptions_df["end_date"])
    dates_norm = (
        (disruptions_df["start_date"] - disruptions_df["start_date"].min()) /
        (disruptions_df["start_date"].max() - disruptions_df["start_date"].min())
    )
    colors = px.colors.sample_colorscale("Redor", dates_norm)

    for date, name, color in zip(disruptions_df["start_date"], disruptions_df["name"], colors):
        fig.add_trace(
            go.Scatter(
                x=[date, date],
                y=[out_df['ViolationPct'].min(), out_df['ViolationPct'].max()],
                mode="lines",
                line=dict(color=color, dash="dash"),
                name=name,              
                legendgroup=name,      
                showlegend=True
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=out_df['Date'],
            y=derivative,
            mode='lines',
            name=None,
            line=dict(color='lightskyblue'),
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        height=1200,
        width=1800,
        showlegend=True,
        xaxis2_title="Date",
        yaxis1_title="Percent of pairs with |S1| > 2",
        yaxis2_title="Derivative",
    )

    fig.show()
    fig.write_html(output_path)
#========================================================================================================
#Composite Functions (Final)
#========================================================================================================

def yf_stock_data_plotly():
    os.mkdir('yf_stocks') 
    tickers = yf_stocks()
    returns = download_yf_data(tickers, "2000-01-01", date.today().strftime("%Y-%m-%d"))
    out_df = run_computation(returns, "yf_stocks/chsh_agro_violation_yf_stock_data.csv", 20, 0.95, 2)
    plot(out_df, "yf_stocks/chsh_agro_violation_yf_stock_data_plotly.html")

def yf_futures_data_plotly():
    os.mkdir('yf_futures') 
    tickers = yf_futures()
    returns = download_yf_data(tickers, "2000-01-01", date.today().strftime("%Y-%m-%d"))
    out_df = run_computation(returns, "yf_futures/chsh_agro_violation_yf_futures_data.csv", 20, 0.95, 2)
    plot(out_df, "yf_futures/chsh_agro_violation_yf_futures_data_plotly.html")

yf_stock_data_plotly()

