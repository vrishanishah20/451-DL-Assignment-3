import argparse, os, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ['SPY','QQQ','IWM','EFA','EEM','TLT','LQD']

@dataclass
class Costs:
    commission_bps: float = 0.0
    slippage_bps: float = 3.0

def annualized_vol(daily_ret: pd.Series) -> float:
    return daily_ret.std(ddof=0) * np.sqrt(252)

def sharpe(daily_ret: pd.Series, rf_daily: float = 0.0) -> float:
    ex = daily_ret - rf_daily
    vol = ex.std(ddof=0)
    return 0.0 if vol == 0 else np.sqrt(252) * ex.mean() / vol

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity/roll_max - 1.0
    return dd.min()

def cagr(equity: pd.Series) -> float:
    if equity.index.size < 2: return 0.0
    yrs = (equity.index[-1] - equity.index[0]).days/365.25
    return (equity.iloc[-1]/equity.iloc[0])**(1/yrs) - 1 if yrs>0 else 0.0

def download_prices(tickers, start, end):
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how='all').ffill().dropna()

def momentum_signal(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    ret = prices.pct_change(lookback)
    sig = (ret > 0).astype(float)  # long or flat
    return sig

def meanrev_signal(prices: pd.DataFrame, lookback: int = 20, z_entry=1.0, z_exit=0.2, z_cap=2.5):
    ma = prices.rolling(lookback).mean()
    sd = prices.rolling(lookback).std(ddof=0)
    z = (prices - ma) / sd
    # Position is opposite of z (fade): cap between [-1,1]
    raw = -z.clip(-z_cap, z_cap) / z_cap
    # Hysteresis: shrink near zero to reduce churn
    pos = raw.where(raw.abs() > z_exit, 0.0)
    pos = pos.where(raw.abs() < z_entry, pos)  # enter only beyond entry
    return pos.fillna(0.0)

def vol_target_weights(returns: pd.DataFrame, target_ann_vol=0.10, floor=1e-8, cap=0.35):
    # inverse volatility per asset over last 60d → scale to target portfolio vol
    win = 60
    rolling_vol = returns.rolling(win).std(ddof=0) * np.sqrt(252)
    inv_vol = 1.0 / (rolling_vol.replace(0, np.nan))
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    # soft cap
    w = w.clip(upper=cap)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    # scale portfolio to target vol (approx): adjust with scalar k
    port_vol = (w * returns).sum(axis=1).rolling(252).std(ddof=0) * np.sqrt(252)
    k = (target_ann_vol / port_vol.replace(0, np.nan)).clip(upper=3.0).fillna(1.0)
    adj_w = (w.T * k).T
    # renormalize
    adj_w = adj_w.div(adj_w.sum(axis=1), axis=0).fillna(0.0)
    return adj_w

def apply_costs(weight_prev: pd.Series, weight_new: pd.Series, costs: Costs):
    # turnover cost in bps
    turn = (weight_new - weight_prev).abs().sum()
    bps = (costs.commission_bps + costs.slippage_bps) * turn
    return -bps / 10000.0

def backtest(prices: pd.DataFrame,
             mode: str = 'momentum',
             start_train: str = '2015-01-01',
             start_oos: str = '2022-01-01',
             target_vol: float = 0.10,
             costs: Costs = Costs()):
    prices = prices.sort_index()
    rets = prices.pct_change().fillna(0.0)

    if mode == 'momentum':
        sig = momentum_signal(prices)  # 1 or 0
    elif mode == 'meanrev':
        sig = meanrev_signal(prices)
    else:
        raise ValueError("mode must be 'momentum' or 'meanrev'")

    # Monthly rebalance on first business day
    month_flag = ~prices.index.to_period('M').duplicated()
    rebal_dates = prices.index[month_flag]

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    prev_w = pd.Series(0.0, index=prices.columns)

    for dt in rebal_dates:
        lo = max(prices.index.get_loc(dt) - 252, 0)
        hist = rets.iloc[:prices.index.get_loc(dt)+1]
        # base unscaled weights from signal
        w_base = sig.loc[dt]
        if mode == 'meanrev':
            # meanrev signal already continuous in [-1,1]; allow short? here: long/flat only
            w_base = w_base.clip(lower=0.0)  # comment this line to allow shorts

        # volatility targeting on last 60d window
        w_vol = vol_target_weights(hist[prices.columns].iloc[-252:], target_ann_vol=target_vol).iloc[-1]
        # combine: mask by signal
        w_target = (w_vol * (w_base > 0).astype(float))
        # normalize
        if w_target.sum() > 0:
            w_target = w_target / w_target.sum()
        else:
            w_target[:] = 0.0

        # trading costs on the rebal day
        cost_ret = apply_costs(prev_w, w_target, costs)
        weights.loc[dt] = w_target
        prev_w = w_target.copy()
        # record cost as an extra series at dt (we’ll add in PnL stream)
        rets.loc[dt, 'COSTS'] = cost_ret

    # forward-fill weights until next rebalance
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)

    port_ret = (weights.shift(1) * rets[prices.columns]).sum(axis=1)
    port_ret = port_ret + rets.get('COSTS', pd.Series(0.0, index=rets.index)).fillna(0.0)

    # Split IS / OOS
    is_ret = port_ret.loc[(port_ret.index >= pd.to_datetime(start_train)) & (port_ret.index < pd.to_datetime(start_oos))]
    oos_ret = port_ret.loc[port_ret.index >= pd.to_datetime(start_oos)]

    equity = (1+port_ret).cumprod()
    equity_is = (1+is_ret).cumprod()
    equity_oos = (1+oos_ret).cumprod()

    # Benchmarks buy-and-hold
    def bh(ticker):
        s = prices[ticker].pct_change().fillna(0.0)
        return s

    bench = {
        'SPY': bh('SPY') if 'SPY' in prices.columns else pd.Series(index=prices.index, dtype=float),
        'QQQ': bh('QQQ') if 'QQQ' in prices.columns else pd.Series(index=prices.index, dtype=float),
        'TLT': bh('TLT') if 'TLT' in prices.columns else pd.Series(index=prices.index, dtype=float),
    }
    bench_equity = {k: (1+v).cumprod() for k,v in bench.items() if not v.empty}

    # Metrics
    def metrics(r):
        eq = (1+r).cumprod()
        return {
            'CAGR': cagr(eq),
            'Sharpe': sharpe(r),
            'MaxDD': max_drawdown(eq),
            'AnnVol': annualized_vol(r),
        }

    out = {
        'port_ret': port_ret,
        'equity': equity,
        'equity_is': equity_is,
        'equity_oos': equity_oos,
        'bench_equity': bench_equity,
        'metrics_total': metrics(port_ret),
        'metrics_is': metrics(is_ret) if len(is_ret)>0 else None,
        'metrics_oos': metrics(oos_ret) if len(oos_ret)>0 else None,
    }
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['momentum','meanrev'], default='momentum')
    p.add_argument('--tickers', nargs='*', default=DEFAULT_TICKERS)
    p.add_argument('--start', default='2014-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--start_oos', default='2022-01-01')
    p.add_argument('--target_vol', type=float, default=0.10)
    p.add_argument('--commission_bps', type=float, default=0.0)
    p.add_argument('--slippage_bps', type=float, default=3.0)
    p.add_argument('--outdir', default='./reports')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'figures'), exist_ok=True)

    prices = download_prices(args.tickers, args.start, args.end)
    res = backtest(
        prices,
        mode=args.mode,
        start_train='2015-01-01',
        start_oos=args.start_oos,
        target_vol=args.target_vol,
        costs=Costs(args.commission_bps, args.slippage_bps),
    )

    # Save metrics
    def fmt(m):
        return {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k,v in m.items()}

    summary = {
        'Mode': args.mode,
        'Universe': args.tickers,
        'Total': fmt(res['metrics_total']),
        'InSample': fmt(res['metrics_is']) if res['metrics_is'] else None,
        'OutOfSample': fmt(res['metrics_oos']) if res['metrics_oos'] else None,
    }
    pd.DataFrame([summary]).to_json(os.path.join(args.outdir, 'summary.json'), orient='records', indent=2)

    # Save equity curves
    res['equity'].rename('Strategy').to_frame().to_csv(os.path.join(args.outdir, 'equity_strategy.csv'))
    for k,eq in res['bench_equity'].items():
        eq.rename(k).to_frame().to_csv(os.path.join(args.outdir, f'equity_{k}.csv'))

    # Quick plot
    import matplotlib.pyplot as plt
    plt.figure()
    (res['equity'] / res['equity'].iloc[0]).plot(label='Strategy')
    for k,eq in res['bench_equity'].items():
        (eq / eq.iloc[0]).plot(label=k, alpha=0.7)
    plt.legend()
    plt.title(f'Equity Curve – {args.mode.upper()}')
    plt.ylabel('Growth of $1')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'figures', f'equity_{args.mode}.png'), dpi=150)

if __name__ == '__main__':
    main()
