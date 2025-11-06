# 451-DL-Assignment-3

# Momentum-Based Multi-Asset Algorithmic Trading Strategy

This repository contains the implementation, backtesting, and performance evaluation of a systematic algorithmic trading strategy for MSDS 451 (Fall 2025). The goal is to design an automated trading approach, execute a historical backtest, and compare fund performance against standard market benchmarks.

The focus of this assignment is **automated trading**, not machine learning. The implemented strategy follows a **time-series momentum** approach inspired by Clenow (2019), with volatility-based position sizing and monthly portfolio rebalancing.

---

## 📈 Strategy Overview

**Universe:**
- SPY (S&P 500)
- QQQ (NASDAQ 100)
- IWM (US Small Caps)
- EFA (Developed Markets ex-US)
- EEM (Emerging Markets)
- TLT (Long-Term Treasuries)
- LQD (Investment Grade Corporate Bonds)

**Signal:**  
A security is held long if its **126-day return is positive**; otherwise the strategy holds no position in that asset.

**Position Sizing:**  
Weights are allocated using **inverse volatility weighting** and scaled to target **10% annualized portfolio volatility**.

**Rebalancing Frequency:**  
Monthly (first trading day of each month).

**Transaction Costs:**  
Slippage = **3 bps** per rebalance.

---

## 🧮 Backtesting Pipeline

1. Fetch adjusted daily close prices via `yfinance`
2. Compute 126-day momentum signal
3. Allocate positions using volatility targeting
4. Apply slippage costs during rebalancing
5. Compare portfolio equity curve to:
   - SPY (broad US equities)
   - QQQ (large-cap growth)
   - TLT (long-duration bonds)

---

## 📊 Key Performance Results

Performance results are stored in:

```

reports/performance.csv
reports/summary.json
reports/figures/equity_momentum.png

```

Example Figure (Strategy vs Benchmarks):

```

reports/figures/equity_momentum.png

````

Performance metrics include:
- CAGR
- Sharpe Ratio
- Maximum Drawdown
- Annualized Volatility
- In-sample vs Out-of-sample evaluation

---

## 🏁 How to Run the Backtest

Run the Strategy

```bash
python src/backtest.py \
    --mode momentum \
    --tickers SPY QQQ IWM EFA EEM TLT LQD \
    --start 2014-01-01 \
    --start_oos 2022-01-01 \
    --slippage_bps 3 \
    --target_vol 0.10
```

### 3. Output Files

Generated files will appear under:

```
reports/
├── summary.json
├── performance.csv
└── figures/
    └── equity_momentum.png
```

---

## 📝 Report

The written PDF report summarizing:

* research motivation,
* data sources,
* strategy logic,
* backtesting methodology,
* results interpretation,
* and conclusions

is located at:

```
report.pdf
```

---

## Use of AI Assistance (Required Disclosure)

I used AI (GPT-5) to help with:

* Structuring the repository layout
* Drafting documentation language
* Formatting results tables

---

## 📚 Reference

Clenow, Andreas F. (2019). *Trading Evolved: Anyone Can Build Killer Trading Strategies in Python.* Independently Published.

