# Hands-on: PSO Optimization of Two Real Indonesian Stocks

**Course:** Metode Optimasi Metaheuristik  
**Level:** Beginner | **Duration:** 60–75 minutes | **Environment:** Google Colab

## Learning outcomes
Students will (1) download real stock prices, (2) compute annualized returns and covariance, (3) formulate a one-variable risk–return objective, (4) implement PSO manually, and (5) interpret optimal weights and convergence.

## Case study
An investor allocates 100% of capital between **Bank Central Asia (`BBCA.JK`)** and **Telkom Indonesia (`TLKM.JK`)**. Determine the weights maximizing estimated expected return minus a risk penalty. **Real market data**, not simulated prices, are downloaded for **2025-01-01 through 2025-12-31** from Yahoo Finance via `yfinance`. The notebook downloads the adjusted closing price (adjusted for corporate actions) using `auto_adjust=True`.

> Educational use only. Historical returns do not predict future performance. An internet connection is needed to retrieve prices. Market data providers may revise their histories.

## Step 1 — Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/GA-PSO-Stock-Portfolio-Optimization/blob/main/PSO_Two_Real_Stocks_HandsOn.ipynb)

Replace `USERNAME` with your GitHub username or organization. You can also open the `.ipynb` file manually in Colab.

## Step 2 — Install libraries

```python
%pip install yfinance pandas numpy matplotlib
```

No PSO package is required: the class implements PSO from scratch.

## Step 3 — Retrieve actual market prices

```python
import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["BBCA.JK", "TLKM.JK"]
raw = yf.download(tickers, start="2025-01-01", end="2026-01-01",
                  auto_adjust=True, progress=False, threads=False)
prices = raw["Close"].reindex(columns=tickers).dropna()
print(prices.head())
```

Use the **same trading dates** for both stocks. If no data are returned, check network access and the provider's availability. The `end` date is exclusive.

## Step 4 — Estimate return and risk

Daily simple returns:

$$r_{i,t}=\frac{P_{i,t}}{P_{i,t-1}}-1$$

Annualized estimates, assuming 252 trading days:

$$ \boldsymbol{\mu}=252\,\overline{\mathbf r},\qquad \Sigma=252\,\operatorname{Cov}(\mathbf r)$$

```python
returns = prices.pct_change().dropna()
mu = 252 * returns.mean().to_numpy()
Sigma = 252 * returns.cov().to_numpy()
```

## Step 5 — Define the decision variable

Let $x$ be the BBCA weight; TLKM receives $1-x$.

$$\mathbf w=(x,1-x)^T,\qquad 0\le x\le1$$

For example, $x=0.30$ means 30% BBCA and 70% TLKM.

## Step 6 — Define the risk–return objective

$$\boxed{\max_{0\le x\le1} f(x)=\mathbf w^T\boldsymbol{\mu}-\lambda\mathbf w^T\Sigma\mathbf w}$$

Use $\lambda=3$ for the first run. **The numerical objective is data-dependent**: unlike a simulated example, no predetermined optimum is assumed.

```python
risk_penalty = 3.0

def fitness(x):
    w = np.array([x, 1-x])
    return float(w @ mu - risk_penalty * (w @ Sigma @ w))
```

## Step 7 — Initialize particles

Each particle has one position $x_i$ (BBCA weight) and one velocity $v_i$. The particle's TLKM weight is automatically $1-x_i$.

```python
rng = np.random.default_rng(42)
n_particles = 8
x = rng.uniform(0, 1, n_particles)
v = np.zeros(n_particles)
pbest = x.copy()
pbest_scores = np.array([fitness(z) for z in x])
gbest = float(pbest[np.argmax(pbest_scores)])
```

## Step 8 — Update velocity and position iteratively

$$v_i^{t+1}=\omega v_i^t+c_1r_1(pbest_i-x_i^t)+c_2r_2(gbest-x_i^t)$$

$$x_i^{t+1}=\operatorname{clip}(x_i^t+v_i^{t+1},0,1)$$

Use $\omega=0.7$, $c_1=c_2=1.5$, and 60 iterations. **Run the linked notebook** for the full PSO loop including updates to personal best and global best.

## Step 9 — Interpret results

Report the optimal BBCA and TLKM weights, expected annual return, annualized volatility, and best fitness. Plot the best-so-far fitness against iteration. Compare PSO with a 50/50 portfolio and a fine one-dimensional grid search for validation.

## Step 10 — Interactive experiments

Change one setting at a time and rerun the notebook:

1. Risk penalty $\lambda$: 1, 3, 5, 10.
2. Number of particles: 3, 8, 30.
3. Inertia weight $\omega$: 0.3, 0.7, 0.9.

**Discussion:** Why might the optimum allocate 100% to one stock? Why can changing $\lambda$ change the weights? Does a larger swarm always yield a better result?

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook PSO_Two_Real_Stocks_HandsOn.ipynb
```

## Transition to the assignment

This example uses **two real stocks and one decision variable**. The subsequent GA–PSO assignment uses five stocks and a five-dimensional weight vector with a sum-to-one constraint.

## Data source

Yahoo Finance historical market data: [BBCA](https://finance.yahoo.com/quote/BBCA.JK/history/) and [TLKM](https://finance.yahoo.com/quote/TLKM.JK/history/).

