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

## Step 4 — Estimate Return and Risk

Daily simple returns:

$$
r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1
$$

where:
- $P_{i,t}$ is the adjusted closing price of stock $i$ at time $t$.
- $r_{i,t}$ is the daily return of stock $i$.

### Annualized Expected Return

Assuming 252 trading days per year:

$$
\boldsymbol{\mu} = 252 \times \overline{\mathbf{r}}
$$

where:
- $\boldsymbol{\mu}$ is the vector of annualized expected returns.
- $\overline{\mathbf{r}}$ is the vector of average daily returns.

### Annualized Covariance Matrix

$$
\Sigma = 252 \times \mathrm{Cov}(\mathbf{r})
$$

where:
- $\Sigma$ is the annualized covariance matrix.
- $\mathrm{Cov}(\mathbf{r})$ is the covariance matrix of daily stock returns.

### Python Implementation

```python
# Calculate daily returns
returns = prices.pct_change().dropna()

# Annualized expected returns
mu = 252 * returns.mean().to_numpy()

# Annualized covariance matrix
Sigma = 252 * returns.cov().to_numpy()

print("Annualized Expected Returns:")
print(mu)

print("\nAnnualized Covariance Matrix:")
print(Sigma)
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


## Step 8 — Update Velocity and Position Iteratively

At each iteration, particles update their velocities based on three components: inertia, cognitive influence, and social influence.

### 1. Velocity Update

At each iteration, the velocity of particle $i$ is updated as follows:

$$
v_i^{t+1} = \omega v_i^t + c_1 r_1 (p_{best,i} - x_i^t) + c_2 r_2 (g_{best} - x_i^t)
$$

The velocity update consists of three components:

- **Inertia:** $\omega v_i^t$
- **Cognitive component:** $c_1 r_1 (p_{best,i} - x_i^t)$
- **Social component:** $c_2 r_2 (g_{best} - x_i^t)$

where:

- $v_i^t$: Current velocity of particle $i$.
- $x_i^t$: Current position of particle $i$.
- $\omega$: Inertia weight.
- $c_1$: Cognitive coefficient.
- $c_2$: Social coefficient.
- $r_1, r_2$: Random numbers between 0 and 1.
- $p_{best,i}$: Personal best position of particle $i$.
- $g_{best}$: Global best position of the swarm.

### 2. Position Update

After updating the velocity, calculate the new position:

$$
x_i^{t+1} = x_i^t + v_i^{t+1}
$$

Since the portfolio weight must remain between 0 and 1,
apply a clipping operation:

$$
x_i^{t+1} = \min\left(1,\max\left(0,x_i^t+v_i^{t+1}\right)\right)
$$

This ensures that:

$$
0 \leq x_i^{t+1} \leq 1
$$

### 3. PSO Parameters

For this hands-on exercise, use:

| Parameter | Value |
|---|---|
| Inertia weight ($\omega$) | 0.7 |
| Cognitive coefficient ($c_1$) | 1.5 |
| Social coefficient ($c_2$) | 1.5 |
| Number of iterations | 60 |

### 4. Python Implementation

```python
# Update velocity
r1 = np.random.rand(n_particles)
r2 = np.random.rand(n_particles)

velocity = (
    0.7 * velocity
    + 1.5 * r1 * (pbest - position)
    + 1.5 * r2 * (gbest - position)
)

# Update position
position = np.clip(position + velocity, 0, 1)
```

### 5. Update Personal Best and Global Best

After updating the positions:

1. Evaluate the fitness of each particle.
2. Update personal best if the new fitness is better.
3. Update global best if any particle finds a better solution.
4. Repeat until 60 iterations are completed.

**Run the linked notebook** for the complete PSO implementation.


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
