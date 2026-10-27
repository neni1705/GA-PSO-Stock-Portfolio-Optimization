# MINI PROJECT — Portfolio Risk–Return Optimization Using GA and PSO

**Course:** Metode Optimasi Metaheuristik (SD234503)  
**Type:** Group assignment (2–3 students)  
**Tools:** Python / Google Colab  
**Level:** Beginner–Intermediate

## Case study
An investor wishes to allocate capital across five stocks (BBCA, BBRI, TLKM, ASII, UNTR). Each stock has different expected returns and risk. Your task is to determine portfolio allocation weights using **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)**, then compare their solutions. Use the supplied **simulated** closing-price dataset. Tickers are illustrative; prices are not actual market observations or investment recommendations.

## Learning objectives
1. Represent a portfolio as a continuous-valued chromosome (GA) or particle position (PSO).
2. Formulate and implement a risk–return objective function with feasible portfolio weights.
3. Apply GA operators (selection, crossover, mutation) and PSO updates (velocity, position, personal best, global best).
4. Interpret and compare optimization outcomes and convergence.

## Optimization problem
Let w be a vector of five weights, mu the annualized mean-return vector, and Sigma the annualized covariance matrix. Maximize

**f(w) = wᵀ mu − λ wᵀ Sigma w**

subject to **sum(w) = 1** and **0 ≤ wᵢ ≤ 1**. Set **λ = 3** for the main comparison. Use annualized simple daily returns: mu = 252 × mean(daily returns), Sigma = 252 × covariance(daily returns). Here lambda is a chosen trade-off coefficient, not a universally optimal investor preference.

## Required tasks (100 points)

### Task 1 — Data and formulation (15 points)
- Load prices, inspect dimensions and missing values, compute simple daily returns.
- Estimate annualized mean returns and covariance matrix.
- Explain the role of weights, risk, expected return and λ.

### Task 2 — GA implementation (25 points)
- Implement GA using PyGAD and a real-valued chromosome of length five.
- Implement fitness evaluation and feasible-weight handling.
- Report population size, generations, selection, crossover, mutation, seed and stopping criterion.
- Present the best portfolio and its fitness/convergence curve.

### Task 3 — PSO implementation (25 points)
- Implement global-best PSO using PySwarms and a five-dimensional particle.
- Implement the same objective and feasible-weight handling as GA.
- Report swarm size, iterations, inertia, cognitive/social coefficients and seed.
- Present the best portfolio and its objective/convergence curve.

### Task 4 — Comparison (25 points)
- Compare GA, PSO and an equal-weight (20% each) baseline on annualized expected return, annualized volatility, variance, objective score and computation time.
- Plot allocation weights and convergence curves. Explain that GA fitness is maximized while PSO cost is minimized; use a consistent sign when comparing.
- Repeat each method with **three random seeds** and discuss result stability (minimum, mean, standard deviation of objective score). For this part, you may wrap your completed algorithms in functions.

### Task 5 — Reflection (10 points)
- Explain the effects of selection/crossover/mutation vs pbest/gbest/velocity updates.
- Re-run with λ = 1 and λ = 8 and describe how the risk–return balance changes.
- State whether the algorithm that performs best in one run is necessarily always better, and why.

## Submission
Submit (1) a runnable .ipynb notebook with all TODO sections completed, (2) a short PDF report (3–5 pages) with methodology, results, charts, discussion, and (3) a short declaration of any external or AI assistance. Include dataset filename and random seeds. Do not claim guaranteed global optimality.

## Fair comparison
Use identical data, objective function, constraints and evaluation metrics for both methods. The default budget is 30 candidate solutions × 100 generations/iterations for each algorithm. This is an approximate comparison of fitness-evaluation budgets, not a claim of identical computational cost.
