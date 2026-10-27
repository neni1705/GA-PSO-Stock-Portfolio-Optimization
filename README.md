
# GA & PSO Stock Portfolio Optimization — Student Mini Project

Mini project for **Metode Optimasi Metaheuristik (SD234503)**. Students use Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) to optimize the allocation of five assets under a risk–return objective.

> **Educational use only.** `simulated_stock_prices.csv` contains simulated stock prices, not historical market quotations. Results are not investment advice.

## Files

| File | Purpose |
| --- | --- |
| `assignment.md` | Case study, instructions, constraints and grading criteria |
| `student_template_GA_PSO.ipynb` | Guided Jupyter Notebook with exercises/TODOs |
| `simulated_stock_prices.csv` | Simulated input data |
| `requirements.txt` | Python dependencies |

## Quick start (Google Colab)

1. Open `student_template_GA_PSO.ipynb` in Google Colab (from GitHub after the repository is published).
2. Upload `simulated_stock_prices.csv` to the Colab session's working directory.
3. Run cells in order. The notebook includes a `%pip install` cell.
4. Complete the `TODO` sections and submit the notebook and a brief report as described in `assignment.md`.

## Optimization model

Maximize `w.T @ mu - lambda_ * (w.T @ Sigma @ w)`, subject to `sum(w) = 1` and `0 <= w_i <= 1`. The starter configuration uses `lambda = 3` and annualised return/covariance estimates. The project compares GA and PSO with an equal-weight baseline.

