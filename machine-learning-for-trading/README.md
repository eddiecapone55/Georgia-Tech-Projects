# CS7646 — Strategy Evaluation (Final Project)

Manual rules vs. **Q-learning StrategyLearner** on **JPM** with classic technical indicators. Includes in-sample vs. out-of-sample evaluation and market impact sensitivity.

## Overview
- **Course:** Georgia Tech CS7646 (Machine Learning for Trading)
- **Goal:** Compare a hand-crafted **ManualStrategy** to a learned **StrategyLearner** (Q-learning).
- **Indicators:** Bollinger %B, MACD, RSI, Momentum.
- **Metrics:** Cumulative return, daily volatility (std), Sharpe ratio (risk-free = 0.0).

## Data
- **Symbol:** JPM (JPMorgan Chase)
- **In-sample:** 2008-01-01 → 2009-12-31  
- **Out-of-sample:** 2010-01-01 → 2011-12-31
- Prices pulled from the ML4T data utility; indicators are computed from the price series.

## Methods
- **ManualStrategy:** Rule-based signals derived from indicators.
- **StrategyLearner:** Q-learning policy over indicator-derived states.
- **Sensitivity:** Performance under varying **market impact**.

## How to Run
> Replace the filenames below with yours if different.

```bash
# Set up the ML4T conda env (Python 3.10; no pandas usage in code)
conda activate ml4t

# Generate trades & evaluate (manual vs. learner)
python <script_to_train_learner>.py
python <script_to_eval_manual>.py
python <script_to_eval_learner>.py
