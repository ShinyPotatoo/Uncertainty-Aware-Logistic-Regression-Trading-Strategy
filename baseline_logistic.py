import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =====================================================
# Feature Engineering
# =====================================================

def compute_rsi(prices, period=14):
    prices = np.asarray(prices, dtype=float).reshape(-1)
    delta = np.diff(prices)

    up = np.maximum(delta, 0)
    down = -np.minimum(delta, 0)

    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()

    rs = roll_up / (roll_down + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return np.concatenate([[np.nan], rsi.values])

def prepare_data(df):
    # --- Ensure flat columns (yfinance safety) ---
    df = df.copy()
    df.columns = df.columns.get_level_values(0)

    close = df["Close"].to_numpy(dtype=float).reshape(-1)
    volume = df["Volume"].to_numpy(dtype=float).reshape(-1)

    returns = np.diff(np.log(close))

    rsi = compute_rsi(close)
    ema20 = pd.Series(close).ewm(span=20).mean().values
    vol = pd.Series(returns).rolling(20).std().values

    # Target: meaningful next-day move
    future_return = np.roll(returns, -1)
    target = (future_return > 0.002).astype(int)

    features = np.column_stack([
        returns,
        volume[1:],
        rsi[1:],
        ema20[1:],
        vol
    ])

    valid = ~np.isnan(features).any(axis=1)
    X = features[valid]
    y = target[valid]

    split = int(len(X) * 0.8)

    # Prices aligned to test set
    prices = close[-len(X[split:]) - 1:]

    return X[:split], y[:split], X[split:], y[split:], prices

# =====================================================
# Backtest + Metrics
# =====================================================
# change threshold
def backtest(model, X_test, prices, threshold=0.52, cost=0.0005):
    probs = model.predict_proba(X_test)[:, 1]
    positions = (probs > threshold).astype(int)

    returns = np.diff(np.log(prices))

    # Strategy returns (yesterday's position)
    strat_returns = positions[:-1] * returns[1:]

    # Transaction costs on position changes
    trades = np.abs(np.diff(positions))
    strat_returns -= trades * cost

    equity = np.exp(np.cumsum(strat_returns))
    return strat_returns, equity, positions

def performance(returns):
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    total_return = np.exp(np.sum(returns)) - 1
    max_dd = np.max(
        np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns)
    )

    return sharpe, total_return, max_dd

# =====================================================
# Main Pipeline
# =====================================================

def main():
    df = yf.download("NVDA", start="2019-01-01", end = "2025-12-31", auto_adjust=True)

    X_train, y_train, X_test, y_test, prices = prepare_data(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    returns, equity, positions = backtest(model, X_test, prices)

    sharpe, total_ret, dd = performance(returns)

    print("\n===== STRATEGY RESULTS =====")
    print(f"Sharpe Ratio     : {sharpe:.2f}")
    print(f"Total Return    : {total_ret:.2%}")
    print(f"Max Drawdown    : {dd:.2%}")
    print(f"Time in Market  : {positions.mean():.2%}")

if __name__ == "__main__":
    main()
