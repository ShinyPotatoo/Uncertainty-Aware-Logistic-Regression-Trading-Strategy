import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

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
    df = df.copy()
    df.columns = df.columns.get_level_values(0)

    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)

    returns = np.diff(np.log(close))

    rsi = compute_rsi(close)
    ema20 = pd.Series(close).ewm(span=20).mean().values
    vol = pd.Series(returns).rolling(20).std().values

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

    prices = close[-len(X[split:]) - 1:]
    vol_test = vol[-len(X[split:]):]

    return (
        X[:split], y[:split],
        X[split:], y[split:],
        prices, vol_test
    )

# =====================================================
# Bayesian-style Ensemble + Calibration
# =====================================================

def train_ensemble(X, y, n_models=10):
    models = []

    for _ in range(n_models):
        X_bs, y_bs = resample(X, y)

        base = LogisticRegression(max_iter=1000)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X_bs, y_bs)

        models.append(model)

    return models


def ensemble_predict(models, X):
    probs = np.array([
        m.predict_proba(X)[:, 1]
        for m in models
    ])

    mean_prob = probs.mean(axis=0)
    std_prob = probs.std(axis=0)

    return mean_prob, std_prob

# =====================================================
# Backtest (scaled Bayesian-style positions)
# =====================================================

def backtest(
    probs,
    uncertainty,
    prices,
    vol,
    cost=0.0005
):
    returns = np.diff(np.log(prices))

    # --- Volatility regime prior (relaxed) ---
    vol_threshold = np.nanpercentile(vol, 75)  # relaxed from 70%
    regime = (vol < vol_threshold).astype(float)

    # --- Bayesian position sizing ---
    edge = probs - 0.5
    confidence = edge / (uncertainty + 0.001)  # slightly more confident

    position_size = np.clip(confidence * 4, 0, 0.8)  # scale exposure
    position_size *= regime[-len(position_size):]

    # Strategy returns using yesterday's position
    strat_returns = position_size[:-1] * returns[1:]

    trades = np.abs(np.diff(position_size))
    strat_returns -= trades * cost

    equity = np.exp(np.cumsum(strat_returns))

    return strat_returns, equity, position_size

# =====================================================
# Performance metrics
# =====================================================

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
    df = yf.download(
        "NVDA",
        start="2019-01-01",
        end="2025-12-29",
        auto_adjust=True
    )

    X_train, y_train, X_test, y_test, prices, vol = prepare_data(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = train_ensemble(X_train, y_train, n_models=10)

    probs, uncertainty = ensemble_predict(models, X_test)

    returns, equity, positions = backtest(
        probs, uncertainty, prices, vol
    )

    sharpe, total_ret, dd = performance(returns)

    print("\n===== STRATEGY RESULTS =====")
    print(f"Sharpe Ratio     : {sharpe:.2f}")
    print(f"Total Return     : {total_ret:.2%}")
    print(f"Max Drawdown     : {dd:.2%}")
    print(f"Avg Exposure     : {positions.mean():.2%}")
    print(f"Trades           : {np.sum(np.abs(np.diff(positions)) > 0)}")


if __name__ == "__main__":
    main()
