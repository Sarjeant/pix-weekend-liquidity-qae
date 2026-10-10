# data.py - turns the Central Bank of Brazil's Conta PI balances into a weekend drawdown distribution.
# Source: Banco Central do Brasil, Estatisticas do SPI (PixRemuneracaoContaPI), Portal de Dados Abertos, ODbL.
import numpy as np
import pandas as pd

BALANCE_FILE = "PixRemuneracaoContaPI.csv"
BALANCE_COLUMN = "BaseRemuneracaoTotal"  # total remunerated Conta PI balance, reais
START_DATE = "2025-01-01"


def load_balances(path=BALANCE_FILE, start=START_DATE):
    """Return a date-indexed Series of total Conta PI balances (reais), oldest first, from `start`."""
    df = pd.read_csv(path, decimal=",")
    df["DataBase"] = pd.to_datetime(df["DataBase"])
    s = df.set_index("DataBase")[BALANCE_COLUMN].astype(float).sort_index()
    return s[s.index >= pd.Timestamp(start)]


def weekend_drawdowns(balances):
    """Return a DataFrame of Friday/Monday pairs and drawdown = 1 - Monday/Friday.

    A pair counts only when the next business day in the data is exactly 3 days
    after the Friday, so holiday weekends are left out rather than stretched.
    """
    rows = []
    dates = balances.index
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        if d.weekday() == 4 and (nxt - d).days == 3:
            fri, mon = balances.iloc[i], balances.iloc[i + 1]
            rows.append({"friday": d.date(), "monday": nxt.date(),
                         "friday_balance": fri, "monday_balance": mon,
                         "drawdown": 1.0 - mon / fri})
    return pd.DataFrame(rows)


def bin_probabilities(drawdowns, buffer=0.10):
    """Return (four bin probabilities, breach probability) for a 2-qubit encoding.

    Bins: below buffer/2, buffer/2 to buffer, buffer to 1.5*buffer, above 1.5*buffer.
    Bins 2 and 3 have the top qubit set, so the top qubit means 'breached'.
    """
    x = np.asarray(drawdowns, dtype=float)
    edges = [-np.inf, buffer / 2, buffer, 1.5 * buffer, np.inf]
    counts, _ = np.histogram(x, bins=edges)
    probs = counts / counts.sum()
    return probs, float(probs[2] + probs[3])


def summary(buffer=0.10, path=BALANCE_FILE):
    """Return a dict with the weekend statistics used on the project page."""
    w = weekend_drawdowns(load_balances(path))
    dd = w["drawdown"].to_numpy()
    probs, p_breach = bin_probabilities(dd, buffer)
    return {
        "weekends": int(len(dd)),
        "first_friday": str(w["friday"].iloc[0]),
        "last_friday": str(w["friday"].iloc[-1]),
        "mean_drawdown": round(float(dd.mean()), 4),
        "p95_drawdown": round(float(np.percentile(dd, 95)), 4),
        "max_drawdown": round(float(dd.max()), 4),
        "max_drawdown_friday": str(w.loc[w["drawdown"].idxmax(), "friday"]),
        "buffer": buffer,
        "bin_probs": [round(float(p), 4) for p in probs],
        "p_breach": round(p_breach, 4),
    }