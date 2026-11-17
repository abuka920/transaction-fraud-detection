"""Synthetic credit card transaction generator — ~100k transactions, ~0.5% fraud."""
import numpy as np
import pandas as pd
import hashlib
import time
from pathlib import Path

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "travel", "entertainment", "healthcare", "atm", "luxury_goods", "other"
]
MERCHANT_RISK = {
    "grocery": 0.05, "restaurant": 0.08, "gas_station": 0.15,
    "online_retail": 0.25, "travel": 0.30, "entertainment": 0.12,
    "healthcare": 0.04, "atm": 0.35, "luxury_goods": 0.40, "other": 0.18,
}


def _txn_id(seed: int) -> str:
    return f"TXN-{hashlib.md5(str(seed).encode()).hexdigest()[:8].upper()}"


def generate_transactions(n_total: int = 100_000, fraud_rate: float = 0.005,
                           random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    n_fraud = int(n_total * fraud_rate)
    n_legit = n_total - n_fraud
    base_time = int(time.time()) - 90 * 86400

    def make_legit(n):
        timestamps = base_time + rng.integers(0, 90 * 86400, n)
        traveller = rng.random(n) < 0.05
        amounts = np.where(
            rng.random(n) < 0.6,
            rng.lognormal(3.5, 0.8, n),
            rng.lognormal(5.0, 1.0, n),
        ).clip(1, 5000)
        cats = rng.choice(MERCHANT_CATEGORIES, n, p=[0.25,0.20,0.12,0.15,0.08,0.07,0.05,0.03,0.02,0.03])
        return pd.DataFrame({
            "timestamp": timestamps,
            "amount": np.round(amounts, 2),
            "merchant_category": cats,
            "hour_of_day": (timestamps // 3600) % 24,
            "day_of_week": (timestamps // 86400) % 7,
            # ~5% of customers are travelling: far from home and often abroad
            "distance_from_home_km": np.round(np.where(
                traveller, rng.exponential(800, n), rng.exponential(15, n)).clip(0, 10000), 1),
            "is_foreign_transaction": (rng.random(n) < np.where(traveller, 0.6, 0.02)).astype(int),
            "card_present": (rng.random(n) < 0.72).astype(int),
            "velocity_1h": rng.poisson(1.8, n).clip(0, 12),
            "merchant_risk_score": np.array([MERCHANT_RISK[c] for c in cats]),
            "is_fraud": 0,
        })

    def make_fraud(n):
        """Two kinds of fraud, like real card fraud:
        - obvious (~65%): card-not-present, foreign, late night, high velocity
        - stealthy (~35%): domestic, normal hours, modest amounts, so it blends in
        """
        timestamps = base_time + rng.integers(0, 90 * 86400, n)
        stealthy = rng.random(n) < 0.35
        night_hours = rng.choice(list(range(0, 6)) + list(range(22, 24)), n)
        day_hours = (timestamps // 3600) % 24
        hours = np.where(stealthy & (rng.random(n) < 0.7), day_hours, night_hours)
        # Keep the timestamp consistent with hour_of_day
        timestamps = (timestamps // 86400) * 86400 + hours * 3600 + rng.integers(0, 3600, n)
        amounts = np.where(
            stealthy,
            rng.lognormal(4.3, 1.0, n),          # close to normal spending
            rng.lognormal(6.0, 1.0, n),          # big-ticket cash-out
        ).clip(5, 15000)
        cats = rng.choice(MERCHANT_CATEGORIES, n, p=[0.08,0.08,0.10,0.25,0.15,0.08,0.03,0.10,0.10,0.03])
        foreign_p = np.where(stealthy, 0.08, 0.55)
        card_present_p = np.where(stealthy, 0.45, 0.10)
        return pd.DataFrame({
            "timestamp": timestamps,
            "amount": np.round(amounts, 2),
            "merchant_category": cats,
            "hour_of_day": hours,
            "day_of_week": (timestamps // 86400) % 7,
            "distance_from_home_km": np.round(
                np.where(stealthy, rng.exponential(40, n), rng.exponential(350, n)).clip(0, 10000), 1),
            "is_foreign_transaction": (rng.random(n) < foreign_p).astype(int),
            "card_present": (rng.random(n) < card_present_p).astype(int),
            "velocity_1h": np.where(stealthy, rng.poisson(3, n), rng.poisson(5, n)).clip(0, 20),
            "merchant_risk_score": np.array([MERCHANT_RISK[c] for c in cats]),
            "is_fraud": 1,
        })

    df = pd.concat([make_legit(n_legit), make_fraud(n_fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df["amount_z_score"] = np.round((df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-8), 4)
    df.insert(0, "transaction_id", [_txn_id(i) for i in range(len(df))])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "sample_data" / "sample_transactions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_transactions()
    df.to_csv(out, index=False)
    n_fraud = df["is_fraud"].sum()
    print(f"Generated {len(df):,} transactions — {n_fraud} fraud ({n_fraud/len(df)*100:.2f}%) → {out}")
