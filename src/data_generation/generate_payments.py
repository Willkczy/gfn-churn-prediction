import numpy as np
import pandas as pd

# --- Subscription pricing ---
TIER_MONTHLY_PRICE = {
    "free": 0.0,
    "priority": 9.99,
    "ultimate": 19.99,
}

DAY_PASS_PRICE = 3.99

# --- Payment type probabilities by persona ---
# subscription: monthly auto-pay (only for paid tiers)
# day_pass: one-off purchase (common for free tier users)
# in_app: hypothetical small purchases

PERSONA_PAYMENT_BEHAVIOR = {
    "hardcore": {
        "day_pass_per_month": (0.2, 0.1),    # rarely needs day pass (already subscribed)
        "in_app_per_month": (1.5, 0.5),       # buys extras often
        "failed_prob": 0.02,                   # very low failure rate
        "refund_prob": 0.01,
    },
    "regular": {
        "day_pass_per_month": (0.5, 0.3),
        "in_app_per_month": (0.8, 0.4),
        "failed_prob": 0.03,
        "refund_prob": 0.02,
    },
    "casual": {
        "day_pass_per_month": (1.0, 0.5),     # free tier users buy day passes
        "in_app_per_month": (0.3, 0.2),
        "failed_prob": 0.05,
        "refund_prob": 0.03,
    },
    "about_to_churn": {
        "day_pass_per_month": (0.3, 0.2),
        "in_app_per_month": (0.2, 0.2),       # spending drops
        "failed_prob": 0.10,                   # higher failure rate (expired card, don't bother updating)
        "refund_prob": 0.08,                   # more refund requests
    },
}

IN_APP_AMOUNTS = [0.99, 1.99, 2.99, 4.99, 9.99]
IN_APP_WEIGHTS = [0.30, 0.30, 0.20, 0.15, 0.05]

PAYMENT_METHODS = ["credit_card", "paypal", "gift_card"]
PAYMENT_METHOD_WEIGHTS = [0.60, 0.25, 0.15]


def _assign_status(rng: np.random.Generator, failed_prob: float, refund_prob: float) -> str:
    """Determine payment status: success, failed, or refunded."""
    roll = rng.random()
    if roll < failed_prob:
        return "failed"
    elif roll < failed_prob + refund_prob:
        return "refunded"
    return "success"


def generate_payments(
    users_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    obs_start = pd.Timestamp("2024-01-01")
    obs_months = 3  # ~12 weeks = 3 months

    all_payments = []

    for _, row in users_df.iterrows():
        user_id = row["user_id"]
        persona = row["persona"]
        tier = row["subscription_tier"]
        behavior = PERSONA_PAYMENT_BEHAVIOR[persona]

        payment_method = rng.choice(PAYMENT_METHODS, p=PAYMENT_METHOD_WEIGHTS)

        # --- 1. Monthly subscription payments (paid tiers only) ---
        if tier != "free":
            monthly_price = TIER_MONTHLY_PRICE[tier]
            for month_offset in range(obs_months):
                # Payment around the 1st~5th of each month
                pay_day = int(rng.integers(1, 6))
                payment_date = obs_start + pd.DateOffset(months=month_offset) + pd.Timedelta(days=pay_day)

                status = _assign_status(rng, behavior["failed_prob"], behavior["refund_prob"])

                all_payments.append({
                    "user_id": user_id,
                    "payment_date": payment_date,
                    "amount_usd": monthly_price,
                    "payment_type": "subscription",
                    "payment_method": payment_method,
                    "status": status,
                })

                # about_to_churn: skip 3rd month payment with 50% chance (stopped caring)
                if persona == "about_to_churn" and month_offset == 2 and rng.random() < 0.5:
                    break

        # --- 2. Day pass purchases ---
        total_day_passes = max(0, int(rng.normal(
            behavior["day_pass_per_month"][0] * obs_months,
            behavior["day_pass_per_month"][1] * obs_months,
        )))
        for _ in range(total_day_passes):
            days_offset = int(rng.integers(0, 84))  # 12 weeks = 84 days
            payment_date = obs_start + pd.Timedelta(days=days_offset)

            all_payments.append({
                "user_id": user_id,
                "payment_date": payment_date,
                "amount_usd": DAY_PASS_PRICE,
                "payment_type": "day_pass",
                "payment_method": payment_method,
                "status": _assign_status(rng, behavior["failed_prob"], behavior["refund_prob"]),
            })

        # --- 3. In-app purchases ---
        total_in_app = max(0, int(rng.normal(
            behavior["in_app_per_month"][0] * obs_months,
            behavior["in_app_per_month"][1] * obs_months,
        )))
        for _ in range(total_in_app):
            days_offset = int(rng.integers(0, 84))
            payment_date = obs_start + pd.Timedelta(days=days_offset)
            amount = rng.choice(IN_APP_AMOUNTS, p=IN_APP_WEIGHTS)

            all_payments.append({
                "user_id": user_id,
                "payment_date": payment_date,
                "amount_usd": amount,
                "payment_type": "in_app",
                "payment_method": payment_method,
                "status": _assign_status(rng, behavior["failed_prob"], behavior["refund_prob"]),
            })

    df = pd.DataFrame(all_payments)

    # Add payment_id
    df.insert(0, "payment_id", [f"PAY{i:07d}" for i in range(1, len(df) + 1)])

    df["payment_date"] = pd.to_datetime(df["payment_date"]).dt.date
    df = df.sort_values("payment_date").reset_index(drop=True)

    return df

if __name__ == "__main__":
    import os

    output_dir = os.path.join("data", "raw")

    users_df = pd.read_parquet(os.path.join(output_dir, "users.parquet"))
    print(f"Loaded {len(users_df)} users")
    print("Generating payments...")

    df = generate_payments(users_df)

    output_path = os.path.join(output_dir, "payments.parquet")
    df.to_parquet(output_path, index=False)

    print(f"\nGenerated {len(df):,} payments → {output_path}")
    print(f"\nPayment type distribution:")
    print(df["payment_type"].value_counts())
    print(f"\nStatus distribution:")
    print(df["status"].value_counts(normalize=True).round(3))
    print(f"\nStatus by persona:")
    merged = df.merge(users_df[["user_id", "persona"]], on="user_id")
    print(pd.crosstab(merged["persona"], merged["status"], normalize="index").round(3))
    print(f"\nAvg spend per persona:")
    success_only = merged[merged["status"] == "success"]
    print(success_only.groupby("persona")["amount_usd"].sum() / users_df["persona"].value_counts())
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))