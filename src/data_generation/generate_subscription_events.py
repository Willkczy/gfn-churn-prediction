import numpy as np
import pandas as pd

# --- Persona-based event probabilities ---
# How likely each persona is to have subscription events during the 12-week window
# Format: {event_type: probability_of_at_least_one_event}

PERSONA_EVENT_PROBS = {
    "hardcore": {
        "upgrade": 0.05,    # already on high tier, rarely upgrades
        "downgrade": 0.02,
        "cancel": 0.01,
        "renew": 0.80,      # loyal, high renew rate
    },
    "regular": {
        "upgrade": 0.15,
        "downgrade": 0.10,
        "cancel": 0.05,
        "renew": 0.60,
    },
    "casual": {
        "upgrade": 0.10,    # occasionally tries higher tier
        "downgrade": 0.15,  # then downgrades back
        "cancel": 0.10,
        "renew": 0.40,
    },
    "about_to_churn": {
        "upgrade": 0.03,
        "downgrade": 0.35,  # strong signal
        "cancel": 0.40,     # many will cancel
        "renew": 0.15,
    },
}

TIER_ORDER = ["free", "priority", "ultimate"]

def _get_upgrade_tier(current_tier: str, rng: np.random.Generator) -> str | None:
    """Return a tier higher than current, or None if already at max."""
    idx = TIER_ORDER.index(current_tier)
    if idx >= len(TIER_ORDER) - 1:
        return None
    # Could jump one or two tiers up
    return rng.choice(TIER_ORDER[idx + 1:])


def _get_downgrade_tier(current_tier: str, rng: np.random.Generator) -> str | None:
    """Return a tier lower than current, or None if already at min."""
    idx = TIER_ORDER.index(current_tier)
    if idx <= 0:
        return None
    return rng.choice(TIER_ORDER[:idx])


def generate_subscription_events(
    users_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    obs_start = pd.Timestamp("2024-01-01")
    obs_end = pd.Timestamp("2024-03-25")
    obs_days = (obs_end - obs_start).days

    all_events = []

    for _, row in users_df.iterrows():
        user_id = row["user_id"]
        persona = row["persona"]
        current_tier = row["subscription_tier"]
        probs = PERSONA_EVENT_PROBS[persona]

        # --- Renew ---
        if rng.random() < probs["renew"]:
            event_date = obs_start + pd.Timedelta(days=int(rng.integers(0, obs_days)))
            all_events.append({
                "user_id": user_id,
                "event_date": event_date,
                "event_type": "renew",
                "from_tier": current_tier,
                "to_tier": current_tier,
            })

        # --- Upgrade ---
        if rng.random() < probs["upgrade"]:
            new_tier = _get_upgrade_tier(current_tier, rng)
            if new_tier:
                event_date = obs_start + pd.Timedelta(days=int(rng.integers(0, obs_days)))
                all_events.append({
                    "user_id": user_id,
                    "event_date": event_date,
                    "event_type": "upgrade",
                    "from_tier": current_tier,
                    "to_tier": new_tier,
                })
                current_tier = new_tier  # update for subsequent events

        # --- Downgrade ---
        if rng.random() < probs["downgrade"]:
            new_tier = _get_downgrade_tier(current_tier, rng)
            if new_tier:
                # Downgrade tends to happen later in the window
                event_date = obs_start + pd.Timedelta(days=int(rng.integers(obs_days // 2, obs_days)))
                all_events.append({
                    "user_id": user_id,
                    "event_date": event_date,
                    "event_type": "downgrade",
                    "from_tier": current_tier,
                    "to_tier": new_tier,
                })
                current_tier = new_tier

        # --- Cancel ---
        if rng.random() < probs["cancel"]:
            # Cancel tends to happen in the last few weeks
            event_date = obs_start + pd.Timedelta(days=int(rng.integers(obs_days * 2 // 3, obs_days)))
            all_events.append({
                "user_id": user_id,
                "event_date": event_date,
                "event_type": "cancel",
                "from_tier": current_tier,
                "to_tier": None,
            })

    df = pd.DataFrame(all_events)

    # Add event_id
    df.insert(0, "event_id", [f"SE{i:06d}" for i in range(1, len(df) + 1)])

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df = df.sort_values("event_date").reset_index(drop=True)

    return df

if __name__ == "__main__":
    import os

    output_dir = os.path.join("data", "raw")

    users_df = pd.read_parquet(os.path.join(output_dir, "users.parquet"))
    print(f"Loaded {len(users_df)} users")
    print("Generating subscription events...")

    df = generate_subscription_events(users_df)

    output_path = os.path.join(output_dir, "subscription_events.parquet")
    df.to_parquet(output_path, index=False)

    print(f"\nGenerated {len(df):,} events → {output_path}")
    print("\nEvent type distribution:")
    print(df["event_type"].value_counts())
    print("\nEvents per persona:")
    merged = df.merge(users_df[["user_id", "persona"]], on="user_id")
    print(pd.crosstab(merged["persona"], merged["event_type"]))
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))