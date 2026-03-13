import numpy as np
import pandas as pd

# --- Scale config ---
SCALE = {
    "small": 50_000,
    "full": 200_000,
}

# --- Persona distribution ---
PERSONAS = ["hardcore", "regular", "casual", "about_to_churn"]
PERSONA_WEIGHTS = [0.20, 0.35, 0.30, 0.15]

# --- Subscription tier by persona ---
# Each persona has different probability of being on each tier
TIER_PROBS_BY_PERSONA = {
    "hardcore":       {"free": 0.05, "priority": 0.35, "ultimate": 0.60},
    "regular":        {"free": 0.20, "priority": 0.55, "ultimate": 0.25},
    "casual":         {"free": 0.70, "priority": 0.25, "ultimate": 0.05},
    "about_to_churn": {"free": 0.50, "priority": 0.35, "ultimate": 0.15},
}

DEVICE_TYPES = ["PC", "Mac", "Chromebook", "SHIELD", "Mobile"]
DEVICE_WEIGHTS = [0.45, 0.15, 0.10, 0.10, 0.20]

REGIONS = [
    "Taipei", "New_Taipei", "Taoyuan", "Taichung", "Tainan", "Kaohsiung",
    "Hsinchu_City", "Hsinchu_County", "Miaoli", "Changhua", "Nantou",
    "Yunlin", "Chiayi_City", "Chiayi_County", "Pingtung", "Yilan",
    "Hualien", "Taitung", "Keelung", "Penghu", "Kinmen", "Lienchiang",
]

# Weights roughly reflect population distribution
REGION_WEIGHTS = [
    0.11, 0.17, 0.10, 0.12, 0.08, 0.12,  # six major cities
    0.03, 0.02, 0.02, 0.05, 0.02,          # Hsinchu~Nantou
    0.03, 0.01, 0.02, 0.03, 0.02,          # Yunlin~Yilan
    0.01, 0.01, 0.02, 0.005, 0.003, 0.002, # East + islands
]

REFERRAL_SOURCES = ["organic", "ad", "friend_referral", "bundled"]
REFERRAL_WEIGHTS = [0.40, 0.25, 0.20, 0.15]

AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"]
AGE_WEIGHTS = [0.30, 0.35, 0.20, 0.15]

GENDERS = ["M", "F", "other", "undisclosed"]
GENDER_WEIGHTS = [0.55, 0.30, 0.05, 0.10]


def generate_users(scale: str = "small", seed: int = 42) -> pd.DataFrame:
    num_users = SCALE[scale]
    rng = np.random.default_rng(seed)

    user_ids = [f"U{i:06d}" for i in range(1, num_users + 1)]

    # Assign persona (controls downstream generation, not a model feature)
    personas = rng.choice(PERSONAS, size=num_users, p=PERSONA_WEIGHTS).tolist()

    # Signup date: spread over 2023-01-01 ~ 2023-12-31
    signup_start = pd.Timestamp("2023-01-01")
    signup_days = rng.integers(0, 365, size=num_users)
    signup_dates = [signup_start + pd.Timedelta(days=int(d)) for d in signup_days]

    # Subscription tier: depends on persona
    tiers = []
    for persona in personas:
        probs = TIER_PROBS_BY_PERSONA[persona]
        tier_choices = list(probs.keys())
        tier_weights = list(probs.values())
        tier = rng.choice(tier_choices, p=tier_weights)
        tiers.append(tier)

    device_types = rng.choice(DEVICE_TYPES, size=num_users, p=DEVICE_WEIGHTS).tolist()
    regions = rng.choice(REGIONS, size=num_users, p=REGION_WEIGHTS).tolist()
    referral_sources = rng.choice(REFERRAL_SOURCES, size=num_users, p=REFERRAL_WEIGHTS).tolist()
    age_groups = rng.choice(AGE_GROUPS, size=num_users, p=AGE_WEIGHTS).tolist()
    genders = rng.choice(GENDERS, size=num_users, p=GENDER_WEIGHTS).tolist()

    df = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": signup_dates,
        "subscription_tier": tiers,
        "device_type": device_types,
        "region": regions,
        "referral_source": referral_sources,
        "age_group": age_groups,
        "gender": genders,
        "persona": personas,  # internal use only, excluded from model features
    })

    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.date

    return df

if __name__ == "__main__":
    import os

    scale = os.environ.get("SCALE", "small")
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    df = generate_users(scale=scale)

    output_path = os.path.join(output_dir, "users.parquet")
    df.to_parquet(output_path, index=False)

    print(f"Generated {len(df)} users (scale={scale}) → {output_path}")
    print(f"\nPersona distribution:")
    print(df["persona"].value_counts())
    print(f"\nSubscription tier distribution:")
    print(df["subscription_tier"].value_counts())
    print(f"\nTier by persona:")
    print(pd.crosstab(df["persona"], df["subscription_tier"], normalize="index").round(3))
    print(f"\nRegion top 10:")
    print(df["region"].value_counts().head(10))