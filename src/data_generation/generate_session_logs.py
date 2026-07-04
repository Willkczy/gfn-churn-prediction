import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- Temporal config ---
OBS_START = pd.Timestamp("2024-01-01")
OBS_WEEKS = 12

# --- Persona behavior profiles ---
PERSONA_PROFILES = {
    "hardcore": {
        "sessions_per_week": (14, 3),
        "session_duration_min": (150, 40),
        "peak_hour_ratio": 0.5,
        "weekend_ratio": 0.4,
    },
    "regular": {
        "sessions_per_week": (6, 2),
        "session_duration_min": (90, 30),
        "peak_hour_ratio": 0.6,
        "weekend_ratio": 0.5,
    },
    "casual": {
        "sessions_per_week": (2, 1),
        "session_duration_min": (40, 15),
        "peak_hour_ratio": 0.7,
        "weekend_ratio": 0.6,
    },
    "about_to_churn": {
        "sessions_per_week": (5, 2),
        "session_duration_min": (80, 25),
        "peak_hour_ratio": 0.6,
        "weekend_ratio": 0.5,
    },
}

# --- Streaming quality by subscription tier ---
QUALITY_PROFILES = {
    "free":     {"latency": (50, 20), "fps": (45, 10), "bitrate": (10, 3),  "jitter": (30, 15), "packet_loss": (0.05, 0.03)},
    "priority": {"latency": (30, 10), "fps": (55, 8),  "bitrate": (25, 5),  "jitter": (15, 8),  "packet_loss": (0.02, 0.01)},
    "ultimate": {"latency": (18, 5),  "fps": (58, 5),  "bitrate": (40, 8),  "jitter": (8, 4),   "packet_loss": (0.008, 0.005)},
}

RESOLUTIONS = ["720p", "1080p", "1440p", "4K"]
RESOLUTION_BY_TIER = {
    "free":     [0.50, 0.40, 0.08, 0.02],
    "priority": [0.10, 0.45, 0.35, 0.10],
    "ultimate": [0.02, 0.15, 0.40, 0.43],
}

EXIT_TYPES = ["normal", "crash", "disconnect", "timeout"]
EXIT_WEIGHTS_NORMAL = [0.85, 0.05, 0.07, 0.03]
EXIT_WEIGHTS_DEGRADED = [0.50, 0.18, 0.22, 0.10]


# --- Helper functions ---

def _apply_quality_impact(
    duration_min: float,
    avg_latency: float,
    disconnect_count: int,
    rng: np.random.Generator,
) -> tuple[float, str | None]:
    """
    Let poor quality dynamically cut session short and force exit type.
    Returns adjusted (duration_min, forced_exit_type or None).
    """
    forced_exit = None

    # High latency frustration: >100ms, chance to quit early
    if avg_latency > 100:
        quit_prob = min(0.6, (avg_latency - 100) / 200)
        if rng.random() < quit_prob:
            duration_min *= rng.uniform(0.2, 0.7)
            forced_exit = "disconnect"

    # Disconnect during session: each disconnect has chance to end session
    for _ in range(disconnect_count):
        if rng.random() < 0.3:
            duration_min *= rng.uniform(0.3, 0.8)
            forced_exit = "disconnect"
            break

    return max(5, duration_min), forced_exit


def _inject_noise(sessions: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Add realistic noise and outliers to generated sessions.
    """
    noisy = []

    for s in sessions:
        # 2% chance: AFK / forgot to close → extremely long session (3-12 hours)
        if rng.random() < 0.02:
            afk_duration = rng.uniform(180, 720)
            s["end_time"] = s["start_time"] + pd.Timedelta(minutes=afk_duration)
            s["avg_fps"] = round(rng.uniform(1, 10), 1)
            s["exit_type"] = "timeout"

        # 3% chance: latency spike
        if rng.random() < 0.03:
            s["avg_latency_ms"] = round(rng.uniform(150, 400), 1)
            s["avg_jitter_ms"] = round(rng.uniform(50, 150), 1)
            s["packet_loss_rate"] = round(rng.uniform(0.05, 0.2), 4)

        # 1% chance: late night session (0:00 - 5:00)
        if rng.random() < 0.01:
            late_hour = int(rng.integers(0, 5))
            original_start = s["start_time"]
            duration = s["end_time"] - s["start_time"]
            new_start = original_start.replace(hour=late_hour, minute=int(rng.integers(0, 60)))
            s["start_time"] = new_start
            s["end_time"] = new_start + duration

        noisy.append(s)

    return noisy


def _generate_hot_weeks(rng: np.random.Generator) -> dict[int, float]:
    """
    Randomly select 2-3 weeks as 'hot event' weeks (new game launch, major update).
    Returns {calendar_week (1-indexed): activity_multiplier}.
    See configs/temporal_conventions.md for the week-numbering convention.
    """
    num_events = rng.integers(2, 4)
    hot_weeks = rng.choice(range(1, OBS_WEEKS + 1), size=num_events, replace=False)
    return {int(w): round(rng.uniform(1.3, 1.8), 2) for w in hot_weeks}


# --- Core generation ---

def _generate_user_sessions(
    user_id: str,
    persona: str,
    tier: str,
    game_ids: list[str],
    game_popularity_weights: np.ndarray,
    hot_weeks: dict[int, float],
    rng: np.random.Generator,
) -> list[dict]:
    profile = PERSONA_PROFILES[persona]
    quality = QUALITY_PROFILES[tier]
    sessions = []

    base_sessions_per_week = max(1, rng.normal(*profile["sessions_per_week"]))
    had_bad_experience = False

    for week in range(OBS_WEEKS):
        # `week` is 0-indexed (0..OBS_WEEKS-1).
        # `calendar_week` is 1-indexed (1..12), matches the timeline used in
        # docs, notebooks, and feature engineering. ALWAYS branch on
        # calendar_week to avoid off-by-one bugs.
        # See configs/temporal_conventions.md for the canonical timeline.
        calendar_week = week + 1
        week_start = OBS_START + pd.Timedelta(weeks=week)

        # --- Decay logic for about_to_churn users ---
        # Decay starts at calendar week 6 (mid-obs-window) and reaches 0.0
        # at calendar week 9 (start of prediction window), so users with
        # this persona produce zero sessions during the pred window (9-10)
        # and yield churn=1.
        #   cal week 6 -> 0.75   cal week 7 -> 0.50
        #   cal week 8 -> 0.25   cal week 9+ -> 0.00
        if persona == "about_to_churn" and calendar_week >= 6:
            decay_factor = 1.0 - 0.25 * (calendar_week - 5)
            decay_factor = max(decay_factor, 0.0)
            duration_multiplier = 0.7 - 0.15 * (calendar_week - 6)
            duration_multiplier = max(duration_multiplier, 0.3)
            use_degraded_exit = True
        else:
            decay_factor = 1.0
            duration_multiplier = 1.0
            use_degraded_exit = False

        # Number of sessions this week
        weekly_count = int(round(base_sessions_per_week * decay_factor))

        # Hot event week: boost activity (keyed by 1-indexed calendar_week)
        if calendar_week in hot_weeks:
            weekly_count = int(round(weekly_count * hot_weeks[calendar_week]))

        weekly_count = max(0, int(rng.normal(weekly_count, 1)))

        for _ in range(weekly_count):
            # --- Bad experience carry-over: chance to skip this session ---
            if had_bad_experience:
                skip_prob = 0.35
                if rng.random() < skip_prob:
                    had_bad_experience = False
                    continue
                had_bad_experience = False

            # --- Timestamp ---
            is_weekend = rng.random() < profile["weekend_ratio"]
            if is_weekend:
                day_offset = rng.choice([5, 6])
            else:
                day_offset = rng.choice([0, 1, 2, 3, 4])

            is_peak = rng.random() < profile["peak_hour_ratio"]
            if is_peak:
                hour = rng.integers(19, 24)
            else:
                hour = rng.integers(8, 19)

            minute = rng.integers(0, 60)
            start_time = week_start + pd.Timedelta(
                days=int(day_offset), hours=int(hour), minutes=int(minute)
            )

            # --- Duration ---
            base_dur = max(10, rng.normal(*profile["session_duration_min"]))
            duration_min = max(10, base_dur * duration_multiplier)

            # Free tier: cap at 60 min
            if tier == "free":
                duration_min = min(duration_min, 60)

            # --- Game selection (weighted by popularity) ---
            game_id = rng.choice(game_ids, p=game_popularity_weights)

            # --- Streaming quality ---
            avg_latency = max(5, rng.normal(*quality["latency"]))
            avg_fps = np.clip(rng.normal(*quality["fps"]), 15, 120)
            avg_bitrate = max(3, rng.normal(*quality["bitrate"]))
            avg_jitter = max(0.5, rng.normal(*quality["jitter"]))
            packet_loss = np.clip(rng.normal(*quality["packet_loss"]), 0, 0.3)

            total_frames = int(avg_fps * duration_min * 60)
            frame_drop_rate = (
                rng.uniform(0.001, 0.02) if not use_degraded_exit
                else rng.uniform(0.01, 0.08)
            )
            total_frame_drops = int(total_frames * frame_drop_rate)

            disconnect_count = int(rng.poisson(0.3 if not use_degraded_exit else 1.5))
            input_lag = max(5, avg_latency * rng.uniform(0.8, 1.5))

            # --- Resolution ---
            resolution = rng.choice(RESOLUTIONS, p=RESOLUTION_BY_TIER[tier])

            # --- Quality impacts duration and exit ---
            duration_min, forced_exit = _apply_quality_impact(
                duration_min, avg_latency, disconnect_count, rng
            )
            end_time = start_time + pd.Timedelta(minutes=duration_min)

            # --- Exit type ---
            if forced_exit:
                exit_type = forced_exit
            else:
                exit_weights = EXIT_WEIGHTS_DEGRADED if use_degraded_exit else EXIT_WEIGHTS_NORMAL
                exit_type = rng.choice(EXIT_TYPES, p=exit_weights)

            sessions.append({
                "user_id": user_id,
                "game_id": game_id,
                "start_time": start_time,
                "end_time": end_time,
                "stream_resolution": resolution,
                "avg_latency_ms": round(avg_latency, 1),
                "avg_fps": round(avg_fps, 1),
                "total_frame_drops": total_frame_drops,
                "disconnect_count": disconnect_count,
                "input_lag_ms": round(input_lag, 1),
                "avg_bitrate_mbps": round(avg_bitrate, 1),
                "avg_jitter_ms": round(avg_jitter, 1),
                "packet_loss_rate": round(packet_loss, 4),
                "exit_type": exit_type,
            })

            # --- Track bad experience for next session ---
            had_bad_experience = exit_type in ("crash", "disconnect") or avg_latency > 120

    # Inject noise and outliers
    sessions = _inject_noise(sessions, rng)

    return sessions


def generate_session_logs(
    users_df: pd.DataFrame,
    game_catalog_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Prepare game selection weights based on popularity tier
    popularity_multiplier = {"S": 10, "A": 5, "B": 2, "C": 1}
    game_ids = game_catalog_df["game_id"].tolist()
    raw_weights = game_catalog_df["popularity_tier"].map(popularity_multiplier).values.astype(float)
    game_popularity_weights = raw_weights / raw_weights.sum()

    # Generate hot weeks (shared across all users — global event)
    hot_weeks = _generate_hot_weeks(rng)
    print(f"Hot event weeks: {hot_weeks}")

    all_sessions = []
    total_users = len(users_df)

    for idx, row in users_df.iterrows():
        if idx % 5000 == 0:
            print(f"  Generating sessions for user {idx + 1}/{total_users}...")

        user_sessions = _generate_user_sessions(
            user_id=row["user_id"],
            persona=row["persona"],
            tier=row["subscription_tier"],
            game_ids=game_ids,
            game_popularity_weights=game_popularity_weights,
            hot_weeks=hot_weeks,
            rng=rng,
        )
        all_sessions.extend(user_sessions)

    df = pd.DataFrame(all_sessions)

    # Add session_id
    df.insert(0, "session_id", [f"S{i:08d}" for i in range(1, len(df) + 1)])

    # Sort by time
    df = df.sort_values("start_time").reset_index(drop=True)

    return df


if __name__ == "__main__":
    import os

    output_dir = os.path.join("data", "raw")

    # Load dependencies
    users_df = pd.read_parquet(os.path.join(output_dir, "users.parquet"))
    game_catalog_df = pd.read_parquet(os.path.join(output_dir, "game_catalog.parquet"))

    print(f"Loaded {len(users_df)} users, {len(game_catalog_df)} games")
    print("Generating session logs...")

    df = generate_session_logs(users_df, game_catalog_df)

    table = pa.Table.from_pandas(df, preserve_index=False)
    for col_name in ["start_time", "end_time"]:
        idx = table.schema.get_field_index(col_name)
        table = table.set_column(idx, col_name, table.column(idx).cast(pa.timestamp("us")))

    output_path = os.path.join(output_dir, "session_logs.parquet")
    pq.write_table(table, output_path)

    print(f"\nGenerated {len(df):,} sessions → {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"\nDate range: {df['start_time'].min()} ~ {df['start_time'].max()}")
    print("\nSessions per persona:")
    merged = df.merge(users_df[["user_id", "persona"]], on="user_id")
    print(merged.groupby("persona").size())
    print("\nExit type distribution:")
    print(df["exit_type"].value_counts(normalize=True).round(3))
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))