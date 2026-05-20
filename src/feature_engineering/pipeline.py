"""End-to-end Phase 2 pipeline: read raw parquet, compute weekly features,
write `weekly_features.parquet`.

Mirrors `notebooks/phase2_feature_engineering.ipynb`. The notebook stays as
the exploratory artifact; this module is the canonical pipeline used by
Phase 3 and downstream modeling.

Run inside the Spark dev container:
    python -m src.feature_engineering.pipeline
"""

from pyspark.sql import DataFrame, SparkSession

from .base import build_obs_sessions, build_user_date_scaffold
from .engagement_decay import compute_engagement_features
from .game_diversity import compute_game_features
from .playtime_volatility import compute_volatility_features
from .session_patterns import compute_session_features
from .streaming_quality import compute_streaming_features
from .subscription_payment import compute_payment_features

DEFAULT_INPUT_DIR = "/home/spark/work/data/raw"
DEFAULT_OUTPUT_PATH = "/home/spark/work/data/processed/weekly_features.parquet"

ZERO_FILL_COLS = [
    "weekly_session_count",
    "avg_session_duration_min",
    "total_playtime_min",
    "weekly_session_count_norm",
    "total_playtime_min_norm",
    "peak_hour_ratio",
    "weekend_ratio",
    "session_regularity",
    "longest_inactive_days",
    "daily_playtime_std",
]


def build_weekly_features(
    spark: SparkSession,
    users: DataFrame,
    sessions: DataFrame,
    games: DataFrame,
    sub_events: DataFrame,
    payments: DataFrame,
) -> DataFrame:
    obs_sessions = build_obs_sessions(sessions)
    user_date_scaffold = build_user_date_scaffold(spark, obs_sessions)

    session_features = compute_session_features(obs_sessions)
    session_features_filled, engagement_features = compute_engagement_features(
        spark, sessions, obs_sessions, session_features, user_date_scaffold
    )
    streaming_features = compute_streaming_features(obs_sessions)
    game_features = compute_game_features(obs_sessions, games)
    volatility_features = compute_volatility_features(obs_sessions, user_date_scaffold)
    payment_features = compute_payment_features(spark, users, sub_events, payments)

    weekly_features = (
        session_features_filled.join(
            engagement_features, on=["user_id", "week_num"], how="outer"
        )
        .join(streaming_features, on=["user_id", "week_num"], how="outer")
        .join(game_features, on=["user_id", "week_num"], how="outer")
        .join(volatility_features, on=["user_id", "week_num"], how="outer")
        .join(payment_features, on=["user_id", "week_num"], how="outer")
        .fillna(0, subset=ZERO_FILL_COLS)
    )
    return weekly_features


def run(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    spark = (
        SparkSession.builder.appName("gfn-feature-engineering")
        .master("local[*]")
        .getOrCreate()
    )

    users = spark.read.parquet(f"{input_dir}/users.parquet")
    sessions = spark.read.parquet(f"{input_dir}/session_logs.parquet")
    games = spark.read.parquet(f"{input_dir}/game_catalog.parquet")
    sub_events = spark.read.parquet(f"{input_dir}/subscription_events.parquet")
    payments = spark.read.parquet(f"{input_dir}/payments.parquet")

    weekly_features = build_weekly_features(
        spark, users, sessions, games, sub_events, payments
    )
    weekly_features.cache()
    n_rows = weekly_features.count()
    print(
        f"weekly_features: {n_rows:,} rows, {len(weekly_features.columns)} cols"
    )

    weekly_features.write.mode("overwrite").parquet(output_path)
    print(f"Saved weekly_features to {output_path}")

    spark.stop()


if __name__ == "__main__":
    run()
