"""Section 2: Engagement Decay.

Produces 5 features per (user_id, week_num):
    session_count_wow_change, playtime_wow_change,
    session_count_vs_baseline, playtime_vs_baseline,
    longest_inactive_days

Also returns the zero-filled `session_features_filled` table because Section 2
needs it as a scaffold and the final join also uses it as the per-week base.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .config import BASELINE_END, BASELINE_WEEKS, DATA_START, OBS_WEEKS


def compute_engagement_features(
    spark: SparkSession,
    sessions: DataFrame,
    obs_sessions: DataFrame,
    session_features: DataFrame,
    user_date_scaffold: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    all_user_weeks = obs_sessions.select("user_id").distinct().crossJoin(
        spark.createDataFrame(
            [(i,) for i in range(1, OBS_WEEKS + 1)], ["week_num"]
        )
    )

    session_features_filled = all_user_weeks.join(
        session_features, on=["user_id", "week_num"], how="left"
    ).fillna(
        0,
        subset=[
            "weekly_session_count",
            "total_playtime_min",
            "avg_session_duration_min",
            "peak_hour_ratio",
            "weekend_ratio",
            "session_regularity",
            "weekly_session_count_norm",
            "total_playtime_min_norm",
        ],
    )

    # wow_change uses normalized values so hot-week boosts cancel out
    wow_window = Window.partitionBy("user_id").orderBy("week_num")
    wow_base = (
        session_features_filled.select(
            "user_id",
            "week_num",
            "weekly_session_count_norm",
            "total_playtime_min_norm",
        )
        .withColumn(
            "prev_session_count", F.lag("weekly_session_count_norm").over(wow_window)
        )
        .withColumn("prev_playtime", F.lag("total_playtime_min_norm").over(wow_window))
    )
    wow_features = wow_base.select(
        "user_id",
        "week_num",
        F.when(
            F.col("prev_session_count").isNull() | (F.col("prev_session_count") == 0),
            F.lit(None),
        )
        .otherwise(
            (F.col("weekly_session_count_norm") - F.col("prev_session_count"))
            / F.col("prev_session_count")
        )
        .alias("session_count_wow_change"),
        F.when(
            F.col("prev_playtime").isNull() | (F.col("prev_playtime") == 0),
            F.lit(None),
        )
        .otherwise(
            (F.col("total_playtime_min_norm") - F.col("prev_playtime"))
            / F.col("prev_playtime")
        )
        .alias("playtime_wow_change"),
    )

    # vs_baseline uses raw values (per-user ratio, hot week effect washes out)
    baseline_sessions = sessions.filter(
        (F.col("start_time").cast("date") >= F.lit(DATA_START))
        & (F.col("start_time").cast("date") < F.lit(BASELINE_END))
    ).withColumn(
        "duration_min",
        (F.unix_timestamp("end_time") - F.unix_timestamp("start_time")) / 60.0,
    )
    user_baseline = baseline_sessions.groupBy("user_id").agg(
        (F.count("session_id") / F.lit(BASELINE_WEEKS)).alias(
            "baseline_avg_session_count"
        ),
        (F.sum("duration_min") / F.lit(BASELINE_WEEKS)).alias("baseline_avg_playtime"),
    )
    baseline_features = (
        session_features_filled.select(
            "user_id", "week_num", "weekly_session_count", "total_playtime_min"
        )
        .join(user_baseline, on="user_id", how="left")
        .select(
            "user_id",
            "week_num",
            F.when(
                F.col("baseline_avg_session_count").isNull()
                | (F.col("baseline_avg_session_count") == 0),
                F.lit(None),
            )
            .otherwise(
                F.col("weekly_session_count") / F.col("baseline_avg_session_count")
            )
            .alias("session_count_vs_baseline"),
            F.when(
                F.col("baseline_avg_playtime").isNull()
                | (F.col("baseline_avg_playtime") == 0),
                F.lit(None),
            )
            .otherwise(F.col("total_playtime_min") / F.col("baseline_avg_playtime"))
            .alias("playtime_vs_baseline"),
        )
    )

    # Longest inactive streak (consecutive 0-session days within obs window)
    session_dates = (
        obs_sessions.select("user_id", F.col("start_time").cast("date").alias("date"))
        .distinct()
        .withColumn("had_session", F.lit(1))
    )
    daily_activity = user_date_scaffold.join(
        session_dates, on=["user_id", "date"], how="left"
    ).fillna(0, subset=["had_session"])

    streak_window = Window.partitionBy("user_id").orderBy("date")
    daily_activity = daily_activity.withColumn(
        "session_cumsum", F.sum("had_session").over(streak_window)
    )
    inactive_streaks = daily_activity.filter(F.col("had_session") == 0)
    streak_group_window = Window.partitionBy("user_id", "session_cumsum").orderBy("date")
    inactive_streaks = inactive_streaks.withColumn(
        "streak_len", F.row_number().over(streak_group_window)
    )
    longest_inactive = inactive_streaks.groupBy("user_id", "week_num").agg(
        F.max("streak_len").alias("longest_inactive_days")
    )

    engagement_features = (
        wow_features.join(baseline_features, on=["user_id", "week_num"], how="outer")
        .join(longest_inactive, on=["user_id", "week_num"], how="outer")
        .fillna(0, subset=["longest_inactive_days"])
    )

    return session_features_filled, engagement_features
