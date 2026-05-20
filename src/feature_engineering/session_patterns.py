"""Section 1 + 1.5: Session Patterns and Population Normalization.

Produces 8 features per (user_id, week_num):
    weekly_session_count, avg_session_duration_min, total_playtime_min,
    peak_hour_ratio, weekend_ratio, session_regularity,
    weekly_session_count_norm, total_playtime_min_norm
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def compute_session_features(obs_sessions: DataFrame) -> DataFrame:
    session_patterns = obs_sessions.groupBy("user_id", "week_num").agg(
        F.count("session_id").alias("weekly_session_count"),
        F.avg("duration_min").alias("avg_session_duration_min"),
        F.sum("duration_min").alias("total_playtime_min"),
        F.avg(F.when(F.hour("start_time").between(19, 23), 1).otherwise(0)).alias(
            "peak_hour_ratio"
        ),
        F.avg(F.when(F.dayofweek("start_time").isin(1, 7), 1).otherwise(0)).alias(
            "weekend_ratio"
        ),
    )

    w_session_order = Window.partitionBy("user_id", "week_num").orderBy("start_time")
    session_regularity = (
        obs_sessions.withColumn("prev_end", F.lag("end_time").over(w_session_order))
        .withColumn(
            "inter_session_gap_min",
            (F.unix_timestamp("start_time") - F.unix_timestamp("prev_end")) / 60.0,
        )
        .filter(F.col("inter_session_gap_min").isNotNull())
        .groupBy("user_id", "week_num")
        .agg(F.stddev("inter_session_gap_min").alias("session_regularity"))
    )

    session_features = session_patterns.join(
        session_regularity, on=["user_id", "week_num"], how="left"
    ).fillna(0, subset=["session_regularity"])

    # Population-level normalization (medians from active users only)
    pop_medians = session_features.groupBy("week_num").agg(
        F.expr("percentile_approx(weekly_session_count, 0.5)").alias(
            "pop_median_session_count"
        ),
        F.expr("percentile_approx(total_playtime_min, 0.5)").alias(
            "pop_median_playtime_min"
        ),
    )

    return (
        session_features.join(pop_medians, on="week_num", how="left")
        .withColumn(
            "weekly_session_count_norm",
            F.col("weekly_session_count") / F.col("pop_median_session_count"),
        )
        .withColumn(
            "total_playtime_min_norm",
            F.col("total_playtime_min") / F.col("pop_median_playtime_min"),
        )
        .drop("pop_median_session_count", "pop_median_playtime_min")
    )
