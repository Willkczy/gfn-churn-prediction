"""Section 5: Playtime Volatility.

Produces 3 features per (user_id, week_num):
    daily_playtime_std, daily_playtime_cv, session_duration_std

`daily_playtime_*` includes 0-session days (using `user_date_scaffold`);
`session_duration_std` is computed across actual sessions only.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_volatility_features(
    obs_sessions: DataFrame, user_date_scaffold: DataFrame
) -> DataFrame:
    daily_playtime = obs_sessions.groupBy(
        "user_id", F.col("start_time").cast("date").alias("date")
    ).agg(F.sum("duration_min").alias("daily_playtime_min"))

    daily_playtime_filled = user_date_scaffold.join(
        daily_playtime, on=["user_id", "date"], how="left"
    ).fillna(0, subset=["daily_playtime_min"])

    daily_volatility = daily_playtime_filled.groupBy("user_id", "week_num").agg(
        F.stddev("daily_playtime_min").alias("daily_playtime_std"),
        F.avg("daily_playtime_min").alias("daily_playtime_mean"),
    )
    daily_volatility = daily_volatility.withColumn(
        "daily_playtime_cv",
        F.when(F.col("daily_playtime_mean") == 0, F.lit(None)).otherwise(
            F.col("daily_playtime_std") / F.col("daily_playtime_mean")
        ),
    ).drop("daily_playtime_mean")

    session_dur_std = obs_sessions.groupBy("user_id", "week_num").agg(
        F.stddev("duration_min").alias("session_duration_std")
    )

    return daily_volatility.join(
        session_dur_std, on=["user_id", "week_num"], how="outer"
    )
