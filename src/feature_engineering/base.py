"""Shared base DataFrames used by multiple feature sections.

`obs_sessions` is the obs-window-filtered sessions table with `week_num`
(window-local 1..4) and `duration_min` derived columns. `user_date_scaffold`
is the user × calendar-day cross product used by anything that needs to count
0-session days (longest inactive streak, daily playtime volatility).
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import OBS_END, OBS_START


def build_obs_sessions(sessions: DataFrame) -> DataFrame:
    return (
        sessions.filter(
            (F.col("start_time").cast("date") >= F.lit(OBS_START))
            & (F.col("start_time").cast("date") < F.lit(OBS_END))
        )
        .withColumn(
            "week_num",
            (F.datediff(F.col("start_time").cast("date"), F.lit(OBS_START)) / 7).cast("int") + 1,
        )
        .withColumn(
            "duration_min",
            (F.unix_timestamp("end_time") - F.unix_timestamp("start_time")) / 60.0,
        )
    )


def build_user_date_scaffold(spark: SparkSession, obs_sessions: DataFrame) -> DataFrame:
    obs_dates = spark.sql(
        f"""
        SELECT explode(sequence(
            to_date('{OBS_START}'),
            date_sub(to_date('{OBS_END}'), 1),
            interval 1 day
        )) AS date
        """
    ).withColumn(
        "week_num",
        (F.datediff(F.col("date"), F.lit(OBS_START)) / 7).cast("int") + 1,
    )
    obs_user_ids = obs_sessions.select("user_id").distinct()
    return obs_user_ids.crossJoin(obs_dates)
