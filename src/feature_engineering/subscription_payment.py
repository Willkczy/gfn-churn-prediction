"""Section 6: Subscription & Payment.

Produces 10 user-level features (broadcast to all 4 obs-window weeks):
    current_tier_numeric, days_since_signup,
    tier_changes_count, has_downgraded,
    total_spend_last_4w, payment_count_last_4w,
    failed_payment_count, refund_count,
    days_since_last_payment, payment_frequency_change

`days_since_last_payment` and `payment_frequency_change` stay null for users
with zero payments — that null itself is a meaningful churn signal.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import BASELINE_END, DATA_START, OBS_END, OBS_START, OBS_WEEKS


def compute_payment_features(
    spark: SparkSession,
    users: DataFrame,
    sub_events: DataFrame,
    payments: DataFrame,
) -> DataFrame:
    # --- 6a. User-level static features ---
    user_static = (
        users.select("user_id", "subscription_tier", "signup_date")
        .withColumn(
            "current_tier_numeric",
            F.when(F.col("subscription_tier") == "free", 0)
            .when(F.col("subscription_tier") == "priority", 1)
            .when(F.col("subscription_tier") == "ultimate", 2),
        )
        .withColumn(
            "days_since_signup",
            F.datediff(F.lit(OBS_START), F.col("signup_date")),
        )
        .select("user_id", "current_tier_numeric", "days_since_signup")
    )

    # --- 6b. Subscription events (obs window only) ---
    obs_sub_events = sub_events.filter(
        (F.col("event_date") >= F.lit(OBS_START))
        & (F.col("event_date") < F.lit(OBS_END))
    )
    sub_features = obs_sub_events.groupBy("user_id").agg(
        F.count("event_id").alias("tier_changes_count"),
        F.max(F.when(F.col("event_type") == "downgrade", 1).otherwise(0)).alias(
            "has_downgraded"
        ),
    )

    # --- 6c. Payment aggregates (obs window only) ---
    obs_payments = payments.filter(
        (F.col("payment_date") >= F.lit(OBS_START))
        & (F.col("payment_date") < F.lit(OBS_END))
    )
    payment_agg = obs_payments.groupBy("user_id").agg(
        F.sum(
            F.when(F.col("status") == "success", F.col("amount_usd")).otherwise(0)
        ).alias("total_spend_last_4w"),
        F.count("payment_id").alias("payment_count_last_4w"),
        F.sum(F.when(F.col("status") == "failed", 1).otherwise(0)).alias(
            "failed_payment_count"
        ),
        F.sum(F.when(F.col("status") == "refunded", 1).otherwise(0)).alias(
            "refund_count"
        ),
        F.datediff(
            F.lit(OBS_END),
            F.max(F.when(F.col("status") == "success", F.col("payment_date"))),
        ).alias("days_since_last_payment"),
    )

    # --- 6d. Payment frequency change vs baseline ---
    baseline_payment_count = (
        payments.filter(
            (F.col("payment_date") >= F.lit(DATA_START))
            & (F.col("payment_date") < F.lit(BASELINE_END))
        )
        .groupBy("user_id")
        .agg(F.count("payment_id").alias("baseline_payment_count"))
    )
    obs_payment_count = obs_payments.groupBy("user_id").agg(
        F.count("payment_id").alias("obs_payment_count")
    )
    payment_freq_change = (
        obs_payment_count.join(baseline_payment_count, on="user_id", how="outer")
        .fillna(0, subset=["obs_payment_count", "baseline_payment_count"])
        .withColumn(
            "payment_frequency_change",
            F.when(F.col("baseline_payment_count") == 0, F.lit(None)).otherwise(
                (F.col("obs_payment_count") - F.col("baseline_payment_count"))
                / F.col("baseline_payment_count")
            ),
        )
        .select("user_id", "payment_frequency_change")
    )

    # --- 6e. Combine + broadcast to all obs-window weeks ---
    sub_pay_features = (
        user_static.join(sub_features, on="user_id", how="left")
        .join(payment_agg, on="user_id", how="left")
        .join(payment_freq_change, on="user_id", how="left")
        .fillna(
            0,
            subset=[
                "tier_changes_count",
                "has_downgraded",
                "total_spend_last_4w",
                "payment_count_last_4w",
                "failed_payment_count",
                "refund_count",
            ],
        )
    )

    week_scaffold = spark.createDataFrame(
        [(i,) for i in range(1, OBS_WEEKS + 1)], ["week_num"]
    )
    return sub_pay_features.crossJoin(week_scaffold)
