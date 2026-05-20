"""Section 3: Streaming Quality.

Produces 8 features per (user_id, week_num):
    avg_latency, avg_fps, frame_drop_rate, disconnect_rate,
    avg_bitrate, avg_jitter, packet_loss_avg, crash_exit_ratio
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_streaming_features(obs_sessions: DataFrame) -> DataFrame:
    return obs_sessions.groupBy("user_id", "week_num").agg(
        F.avg("avg_latency_ms").alias("avg_latency"),
        F.avg("avg_fps").alias("avg_fps"),
        F.avg("total_frame_drops").alias("frame_drop_rate"),
        F.avg("disconnect_count").alias("disconnect_rate"),
        F.avg("avg_bitrate_mbps").alias("avg_bitrate"),
        F.avg("avg_jitter_ms").alias("avg_jitter"),
        F.avg("packet_loss_rate").alias("packet_loss_avg"),
        F.avg(
            F.when(F.col("exit_type").isin("crash", "disconnect", "timeout"), 1).otherwise(0)
        ).alias("crash_exit_ratio"),
    )
