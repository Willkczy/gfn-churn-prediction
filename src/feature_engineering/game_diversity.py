"""Section 4: Game Diversity.

Produces 4 features per (user_id, week_num):
    unique_games_played, top_game_concentration,
    genre_entropy, new_game_trial_rate

`new_game_trial_rate` is null for week_num=1 (no prior weeks inside the obs
window to compare against).
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def compute_game_features(obs_sessions: DataFrame, games: DataFrame) -> DataFrame:
    # --- 4a. unique_games_played & top_game_concentration ---
    game_playtime = obs_sessions.groupBy("user_id", "week_num", "game_id").agg(
        F.sum("duration_min").alias("game_playtime"),
        F.count("session_id").alias("game_sessions"),
    )
    w_game = Window.partitionBy("user_id", "week_num")
    game_concentration = game_playtime.withColumn(
        "total_playtime", F.sum("game_playtime").over(w_game)
    ).withColumn(
        "playtime_share", F.col("game_playtime") / F.col("total_playtime")
    )
    game_basic = game_concentration.groupBy("user_id", "week_num").agg(
        F.countDistinct("game_id").alias("unique_games_played"),
        F.max("playtime_share").alias("top_game_concentration"),
    )

    # --- 4b. genre_entropy (Shannon entropy over per-week genre distribution) ---
    game_with_genre = obs_sessions.join(
        games.select("game_id", "genre"), on="game_id", how="left"
    )
    genre_counts = game_with_genre.groupBy("user_id", "week_num", "genre").agg(
        F.count("session_id").alias("genre_sessions")
    )
    w_genre = Window.partitionBy("user_id", "week_num")
    genre_probs = (
        genre_counts.withColumn("total_sessions", F.sum("genre_sessions").over(w_genre))
        .withColumn("p", F.col("genre_sessions") / F.col("total_sessions"))
        .withColumn("neg_p_log2_p", -F.col("p") * F.log2(F.col("p")))
    )
    genre_entropy = genre_probs.groupBy("user_id", "week_num").agg(
        F.sum("neg_p_log2_p").alias("genre_entropy")
    )

    # --- 4c. new_game_trial_rate (week 1 = null, no prior weeks) ---
    user_week_games = obs_sessions.groupBy("user_id", "week_num").agg(
        F.collect_set("game_id").alias("current_week_games")
    )
    prior_games = obs_sessions.select("user_id", "week_num", "game_id").distinct()
    prior_games_agg = (
        prior_games.alias("a")
        .join(
            prior_games.alias("b"),
            (F.col("a.user_id") == F.col("b.user_id"))
            & (F.col("a.week_num") > F.col("b.week_num")),
        )
        .groupBy(
            F.col("a.user_id").alias("user_id"),
            F.col("a.week_num").alias("week_num"),
        )
        .agg(F.collect_set(F.col("b.game_id")).alias("prior_games"))
    )
    new_game_rate = (
        user_week_games.join(prior_games_agg, on=["user_id", "week_num"], how="left")
        .withColumn(
            "new_games",
            F.when(F.col("prior_games").isNull(), F.lit(None)).otherwise(
                F.size(F.array_except(F.col("current_week_games"), F.col("prior_games")))
            ),
        )
        .withColumn(
            "new_game_trial_rate",
            F.when(F.col("new_games").isNull(), F.lit(None)).otherwise(
                F.col("new_games") / F.size(F.col("current_week_games"))
            ),
        )
        .select("user_id", "week_num", "new_game_trial_rate")
    )

    return (
        game_basic.join(genre_entropy, on=["user_id", "week_num"], how="outer").join(
            new_game_rate, on=["user_id", "week_num"], how="outer"
        )
    )
