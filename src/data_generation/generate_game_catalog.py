import numpy as np
import pandas as pd

NUM_GAMES = 200

NUM_GAMES = 200

GENRES = ["FPS", "RPG", "Racing", "Strategy", "Sports", "Casual", "Simulation", "Battle_Royale"]

PUBLISHERS = [f"Publisher_{chr(65 + i)}" for i in range(20)]  # Publisher_A ~ Publisher_T

POPULARITY_TIERS = ["S", "A", "B", "C"]
POPULARITY_WEIGHTS = [0.05, 0.15, 0.35, 0.45]  # S-tier is rare, C-tier is most common

MAX_RESOLUTIONS = ["1080p", "1440p", "4K"]
RESOLUTION_WEIGHTS = [0.4, 0.35, 0.25]

def generate_game_catalog(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    game_ids = [f"G{i:04d}" for i in range(1, NUM_GAMES + 1)]

    genres = rng.choice(GENRES, size=NUM_GAMES).tolist()

    publishers = rng.choice(PUBLISHERS, size=NUM_GAMES).tolist()

    popularity_tiers = rng.choice(
        POPULARITY_TIERS, size=NUM_GAMES, p=POPULARITY_WEIGHTS
    ).tolist()

    max_resolutions = rng.choice(
        MAX_RESOLUTIONS, size=NUM_GAMES, p=RESOLUTION_WEIGHTS
    ).tolist()

    # Game titles: combine genre + a unique number for readability
    game_titles = [f"{genre}_Game_{i:03d}" for i, genre in zip(range(1, NUM_GAMES + 1), genres)]

    df = pd.DataFrame({
        "game_id": game_ids,
        "game_title": game_titles,
        "genre": genres,
        "publisher": publishers,
        "popularity_tier": popularity_tiers,
        "supported_max_resolution": max_resolutions,
    })

    return df


if __name__ == "__main__":
    import os

    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    df = generate_game_catalog()

    output_path = os.path.join(output_dir, "game_catalog.parquet")
    df.to_parquet(output_path, index=False)

    # Quick validation
    print(f"Generated {len(df)} games → {output_path}")
    print("\nPopularity distribution:")
    print(df["popularity_tier"].value_counts().sort_index())
    print("\nGenre distribution:")
    print(df["genre"].value_counts())
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))