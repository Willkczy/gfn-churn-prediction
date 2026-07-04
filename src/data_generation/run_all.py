import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_generation.generate_game_catalog import generate_game_catalog
from src.data_generation.generate_payments import generate_payments
from src.data_generation.generate_session_logs import generate_session_logs
from src.data_generation.generate_subscription_events import generate_subscription_events
from src.data_generation.generate_users import generate_users


def main():
    scale = os.environ.get("SCALE", "small")
    seed = int(os.environ.get("SEED", "42"))
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== GFN Churn Data Generation (scale={scale}, seed={seed}) ===\n")
    total_start = time.time()

    # 1. Game catalog
    print("[1/5] Generating game catalog...")
    t = time.time()
    game_catalog_df = generate_game_catalog(seed=seed)
    game_catalog_df.to_parquet(os.path.join(output_dir, "game_catalog.parquet"), index=False)
    print(f"  → {len(game_catalog_df):,} games ({time.time() - t:.1f}s)\n")

    # 2. Users
    print("[2/5] Generating users...")
    t = time.time()
    users_df = generate_users(scale=scale, seed=seed)
    users_df.to_parquet(os.path.join(output_dir, "users.parquet"), index=False)
    print(f"  → {len(users_df):,} users ({time.time() - t:.1f}s)\n")

    # 3. Session logs
    print("[3/5] Generating session logs (this may take a few minutes)...")
    t = time.time()
    session_logs_df = generate_session_logs(users_df, game_catalog_df, seed=seed)
    table = pa.Table.from_pandas(session_logs_df, preserve_index=False)
    for col_name in ["start_time", "end_time"]:
        i = table.schema.get_field_index(col_name)
        table = table.set_column(i, col_name, table.column(i).cast(pa.timestamp("us"), safe=False))
    pq.write_table(table, os.path.join(output_dir, "session_logs.parquet"))
    print(f"  → {len(session_logs_df):,} sessions ({time.time() - t:.1f}s)\n")

    # 4. Subscription events
    print("[4/5] Generating subscription events...")
    t = time.time()
    sub_events_df = generate_subscription_events(users_df, seed=seed)
    sub_events_df.to_parquet(os.path.join(output_dir, "subscription_events.parquet"), index=False)
    print(f"  → {len(sub_events_df):,} events ({time.time() - t:.1f}s)\n")

    # 5. Payments
    print("[5/5] Generating payments...")
    t = time.time()
    payments_df = generate_payments(users_df, seed=seed)
    payments_df.to_parquet(os.path.join(output_dir, "payments.parquet"), index=False)
    print(f"  → {len(payments_df):,} payments ({time.time() - t:.1f}s)\n")

    # Summary
    total_time = time.time() - total_start
    print("=" * 50)
    print(f"All data generated in {total_time:.1f}s")
    print(f"Output directory: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".parquet"):
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
            print(f"  {f:40s} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()