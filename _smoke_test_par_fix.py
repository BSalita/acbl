"""
Smoke test for the _list_to_ddtable par fix.

Loads only 2026 club hand records (~1M rows) and the full cache,
runs AllHandRecordAugmentations with max_sd_adds=0 (skip slow SD),
and checks that 2026 data survives the inner join.

Expected runtime: ~5-10 minutes.
Does NOT write any production files.
"""

import pathlib
import sys
import time
import traceback

import polars as pl

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / 'mlBridge'
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f'mlBridge not found at {_MLBRIDGE}')
for _p in (_SRC_DIR, _MLBRIDGE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)
import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')

print(time.strftime("%Y-%m-%d %H:%M:%S"))
t_start = time.time()

# --- Load cleaned hand records, filter to 2026 only ---
hrs_file = acblPath / 'acbl_club_hand_records_cleaned.parquet'
hrs_df_full = pl.read_parquet(hrs_file)
print(f"Full cleaned hand records: {hrs_df_full.shape}")

date_col = [c for c in hrs_df_full.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in hrs_df_full.columns) else None
if date_col:
    print(f"  Using date column: {date_col} (dtype: {hrs_df_full[date_col].dtype})")
    if hrs_df_full[date_col].dtype == pl.Utf8:
        hrs_df = hrs_df_full.filter(pl.col(date_col).str.starts_with('2026'))
    else:
        hrs_df = hrs_df_full.filter(pl.col(date_col).dt.year() == 2026)
else:
    print("  No date column found. Trying hand_record_id-based heuristic...")
    hrs_df = hrs_df_full.tail(1_000_000)

del hrs_df_full
print(f"2026-only hand records: {hrs_df.shape}")

assert hrs_df.height > 0, "No 2026 rows found in cleaned hand records!"
assert hrs_df['PBN'].null_count() == 0
assert hrs_df['Dealer'].null_count() == 0
assert hrs_df['Vul'].null_count() == 0

# --- Load cache ---
cache_file = acblPath / 'acbl_club_hand_records_cache_df.parquet'
hrs_cache_df = pl.read_parquet(cache_file)
print(f"Cache: {hrs_cache_df.shape}")
print(f"  Cache null Dealer count: {hrs_cache_df['Dealer'].null_count()}")
print(f"  Cache null Vul count: {hrs_cache_df['Vul'].null_count()}")

# --- Check PBN overlap before augmentation ---
input_pbns = set(hrs_df['PBN'].unique().to_list())
cache_composite_keys = set(
    hrs_cache_df
    .filter(pl.col('Dealer').is_not_null() & pl.col('Vul').is_not_null())
    .select(pl.concat_str(['PBN', 'Dealer', 'Vul'], separator='|'))
    .to_series().to_list()
)
input_composite_keys = set(
    hrs_df
    .select(pl.concat_str(['PBN', 'Dealer', 'Vul'], separator='|'))
    .to_series().to_list()
)
pre_match = len(input_composite_keys & cache_composite_keys)
print(f"  Input unique PBNs: {len(input_pbns)}")
print(f"  Input unique (PBN,Dealer,Vul): {len(input_composite_keys)}")
print(f"  Cache composite keys matching input (before augmentation): {pre_match}")

print(f"\nLoad time: {time.time() - t_start:.1f}s")

# --- Run augmentation (the part that had the bug) ---
print("\n--- Running AllHandRecordAugmentations (max_sd_adds=0) ---")
t_aug = time.time()
try:
    all_augmentations = mlBridgeAugmentLib.AllHandRecordAugmentations(
        hrs_df,
        hrs_cache_df,
        sd_productions=10,
        max_dd_adds=None,
        max_sd_adds=0,
        cache_file_path=None,  # don't persist cache to disk
    )
    hrs_df_out, hrs_cache_df_out = all_augmentations.perform_all_hand_record_augmentations()
except Exception:
    traceback.print_exc()
    print("\nSMOKE TEST FAILED: augmentation raised an exception.")
    sys.exit(1)

aug_time = time.time() - t_aug
print(f"\nAugmentation completed in {aug_time:.1f}s")

# --- Validate results ---
print("\n--- Results ---")
print(f"Output rows:    {hrs_df_out.height}")
print(f"Output columns: {hrs_df_out.width}")

expected_cols = ['DD_N_S', 'DD_N_H', 'DD_N_D', 'DD_N_C', 'DD_N_N', 'ParScore']
missing_cols = [c for c in expected_cols if c not in hrs_df_out.columns]
if missing_cols:
    print(f"MISSING expected columns: {missing_cols}")

if 'ParScore' in hrs_df_out.columns:
    par_null = hrs_df_out['ParScore'].null_count()
    par_total = hrs_df_out.height
    print(f"ParScore: {par_total - par_null}/{par_total} non-null ({100*(par_total-par_null)/par_total:.1f}%)")

# The critical check: did 2026 data survive the inner join?
input_rows = hrs_df.height
output_rows = hrs_df_out.height
retention_pct = 100 * output_rows / input_rows if input_rows > 0 else 0

print(f"\nInput rows:     {input_rows:>12,}")
print(f"Output rows:    {output_rows:>12,}")
print(f"Retention:      {retention_pct:>11.1f}%")

if output_rows < input_rows * 0.5:
    print(f"\nSMOKE TEST FAILED: only {retention_pct:.1f}% of input rows survived.")
    print("The inner join is still dropping 2026 data.")
    sys.exit(1)
elif missing_cols:
    print(f"\nSMOKE TEST FAILED: missing columns {missing_cols}")
    sys.exit(1)
else:
    print(f"\nSMOKE TEST PASSED: {retention_pct:.1f}% retention, all expected columns present.")

print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
print(time.strftime("%Y-%m-%d %H:%M:%S"))
