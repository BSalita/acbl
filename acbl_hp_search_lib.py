"""
Shared hyperparameter search utilities for ACBL prediction targets.

Strategy: coordinate descent.
  * Start with a baseline config.
  * For each search axis (in order), train one model per candidate value
    while holding all other params at the current best.
  * Pick the best value for that axis (per the selection metric), update
    baseline, move to next axis.
  * Total trials = sum(len(candidates) for each axis), much smaller than
    a full grid.

Optimization: schema + shards are built ONCE at the start of the search
(features and scaling don't depend on model architecture), then each trial
just modifies `mlp_layers` / `mlp_dropout` / `y_range` in the schema dict
and calls `train_model_from_shards` again. Re-creating shards is the slow
step (~1-2 min) so reusing them turns ~5 min/trial into ~2-3 min/trial.

Each trial saves the model to a fixed name and overwrites previous trials.
A CSV of all trial results is written incrementally so partial runs are
not lost.
"""

import os
# Silence GT 1030 / multi-GPU mismatch warnings; set BEFORE importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import copy
import gc
import json
import math
import pathlib
import sys
import time

# Windows powershell defaults to cp1252; force UTF-8 so progress bars and
# arrow characters (used in summary lines) don't crash the script.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

sys.path.append(str(pathlib.Path.cwd().parent.joinpath('mlBridgeLib')))

import mlBridge  # noqa: F401  (used for pd_options_display in callers)
from mlBridge.mlBridgeAiLib import (
    generate_and_save_schema,
    create_torch_shards,
    train_model_from_shards,
    predict_model,
)

ROOT_PATH = pathlib.Path('e:/bridge/data')
ACBL_PATH = ROOT_PATH.joinpath('acbl')
SAVED_MODELS_PATH = ACBL_PATH.joinpath('SavedModels')

Y_NAMES = ['Declarer_Direction', 'Contract', 'Pct_NS']

# Pct_NS now uses ONLY pre-board features — Contract / Declarer_Direction
# are auction outcomes and are excluded. Mirrors the rule in
# acbl_prediction_train.py.
TARGETS_TO_KEEP: Dict[str, List[str]] = {
    'Declarer_Direction': [],
    'Contract': [],
    'Pct_NS': [],
}

SPLIT_INDICATOR_COLS = ['is_train_set', 'is_val_set', 'is_test_set']

# ──────────────────────────────────────────────────────────────────────
# Pct_NS feature pruning
#
# After the first run on the pre-auction-only feature set the importance
# report (acbl_club_predicted_pct_ns_torch_model_importance.csv) made it
# clear that 78 % of the 6035 inputs sit at the noise floor (importance ≈
# 13–14, vs MasterPoints_* ≈ 90 and Elo_* ≈ 22). They add training cost,
# inflate the input dim, and act as random feature noise for SGD without
# meaningfully informing Pct_NS. Drop the three biggest, weakest families:
#
#   1. EV_<pair>_<dir>_<strain>_<level>_<vul>_<tricks>_<score>  (3920 cols)
#      The fully-broken-out expected-value cube. The compact summaries
#      (EV_*_Max, EV_<pair>_<dir>_<strain>_<level>, EV_<other>) are kept.
#   2. Probs_<pair>_<dir>_<strain>_<level>                       (560 cols)
#      Per-(pair,strain,level) make-probability table. Same story — the
#      coarser DD signals retain the equity information.
#   3. C_<seat><suit><rank>                                      (208 cols)
#      Per-card Boolean indicators (does East hold ♦4?). Fully redundant
#      with HCP_*, SL_*, DP_*, QT_*, LTC_*, DD_* summaries that we keep.
#
# Total drop: ~4688 features → leaves ~1347 informative inputs.
# Re-enable any of these by removing the matching pattern below.
# ──────────────────────────────────────────────────────────────────────
import re as _re

PCT_NS_DROP_PATTERNS: List[str] = [
    r'^EV_(NS|EW)_[NESW]_[CDHSN]_\d+_(NV|V)_\d+_-?\d+$',
    r'^Probs_(NS|EW)_[NESW]_[CDHSN]_\d+$',
    r'^C_[NESW][CDHS][2-9TJQKA]$',
]
_PCT_NS_DROP_REGEXES = [_re.compile(p) for p in PCT_NS_DROP_PATTERNS]


def prune_pct_ns_features(df: pl.DataFrame, *, verbose: bool = True) -> pl.DataFrame:
    """Drop the noisy high-cardinality feature families for Pct_NS.

    No-op for any column not matching `PCT_NS_DROP_PATTERNS`. Safe to call
    on the test DataFrame too — the Pct_NS column itself never matches.
    """
    to_drop = [c for c in df.columns if any(rx.match(c) for rx in _PCT_NS_DROP_REGEXES)]
    if not to_drop:
        return df
    if verbose:
        print(f"[Pct_NS prune] Dropping {len(to_drop):,} noisy features "
              f"(of {len(df.columns):,}); keeping {len(df.columns)-len(to_drop):,}")
    return df.drop(to_drop)


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_data(
    club_or_tournament: str,
    *,
    input_suffix: str = "",
    train_sample_rows: Optional[int] = None,
    test_sample_rows: Optional[int] = None,
    drop_regex_patterns: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load the train and test parquets produced by acbl_prediction_data.py.

    `input_suffix` is appended to the filename stem so callers can pick the
    regenerated `_v2` parquets (corrected pair Elo, no MasterPoint cap) without
    overwriting the legacy files. Default '' preserves the historical behavior.

    `train_sample_rows` / `test_sample_rows`: when provided, only the LAST N
    rows are read via `pl.scan_parquet().slice(...)`. The slice push-down
    skips earlier row groups so RAM usage and disk I/O are bounded. CRITICAL
    for the v2 club parquet (165 GB on disk, ~250 GB decoded — eager load
    exceeds 192 GB RAM and triggers pagefile thrashing).

    We deliberately take the TAIL rather than the HEAD: the train parquet is
    written year-by-year (oldest first), and the test split is the newest
    year. A head() smoke sample fits a model on 2019 data and evaluates on
    2026, producing nonsense (RMSE >> target range, var_ratio > 1). The tail
    is temporally adjacent to the test set, giving a representative read.

    `drop_regex_patterns`: when provided, columns matching ANY of these
    regexes are excluded at scan time via `.select(pl.exclude(...))`. This
    pushes the projection into the parquet reader so the dropped columns are
    never decoded. Essential for Pct_NS — the v2 club parquet has 6,042
    columns and ~4,700 of them are EV/Probs/per-card noise-floor families
    that get dropped immediately by `prune_pct_ns_features`. Pruning at scan
    time changes a 10M-row slice from 240 GB to ~50 GB.
    """
    train_file = ACBL_PATH.joinpath(f"acbl_{club_or_tournament}_prediction_data_train{input_suffix}.parquet")
    test_file = ACBL_PATH.joinpath(f"acbl_{club_or_tournament}_prediction_data_test{input_suffix}.parquet")
    if not train_file.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_file}")

    compiled_drops = (
        [_re.compile(p) for p in drop_regex_patterns] if drop_regex_patterns else None
    )

    def _scan_with_projection(path) -> pl.LazyFrame:
        """scan + drop matching columns at the lazy-frame level."""
        lf = pl.scan_parquet(path)
        if compiled_drops is None:
            return lf
        all_cols = lf.collect_schema().names()
        keep = [c for c in all_cols if not any(rx.match(c) for rx in compiled_drops)]
        n_drop = len(all_cols) - len(keep)
        if n_drop:
            print(f"  scan-time prune {path.name}: dropping {n_drop:,} of {len(all_cols):,} cols, keeping {len(keep):,}")
        return lf.select(keep)

    def _tail_or_full(path, sample_rows: Optional[int]) -> pl.DataFrame:
        # Get total row count cheaply from parquet metadata.
        n_total = pl.scan_parquet(path).select(pl.len()).collect().item()
        lf = _scan_with_projection(path)
        if sample_rows is None or sample_rows >= n_total:
            print(f"Reading {path.name} (full; {n_total:,} rows) ...")
            return lf.collect()
        offset = n_total - sample_rows
        print(f"Scanning {path.name} (tail {sample_rows:,} of {n_total:,} rows, offset={offset:,}) ...")
        return lf.slice(offset, sample_rows).collect()

    train_df = _tail_or_full(train_file, train_sample_rows)
    print(f"  shape={train_df.shape}")
    test_df = _tail_or_full(test_file, test_sample_rows)
    print(f"  shape={test_df.shape}")
    return train_df, test_df


def build_working_dfs(
    model_df: pl.DataFrame,
    df_test: pl.DataFrame,
    y_name: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Drop the OTHER y targets to avoid leakage; keep temporally-prior ones.

    Also applies target-specific feature pruning (e.g. Pct_NS noise-floor
    families — see `PCT_NS_DROP_PATTERNS`).
    """
    keep = TARGETS_TO_KEEP.get(y_name, [])
    other_targets = [t for t in Y_NAMES if t != y_name and t not in keep and t in model_df.columns]
    working_df = model_df.select(pl.exclude(other_targets))
    working_test_df = df_test.select(pl.exclude(other_targets))
    if y_name == 'Pct_NS':
        working_df = prune_pct_ns_features(working_df, verbose=True)
        working_test_df = prune_pct_ns_features(working_test_df, verbose=False)
    return working_df, working_test_df


# ──────────────────────────────────────────────────────────────────────
# Class-weight helpers (Contract uses these)
# ──────────────────────────────────────────────────────────────────────

def compute_class_counts(
    working_df_for_training: pl.DataFrame,
    schema_d: Dict[str, Any],
    y_name: str,
) -> Optional[np.ndarray]:
    """Return counts[k] = number of training rows for class index k, or None."""
    class_to_idx = schema_d.get('class_to_idx', {}) or {}
    if not class_to_idx:
        return None
    num_classes = len(class_to_idx)
    vc_df = (
        working_df_for_training
        .group_by(y_name)
        .len()
        .rename({'len': '_count'})
    )
    counts = np.zeros(num_classes, dtype=np.float64)
    for label, count in zip(vc_df[y_name].to_list(), vc_df['_count'].to_list()):
        idx = class_to_idx.get(label)
        if idx is None:
            idx = class_to_idx.get(str(label))
        if idx is not None and 0 <= idx < num_classes:
            counts[idx] += float(count)
    return counts


def class_weights_from_counts(
    counts: np.ndarray,
    mode: Optional[str],
) -> Optional[List[float]]:
    """Convert raw counts into a normalized weight vector.

    Modes (all normalized so mean(weights) == 1):
      * None / 'none'      → no weighting (return None)
      * 'inverse_freq'     → weights ∝ 1 / count_k (most aggressive)
      * 'sqrt_inverse_freq'→ weights ∝ 1 / sqrt(count_k) (moderate)
      * 'log_inverse_freq' → weights ∝ 1 / log(1 + count_k) (gentle)
    """
    if mode is None or mode == 'none':
        return None
    if counts is None:
        return None
    safe = np.maximum(counts, 1.0)  # avoid div-by-zero
    if mode == 'inverse_freq':
        inv = 1.0 / safe
    elif mode == 'sqrt_inverse_freq':
        inv = 1.0 / np.sqrt(safe)
    elif mode == 'log_inverse_freq':
        inv = 1.0 / np.log1p(safe)
    else:
        raise ValueError(f"Unknown class_weights mode: {mode!r}")
    n = len(safe)
    weights = (inv * (n / inv.sum())).tolist()
    return weights


# ──────────────────────────────────────────────────────────────────────
# Schema / shard helpers
# ──────────────────────────────────────────────────────────────────────

def cleanup_shards(model_name: str) -> None:
    for p in SAVED_MODELS_PATH.glob(f"{model_name}_shard_*.pt"):
        try:
            p.unlink()
        except Exception:
            pass


def align_inference_features(
    predict_df: pl.DataFrame,
    schema: Dict[str, Any],
    y_name: str,
) -> pl.DataFrame:
    """Align test_df columns to the order/dtypes the schema expects."""
    expected_dtypes = schema.get("feature_dtypes")
    if not expected_dtypes:
        raise ValueError("Schema missing 'feature_dtypes'")
    expected_features = list(expected_dtypes.keys())

    def default_for_dtype(dtype_str):
        if dtype_str is None:
            return None
        ds = str(dtype_str).lower()
        if "int" in ds:
            return 0
        if "float" in ds:
            return 0.0
        if "bool" in ds:
            return False
        return None

    out = predict_df
    for col in expected_features:
        if col not in out.columns:
            out = out.with_columns(pl.lit(default_for_dtype(expected_dtypes.get(col))).alias(col))
    extra = [c for c in out.columns if c not in expected_features and c != y_name]
    if extra:
        out = out.drop(extra)
    return out.select(expected_features)


def build_shards_once(
    working_df_for_training: pl.DataFrame,
    y_name: str,
    model_name: str,
    initial_layers: List[int],
    initial_dropout: float,
    initial_y_range: Optional[Tuple[float, float]],
    shard_rows_count: int,
) -> Dict[str, Any]:
    """Generate schema + shards once. Returns the schema dict."""
    cleanup_shards(model_name)
    schema_d = generate_and_save_schema(
        working_df_for_training,
        SAVED_MODELS_PATH,
        model_name,
        y_name,
        layers=initial_layers,
        dropout=initial_dropout,
        apply_scaling_parameters=True,
        y_range=initial_y_range,
        verbose=False,
    )
    num_rows = len(working_df_for_training)
    est_shards = math.ceil(num_rows / shard_rows_count) if shard_rows_count else 0
    effective_shard_rows = shard_rows_count
    if est_shards < 2 and num_rows > 0:
        effective_shard_rows = max(1, num_rows // 2)
        print(f"[shards] Adjusted shard_rows_count {shard_rows_count:,} → {effective_shard_rows:,} "
              f"to ensure >=2 shards (rows={num_rows:,})")
    create_torch_shards(
        working_df_for_training,
        schema_d,
        shard_rows_count=effective_shard_rows,
        apply_scaling=True,
    )
    return schema_d


# ──────────────────────────────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────────────────────────────

def _save_schema_to_disk(schema_d: Dict[str, Any], model_name: str) -> None:
    """Persist the (possibly mutated) schema so predict_model picks it up."""
    schema_path = SAVED_MODELS_PATH / f"{model_name}_schema.json"
    with open(schema_path, 'w') as f:
        json.dump(schema_d, f, default=str)


def run_trial(
    *,
    base_schema: Dict[str, Any],
    working_test_df: pl.DataFrame,
    y_name: str,
    model_name: str,
    config: Dict[str, Any],
    class_counts: Optional[np.ndarray],
    device: str = 'cuda',
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one training+evaluation trial on the pre-built shards.

    Mutates a deepcopy of `base_schema` with this trial's architecture
    (`layers`, `dropout`, `y_range`) and writes it to disk so prediction
    sees the right shapes. Returns a metrics dict.
    """
    schema_d = copy.deepcopy(base_schema)
    schema_d['mlp_layers'] = list(config['layers'])
    schema_d['mlp_dropout'] = float(config['dropout'])
    yr = config.get('y_range', None)
    if yr is None:
        schema_d['y_range'] = None
    else:
        schema_d['y_range'] = list(yr) if isinstance(yr, (tuple, list)) else yr
    _save_schema_to_disk(schema_d, model_name)

    # Class weights
    cw_mode = config.get('class_weights', None)
    weights_list: Optional[List[float]] = None
    if cw_mode is not None and class_counts is not None:
        weights_list = class_weights_from_counts(class_counts, cw_mode)
        if weights_list is not None:
            cw_arr = np.asarray(weights_list, dtype=np.float32)
            print(f"   class_weights({cw_mode}): n={len(weights_list)} "
                  f"min={cw_arr.min():.3f} max={cw_arr.max():.3f} mean={cw_arr.mean():.3f}")

    # Train
    train_t0 = time.time()
    _model, _model_path, stats = train_model_from_shards(
        schema_d,
        epochs=int(config['epochs']),
        bs=int(config['bs']),
        lr=float(config['lr']),
        use_amp=bool(config.get('use_amp', False)),
        weight_decay=float(config['weight_decay']),
        device=device,
        seed=int(seed),
        early_stop_patience=int(config.get('early_stop_patience', 3)),
        class_weights=weights_list,
        verbose=False,
    )
    train_time = time.time() - train_t0

    # Predict on test set
    pred_t0 = time.time()
    schema_path = SAVED_MODELS_PATH / f"{model_name}_schema.json"
    with open(schema_path, 'r') as f:
        on_disk_schema = json.load(f)
    features_df = align_inference_features(working_test_df, on_disk_schema, y_name)
    prediction_df = predict_model(SAVED_MODELS_PATH, model_name, features_df)
    pred_time = time.time() - pred_t0

    metrics: Dict[str, Any] = {
        'best_val_loss': stats.get('best_val_loss'),
        'epochs_run': stats.get('total_epochs'),
        'training_time_s': stats.get('training_time'),
        'train_wall_s': train_time,
        'pred_wall_s': pred_time,
    }

    pred_col = f"{y_name}_Pred"
    model_type = schema_d.get('model_type', 'regression')
    if y_name in working_test_df.columns and pred_col in prediction_df.columns:
        if model_type == 'classification':
            y_true = working_test_df[y_name]
            y_pred = prediction_df[pred_col]
            metrics['test_accuracy'] = float((y_true == y_pred).mean())
        else:
            yt = working_test_df[y_name].to_numpy().astype(np.float64)
            yp = prediction_df[pred_col].to_numpy().astype(np.float64)
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.any():
                yt, yp = yt[mask], yp[mask]
                metrics['test_mae'] = float(np.mean(np.abs(yt - yp)))
                metrics['test_rmse'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
                a_std = float(np.std(yt))
                p_std = float(np.std(yp))
                metrics['actual_std'] = a_std
                metrics['pred_std'] = p_std
                metrics['variance_ratio'] = (p_std / a_std) if a_std > 0 else float('nan')

    # Free GPU between trials
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Coordinate-descent search
# ──────────────────────────────────────────────────────────────────────

def _print_summary_line(record: Dict[str, Any], selection_metric: str) -> None:
    mv = record.get(selection_metric)
    parts = []
    if isinstance(mv, (int, float)):
        parts.append(f"{selection_metric}={mv:.6f}")
    else:
        parts.append(f"{selection_metric}={mv}")
    for k, fmt in [
        ('test_accuracy', '.4f'),
        ('test_mae', '.4f'),
        ('test_rmse', '.4f'),
        ('variance_ratio', '.3f'),
        ('best_val_loss', '.6f'),
        ('epochs_run', 'd'),
        ('train_wall_s', '.1f'),
    ]:
        v = record.get(k)
        if v is None:
            continue
        if k == selection_metric:
            continue
        try:
            if fmt == 'd':
                parts.append(f"{k}={int(v)}")
            else:
                parts.append(f"{k}={v:{fmt}}")
        except Exception:
            parts.append(f"{k}={v}")
    print("   → " + "  ".join(parts))


def coordinate_descent_search(
    *,
    y_name: str,
    club_or_tournament: str,
    baseline: Dict[str, Any],
    search_axes: List[Tuple[str, List[Any]]],
    selection_metric: str,
    selection_better: str,           # 'lower' or 'higher'
    output_csv: pathlib.Path,
    device: str = 'cuda',
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Top-level: load data, build shards once, sweep axes, return (best_config, all_trials)."""
    overall_t = time.time()
    print(f"\n{'='*72}")
    print(f"HYPERPARAMETER SEARCH — {y_name} ({club_or_tournament})")
    print(f"{'='*72}")
    print(f"Baseline:")
    for k, v in baseline.items():
        print(f"  {k}: {v!r}")
    print(f"Search axes ({len(search_axes)}):")
    n_total = 0
    for name, candidates in search_axes:
        print(f"  {name}: {candidates}")
        n_total += len(candidates)
    print(f"Total trials: {n_total}")
    print(f"Selection: {selection_metric} ({selection_better} is better)")
    print(f"CSV output: {output_csv}")

    # Load + prep
    train_df, test_df = load_data(club_or_tournament)
    working_df, working_test_df = build_working_dfs(train_df, test_df, y_name)
    del train_df, test_df
    gc.collect()
    working_df_for_training = working_df.drop(SPLIT_INDICATOR_COLS, strict=False)
    print(f"working_df_for_training shape={working_df_for_training.shape}")

    # Build schema + shards ONCE
    model_name = f"acbl_{club_or_tournament}_hp_search_{y_name.lower()}"
    print(f"\n[setup] Generating schema + shards (one-time)...")
    setup_t = time.time()
    base_schema = build_shards_once(
        working_df_for_training,
        y_name,
        model_name,
        initial_layers=baseline['layers'],
        initial_dropout=baseline['dropout'],
        initial_y_range=baseline.get('y_range', None),
        shard_rows_count=int(baseline.get('shard_rows_count', 2_000_000)),
    )
    print(f"[setup] Done in {time.time()-setup_t:.1f}s. model_type={base_schema.get('model_type')}")

    # Precompute class counts (only used by Contract / classification searches)
    class_counts = None
    if base_schema.get('model_type') == 'classification':
        class_counts = compute_class_counts(working_df_for_training, base_schema, y_name)
        if class_counts is not None:
            print(f"[setup] class counts: n={len(class_counts)} "
                  f"min={int(class_counts.min())} max={int(class_counts.max())} "
                  f"sum={int(class_counts.sum())}")

    # Free training DF; shards are on disk and class_counts are extracted
    del working_df_for_training, working_df
    gc.collect()

    # Sweep
    best_config = dict(baseline)
    all_trials: List[Dict[str, Any]] = []
    trial_idx = 0
    for axis_name, candidates in search_axes:
        print(f"\n--- [axis] '{axis_name}' candidates={candidates} (current best={best_config.get(axis_name)!r}) ---")
        axis_records = []
        for value in candidates:
            trial_config = dict(best_config)
            trial_config[axis_name] = value
            print(f"\n[trial {trial_idx}] {axis_name}={value!r}  config={ {k: trial_config[k] for k in baseline.keys()} }")
            t0 = time.time()
            try:
                metrics = run_trial(
                    base_schema=base_schema,
                    working_test_df=working_test_df,
                    y_name=y_name,
                    model_name=model_name,
                    config=trial_config,
                    class_counts=class_counts,
                    device=device,
                )
                record = {
                    'trial': trial_idx,
                    'axis': axis_name,
                    'value': str(value),
                    **{k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in trial_config.items()},
                    **metrics,
                    'wall_s': time.time() - t0,
                }
            except Exception as e:
                print(f"   ❌ trial failed: {type(e).__name__}: {e}")
                record = {
                    'trial': trial_idx,
                    'axis': axis_name,
                    'value': str(value),
                    **{k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in trial_config.items()},
                    'error': f"{type(e).__name__}: {e}",
                    'wall_s': time.time() - t0,
                }
            axis_records.append(record)
            all_trials.append(record)
            _print_summary_line(record, selection_metric)
            trial_idx += 1

            # Incremental CSV save
            try:
                pl.DataFrame(all_trials).write_csv(output_csv)
            except Exception as e:
                print(f"   ⚠️ Could not write CSV: {e}")

        # Pick best for this axis
        scored = [
            r for r in axis_records
            if isinstance(r.get(selection_metric), (int, float)) and not np.isnan(r[selection_metric])
        ]
        if not scored:
            print(f"   ⚠️ No valid scores for axis '{axis_name}'; keeping current best={best_config[axis_name]!r}")
            continue
        if selection_better == 'lower':
            best_record = min(scored, key=lambda r: r[selection_metric])
        else:
            best_record = max(scored, key=lambda r: r[selection_metric])
        # Recover original (un-stringified) candidate value
        best_str = best_record.get('value')
        best_value = next((v for v in candidates if str(v) == best_str), best_str)
        prev = best_config[axis_name]
        best_config[axis_name] = best_value
        print(f"   *** BEST {axis_name}: {best_value!r} (was {prev!r})  "
              f"{selection_metric}={best_record[selection_metric]:.6f} ***")

    total_min = (time.time() - overall_t) / 60.0
    print(f"\n{'='*72}")
    print(f"SEARCH COMPLETE — {len(all_trials)} trials in {total_min:.1f} min")
    print(f"{'='*72}")
    print(f"FINAL BEST CONFIG ({y_name}):")
    for k, v in best_config.items():
        print(f"  {k}: {v!r}")

    # Top-N ranking
    valid = [
        r for r in all_trials
        if isinstance(r.get(selection_metric), (int, float)) and not np.isnan(r[selection_metric])
    ]
    if valid:
        valid.sort(key=lambda r: r[selection_metric], reverse=(selection_better == 'higher'))
        print(f"\nTop 10 trials by {selection_metric}:")
        for r in valid[:10]:
            extras = []
            for k in ('test_accuracy', 'test_mae', 'variance_ratio', 'best_val_loss', 'epochs_run'):
                v = r.get(k)
                if v is None or k == selection_metric:
                    continue
                if isinstance(v, float):
                    extras.append(f"{k}={v:.4f}")
                else:
                    extras.append(f"{k}={v}")
            print(f"  trial={r['trial']:3d}  {r['axis']:>15s}={r['value']:>22s}  "
                  f"{selection_metric}={r[selection_metric]:.6f}  " + "  ".join(extras))

    # Cleanup shards (keep model + schema for inspection if desired)
    cleanup_shards(model_name)
    return best_config, all_trials


def parse_club_tournament_args(default: str = 'club') -> str:
    """Parse a single --club / --tournament flag. Returns 'club' or 'tournament'."""
    import argparse
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--club', action='store_true')
    g.add_argument('--tournament', action='store_true')
    args = parser.parse_args()
    if args.tournament:
        return 'tournament'
    if args.club:
        return 'club'
    return default
