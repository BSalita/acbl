"""
Train ACBL prediction models.

Takes 6h/8.5h for 15m/14m rows x 5963/5963 columns.
Uses 2.8TB/1TB of memory/pagefile and < 1GB of dedicated GPU memory.
Predict Declarer_Direction, Contract, Pct_NS.

WARNING: if pytorch is acting weird, try closing notebook/server (close vscode) and try again.
Make sure 'gpu' is being used to train. If not, try:
  1. pip uninstall torch torchvision torchaudio -y
  2. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"""

import os
# Silence GT 1030 / multi-GPU mismatch warnings by only exposing the primary GPU (index 0).
# Set BEFORE importing torch. Override by exporting CUDA_VISIBLE_DEVICES yourself.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
# Windows powershell defaults to cp1252; force UTF-8 so the various status-line
# emoji/glyphs printed below (✅, 📊, 📦, 🔁, …) don't crash the script with
# UnicodeEncodeError. Same fix as in acbl_hp_search_lib.py.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import polars as pl
from collections import defaultdict
import pathlib
import re
import pickle
import gc
import time
from typing import Any, Optional, Dict, List, Tuple, Union
import numpy as np

sys.path.append(str(pathlib.Path.cwd().parent.joinpath('mlBridgeLib')))

import mlBridge
from mlBridge.mlBridgeAiLib import (
    generate_and_save_schema,
    df_to_scaled_tensors,
    train_model_from_df,
    train_model_from_tensors,
    create_torch_shards,
    train_model_from_shards,
    predict_model,
    analyze_prediction_results,
    display_feature_importances,
)
# Feature-pruning rules for Pct_NS (kept in the HP search lib so that
# acbl_prediction_train.py and acbl_hp_search_pct_ns.py prune identically).
from acbl_hp_search_lib import prune_pct_ns_features

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')
savedModelsPath = acblPath.joinpath('SavedModels')

y_names = ['Declarer_Direction', 'Contract', 'Pct_NS']

def _align_inference_features(predict_df: pl.DataFrame, schema: dict, y_name: str) -> pl.DataFrame:
    """Add missing columns with safe defaults and select features in the schema's training order.

    Mirrors the alignment block already used inline in train_predictions(), so OOF / test
    inference uses the same logic as the main pipeline.
    """
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
    missing = [c for c in expected_features if c not in out.columns]
    for col in missing:
        out = out.with_columns(pl.lit(default_for_dtype(expected_dtypes.get(col))).alias(col))

    extra = [c for c in out.columns if c not in expected_features and c != y_name]
    if extra:
        out = out.drop(extra)
    return out.select(expected_features)


def _train_oof_classifier_fold(
    train_df: pl.DataFrame,
    predict_df: pl.DataFrame,
    y_name: str,
    club_or_tournament: str,
    fold_id: int,
    *,
    layers,
    dropout: float,
    epochs: int,
    bs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    shard_rows_count: int,
    early_stop_patience: int,
    class_weights,
    device: str,
) -> pl.DataFrame:
    """Train a throw-away classifier on `train_df`; return predictions on `predict_df`.

    Used for OOF stacking. Saves under a unique model_name and cleans up shards after.
    `predict_df` row order is preserved in the returned DataFrame.
    """
    import math
    import json as _json
    import importlib
    from mlBridge.mlBridgeAiLib import (
        generate_and_save_schema,
        create_torch_shards,
        train_model_from_shards,
        predict_model,
    )

    model_name = f"acbl_{club_or_tournament}_oof_{y_name.lower()}_fold{fold_id}_torch_model"
    print(f"   [OOF/{y_name} fold {fold_id}] Training on {len(train_df):,} rows; predicting {len(predict_df):,} rows")

    split_indicator_cols = ['is_train_set', 'is_val_set', 'is_test_set']
    train_features_df = train_df.drop([c for c in split_indicator_cols if c in train_df.columns], strict=False)

    schema_d = generate_and_save_schema(
        train_features_df,
        savedModelsPath,
        model_name,
        y_name,
        layers=layers,
        dropout=dropout,
        apply_scaling_parameters=True,
        y_range=None,
        verbose=False,
    )

    for p in savedModelsPath.glob(f"{model_name}_shard_*.pt"):
        if p.is_file():
            p.unlink()

    num_rows = len(train_features_df)
    est_shards = math.ceil(num_rows / shard_rows_count) if shard_rows_count else 0
    effective_shard_rows = shard_rows_count if est_shards >= 2 else max(1, num_rows // 2)
    create_torch_shards(
        train_features_df,
        schema_d,
        shard_rows_count=effective_shard_rows,
        apply_scaling=True,
    )
    del train_features_df
    gc.collect()

    train_model_from_shards(
        schema_d,
        epochs=epochs,
        bs=bs,
        lr=lr,
        use_amp=use_amp,
        weight_decay=weight_decay,
        device=device,
        seed=42 + fold_id,
        early_stop_patience=early_stop_patience,
        class_weights=class_weights,
        verbose=False,
    )

    schema_path = pathlib.Path(savedModelsPath) / f"{model_name}_schema.json"
    with open(schema_path, "r") as f:
        schema = _json.load(f)
    features_df = _align_inference_features(predict_df, schema, y_name)
    pred_df = predict_model(savedModelsPath, model_name, features_df)

    for p in savedModelsPath.glob(f"{model_name}_shard_*.pt"):
        if p.is_file():
            p.unlink()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return pred_df


def generate_oof_predictions(
    working_df: pl.DataFrame,
    y_name: str,
    club_or_tournament: str,
    *,
    layers,
    dropout: float,
    epochs: int,
    bs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    shard_rows_count: int,
    early_stop_patience: int,
    class_weights,
    device: str,
) -> pl.DataFrame:
    """Run 2-fold OOF for a classifier and return predictions aligned to `working_df` row order.

    Splits deterministically on row_index % 2. Returns a DataFrame with one row per input row,
    in the same order, containing at least `{y_name}_Pred` (predicted class label).
    """
    n = len(working_df)
    df_with_idx = working_df.with_row_index("_oof_row_idx")
    fold_a = df_with_idx.filter(pl.col("_oof_row_idx") % 2 == 0)
    fold_b = df_with_idx.filter(pl.col("_oof_row_idx") % 2 == 1)
    print(f"   [OOF/{y_name}] 2-fold split: fold_a={len(fold_a):,}, fold_b={len(fold_b):,} (total={n:,})")

    common_kwargs = dict(
        layers=layers,
        dropout=dropout,
        epochs=epochs,
        bs=bs,
        lr=lr,
        weight_decay=weight_decay,
        use_amp=use_amp,
        shard_rows_count=shard_rows_count,
        early_stop_patience=early_stop_patience,
        class_weights=class_weights,
        device=device,
    )

    pred_b = _train_oof_classifier_fold(
        train_df=fold_a.drop("_oof_row_idx"),
        predict_df=fold_b.drop("_oof_row_idx"),
        y_name=y_name, club_or_tournament=club_or_tournament, fold_id=0,
        **common_kwargs,
    )
    pred_a = _train_oof_classifier_fold(
        train_df=fold_b.drop("_oof_row_idx"),
        predict_df=fold_a.drop("_oof_row_idx"),
        y_name=y_name, club_or_tournament=club_or_tournament, fold_id=1,
        **common_kwargs,
    )

    # Assert row counts match what we predicted on (predict_model preserves row order/count
    # by default; if it ever drops rows we want to fail loudly rather than misalign labels).
    assert len(pred_a) == len(fold_a), f"OOF fold A: predicted {len(pred_a)} rows but expected {len(fold_a)}"
    assert len(pred_b) == len(fold_b), f"OOF fold B: predicted {len(pred_b)} rows but expected {len(fold_b)}"

    pred_a = pred_a.with_columns(fold_a["_oof_row_idx"].alias("_oof_row_idx"))
    pred_b = pred_b.with_columns(fold_b["_oof_row_idx"].alias("_oof_row_idx"))
    combined = pl.concat([pred_a, pred_b], how="diagonal_relaxed").sort("_oof_row_idx").drop("_oof_row_idx")
    return combined


def check_prediction_variance(pred_df, target_col: str, pred_col: str) -> float:
    """Print std of actual vs predictions and return variance ratio (pred_std/actual_std)."""
    actual_std = float(pred_df[target_col].std())
    pred_std = float(pred_df[pred_col].std())
    ratio = pred_std / actual_std if actual_std != 0 else float('nan')
    print("\n\U0001F4CA VARIANCE ANALYSIS:")
    print(f"   Actual std: {actual_std:.6f}")
    print(f"   Predicted std: {pred_std:.6f}")
    print(f"   Variance ratio: {ratio:.3f}")
    if np.isnan(ratio):
        print("   534 WARNING: Actual std is zero or NaN.")
    elif ratio < 0.3:
        print("   534 SEVERE: Predictions severely underfit (ratio < 0.3)")
    elif ratio < 0.5:
        print("   7e0 MODERATE: Predictions somewhat underfit (ratio < 0.5)")
    elif ratio > 1.2:
        print("   7e0 MODERATE: Predictions may be overfitting (ratio > 1.2)")
    else:
        print("   7e2 GOOD: Prediction variance is reasonable")
    return ratio

def _parse_cli_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train ACBL prediction models.")
    parser.add_argument("--club", action="store_true", help="Process club data only")
    parser.add_argument("--tournament", action="store_true", help="Process tournament data only")
    parser.add_argument(
        "--target", "-y", action="append", choices=y_names, default=None,
        help=("Restrict training to one or more targets. Repeat the flag to train "
              "several (e.g. --target Pct_NS). Default: train all three."),
    )
    parser.add_argument(
        "--input-suffix", default="_v2",
        help=("Suffix on the input prediction parquet filenames. Must match the "
              "--output-suffix passed to acbl_prediction_data.py. Default '_v2' "
              "matches the current acbl_prediction_data.py default. Pass '' to "
              "read the legacy un-suffixed files."),
    )
    args = parser.parse_args()
    if not args.club and not args.tournament:
        modes = ["club", "tournament"]
    else:
        modes = []
        if args.club:
            modes.append("club")
        if args.tournament:
            modes.append("tournament")
    targets = args.target if args.target else list(y_names)
    return modes, targets, args.input_suffix


def train_predictions(club_or_tournament, targets: Optional[List[str]] = None,
                      input_suffix: str = "_v2"):
    """Train ACBL prediction models for the requested targets (default: all three).

    `targets` controls which y-variables are trained this run. Targets are still
    iterated in the canonical `y_names` order so any cross-target dependencies
    (e.g. classifier OOF caches) are produced before they could be consumed.

    `input_suffix` selects which generation of prediction parquets to read.
    Default '_v2' matches `acbl_prediction_data.py --output-suffix _v2`.
    """
    t = time.time()
    selected_targets = [y for y in y_names if y in (targets or y_names)]
    print(f"\nProcessing {club_or_tournament} prediction training... "
          f"targets={selected_targets}  input_suffix='{input_suffix}'")

    acbl_prediction_data_train_filename = f"acbl_{club_or_tournament}_prediction_data_train{input_suffix}.parquet"
    acbl_prediction_data_train_file = acblPath.joinpath(acbl_prediction_data_train_filename)
    if not acbl_prediction_data_train_file.exists():
        raise FileNotFoundError(
            f"Training parquet not found: {acbl_prediction_data_train_file}. "
            f"Re-run acbl_prediction_data.py --{club_or_tournament} "
            f"--output-suffix '{input_suffix}', or pass --input-suffix to match an "
            f"existing file."
        )
    model_df = pl.read_parquet(acbl_prediction_data_train_file)
    print(f"Loaded {acbl_prediction_data_train_filename}: shape:{model_df.shape} size:{acbl_prediction_data_train_file.stat().st_size}")

    acbl_prediction_data_test_filename = f"acbl_{club_or_tournament}_prediction_data_test{input_suffix}.parquet"
    acbl_prediction_data_test_file = acblPath.joinpath(acbl_prediction_data_test_filename)
    df_test = pl.read_parquet(acbl_prediction_data_test_file)
    print(f"Loaded {acbl_prediction_data_test_filename}: shape:{df_test.shape} size:{acbl_prediction_data_test_file.stat().st_size}")

    # ── Enum -> Categorical compatibility shim (2026-04-21) ─────────────────
    # acbl_prediction_data.py emits categorical features as pl.Enum (required
    # for streaming-safe sink_parquet). The training stack in
    # mlBridge/mlBridgeAiLib.py predates that switch and still assumes
    # pl.Categorical in ~20 places (validate_training_dataframe_dtypes,
    # _is_categorical_dtype, schema generators, prediction-code casts, etc.).
    # Cast Enum back to Categorical at load time so the trainer is happy.
    # See TODO.md "Training pipeline does not understand pl.Enum natively".
    enum_cols = [c for c, dt in model_df.schema.items() if isinstance(dt, pl.Enum)]
    if enum_cols:
        print(f"  Casting {len(enum_cols)} Enum -> Categorical for trainer compat: "
              f"{enum_cols[:8]}{'...' if len(enum_cols) > 8 else ''}")
        model_df = model_df.with_columns(
            [pl.col(c).cast(pl.String).cast(pl.Categorical) for c in enum_cols]
        )
        test_enum_cols = [c for c in enum_cols if c in df_test.columns]
        if test_enum_cols:
            df_test = df_test.with_columns(
                [pl.col(c).cast(pl.String).cast(pl.Categorical) for c in test_enum_cols]
            )


    # for y_name in y_names:

    #     # Keep only the current target; drop other targets to avoid leakage
    #     other_targets = [t for t in y_names if t != y_name and t in model_df.columns]
    #     working_df = model_df.select(pl.exclude(other_targets))
    #     working_test_df = df_test.select(pl.exclude(other_targets))

    #     # Verify split indicators are preserved in working dataframes
    #     assert 'is_train_set' in working_df.columns, "Split indicators missing from training data"
    #     assert 'is_test_set' in working_test_df.columns, "Split indicators missing from test data"
    #     print(f"✅ Split indicators preserved for {y_name}")

    #     # save model-ready data (includes split indicators as metadata)
    #     acbl_club_model_data_filename = f"acbl_{club_or_tournament}_working_{y_name.lower()}.parquet"
    #     acbl_club_model_data_file = acblPath.joinpath(acbl_club_model_data_filename)
    #     working_df.write_parquet(acbl_club_model_data_file)
    #     print(f"Saved {acbl_club_model_data_filename}: shape:{working_df.shape} size:{acbl_club_model_data_file.stat().st_size}")

    #model_df.select(pl.col('^.*Elo.*$')).columns

    import importlib
    import mlBridge.mlBridgeAiLib
    importlib.reload(mlBridge.mlBridgeAiLib)
    from mlBridge.mlBridgeAiLib import predict_model

    # ── Classifier prediction caches (currently unused by Pct_NS) ───────────────
    # Reserved for future stacking experiments. We still populate
    # `test_classifier_preds[y]` (full-model test predictions for each classifier
    # `y`) at no extra cost, but the per-row OOF generation step is skipped — see
    # the note where Pct_NS's `targets_to_keep` is set, and the OOF block at the
    # end of this loop. `oof_classifier_preds` is intentionally left empty.
    oof_classifier_preds: Dict[str, pl.DataFrame] = {}
    test_classifier_preds: Dict[str, pl.DataFrame] = {}

    for y_name in selected_targets:

        print('\n', "=" * 50)

        # Drop other targets that would be leakage (determined at same time or after current target).
        #
        # Pct_NS now uses ONLY pre-board information: hand-record data (deals, hands, HCP,
        # suit lengths, distribution/total points, quick tricks, double-dummy tricks, par
        # score, par contracts, expected values, LTC, etc.) and seating data (which players
        # are at which seat, master points, Elo and other quality metrics). It must NOT see
        # any results or auction-derived signals. In particular, Contract and
        # Declarer_Direction are auction outcomes and are excluded — even though they are
        # temporally prior to Pct_NS, they are not "knowable before the boards start to be
        # played" and were creating an information channel from the auction into the
        # regressor (previously bridged via OOF stacking, see comments below).
        # The upstream parquet (acbl_prediction_data.py, game_state=5) already restricts
        # columns to game-state levels 0–4 (board / deal / event / players / session), so
        # the only remaining results-leakage risk is the two classification targets.
        targets_to_keep = {
            'Declarer_Direction': [],
            'Contract': [],
            'Pct_NS': [],
        }
        keep = targets_to_keep.get(y_name, [])
        other_targets = [t for t in y_names if t != y_name and t not in keep and t in model_df.columns]
        if keep:
            kept_in_df = [t for t in keep if t in model_df.columns]
            print(f"Keeping {kept_in_df} as features for {y_name} (temporally prior)")

        working_df = model_df.select(pl.exclude(other_targets))
        working_test_df = df_test.select(pl.exclude(other_targets))

        # Drop the noisy high-cardinality feature families for Pct_NS (per-card
        # Booleans, fully-broken-out EV cube, per-(strain,level) Probs table).
        # See `PCT_NS_DROP_PATTERNS` in acbl_hp_search_lib.py for the full
        # rationale — based on the importance report from the first
        # post-leakage-fix run, those ~4688 columns sat at the noise floor.
        if y_name == 'Pct_NS':
            working_df = prune_pct_ns_features(working_df, verbose=True)
            working_test_df = prune_pct_ns_features(working_test_df, verbose=False)

        # ── OOF stacking is intentionally NOT applied for Pct_NS ─────────────────────
        # Historically Pct_NS was trained with Declarer_Direction and Contract as input
        # features, replaced at train time with OOF predictions to avoid train/serve
        # skew. Pct_NS now uses only pre-auction information (see `targets_to_keep`),
        # so neither the OOF replacement nor the classifier OOF generation block at the
        # bottom of this loop runs. The `oof_classifier_preds` / `test_classifier_preds`
        # caches remain available for future stacking experiments but are unused.

        # CRITICAL: Define split indicator columns - these must be excluded from training
        # to prevent data leakage. They are metadata columns for tracking which rows
        # were used in train/val/test splits, useful for inference and analysis.
        split_indicator_cols = ['is_train_set', 'is_val_set', 'is_test_set']

        # Verify split indicators are present
        for col in split_indicator_cols:
            if col not in working_df.columns:
                print(f"⚠️  WARNING: Split indicator '{col}' not found in working_df")

        # Create training dataframe with split indicators EXCLUDED from features
        # Split indicators will remain in the dataframe for metadata/tracking but won't be trained on
        working_df_for_training = working_df.drop(split_indicator_cols, strict=False)
        print(f"✅ Excluded {len([c for c in split_indicator_cols if c in working_df.columns])} split indicator columns from training features")
        print(f"   Training on {len(working_df_for_training.columns)} features (including target '{y_name}')")

        model_name = f"acbl_{club_or_tournament}_predicted_{y_name.lower()}_torch_model"
        print(f"{model_name=}")

        # takes 10m/
        # Define architecture first so it can be saved in schema
        # Use more moderate architecture to prevent NaN issues
        # optimal_layers = [1536, 768, 384, 192]
        # optimal_dropout = 0.10
        # optimal_epochs = 20
        # optimal_lr = 3e-4
        # optimal_weight_decay = 1e-5
        # optimal_bs = 1024  # use 512 if VRAM constrained

        # Per-target hyperparameter overrides come from the
        # acbl_hp_search_<target>.py coordinate-descent searches. Anything not
        # explicitly overridden falls through to the shared defaults below.
        optimal_layers = [2048, 1024, 512, 256]
        optimal_dropout = 0.10
        optimal_epochs = 20 # 3
        optimal_lr = 1e-3 # reduce LR to improve stability
        optimal_weight_decay = 1e-4 #1e-5
        optimal_bs = 8192
        optimal_shard_rows_count = 2_000_000
        optimal_use_amp =  False # False
        optimal_y_range = None
        # Early-stopping patience (epochs without val_loss improvement). 0 disables.
        optimal_early_stop_patience = 3

        # ── Declarer_Direction (club): hp search results, 18 trials, 156.9 min ─
        # Best test_accuracy = 0.6016 (vs 0.5923 with old defaults). Smaller
        # network won by ~1pp; weight_decay was unnecessary; longer patience
        # helped weaker configs converge but didn't help the winner. See
        # e:/bridge/data/acbl/acbl_hp_search_declarer_direction_club.csv.
        if y_name == 'Declarer_Direction':
            optimal_layers = [1024, 512, 256, 128]
            optimal_weight_decay = 0.0
            optimal_early_stop_patience = 5

        # ── Contract (club): hp search results, 17 trials, 158.6 min ──────────
        # Best test_accuracy = 0.3131 (vs 0.1942 with old defaults; +11.9 pp).
        # The biggest single win was DISABLING inverse-frequency class
        # weighting — the gentler sqrt/log variants also hurt accuracy. Higher
        # dropout (0.20) and lower lr (3e-4) added small additional gains.
        # See e:/bridge/data/acbl/acbl_hp_search_contract_club.csv.
        if y_name == 'Contract':
            optimal_dropout = 0.20
            optimal_lr = 3e-4
            optimal_weight_decay = 0.0
            optimal_early_stop_patience = 5

        # ── Pct_NS (club): from acbl_hp_search_pct_ns.py --club ───────────────
        # 20-trial coordinate-descent search on the pruned 1347-feature input
        # (acbl_hp_search_pct_ns_club.csv). Best test_mae=0.263274,
        # variance_ratio=0.165, best_val_loss=0.092705 @ epoch 17.
        #   - The smallest of 4 candidate architectures won. Pre-board signal is
        #     limited; bigger nets just memorize noise without generalizing.
        #   - lr 3e-4 (3× the manual quick-fix). The pruned input is well-
        #     conditioned enough to tolerate a faster step.
        #   - y_range=(0,1) saturated again (test_mae=0.434, ~"predict the
        #     mean"). Sigmoid bound stays off — clip downstream if needed.
        #   - weight_decay and bs were essentially flat across candidates;
        #     winners shown but ±0.0005 in mae either way.
        if y_name == 'Pct_NS':
            optimal_layers = [512, 256, 128]
            optimal_dropout = 0.05
            optimal_lr = 3e-4
            optimal_weight_decay = 1e-4
            optimal_bs = 8192
            optimal_y_range = None
            optimal_early_stop_patience = 5
        # optimal_grad_accum_steps = 2
        # optimal_num_workers = 16
        # optimal_pin_memory = True

        # CRITICAL: Use working_df_for_training (without split indicators) to generate schema
        # This ensures split indicators are not included as features in the model
        schema_d = generate_and_save_schema(
            working_df_for_training,  # WITHOUT split indicators
            savedModelsPath, 
            model_name, 
            y_name,
            # 🔧 SHARED PARAMETERS: Must match between training and inference
            layers=optimal_layers,  # Network architecture
            dropout=optimal_dropout,  # Dropout rate for model structure
            apply_scaling_parameters=True,  # Enable feature scaling (targets are not scaled)
            y_range=optimal_y_range,  # None today (no sigmoid bound); set to (lo,hi) to enable
            verbose=True
        )
        model_type = schema_d['model_type']
        print(f"Model type detected: {model_type}")
        print(f"🏗️ Architecture saved in schema: {optimal_layers} (dropout={optimal_dropout}, y_range={optimal_y_range})")

        # Verify schema contains correct architecture
        schema_layers = schema_d.get('mlp_layers', schema_d.get('layers', []))
        print(f"🔍 Schema layers: {schema_layers}")
        if schema_layers != optimal_layers:
            print(f"⚠️ WARNING: Schema layers don't match! Expected {optimal_layers}, got {schema_layers}")

        # remove all shards to free up space.
        print("Removing all shards to free up space...")
        for p in savedModelsPath.glob("*model_shard_*.pt"):
            if p.is_file():
                p.unlink()

        # takes 10m/?m
        # Create shards from training data WITHOUT split indicators
        # Ensure at least 2 shards so validation has data
        import math
        num_rows = len(working_df_for_training)
        est_shards = math.ceil(num_rows / optimal_shard_rows_count) if optimal_shard_rows_count else 0
        effective_shard_rows_count = optimal_shard_rows_count
        if est_shards < 2 and num_rows > 0:
            effective_shard_rows_count = max(1, num_rows // 2)
            print(f"[shards] Adjusted shard_rows_count from {optimal_shard_rows_count:,} to {effective_shard_rows_count:,} to ensure a validation shard (rows={num_rows:,})")
        shards_path = create_torch_shards(
            working_df_for_training,  # WITHOUT split indicators
            schema_d,
            shard_rows_count=effective_shard_rows_count,
            apply_scaling=True,  # Enable scaling of features
        )
        print(f"✅ Feature scaling enabled for inputs; targets remain unscaled")
        print(f"✅ Split indicators excluded from training shards")

        # takes 10m/12m
        # Remove old shards if they exist. Determine scale parameters, scale, create shard files.
        # todo: CategoricalRemappingWarning: Local categoricals have different encodings, expensive re-encoding is done to perform this merge operation.
        #     Consider using a StringCache or an Enum type if the categories are known in advance. shard_df.select(
        # This will automatically detect that 'Pct_NS' is regression and 'Declarer_Direction' is classification
        device = 'cuda' # 'cuda' or 'cpu'

        # CUDA diagnostics to catch unintended CPU fallback
        try:
            import torch, sys
            print(f"[CUDA] requested={device} available={torch.cuda.is_available()} exe={sys.executable}")
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available in this process. Check environment/kernel.")
        except Exception as e:
            print(f"[CUDA] diagnostic error: {e}")

        # Check actual target distribution
        if model_type == 'regression':
            target_stats = working_df_for_training[y_name].describe()
            print(f"📊 Target distribution: {target_stats}")
            actual_min = float(working_df_for_training[y_name].min())
            actual_max = float(working_df_for_training[y_name].max())
            print(f"📊 Actual range: {actual_min:.3f} to {actual_max:.3f}")

        # Class weights: disabled for both classifiers.
        # The hp search (acbl_hp_search_contract.py, 17 trials) showed that
        # ALL inverse-frequency variants — full, sqrt, and log — degraded
        # Contract test_accuracy vs unweighted training:
        #   class_weights=None              → 0.3131
        #   class_weights=log_inverse_freq  → 0.3087  (-0.4 pp)
        #   class_weights=sqrt_inverse_freq → 0.2788  (-3.4 pp)
        #   class_weights=inverse_freq (old)→ 0.1942  (-11.9 pp; broke training)
        # The model learns the natural class distribution better than any
        # reweighting. Declarer_Direction is balanced across N/E/S/W and was
        # never weighted. To re-enable an experiment, see compute_class_counts
        # + class_weights_from_counts in acbl_hp_search_lib.py.
        class_weights: Optional[List[float]] = None

        # Free training features DataFrame; shards are on disk now
        try:
            del working_df_for_training
        except Exception:
            pass
        gc.collect()

        # Adjust batch size for small dataset
        # optimal_bs = 512  # Smaller batch to add gradient noise
        # print(f"🔧 Adjusting batch size from 32768 to {optimal_bs} for dataset size {len(working_df)}")

        model, model_path, stats = train_model_from_shards(
            schema_d,
            # Training-only parameters (do not affect inference)
            epochs=optimal_epochs,
            bs=optimal_bs,
            lr=optimal_lr,
            use_amp=optimal_use_amp,  # Disable AMP to prevent numerical instability
            weight_decay=optimal_weight_decay,  # Minimal regularization
            device=device,
            seed=42,
            early_stop_patience=optimal_early_stop_patience,
            class_weights=class_weights,
            verbose=True
        )

        # 🔍 CAPTURE INPUT AND PREDICTIONS FOR DEBUGGING
        print(f"📊 Capturing input data for model: {model_name}")

        # Save input data for debugging
        input_capture_path = acblPath.joinpath(f"debug_input_{y_name.lower()}.parquet")
        working_test_df.write_parquet(input_capture_path)
        print(f"✅ Input data saved to: {input_capture_path}")
        print(f"   Shape: {working_test_df.shape}")
        print(f"   Columns: {len(working_test_df.columns)}")

        # === Schema alignment + prediction ===
        import json
        from pathlib import Path

        schema_path = Path(savedModelsPath) / f"{model_name}_schema.json"
        print(f"Loading schema from: {schema_path}")
        with open(schema_path, "r") as f:
            schema = json.load(f)

        expected_dtypes = schema.get("feature_dtypes")
        if not expected_dtypes:
            raise ValueError(f"Schema missing 'feature_dtypes'. Keys: {list(schema.keys())}")
        expected_features = list(expected_dtypes.keys())
        print(f"Expected features: {len(expected_features)}")

        # Identify differences vs inference df
        extra = [c for c in working_test_df.columns if c not in expected_features]
        missing = [c for c in expected_features if c not in working_test_df.columns]
        print(f"Extra columns ({len(extra)}): {extra}")
        print(f"Missing columns ({len(missing)}): {missing}")

        # Add missing columns with safe defaults
        def default_for_dtype(dtype_str):
            if dtype_str is None: return None
            ds = str(dtype_str).lower()
            if "int" in ds: return 0
            if "float" in ds: return 0.0
            if "bool" in ds: return False
            return None

        if missing:
            for col in missing:
                dtype = expected_dtypes.get(col)
                working_test_df = working_test_df.with_columns(pl.lit(default_for_dtype(dtype)).alias(col))
                print(f"  Added missing column: {col} with dtype {dtype}")

        # Preserve target (if present) and drop only truly extra features
        y_series = working_test_df[y_name] if y_name in working_test_df.columns else None
        extra_features = [c for c in extra if c != y_name]
        if extra_features:
            working_test_df = working_test_df.drop(extra_features)
            print(f"  Dropped {len(extra_features)} extra columns (kept target '{y_name}')")

        # Select features in training order
        features_df = working_test_df.select(expected_features)

        # Predict on features-only df
        prediction_df = predict_model(savedModelsPath, model_name, features_df)

        # Attach target back for evaluation/analysis (if preserved)
        if y_series is not None:
            prediction_df = prediction_df.with_columns([y_series.alias(y_name)])

        # Save predictions for debugging
        predictions_capture_path = acblPath.joinpath(f"debug_predictions_{y_name.lower()}.parquet")
        prediction_df.write_parquet(predictions_capture_path)
        print(f"✅ Predictions saved to: {predictions_capture_path}")
        print(f"   Shape: {prediction_df.shape}")

        print(prediction_df)

        if model_type == 'regression':
            # Quick variance check
            check_prediction_variance(prediction_df, y_name, f"{y_name}_Pred")

        match y_name:
            case 'Contract':
                # only need level strain for confusion matrix
                contract_df = prediction_df.with_columns([
                    pl.col(y_name).cast(pl.Utf8).str.slice(0, 2).cast(pl.Categorical).alias('Level_Strain'),
                    pl.col(f'{y_name}_Pred').cast(pl.Utf8).str.slice(0, 2).cast(pl.Categorical).alias('Level_Strain_Pred'),
                ])
                analyze_prediction_results(contract_df, 'Level_Strain')
            case _:
                analyze_prediction_results(prediction_df, y_name)

        fi = display_feature_importances(savedModelsPath, model_name, top_n=50, bottom_n=50, return_df=False)
        print(fi.head(10))
        acbl_club_model_data_importance_filename = savedModelsPath.joinpath(f"{model_name}_importance.csv")
        acbl_club_model_data_importance_file = acblPath.joinpath(acbl_club_model_data_importance_filename)
        fi.write_csv(acbl_club_model_data_importance_file, quote_style='non_numeric', include_bom=True)
        print(f"Saved {acbl_club_model_data_importance_filename}: shape:{fi.shape} size:{acbl_club_model_data_importance_file.stat().st_size}")

        # ── OOF stacking generation: disabled ───────────────────────────────────────
        # Pct_NS no longer consumes Declarer_Direction or Contract as features (see
        # `targets_to_keep` above), so there is no downstream consumer for OOF
        # predictions. The 2-fold OOF generation step (~equal in cost to the main
        # classifier training itself) is therefore skipped. Only the full-model test
        # predictions are cached, in case future code re-enables stacking.
        if y_name in ('Declarer_Direction', 'Contract'):
            pred_col = f"{y_name}_Pred"
            if pred_col in prediction_df.columns:
                test_classifier_preds[y_name] = prediction_df.select([pred_col])
                print(f"📦 Cached full-model test predictions for {y_name}: {len(prediction_df):,} rows")

        # Free per-target intermediates and GPU cache
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        for _v in ['prediction_df', 'features_df', 'y_series', 'contract_df']:
            if _v in locals():
                try:
                    del locals()[_v]
                except Exception:
                    pass
        gc.collect()

    assert model_df.select(pl.col(pl.Float64)).columns == [], model_df.select(pl.col(pl.Float64)).columns

    print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
    print("-" * 70, "\n")


if __name__ == "__main__":
    mlBridge.pd_options_display()
    savedModelsPath.mkdir(parents=True, exist_ok=True)

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    cli_modes, cli_targets, cli_suffix = _parse_cli_args()
    for club_or_tournament in cli_modes:
        train_predictions(club_or_tournament, targets=cli_targets,
                          input_suffix=cli_suffix)

    print_ended(program_start_time)

# # important! make sure saved model files are moved to the proper place for inferencing.

# # 🔍 DEBUG COMPARISON CELL - Read captured data and compare accuracy
# print("=" * 80)
# print("🔍 DEBUGGING CAPTURED DATA AND PREDICTIONS")
# print("=" * 80)

# # Define paths for captured data
# debug_files = {}
# for y_name in y_names:
#     input_path = acblPath.joinpath(f"debug_input_{y_name.lower()}.parquet")
#     predictions_path = acblPath.joinpath(f"debug_predictions_{y_name.lower()}.parquet")
    
#     if input_path.exists() and predictions_path.exists():
#         debug_files[y_name] = {
#             'input_path': input_path,
#             'predictions_path': predictions_path
#         }
#         print(f"✅ Found debug files for {y_name}")
#     else:
#         print(f"❌ Missing debug files for {y_name}")
#         if not input_path.exists():
#             print(f"   Missing: {input_path}")
#         if not predictions_path.exists():
#             print(f"   Missing: {predictions_path}")

# print(f"\n📊 Found debug files for {len(debug_files)} targets: {list(debug_files.keys())}")

# # Process each captured dataset
# for y_name, paths in debug_files.items():
#     print(f"\n{'='*60}")
#     print(f"🎯 ANALYZING {y_name.upper()}")
#     print(f"{'='*60}")
    
#     # Load captured input data
#     print(f"📥 Loading input data from: {paths['input_path']}")
#     captured_input_df = pl.read_parquet(paths['input_path'])
#     print(f"   Input shape: {captured_input_df.shape}")
    
#     # Load captured predictions
#     print(f"📥 Loading predictions from: {paths['predictions_path']}")
#     captured_predictions_df = pl.read_parquet(paths['predictions_path'])
#     print(f"   Predictions shape: {captured_predictions_df.shape}")
    
#     # Show basic info about the captured data
#     print(f"\n📊 INPUT DATA SUMMARY:")
#     print(f"   Columns: {len(captured_input_df.columns)}")
#     print(f"   Rows: {len(captured_input_df):,}")
#     if 'Date' in captured_input_df.columns:
#         date_range = captured_input_df['Date'].describe()
#         print(f"   Date range: {date_range}")
    
#     print(f"\n📊 PREDICTIONS SUMMARY:")
#     pred_col = f"{y_name}_Pred"
#     if pred_col in captured_predictions_df.columns:
#         if y_name == 'Pct_NS':  # Regression
#             pred_stats = captured_predictions_df[pred_col].describe()
#             actual_stats = captured_predictions_df[y_name].describe()
#             print(f"   Predicted {pred_col}: {pred_stats}")
#             print(f"   Actual {y_name}: {actual_stats}")
            
#             # Calculate accuracy metrics for regression
#             mse = ((captured_predictions_df[pred_col] - captured_predictions_df[y_name]) ** 2).mean()
#             mae = (captured_predictions_df[pred_col] - captured_predictions_df[y_name]).abs().mean()
#             print(f"   MSE: {mse:.6f}")
#             print(f"   MAE: {mae:.6f}")
            
#             # Variance analysis
#             check_prediction_variance(captured_predictions_df, y_name, pred_col)
            
#         else:  # Classification (Contract, Declarer_Direction)
#             # Show prediction distribution
#             pred_counts = captured_predictions_df[pred_col].value_counts().sort(pred_col)
#             actual_counts = captured_predictions_df[y_name].value_counts().sort(y_name)
#             print(f"   Unique predictions: {captured_predictions_df[pred_col].n_unique()}")
#             print(f"   Unique actuals: {captured_predictions_df[y_name].n_unique()}")
            
#             # Calculate accuracy
#             accuracy = (captured_predictions_df[pred_col] == captured_predictions_df[y_name]).mean()
#             print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
#             # Show top predictions vs actuals
#             print(f"\n   Top 10 predicted values:")
#             print(pred_counts.head(10))
#             print(f"\n   Top 10 actual values:")
#             print(actual_counts.head(10))
    
#     # 🔄 RE-RUN PREDICTION ON CAPTURED INPUT FOR COMPARISON
#     print(f"\n🔄 RE-RUNNING PREDICTION ON CAPTURED INPUT...")
#     model_name = f"acbl_{club_or_tournament}_predicted_{y_name.lower()}_torch_model"
    
#     try:
#         # Re-run prediction using the same input
#         new_prediction_df = predict_model(savedModelsPath, model_name, captured_input_df)
#         new_prediction_df = new_prediction_df.with_columns([
#             captured_input_df[y_name].alias(y_name),
#         ])
        
#         print(f"✅ New prediction completed")
#         print(f"   New predictions shape: {new_prediction_df.shape}")
        
#         # Compare old vs new predictions
#         print(f"\n🔍 COMPARING OLD vs NEW PREDICTIONS:")
        
#         if pred_col in captured_predictions_df.columns and pred_col in new_prediction_df.columns:
#             # Check if predictions are identical
#             old_preds = captured_predictions_df[pred_col]
#             new_preds = new_prediction_df[pred_col]
            
#             if y_name == 'Pct_NS':  # Regression - use tolerance for floating point comparison
#                 tolerance = 1e-6
#                 identical = (old_preds - new_preds).abs().max() < tolerance
#                 max_diff = (old_preds - new_preds).abs().max()
#                 mean_diff = (old_preds - new_preds).abs().mean()
#                 print(f"   Max difference: {max_diff:.8f}")
#                 print(f"   Mean difference: {mean_diff:.8f}")
#                 print(f"   Identical (within {tolerance}): {identical}")
#             else:  # Classification - exact comparison
#                 identical = (old_preds == new_preds).all()
#                 different_count = (old_preds != new_preds).sum()
#                 print(f"   Identical predictions: {identical}")
#                 print(f"   Different predictions: {different_count:,} out of {len(old_preds):,}")
                
#                 if different_count > 0:
#                     print(f"   Percentage different: {different_count/len(old_preds)*100:.4f}%")
            
#             # Compare accuracies
#             if y_name != 'Pct_NS':  # Classification accuracy
#                 old_accuracy = (captured_predictions_df[pred_col] == captured_predictions_df[y_name]).mean()
#                 new_accuracy = (new_prediction_df[pred_col] == new_prediction_df[y_name]).mean()
#                 print(f"   Old accuracy: {old_accuracy:.6f} ({old_accuracy*100:.4f}%)")
#                 print(f"   New accuracy: {new_accuracy:.6f} ({new_accuracy*100:.4f}%)")
#                 print(f"   Accuracy difference: {new_accuracy - old_accuracy:.6f}")
#             else:  # Regression metrics
#                 old_mse = ((captured_predictions_df[pred_col] - captured_predictions_df[y_name]) ** 2).mean()
#                 new_mse = ((new_prediction_df[pred_col] - new_prediction_df[y_name]) ** 2).mean()
#                 print(f"   Old MSE: {old_mse:.8f}")
#                 print(f"   New MSE: {new_mse:.8f}")
#                 print(f"   MSE difference: {new_mse - old_mse:.8f}")
        
#     except Exception as e:
#         print(f"❌ Error re-running prediction: {str(e)}")
#         print(f"   This might indicate a problem with the model or input data")

# print(f"\n{'='*80}")
# print("🎯 DEBUGGING SUMMARY COMPLETE")
# print("📝 Use this information to compare with results from the other program")
# print("🔍 Look for differences in:")
# print("   - Input data shape and content")
# print("   - Prediction distributions")
# print("   - Accuracy metrics")
# print("   - Consistency between runs")
# print("="*80)
# # Test the updated schema generation with feature dtypes
# print("=" * 80)
# print("🧪 TESTING UPDATED SCHEMA GENERATION WITH FEATURE DTYPES")
# print("=" * 80)

# # Create a small test dataframe to verify schema generation
# test_df = model_df.head(100).select([
#     'Contract', 'Declarer_Direction', 'Pct_NS',  # targets
#     'Board', 'Dealer', 'Vul_NS', 'Vul_EW',      # basic features
#     'C_ED7', 'C_WH2', 'QT_W',                   # sample features of different types
#     'Date'                                       # date feature
# ])

# print(f"📊 Test dataframe shape: {test_df.shape}")
# print(f"📊 Test dataframe schema:")
# for col, dtype in test_df.schema.items():
#     print(f"  {col}: {dtype}")

# # Test schema generation for Contract (classification)
# print(f"\n🔧 Testing schema generation for Contract (classification)...")
# test_model_name = "test_contract_schema"
# test_schema = generate_and_save_schema(
#     test_df, 
#     savedModelsPath, 
#     test_model_name, 
#     'Contract',
#     layers=[64, 32],  # Small architecture for test
#     dropout=0.1,
#     apply_scaling_parameters=True,
#     verbose=True
# )

# print(f"\n✅ Schema generated successfully!")
# print(f"📋 Schema keys: {list(test_schema.keys())}")

# # Check if feature_dtypes is in the schema
# if 'feature_dtypes' in test_schema:
#     print(f"\n🎯 SUCCESS: feature_dtypes found in schema!")
#     print(f"📊 Number of features with dtypes: {len(test_schema['feature_dtypes'])}")
#     print(f"📊 Sample feature dtypes:")
#     for i, (feature, dtype) in enumerate(test_schema['feature_dtypes'].items()):
#         if i < 10:  # Show first 10
#             print(f"  {feature}: {dtype}")
#         elif i == 10:
#             print(f"  ... and {len(test_schema['feature_dtypes']) - 10} more features")
#             break
# else:
#     print(f"❌ ERROR: feature_dtypes not found in schema!")

# # Verify schema version
# schema_version = test_schema.get('schema_version', 'unknown')
# print(f"\n📋 Schema version: {schema_version}")
# if schema_version == "1.1":
#     print(f"✅ Schema version updated correctly to 1.1")
# else:
#     print(f"⚠️  Schema version is {schema_version}, expected 1.1")

# # Clean up test schema file
# test_schema_file = savedModelsPath / f"{test_model_name}_schema.json"
# if test_schema_file.exists():
#     test_schema_file.unlink()
#     print(f"\n🧹 Cleaned up test schema file: {test_schema_file}")

# print(f"\n{'='*80}")
# print(f"🎉 SCHEMA TESTING COMPLETE")
# print(f"{'='*80}")

# # 📋 SUMMARY: Training Schema Enhanced with Feature Dtypes

# print("=" * 80)
# print("📋 TRAINING SCHEMA ENHANCEMENT SUMMARY")
# print("=" * 80)

# print("""
# 🎯 CHANGES MADE:

# 1. ✅ Enhanced generate_and_save_schema_core() function in mlBridgeAiLib.py
#    - Added 'feature_dtypes' mapping to schema
#    - Maps each training feature to its original Polars dtype
#    - Enables precise dtype tracking for training features

# 2. ✅ Updated schema version from "1.0" to "1.1"
#    - Reflects the new feature dtype information
#    - Maintains backward compatibility awareness

# 3. ✅ Enhanced verbose output
#    - Shows count of features with recorded dtypes
#    - Provides better visibility into schema contents

# 4. ✅ Updated both library files
#    - mlBridgeAiLib.py (main file)
#    - mlBridgeAiLib copy.py (backup/alternative file)

# 5. ✅ Added test cell to verify functionality
#    - Tests schema generation with sample data
#    - Validates feature_dtypes presence and content
#    - Confirms schema version update

# 📊 SCHEMA STRUCTURE (New in v1.1):
# {
#   "schema_version": "1.1",
#   "feature_column_list": [...],
#   "target_column": "...",
#   "feature_dtypes": {
#     "feature1": "Float32",
#     "feature2": "Boolean", 
#     "feature3": "Int32",
#     "feature4": "Categorical",
#     ...
#   },
#   ... (existing fields)
# }

# 🔧 USAGE:
# - The feature_dtypes field contains a mapping of feature_name -> dtype_string
# - Dtype strings are Polars dtype representations (e.g., "Float32", "Boolean", "Categorical")
# - This enables precise tracking of what dtypes were used during training
# - Useful for debugging, validation, and ensuring consistency between training and inference

# 🎉 BENEFITS:
# - Better debugging capabilities for dtype mismatches
# - Precise documentation of training feature types
# - Enhanced schema validation possibilities
# - Improved reproducibility and consistency tracking
# """)

# print("=" * 80)
# print("✅ SCHEMA ENHANCEMENT COMPLETE")
# print("=" * 80)

