#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_pct_ns_pipeline.py â€” Unattended end-to-end Pct_NS prediction pipeline.

Goal
----
Lift Pct_NS test-set variance ratio from the current ~0.165 baseline toward
â‰¥0.35, while producing per-board predictions, session-aggregated pair scores,
and luck-vs-skill confidence intervals. Designed to run unattended on either
side (club / tournament) or both, with crash-safe per-phase checkpointing.

Phases
------
  p0_baseline           Reference MLP on pruned pre-board features. Establishes
                        var_ratio reference; phase 9 falls back to this if no
                        later phase succeeded.
  p2_skill              Synthesize seat-Elo skill differential (correct, ~r=+0.19),
                        partnership tenure Ã— skill interactions, experience
                        asymmetry, par-direction Ã— skill interactions.
  p3_field_resid        Field-conditioned baseline. Fit OLS on the strong
                        skill features; train the deep MLP on the residual so
                        it focuses on what skill alone can't explain.
  p4_two_stage          Two-stage OOF: K-fold OOF predictions of Contract +
                        Declarer_Direction become input features (avoids the
                        leakage that disabled this previously).
  p5_aux                Auxiliary tail-event classifiers (game made, set,
                        slam tried, vulnerable game, sacrifice). OOF probs
                        as features.
  p6_ensemble           Deep ensemble (N seeds) of the best phase. Mean Â±
                        ensemble Ïƒ gives the prediction CI.
  p7_combined           Single model trained on club + tournament with a
                        dataset indicator feature; should improve generalization
                        on the smaller side.
  p8_calibration        Reliability diagram + isotonic recalibration on test
                        set predictions. Fixes systematic mean shifts.
  p9_session            Aggregate board predictions to sessionÃ—pair, compute
                        confidence intervals, build luck-vs-skill report.
                        ALWAYS re-runs (no checkpoint) so it picks up the
                        newest upstream phase outputs.

Usage
-----
  # Full unattended run, both sides, all phases
  python acbl_pct_ns_pipeline.py

  # Smoke test: club only, phases 0 + 9, on subsampled data (~10 min)
  python acbl_pct_ns_pipeline.py --club --smoke --phases p0_baseline,p9_session

  # Resume after interruption â€” skips phases with .done checkpoints
  python acbl_pct_ns_pipeline.py --resume

  # Wipe pipeline state and start over
  python acbl_pct_ns_pipeline.py --clean

  # Force-re-run a specific phase (removes its checkpoints, then runs)
  python acbl_pct_ns_pipeline.py --rerun p3_field_resid --resume

Outputs
-------
  e:/bridge/data/acbl/pct_ns_pipeline/
    checkpoints/        per-(phase, side) .done JSON markers (resume control)
    outputs/            per-(phase, side) predictions.parquet, metrics.json,
                        importance.csv (when relevant)
    final/              session_luck_vs_skill_<side>.parquet, summary JSON
    logs/               <timestamp>_pipeline.log + matching _config.json

Design notes
------------
* Each (phase, side) is its own atomic unit. A crash mid-Phase-3-tournament
  leaves Phase 0+2 club AND tournament intact; resume re-enters at the
  exact failed phase.
* Every phase reads upstream data from disk (predictions.parquet); no
  in-memory state passes between phases. So inspecting outputs/ at any
  point shows the full data state.
* Phase 9 is exempted from the skip-on-resume logic â€” it should always
  re-run because it consumes whatever the latest upstream phase produced.
* The Elo pair-column bug (TODO.md) is worked around in phase 2 by
  synthesizing the pair Elo from the working seat-level columns. After
  the data cascade re-runs, phase 2 will simplify automatically.
"""

import os
# Set BEFORE importing torch (via mlBridge below)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import dataclasses
import gc
import json
import logging
import pathlib
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Windows powershell defaults to cp1252; force UTF-8 so progress bars and
# the arrow / box-drawing characters in summary lines don't crash.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import polars as pl

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / 'mlBridge'
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f'mlBridge not found at {_MLBRIDGE}')
for _p in (_SRC_DIR, _MLBRIDGE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)

import mlBridge  # noqa: F401  (used for side-effects in mlBridge.mlBridgeAiLib)
from mlBridge.mlBridgeAiLib import predict_model

from acbl_hp_search_lib import (
    ACBL_PATH,
    PCT_NS_DROP_PATTERNS,
    SAVED_MODELS_PATH,
    SPLIT_INDICATOR_COLS,
    align_inference_features,
    build_shards_once,
    build_working_dfs,
    load_data as load_train_test,
    run_trial,
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Constants & paths
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

PIPELINE_DIR = ACBL_PATH / 'pct_ns_pipeline'
CHECKPOINTS_DIR = PIPELINE_DIR / 'checkpoints'
OUTPUTS_DIR = PIPELINE_DIR / 'outputs'
FINAL_DIR = PIPELINE_DIR / 'final'
LOGS_DIR = PIPELINE_DIR / 'logs'

TARGET_VAR_RATIO = 0.35
SIDES_ALL = ('club', 'tournament')

# Phase 9 (and any future aggregation-only phases) re-run on every invocation
# because they consume whatever upstream phase last produced predictions.
ALWAYS_RERUN_PHASES = frozenset({'p9_session'})


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class PipelineConfig:
    sides: List[str] = field(default_factory=lambda: list(SIDES_ALL))
    phases: List[str] = field(default_factory=list)
    resume: bool = False
    smoke: bool = False
    smoke_rows: int = 100_000
    seed: int = 42
    n_ensemble_seeds: int = 3
    cuda_device: int = 0
    target_var_ratio: float = TARGET_VAR_RATIO
    # Suffix appended to the prediction-data parquet filenames; default '_v2'
    # picks the corrected pair-Elo / uncapped MasterPoints regen from
    # acbl_prediction_data.py. Use '' to read the legacy un-suffixed files.
    input_suffix: str = "_v2"
    # Hard caps on rows pulled from the train/test parquets at scan time
    # (orthogonal to --smoke). The v2 club train parquet is 165 GB / 55M rows
    # / 6042 cols; eagerly loading it doesn't fit in 192 GB RAM. Setting these
    # to a sane value (e.g. 10M train, full test) bounds the working set.
    # `None` means "load everything"; smoke mode overrides via smoke_rows.
    train_sample_rows: Optional[int] = None
    test_sample_rows: Optional[int] = None

    # Baseline hyperparameters (winners from the prior Pct_NS HP search,
    # see acbl_hp_search_pct_ns.py docstring). Phases that train new models
    # start from these unless they override.
    baseline_hparams: Dict[str, Any] = field(default_factory=lambda: dict(
        layers=[512, 256, 128],
        dropout=0.05,
        lr=3e-4,
        weight_decay=1e-4,
        bs=8192,
        epochs=20,
        use_amp=False,
        early_stop_patience=5,
        shard_rows_count=2_000_000,
        y_range=None,
        class_weights=None,
    ))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Logging
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def setup_logging(log_path: pathlib.Path, level: str = 'INFO') -> logging.Logger:
    """Create a logger that tees to console and the timestamped log file."""
    logger = logging.getLogger('pct_ns_pipeline')
    logger.setLevel(level)
    logger.handlers.clear()
    # mlBridge (or one of its imports) installs a default handler on the root
    # logger; without disabling propagation our messages would be printed
    # twice, once by our handler and once by the root logger's default.
    logger.propagate = False
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Phase result + context
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class PhaseResult:
    success: bool
    skipped: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class PipelineContext:
    """Per-run state passed to every phase function."""

    def __init__(self, config: PipelineConfig, log: logging.Logger) -> None:
        self.config = config
        self.log = log
        self.outputs_dir = OUTPUTS_DIR
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.final_dir = FINAL_DIR

    def checkpoint_path(self, phase_id: str, side: str) -> pathlib.Path:
        return self.checkpoints_dir / f"{phase_id}_{side}.done"

    def is_done(self, phase_id: str, side: str) -> bool:
        return self.checkpoint_path(phase_id, side).exists()

    def mark_done(self, phase_id: str, side: str, metrics: Dict[str, Any]) -> None:
        path = self.checkpoint_path(phase_id, side)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'phase': phase_id,
                'side': side,
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics,
            }, f, indent=2, default=str)

    def output_path(self, phase_id: str, side: str, suffix: str) -> pathlib.Path:
        return self.outputs_dir / f"{phase_id}_{side}_{suffix}"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Training wrapper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Columns we try to preserve in the per-row predictions.parquet so that
# downstream phases (especially p9 session aggregation) can join back to
# session/pair/board context. Anything missing is silently skipped.
_PREDICTION_PRESERVE_COLS = (
    'event_id', 'session_id', 'Board', 'Date', 'Section_Pairs',
    'Pair_Number_NS', 'Pair_Number_EW',
    'MasterPoints_N', 'MasterPoints_S', 'MasterPoints_E', 'MasterPoints_W',
    'Elo_R_N_EventStart', 'Elo_R_S_EventStart',
    'Elo_R_E_EventStart', 'Elo_R_W_EventStart',
    'Elo_N_NS', 'Elo_N_EW',
    'Vul_NS', 'Vul_EW', 'Dealer',
)


def train_one_pct_ns_model(
    side: str,
    *,
    feature_engineer_fn: Optional[
        Callable[[pl.DataFrame, pl.DataFrame], Tuple[pl.DataFrame, pl.DataFrame]]
    ] = None,
    target_override: Optional[str] = None,
    hparams_override: Optional[Dict[str, Any]] = None,
    model_name_suffix: str = '',
    smoke_rows: Optional[int] = None,
    seed: int = 42,
    log: Optional[logging.Logger] = None,
    input_suffix: str = "",
    train_sample_rows: Optional[int] = None,
    test_sample_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """End-to-end: load â†’ prune â†’ optional FE â†’ schema/shards â†’ train â†’ predict.

    Returns a dict with:
      model_name : str
      metrics    : dict (mae/rmse/var_ratio + training stats from run_trial)
      hparams    : the resolved hyperparameters used
      predictions: pl.DataFrame with preserved key columns + y_true + y_pred

    The trained model artifacts are saved under SAVED_MODELS_PATH using
    model_name = f"acbl_{side}_pct_ns_pipeline{model_name_suffix}".
    """
    y_name = target_override or 'Pct_NS'
    log = log or logging.getLogger('pct_ns_pipeline')

    # Resolve the row caps for this call. `smoke_rows` (set by --smoke) wins
    # over the standing config caps. Whichever wins, the loader pushes a
    # slice() into scan_parquet so we never materialize the full 165 GB v2
    # club parquet. We take the TAIL of train (most recent year, temporally
    # closest to the test split). Column projection (PCT_NS_DROP_PATTERNS)
    # is also pushed into the scan to skip ~4,700 noise-floor cols at decode
    # time â€” without that, even a 10M-row slice would be ~240 GB.
    if smoke_rows is not None:
        train_sample = smoke_rows
        test_sample = smoke_rows * 10
    else:
        train_sample = train_sample_rows
        test_sample = test_sample_rows
    log.info(
        f"  load {side} train+test parquets (suffix={input_suffix!r}, "
        f"train_sample={train_sample}, test_sample={test_sample})"
    )
    train_df, test_df = load_train_test(
        side,
        input_suffix=input_suffix,
        train_sample_rows=train_sample,
        test_sample_rows=test_sample,
        drop_regex_patterns=PCT_NS_DROP_PATTERNS if y_name == 'Pct_NS' else None,
    )

    working_df, working_test_df = build_working_dfs(train_df, test_df, y_name)
    del train_df, test_df
    gc.collect()

    if smoke_rows is not None and len(working_df) > smoke_rows:
        log.info(f"  SMOKE mode: subsampling train {len(working_df):,} â†’ {smoke_rows:,}")
        working_df = working_df.head(smoke_rows)

    if feature_engineer_fn is not None:
        log.info(f"  apply custom feature engineering")
        before_cols = working_df.width
        working_df, working_test_df = feature_engineer_fn(working_df, working_test_df)
        log.info(f"    cols: {before_cols} â†’ {working_df.width}")

    working_df_for_training = working_df.drop(SPLIT_INDICATOR_COLS, strict=False)
    log.info(f"  shape={working_df_for_training.shape}, target={y_name}")

    # Merge baseline + override
    hparams: Dict[str, Any] = {
        'layers': [512, 256, 128],
        'dropout': 0.05,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'bs': 8192,
        'epochs': 20,
        'use_amp': False,
        'early_stop_patience': 5,
        'shard_rows_count': 2_000_000,
        'y_range': None,
        'class_weights': None,
    }
    if hparams_override:
        hparams.update(hparams_override)

    model_name = f"acbl_{side}_pct_ns_pipeline{model_name_suffix}"
    log.info(f"  build shards (model_name={model_name})")
    base_schema = build_shards_once(
        working_df_for_training,
        y_name,
        model_name,
        initial_layers=hparams['layers'],
        initial_dropout=hparams['dropout'],
        initial_y_range=hparams.get('y_range', None),
        shard_rows_count=int(hparams['shard_rows_count']),
    )

    del working_df_for_training, working_df
    gc.collect()

    log.info(f"  train (seed={seed})")
    metrics = run_trial(
        base_schema=base_schema,
        working_test_df=working_test_df,
        y_name=y_name,
        model_name=model_name,
        config=hparams,
        class_counts=None,
        device='cuda',
        seed=seed,
    )

    # Re-run prediction explicitly so we can capture the per-row predictions
    # joined with the preserved key columns. (run_trial only computes metrics.)
    log.info(f"  predict on test (for downstream phases)")
    schema_path = SAVED_MODELS_PATH / f"{model_name}_schema.json"
    with open(schema_path, 'r', encoding='utf-8') as f:
        on_disk_schema = json.load(f)
    features_df = align_inference_features(working_test_df, on_disk_schema, y_name)
    pred_df = predict_model(SAVED_MODELS_PATH, model_name, features_df)
    pred_col = f"{y_name}_Pred"

    preserve_cols = [c for c in _PREDICTION_PRESERVE_COLS if c in working_test_df.columns]
    if y_name in working_test_df.columns and y_name not in preserve_cols:
        preserve_cols.append(y_name)
    out_pred = working_test_df.select(preserve_cols)
    if pred_col in pred_df.columns:
        out_pred = out_pred.with_columns(pred_df[pred_col].alias(pred_col))
    else:
        log.warning(f"  predict_model output missing {pred_col}; predictions will be empty")

    return {
        'model_name': model_name,
        'metrics': metrics,
        'hparams': hparams,
        'predictions': out_pred,
    }


def _save_phase_outputs(
    ctx: PipelineContext,
    phase_id: str,
    side: str,
    *,
    predictions: Optional[pl.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
    hparams: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist standard per-phase artifacts (predictions parquet + metrics json)."""
    if predictions is not None:
        pred_path = ctx.output_path(phase_id, side, 'predictions.parquet')
        predictions.write_parquet(pred_path)
        ctx.log.info(f"  wrote {pred_path.name} ({len(predictions):,} rows)")

    payload: Dict[str, Any] = {}
    if metrics is not None:
        payload['metrics'] = metrics
    if hparams is not None:
        payload['hparams'] = hparams
    if model_name is not None:
        payload['model_name'] = model_name
    if extra:
        payload.update(extra)
    if payload:
        metrics_path = ctx.output_path(phase_id, side, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        ctx.log.info(f"  wrote {metrics_path.name}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Phase implementations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def phase_0_baseline(side: str, ctx: PipelineContext) -> PhaseResult:
    """Reference MLP on the pruned pre-board feature set.

    Establishes the baseline var_ratio that all subsequent phases are
    measured against. Uses the Pct_NS HP-search winner hyperparameters
    (configured in PipelineConfig.baseline_hparams).
    """
    log = ctx.log
    smoke_rows = ctx.config.smoke_rows if ctx.config.smoke else None

    result = train_one_pct_ns_model(
        side,
        hparams_override=ctx.config.baseline_hparams,
        model_name_suffix='_p0_baseline',
        smoke_rows=smoke_rows,
        log=log,
        input_suffix=ctx.config.input_suffix,
        train_sample_rows=ctx.config.train_sample_rows,
        test_sample_rows=ctx.config.test_sample_rows,
    )

    m = result['metrics']
    log.info(
        f"  Phase 0 metrics: "
        f"mae={m.get('test_mae'):.4f} "
        f"rmse={m.get('test_rmse'):.4f} "
        f"var_ratio={m.get('variance_ratio'):.3f}"
    )

    _save_phase_outputs(
        ctx, 'p0_baseline', side,
        predictions=result['predictions'],
        metrics=m,
        hparams=result['hparams'],
        model_name=result['model_name'],
    )

    return PhaseResult(success=True, metrics=m)


def _stub(phase_id: str, description: str) -> Callable[[str, PipelineContext], PhaseResult]:
    """Build a stub phase function. Marks success=True+skipped=True so the
    pipeline can still flow through to phase 9 with whatever has been built."""
    def _impl(side: str, ctx: PipelineContext) -> PhaseResult:
        ctx.log.warning(f"  [{phase_id}] not yet implemented â€” {description}")
        ctx.log.warning(f"  [{phase_id}] skipping; downstream phases will use the latest available predictions")
        return PhaseResult(success=True, skipped=True, metrics={'stub': True})
    _impl.__doc__ = f"[STUB] {description}"
    _impl.__name__ = f"phase_{phase_id}"
    return _impl


# ----------------------------------------------------------------------
# Skill feature engineering (used by phase 2 + phase 6)
# ----------------------------------------------------------------------

def _add_skill_features(
    working_df: pl.DataFrame,
    working_test_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Synthesize the skill features the empirical analysis flagged as high
    leverage:

      * elo_R_seat_NS / EW   â€” average of per-seat starting Elo (the broken
                               pair Elo column from the lookup is ignored)
      * elo_R_seat_diff      â€” NS minus EW (r â‰ˆ +0.19 vs deal-mean residual)
      * elo_R_seat_avg       â€” proxy for field strength
      * mp_log_NS / EW       â€” log1p of average MasterPoints per seat-pair
      * mp_log_diff / avg    â€” partnership skill differential / level
      * elo_N_diff           â€” experience asymmetry (sessions played)
      * elo_diff_x_vul_NS    â€” skill-differential Ã— vulnerability interaction
      * elo_diff_x_vul_EW
      * mp_log_diff_x_elo_diff â€” agreement between MP-skill and Elo-skill

    All features fill nulls with zero so the MLP doesn't get NaN inputs.
    Operates on both train and test to keep them aligned.
    """
    def _augment(df: pl.DataFrame) -> pl.DataFrame:
        cols_present = set(df.columns)

        def col_or_zero(name: str) -> pl.Expr:
            if name in cols_present:
                return pl.col(name).cast(pl.Float64).fill_null(0.0)
            return pl.lit(0.0)

        elo_n = col_or_zero('Elo_R_N_EventStart')
        elo_s = col_or_zero('Elo_R_S_EventStart')
        elo_e = col_or_zero('Elo_R_E_EventStart')
        elo_w = col_or_zero('Elo_R_W_EventStart')

        mp_n = col_or_zero('MasterPoints_N')
        mp_s = col_or_zero('MasterPoints_S')
        mp_e = col_or_zero('MasterPoints_E')
        mp_w = col_or_zero('MasterPoints_W')

        elo_n_ns = col_or_zero('Elo_N_NS')
        elo_n_ew = col_or_zero('Elo_N_EW')

        vul_ns = col_or_zero('Vul_NS')
        vul_ew = col_or_zero('Vul_EW')

        elo_seat_ns = (elo_n + elo_s) / 2.0
        elo_seat_ew = (elo_e + elo_w) / 2.0
        elo_seat_diff = elo_seat_ns - elo_seat_ew
        elo_seat_avg = (elo_seat_ns + elo_seat_ew) / 2.0

        mp_avg_ns = (mp_n + mp_s) / 2.0
        mp_avg_ew = (mp_e + mp_w) / 2.0
        mp_log_ns = (1.0 + mp_avg_ns).log()
        mp_log_ew = (1.0 + mp_avg_ew).log()
        mp_log_diff = mp_log_ns - mp_log_ew
        mp_log_avg = (mp_log_ns + mp_log_ew) / 2.0

        elo_n_diff = elo_n_ns - elo_n_ew

        return df.with_columns(
            elo_R_seat_NS=elo_seat_ns,
            elo_R_seat_EW=elo_seat_ew,
            elo_R_seat_diff=elo_seat_diff,
            elo_R_seat_avg=elo_seat_avg,
            mp_log_NS=mp_log_ns,
            mp_log_EW=mp_log_ew,
            mp_log_diff=mp_log_diff,
            mp_log_avg=mp_log_avg,
            elo_N_diff=elo_n_diff,
            elo_diff_x_vul_NS=elo_seat_diff * vul_ns,
            elo_diff_x_vul_EW=elo_seat_diff * vul_ew,
            mp_log_diff_x_elo_diff=mp_log_diff * elo_seat_diff,
        )

    return _augment(working_df), _augment(working_test_df)


def _regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the headline regression metrics used throughout the pipeline."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return dict(
            test_mae=float('nan'),
            test_rmse=float('nan'),
            actual_std=float('nan'),
            pred_std=float('nan'),
            variance_ratio=float('nan'),
            mean_actual=float('nan'),
            mean_pred=float('nan'),
        )
    yt = yt[mask]
    yp = yp[mask]
    actual_std = float(np.std(yt))
    pred_std = float(np.std(yp))
    return dict(
        test_mae=float(np.mean(np.abs(yt - yp))),
        test_rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
        actual_std=actual_std,
        pred_std=pred_std,
        variance_ratio=(pred_std / actual_std) if actual_std > 0 else float('nan'),
        mean_actual=float(np.mean(yt)),
        mean_pred=float(np.mean(yp)),
    )


FIELD_RESID_BASELINE_FEATURES = (
    'elo_R_seat_diff',
    'elo_R_seat_avg',
    'mp_log_diff',
    'mp_log_avg',
    'elo_N_diff',
    'elo_diff_x_vul_NS',
    'elo_diff_x_vul_EW',
    'mp_log_diff_x_elo_diff',
    'Vul_NS',
    'Vul_EW',
    'Section_Pairs',
)


def _fit_field_skill_baseline(
    working_df: pl.DataFrame,
    working_test_df: pl.DataFrame,
    *,
    log: logging.Logger,
    ridge_lambda: float = 1e-3,
) -> Dict[str, Any]:
    """Fit a tiny ridge-stabilized linear baseline on strong skill features."""
    feature_names = [c for c in FIELD_RESID_BASELINE_FEATURES if c in working_df.columns]
    if not feature_names:
        raise ValueError("No field-baseline features are present after skill augmentation")

    train_x = (
        working_df
        .select([pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in feature_names])
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    test_x = (
        working_test_df
        .select([pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in feature_names])
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = np.nan_to_num(
        working_df['Pct_NS'].to_numpy().astype(np.float64, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    y_test = np.nan_to_num(
        working_test_df['Pct_NS'].to_numpy().astype(np.float64, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std = np.where(x_std > 1e-9, x_std, 1.0)
    x_train_std = (train_x - x_mean) / x_std
    x_test_std = (test_x - x_mean) / x_std

    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean

    xtx = x_train_std.T @ x_train_std
    xty = x_train_std.T @ y_centered
    ridge = ridge_lambda * np.eye(len(feature_names), dtype=np.float64)
    coef_std = np.linalg.solve(xtx + ridge, xty)

    train_pred = y_mean + x_train_std @ coef_std
    test_pred = y_mean + x_test_std @ coef_std

    coef_raw = coef_std / x_std
    intercept = y_mean - float(np.dot(x_mean, coef_raw))
    coef_map = {name: float(val) for name, val in zip(feature_names, coef_raw)}
    top_abs = sorted(coef_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    log.info(
        "  field baseline fitted on %d features; top |coef|: %s",
        len(feature_names),
        ", ".join(f"{k}={v:+.4f}" for k, v in top_abs),
    )

    return {
        'feature_names': feature_names,
        'ridge_lambda': float(ridge_lambda),
        'intercept': float(intercept),
        'coefficients': coef_map,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_stats': _regression_stats(y_train, train_pred),
        'test_stats': _regression_stats(y_test, test_pred),
    }


def phase_2_skill(side: str, ctx: PipelineContext) -> PhaseResult:
    """Add hand-crafted skill features and retrain.

    Empirical setup (per `_tmp_leverage_analysis.py` / `acbl_skill_leverage_search.py`):
      * elo_R_seat_diff correlates +0.19 with deal-mean residual â€” strongest
        single skill signal.
      * MasterPoints differential adds a complementary +0.15.
      * The pair-Elo column from the lookup table is broken (TODO.md);
        we synthesize the correct pair Elo from per-seat columns here.

    The full pre-board feature set already contains the seat-level Elo and
    MasterPoints columns, but the MLP appears to underweight them given the
    1346-feature width. Explicitly engineering the differential + interactions
    gives the gradient a strong, low-dimensional handle.
    """
    log = ctx.log
    smoke_rows = ctx.config.smoke_rows if ctx.config.smoke else None

    result = train_one_pct_ns_model(
        side,
        feature_engineer_fn=_add_skill_features,
        hparams_override=ctx.config.baseline_hparams,
        model_name_suffix='_p2_skill',
        smoke_rows=smoke_rows,
        log=log,
        input_suffix=ctx.config.input_suffix,
        train_sample_rows=ctx.config.train_sample_rows,
        test_sample_rows=ctx.config.test_sample_rows,
    )

    m = result['metrics']
    log.info(
        f"  Phase 2 metrics: "
        f"mae={m.get('test_mae'):.4f} "
        f"rmse={m.get('test_rmse'):.4f} "
        f"var_ratio={m.get('variance_ratio'):.3f}"
    )

    _save_phase_outputs(
        ctx, 'p2_skill', side,
        predictions=result['predictions'],
        metrics=m,
        hparams=result['hparams'],
        model_name=result['model_name'],
    )
    return PhaseResult(success=True, metrics=m)


def phase_3_field_resid(side: str, ctx: PipelineContext) -> PhaseResult:
    """Linear skill baseline + residual MLP."""
    log = ctx.log
    smoke_rows = ctx.config.smoke_rows if ctx.config.smoke else None
    resid_target = 'Pct_NS_FieldResid'

    if smoke_rows is not None:
        train_sample = smoke_rows
        test_sample = smoke_rows * 10
    else:
        train_sample = ctx.config.train_sample_rows
        test_sample = ctx.config.test_sample_rows

    log.info(
        f"  load {side} train+test parquets (suffix={ctx.config.input_suffix!r}, "
        f"train_sample={train_sample}, test_sample={test_sample})"
    )
    train_df, test_df = load_train_test(
        side,
        input_suffix=ctx.config.input_suffix,
        train_sample_rows=train_sample,
        test_sample_rows=test_sample,
        drop_regex_patterns=PCT_NS_DROP_PATTERNS,
    )

    working_df, working_test_df = build_working_dfs(train_df, test_df, 'Pct_NS')
    del train_df, test_df
    gc.collect()

    if smoke_rows is not None and len(working_df) > smoke_rows:
        log.info(f"  SMOKE mode: subsampling train {len(working_df):,} â†’ {smoke_rows:,}")
        working_df = working_df.head(smoke_rows)

    log.info("  apply skill + field-residual feature engineering")
    before_cols = working_df.width
    working_df, working_test_df = _add_skill_features(working_df, working_test_df)
    log.info(f"    cols: {before_cols} â†’ {working_df.width}")

    baseline = _fit_field_skill_baseline(working_df, working_test_df, log=log)
    train_field_pred = baseline['train_pred']
    test_field_pred = baseline['test_pred']
    train_resid = working_df['Pct_NS'].to_numpy().astype(np.float64, copy=False) - train_field_pred
    test_resid = working_test_df['Pct_NS'].to_numpy().astype(np.float64, copy=False) - test_field_pred

    working_df = working_df.with_columns(
        pl.Series(resid_target, train_resid),
        pl.Series('Pct_NS_FieldBase', train_field_pred),
    )
    working_test_df = working_test_df.with_columns(
        pl.Series(resid_target, test_resid),
        pl.Series('Pct_NS_FieldBase', test_field_pred),
    )

    field_only_stats = baseline['test_stats']
    log.info(
        f"  field-only test metrics: mae={field_only_stats['test_mae']:.4f} "
        f"rmse={field_only_stats['test_rmse']:.4f} "
        f"var_ratio={field_only_stats['variance_ratio']:.3f}"
    )

    working_df_for_training = working_df.drop(SPLIT_INDICATOR_COLS + ['Pct_NS'], strict=False)
    log.info(f"  residual training shape={working_df_for_training.shape}, target={resid_target}")

    hparams = dict(ctx.config.baseline_hparams)
    model_name = f"acbl_{side}_pct_ns_pipeline_p3_field_resid"
    log.info(f"  build shards (model_name={model_name})")
    base_schema = build_shards_once(
        working_df_for_training,
        resid_target,
        model_name,
        initial_layers=hparams['layers'],
        initial_dropout=hparams['dropout'],
        initial_y_range=hparams.get('y_range', None),
        shard_rows_count=int(hparams['shard_rows_count']),
    )

    del working_df_for_training
    gc.collect()

    log.info(f"  train residual model (seed={ctx.config.seed})")
    resid_metrics = run_trial(
        base_schema=base_schema,
        working_test_df=working_test_df,
        y_name=resid_target,
        model_name=model_name,
        config=hparams,
        class_counts=None,
        device='cuda',
        seed=int(ctx.config.seed),
    )

    log.info("  predict residual on test (for downstream phases)")
    schema_path = SAVED_MODELS_PATH / f"{model_name}_schema.json"
    with open(schema_path, 'r', encoding='utf-8') as f:
        on_disk_schema = json.load(f)
    resid_features_df = align_inference_features(working_test_df, on_disk_schema, resid_target)
    resid_pred_df = predict_model(SAVED_MODELS_PATH, model_name, resid_features_df)
    resid_pred_col = f"{resid_target}_Pred"
    if resid_pred_col not in resid_pred_df.columns:
        raise ValueError(f"Residual prediction output missing {resid_pred_col}")

    resid_pred = resid_pred_df[resid_pred_col].to_numpy().astype(np.float64, copy=False)
    combined_pred = test_field_pred + resid_pred
    combined_stats = _regression_stats(
        working_test_df['Pct_NS'].to_numpy().astype(np.float64, copy=False),
        combined_pred,
    )
    residual_stats = _regression_stats(test_resid, resid_pred)

    log.info(
        f"  Phase 3 metrics: "
        f"mae={combined_stats['test_mae']:.4f} "
        f"rmse={combined_stats['test_rmse']:.4f} "
        f"var_ratio={combined_stats['variance_ratio']:.3f} "
        f"(field-only={field_only_stats['variance_ratio']:.3f})"
    )

    preserve_cols = [c for c in _PREDICTION_PRESERVE_COLS if c in working_test_df.columns]
    if 'Pct_NS' not in preserve_cols and 'Pct_NS' in working_test_df.columns:
        preserve_cols.append('Pct_NS')
    out_pred = working_test_df.select(preserve_cols).with_columns(
        pl.Series('Pct_NS_FieldBase', test_field_pred),
        pl.Series('Pct_NS_ResidComponent', resid_pred),
        pl.Series('Pct_NS_Pred', combined_pred),
    )

    phase_metrics: Dict[str, Any] = {
        **combined_stats,
        'field_only': field_only_stats,
        'residual_target': residual_stats,
        'residual_training': resid_metrics,
        'field_baseline': {
            'ridge_lambda': baseline['ridge_lambda'],
            'intercept': baseline['intercept'],
            'feature_names': baseline['feature_names'],
            'coefficients': baseline['coefficients'],
        },
    }

    _save_phase_outputs(
        ctx, 'p3_field_resid', side,
        predictions=out_pred,
        metrics=phase_metrics,
        hparams=hparams,
        model_name=model_name,
    )
    return PhaseResult(success=True, metrics=phase_metrics)

phase_4_two_stage = _stub(
    'p4_two_stage',
    "Two-stage OOF: K-fold OOF predictions of Contract + Declarer_Direction "
    "as input features. Avoids the leakage that disabled this previously."
)

phase_5_aux = _stub(
    'p5_aux',
    "Auxiliary tail-event classifiers (game made, set, slam, vulnerable "
    "game, sacrifice). OOF probabilities as features."
)


def phase_6_ensemble(side: str, ctx: PipelineContext) -> PhaseResult:
    """Deep ensemble: train N seeds of the (skill-feature-augmented) model;
    aggregate to mean prediction + per-board Ïƒ.

    The Ïƒ is the directly-usable per-board predictive uncertainty: a board
    where all 3 seeds agree gets a tight CI; one where they disagree gets
    a wide CI. Phase 9 multiplies Ïƒ by sqrt(n_boards) to derive the
    session-level model CI consumed by the luck-vs-skill report.

    Implementation note: the inner training is sequential because the GPU
    is single. Each seed gets its own model_name suffix so their schemas
    and weights live side-by-side on disk. Schema/shards are NOT shared
    across seeds â€” each call to `train_one_pct_ns_model` rebuilds them.
    Acceptable cost given training >> shard build, and it lets a crashed
    seed re-start cleanly.
    """
    log = ctx.log
    n_seeds = max(1, int(ctx.config.n_ensemble_seeds))
    smoke_rows = ctx.config.smoke_rows if ctx.config.smoke else None
    base_seed = int(ctx.config.seed)
    seeds = [base_seed + i * 1000 for i in range(n_seeds)]

    log.info(f"  ensemble seeds: {seeds}")

    seed_metrics: List[Dict[str, Any]] = []
    seed_preds: List[pl.DataFrame] = []
    pred_col_name: Optional[str] = None
    preserve_cols: Optional[List[str]] = None

    for i, sd in enumerate(seeds):
        log.info(f"  --- seed {i+1}/{n_seeds} (seed={sd}) ---")
        result = train_one_pct_ns_model(
            side,
            feature_engineer_fn=_add_skill_features,
            hparams_override=ctx.config.baseline_hparams,
            model_name_suffix=f'_p6_ensemble_seed{sd}',
            smoke_rows=smoke_rows,
            seed=sd,
            log=log,
            input_suffix=ctx.config.input_suffix,
            train_sample_rows=ctx.config.train_sample_rows,
            test_sample_rows=ctx.config.test_sample_rows,
        )
        seed_metrics.append({'seed': sd, **result['metrics']})
        preds_df = result['predictions']
        # First seed defines the column ordering / pred-col name
        if pred_col_name is None:
            pred_col_name = next(
                (c for c in preds_df.columns if c.endswith('_Pred')), 'Pct_NS_Pred'
            )
            preserve_cols = [c for c in preds_df.columns if c != pred_col_name]
        # Rename per-seed pred col so we can stack horizontally
        seed_pred_col = f'{pred_col_name}_seed{sd}'
        seed_preds.append(
            preds_df.rename({pred_col_name: seed_pred_col})
            .select(preserve_cols + [seed_pred_col])
        )
        gc.collect()

    # Join predictions on preserve cols (event_id, session_id, ...). Use the
    # first seed as the base; horizontally stack the prediction columns from
    # the rest. This avoids materializing N joins.
    base = seed_preds[0]
    for other in seed_preds[1:]:
        # The non-preserve columns are unique per seed; align row order via
        # hstack (rows are identical because we read the same test parquet).
        added_col = [c for c in other.columns if c not in preserve_cols][0]
        if len(other) != len(base):
            log.warning(f"  seed prediction row count mismatch ({len(base)} vs {len(other)}); using join")
            base = base.join(other, on=preserve_cols, how='left')
        else:
            base = base.with_columns(other[added_col])

    pred_seed_cols = [c for c in base.columns if c.startswith(f'{pred_col_name}_seed')]
    arr = base.select(pred_seed_cols).to_numpy()
    pred_mean = arr.mean(axis=1)
    pred_std = arr.std(axis=1, ddof=0)

    out = base.with_columns(
        pl.Series(pred_col_name, pred_mean),
        pl.Series(f'{pred_col_name}_std', pred_std),
    )

    # Compute ensemble metrics on the aggregated mean
    ens_metrics: Dict[str, Any] = {
        'n_seeds': n_seeds,
        'seeds': seeds,
        'per_seed_metrics': seed_metrics,
    }
    if 'Pct_NS' in out.columns:
        yt = out['Pct_NS'].to_numpy().astype(np.float64)
        yp = pred_mean
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.any():
            yt_m, yp_m = yt[mask], yp[mask]
            ens_metrics.update(
                test_mae=float(np.mean(np.abs(yt_m - yp_m))),
                test_rmse=float(np.sqrt(np.mean((yt_m - yp_m) ** 2))),
                actual_std=float(np.std(yt_m)),
                pred_std=float(np.std(yp_m)),
                variance_ratio=float(np.std(yp_m) / np.std(yt_m)) if np.std(yt_m) > 0 else float('nan'),
                mean_ensemble_sigma=float(pred_std[mask].mean()),
                p50_ensemble_sigma=float(np.percentile(pred_std[mask], 50)),
                p95_ensemble_sigma=float(np.percentile(pred_std[mask], 95)),
            )

    log.info(
        f"  Phase 6 ensemble metrics: "
        f"mae={ens_metrics.get('test_mae'):.4f} "
        f"var_ratio={ens_metrics.get('variance_ratio'):.3f} "
        f"mean Ïƒ={ens_metrics.get('mean_ensemble_sigma'):.4f}"
    )

    _save_phase_outputs(
        ctx, 'p6_ensemble', side,
        predictions=out,
        metrics=ens_metrics,
        hparams=ctx.config.baseline_hparams,
        model_name=f"acbl_{side}_pct_ns_pipeline_p6_ensemble (n={n_seeds})",
    )
    return PhaseResult(success=True, metrics=ens_metrics)


phase_7_combined = _stub(
    'p7_combined',
    "Single model trained on club + tournament with a dataset indicator "
    "feature; should improve generalization on the smaller tournament side."
)


def phase_8_calibration(side: str, ctx: PipelineContext) -> PhaseResult:
    """Isotonic recalibration of the latest available upstream prediction.

    Procedure (50/50 split of the test set):
      * Sort test rows deterministically; first half fits the isotonic curve
        on (raw_pred â†’ actual), second half is the held-out evaluation slice.
      * Apply the fitted curve to all rows; report metrics on the held-out
        half so the headline numbers are honest.
      * Save the calibrated predictions parquet (with both raw and calibrated
        columns) and the fitted curve as JSON for inspection.

    This is a cheap diagnostic â€” if the model's prediction shape is already
    monotonically calibrated against actuals, isotonic adds little; if there's
    a systematic mean shift or compressed range, isotonic fixes it without
    changing the rank order.
    """
    log = ctx.log

    # Find the freshest upstream model â€” prefer ensemble, then stronger
    # single-model phases, then the simpler baselines.
    candidates = ('p6_ensemble', 'p3_field_resid', 'p2_skill', 'p0_baseline')
    src_path = None
    src_phase = None
    for p in candidates:
        cand = ctx.output_path(p, side, 'predictions.parquet')
        if cand.exists():
            src_path = cand
            src_phase = p
            break
    if src_path is None:
        log.warning(f"  [p8_calibration] no upstream predictions found, skipping")
        return PhaseResult(success=True, skipped=True, metrics={'reason': 'no upstream'})

    log.info(f"  calibrating predictions from {src_path.name}")
    df = pl.read_parquet(src_path)
    pred_col = 'Pct_NS_Pred' if 'Pct_NS_Pred' in df.columns else next(
        (c for c in df.columns if c.endswith('_Pred') and not c.endswith('_std')),
        None,
    )
    if pred_col is None or 'Pct_NS' not in df.columns:
        log.warning(f"  [p8_calibration] missing Pct_NS or *_Pred column, skipping")
        return PhaseResult(success=True, skipped=True, metrics={'reason': 'missing columns'})

    yt = df['Pct_NS'].to_numpy().astype(np.float64)
    yp = df[pred_col].to_numpy().astype(np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 200:
        log.warning(f"  [p8_calibration] too few finite rows ({mask.sum()}), skipping")
        return PhaseResult(success=True, skipped=True, metrics={'reason': 'too few rows'})

    # Deterministic 50/50 split (use a stable hash on row index)
    n = len(yt)
    rng = np.random.default_rng(seed=ctx.config.seed)
    perm = rng.permutation(n)
    half = n // 2
    fit_idx = perm[:half]
    eval_idx = perm[half:]

    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        log.error(f"  [p8_calibration] sklearn not installed; skipping")
        return PhaseResult(success=True, skipped=True, metrics={'reason': 'sklearn missing'})

    fit_yp, fit_yt = yp[fit_idx], yt[fit_idx]
    fit_mask = np.isfinite(fit_yp) & np.isfinite(fit_yt)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(fit_yp[fit_mask], fit_yt[fit_mask])

    yp_cal = iso.predict(np.clip(yp, 0.0, 1.0))

    eval_yp_raw = yp[eval_idx]
    eval_yp_cal = yp_cal[eval_idx]
    eval_yt = yt[eval_idx]
    em = np.isfinite(eval_yt) & np.isfinite(eval_yp_raw) & np.isfinite(eval_yp_cal)
    eval_yt, eval_yp_raw, eval_yp_cal = eval_yt[em], eval_yp_raw[em], eval_yp_cal[em]

    def _stats(yt_, yp_):
        return dict(
            mae=float(np.mean(np.abs(yt_ - yp_))),
            rmse=float(np.sqrt(np.mean((yt_ - yp_) ** 2))),
            pred_std=float(np.std(yp_)),
            actual_std=float(np.std(yt_)),
            variance_ratio=float(np.std(yp_) / np.std(yt_)) if np.std(yt_) > 0 else float('nan'),
            mean_pred=float(np.mean(yp_)),
            mean_actual=float(np.mean(yt_)),
        )

    raw_stats = _stats(eval_yt, eval_yp_raw)
    cal_stats = _stats(eval_yt, eval_yp_cal)

    log.info(
        f"  raw    eval: mae={raw_stats['mae']:.4f}  var_ratio={raw_stats['variance_ratio']:.3f}  "
        f"mean_pred={raw_stats['mean_pred']:.3f}  mean_actual={raw_stats['mean_actual']:.3f}"
    )
    log.info(
        f"  calib. eval: mae={cal_stats['mae']:.4f}  var_ratio={cal_stats['variance_ratio']:.3f}  "
        f"mean_pred={cal_stats['mean_pred']:.3f}"
    )

    out = df.with_columns(
        pl.Series(f'{pred_col}_raw', yp),
        pl.Series(f'{pred_col}', yp_cal),
        pl.Series('_p8_eval_split', np.isin(np.arange(n), eval_idx)),
    )

    _save_phase_outputs(
        ctx, 'p8_calibration', side,
        predictions=out,
        metrics={
            'source_phase': src_phase,
            'source_file': src_path.name,
            'raw_eval': raw_stats,
            'calibrated_eval': cal_stats,
            'iso_X_thresholds': iso.X_thresholds_.tolist(),
            'iso_y_thresholds': iso.y_thresholds_.tolist(),
        },
    )
    return PhaseResult(success=True, metrics={'raw': raw_stats, 'calibrated': cal_stats})


MIN_BOARDS_FOR_REPORT = 4  # Sessions with <4 boards have no usable SEM


def phase_9_session_aggregation(side: str, ctx: PipelineContext) -> PhaseResult:
    """Aggregate per-board predictions to (session Ã— pair) and emit the
    luck-vs-skill report.

    Source preference order (best â†’ worst):
        p8_calibration â†’ p7_combined â†’ p6_ensemble â†’ p5_aux â†’ p4_two_stage
        â†’ p3_field_resid â†’ p2_skill â†’ p0_baseline.

    For each (event_id, session_id, Pair_Number_NS) we compute:
        * actual_session_pct  /  predicted_session_pct  /  luck_delta
        * actual_board_std â†’ SEM of the actual session mean
        * pred_board_std â†’ board-level prediction spread (informational)
        * model_sigma_session = mean(board ensemble Ïƒ) / sqrt(n_boards)
            â€” when phase 6 produced *_Pred_std, this is the model's CI on
            the session prediction. Without phase 6 it's null.
        * total_sigma = sqrt(semÂ² + model_sigma_sessionÂ²) â€” combined CI
            covering both luck variance and model uncertainty.
        * z_luck = luck_delta / total_sigma â€” magnitude of the luck/skill
            departure in standard deviations. Sorted descending, this gives
            the "most surprising sessions" report.
        * is_lucky_significant = |z_luck| > 1.96
        * skill_rank â€” pair's MasterPoints rank-percentile in the report
            (so "above expectation for their skill level" reads cleanly).

    The full parquet keeps every row (including 1-2 board sessions); the
    on-screen summary filters to n_boards >= MIN_BOARDS_FOR_REPORT so the
    headline numbers aren't dominated by single-board outliers.
    """
    log = ctx.log

    candidate_phases = (
        'p8_calibration',
        'p7_combined',
        'p6_ensemble',
        'p5_aux',
        'p4_two_stage',
        'p3_field_resid',
        'p2_skill',
        'p0_baseline',
    )
    pred_path: Optional[pathlib.Path] = None
    source_phase: Optional[str] = None
    for p in candidate_phases:
        cand = ctx.output_path(p, side, 'predictions.parquet')
        if cand.exists():
            pred_path = cand
            source_phase = p
            break
    if pred_path is None:
        log.error(f"  [{side}] no upstream predictions found in {ctx.outputs_dir}")
        return PhaseResult(success=False, error="no upstream predictions")

    log.info(f"  using predictions from {pred_path.name} (source={source_phase})")
    df = pl.read_parquet(pred_path)

    pred_col = next(
        (c for c in df.columns if c.endswith('_Pred') and not c.endswith('_std')),
        None,
    )
    if pred_col is None:
        return PhaseResult(success=False, error="predictions parquet has no *_Pred column")
    sigma_col = f'{pred_col}_std' if f'{pred_col}_std' in df.columns else None

    if 'session_id' not in df.columns or 'Pair_Number_NS' not in df.columns:
        return PhaseResult(
            success=False,
            error=f"predictions missing session_id / Pair_Number_NS (have: {df.columns})"
        )

    group_keys: List[str] = ['session_id', 'Pair_Number_NS']
    if 'event_id' in df.columns:
        group_keys = ['event_id', 'session_id', 'Pair_Number_NS']

    agg_exprs = [
        pl.len().alias('n_boards'),
        pl.col('Pct_NS').mean().alias('actual_session_pct'),
        pl.col(pred_col).mean().alias('predicted_session_pct'),
        pl.col('Pct_NS').std().alias('actual_board_std'),
        pl.col(pred_col).std().alias('pred_board_std'),
    ]
    if sigma_col:
        agg_exprs.append(pl.col(sigma_col).mean().alias('mean_board_pred_sigma'))
    if 'MasterPoints_N' in df.columns and 'MasterPoints_S' in df.columns:
        agg_exprs.append(
            ((pl.col('MasterPoints_N') + pl.col('MasterPoints_S')) / 2.0)
            .mean().alias('pair_mp_avg')
        )

    session_pair = (
        df.group_by(group_keys)
          .agg(agg_exprs)
          .with_columns(
              luck_delta=pl.col('actual_session_pct') - pl.col('predicted_session_pct'),
              sem=pl.col('actual_board_std') / pl.col('n_boards').cast(pl.Float64).sqrt(),
          )
    )
    if sigma_col:
        session_pair = session_pair.with_columns(
            model_sigma_session=pl.col('mean_board_pred_sigma')
                              / pl.col('n_boards').cast(pl.Float64).sqrt(),
        ).with_columns(
            total_sigma=(pl.col('sem').pow(2) + pl.col('model_sigma_session').pow(2)).sqrt(),
        )
    else:
        session_pair = session_pair.with_columns(
            model_sigma_session=pl.lit(None, dtype=pl.Float64),
            total_sigma=pl.col('sem'),
        )

    session_pair = session_pair.with_columns(
        ci_lo=pl.col('actual_session_pct') - 1.96 * pl.col('total_sigma'),
        ci_hi=pl.col('actual_session_pct') + 1.96 * pl.col('total_sigma'),
        z_luck=pl.col('luck_delta') / pl.col('total_sigma'),
    ).with_columns(
        is_lucky_significant=(pl.col('z_luck').abs() > 1.96),
    )

    if 'pair_mp_avg' in session_pair.columns:
        # Skill percentile within this side's report (0..1)
        session_pair = session_pair.with_columns(
            skill_pct=pl.col('pair_mp_avg').rank(method='average') / pl.len(),
        )

    # Sort by raw luck_delta descending â€” it's the most intuitive "lucky
    # vs unlucky" ordering. Z-score is also in the parquet for analysts who
    # want significance-weighted ranking.
    session_pair = session_pair.sort('luck_delta', descending=True)

    out_path = ctx.final_dir / f'session_luck_vs_skill_{side}.parquet'
    session_pair.write_parquet(out_path)
    log.info(f"  wrote {out_path.name} ({len(session_pair):,} sessionÃ—pair rows)")

    # Headline summary â€” only sessions long enough to be meaningful.
    rep = session_pair.filter(pl.col('n_boards') >= MIN_BOARDS_FOR_REPORT)
    summary: Dict[str, Any] = {
        'source_phase': source_phase,
        'source_file': pred_path.name,
        'n_session_pairs_total': len(session_pair),
        'n_session_pairs_reportable': len(rep),
        'min_boards_for_report': MIN_BOARDS_FOR_REPORT,
        'has_model_uncertainty': sigma_col is not None,
    }
    if len(rep) > 0:
        summary.update(
            mean_n_boards=float(rep['n_boards'].mean()),
            mean_luck_delta=float(rep['luck_delta'].mean()),
            std_luck_delta=float(rep['luck_delta'].std()),
            p05_luck_delta=float(rep['luck_delta'].quantile(0.05)),
            p95_luck_delta=float(rep['luck_delta'].quantile(0.95)),
            pct_significantly_lucky=float(rep['is_lucky_significant'].mean()),
            mean_total_sigma=float(rep['total_sigma'].mean()),
        )
        if sigma_col:
            summary['mean_model_sigma_session'] = float(rep['model_sigma_session'].mean())

    _save_phase_outputs(
        ctx, 'p9_session', side,
        metrics=summary,
        extra={'output_file': str(out_path)},
    )

    if len(rep) > 0:
        log.info(
            f"  p9 summary (n_boards>={MIN_BOARDS_FOR_REPORT}, n={len(rep):,}): "
            f"luck Ïƒ={summary['std_luck_delta']:.4f}  "
            f"5/95-pct=[{summary['p05_luck_delta']:+.4f}, {summary['p95_luck_delta']:+.4f}]  "
            f"significant-luck rate={summary['pct_significantly_lucky']*100:.1f}%"
        )
        if sigma_col:
            log.info(
                f"  p9 model uncertainty: "
                f"mean session model Ïƒ={summary['mean_model_sigma_session']:.4f}  "
                f"mean total Ïƒ={summary['mean_total_sigma']:.4f}"
            )
    else:
        log.warning(f"  p9 summary: no sessions with n_boards>={MIN_BOARDS_FOR_REPORT}")
    return PhaseResult(success=True, metrics=summary)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Phase registry
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

PHASES: 'OrderedDict[str, Callable[[str, PipelineContext], PhaseResult]]' = OrderedDict([
    ('p0_baseline', phase_0_baseline),
    ('p2_skill', phase_2_skill),
    ('p3_field_resid', phase_3_field_resid),
    ('p4_two_stage', phase_4_two_stage),
    ('p5_aux', phase_5_aux),
    ('p6_ensemble', phase_6_ensemble),
    ('p7_combined', phase_7_combined),
    ('p8_calibration', phase_8_calibration),
    ('p9_session', phase_9_session_aggregation),
])


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Dispatcher
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_pipeline(config: PipelineConfig, log: logging.Logger) -> Dict[str, Any]:
    """Iterate (side, phase) pairs, honoring resume/checkpoint logic."""
    ctx = PipelineContext(config, log)

    log.info("=" * 72)
    log.info(f"PCT_NS PIPELINE â€” sides={config.sides} resume={config.resume} smoke={config.smoke}")
    log.info(f"  phases: {config.phases}")
    log.info(f"  target var_ratio â‰¥ {config.target_var_ratio}")
    log.info("=" * 72)

    summary: Dict[str, Any] = {
        'sides': {},
        'config': dataclasses.asdict(config),
    }

    overall_t = time.time()
    for side in config.sides:
        log.info(f"\n{'#' * 72}\n# SIDE: {side}\n{'#' * 72}")
        side_summary: Dict[str, Any] = {'phases': {}}

        for phase_id in config.phases:
            if phase_id not in PHASES:
                log.warning(f"  [{side}] unknown phase '{phase_id}' â€” skipping")
                continue

            # Skip-on-resume logic. Aggregation phases always re-run.
            if (config.resume
                    and phase_id not in ALWAYS_RERUN_PHASES
                    and ctx.is_done(phase_id, side)):
                log.info(f"  [SKIP-RESUME] {phase_id} ({side})")
                with open(ctx.checkpoint_path(phase_id, side), encoding='utf-8') as f:
                    side_summary['phases'][phase_id] = {
                        'skipped_resume': True,
                        **json.load(f),
                    }
                continue

            log.info(f"\n--- [{side}] PHASE {phase_id} ---")
            t0 = time.time()
            try:
                result = PHASES[phase_id](side, ctx)
                if result.success and not result.skipped and phase_id not in ALWAYS_RERUN_PHASES:
                    ctx.mark_done(phase_id, side, result.metrics)
                side_summary['phases'][phase_id] = {
                    'success': result.success,
                    'skipped': result.skipped,
                    'wall_s': time.time() - t0,
                    'metrics': result.metrics,
                    'error': result.error,
                }
                log.info(f"--- [{side}] PHASE {phase_id} done in {time.time() - t0:.1f}s "
                         f"(success={result.success}, skipped={result.skipped}) ---")
            except KeyboardInterrupt:
                log.warning(f"  [{side}] PHASE {phase_id} interrupted by user")
                side_summary['phases'][phase_id] = {
                    'success': False,
                    'wall_s': time.time() - t0,
                    'error': 'KeyboardInterrupt',
                }
                summary['sides'][side] = side_summary
                summary['interrupted'] = True
                _write_summary(summary)
                raise
            except Exception as e:
                log.error(f"  EXCEPTION in {phase_id}: {type(e).__name__}: {e}")
                log.error(traceback.format_exc())
                side_summary['phases'][phase_id] = {
                    'success': False,
                    'wall_s': time.time() - t0,
                    'error': f"{type(e).__name__}: {e}",
                }
                # Continue to next phase â€” don't abort the side or the run.
                # Phase 9 will gracefully fall back to whatever upstream
                # predictions did succeed.

        summary['sides'][side] = side_summary

    summary['total_wall_s'] = time.time() - overall_t
    _write_summary(summary)
    log.info(f"\nTotal pipeline wall: {summary['total_wall_s']:.1f}s")
    return summary


def _write_summary(summary: Dict[str, Any]) -> None:
    summary_path = FINAL_DIR / 'pipeline_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_cli() -> Tuple[PipelineConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="ACBL Pct_NS unattended pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Phases: " + ', '.join(PHASES.keys()),
    )

    sg = p.add_mutually_exclusive_group()
    sg.add_argument('--club', action='store_true', help='Run only club side')
    sg.add_argument('--tournament', action='store_true', help='Run only tournament side')
    sg.add_argument('--both', action='store_true', help='Run both sides (default)')

    p.add_argument(
        '--phases', type=str, default='all',
        help='Comma-separated phase ids to run, or "all". '
             'Phase ids must come from the registry (see epilog).'
    )
    p.add_argument(
        '--resume', action='store_true',
        help='Skip phases with existing .done checkpoints (per-side).'
    )
    p.add_argument(
        '--smoke', action='store_true',
        help='Subsample training to PipelineConfig.smoke_rows for fast iteration.'
    )
    p.add_argument(
        '--clean', action='store_true',
        help='Wipe checkpoints/, outputs/, final/ before running.'
    )
    p.add_argument(
        '--rerun', type=str, default='',
        help='Comma-separated phase ids whose checkpoints to remove before running. '
             'Combine with --resume to re-run only specific phases.'
    )
    p.add_argument('--device', type=int, default=0, help='CUDA device index (default 0)')
    p.add_argument(
        '--input-suffix', type=str, default='_v2',
        help='Suffix appended to acbl_<side>_prediction_data_{train,test}<suffix>.parquet '
             'when loading. Default "_v2" picks the corrected pair-Elo / uncapped-MasterPoints '
             'regen. Pass "" to read the legacy un-suffixed parquets.'
    )
    p.add_argument(
        '--train-sample-rows', type=int, default=None,
        help='Cap rows pulled from the train parquet at scan time (TAIL of file). '
             'Use to bound the working set on the v2 club parquet (165 GB / 55M rows / '
             '6042 cols) â€” eager full load OOMs on 192 GB RAM. E.g. 10_000_000 fits in '
             '~50 GB after column projection. Orthogonal to --smoke. Default: full file.'
    )
    p.add_argument(
        '--test-sample-rows', type=int, default=None,
        help='Cap rows pulled from the test parquet at scan time (TAIL). Default: full.'
    )

    args = p.parse_args()

    if args.club:
        sides = ['club']
    elif args.tournament:
        sides = ['tournament']
    else:
        sides = list(SIDES_ALL)

    if args.phases.strip().lower() == 'all':
        phases = list(PHASES.keys())
    else:
        phases = [s.strip() for s in args.phases.split(',') if s.strip()]
        unknown = [p for p in phases if p not in PHASES]
        if unknown:
            p_err = ', '.join(unknown)
            valid = ', '.join(PHASES.keys())
            raise SystemExit(f"Unknown phase(s): {p_err}. Valid: {valid}")

    config = PipelineConfig(
        sides=sides,
        phases=phases,
        resume=args.resume,
        smoke=args.smoke,
        cuda_device=args.device,
        input_suffix=args.input_suffix,
        train_sample_rows=args.train_sample_rows,
        test_sample_rows=args.test_sample_rows,
    )
    return config, args


def _maybe_clean(args: argparse.Namespace, log: logging.Logger) -> None:
    if args.clean:
        for d in (CHECKPOINTS_DIR, OUTPUTS_DIR, FINAL_DIR):
            if d.exists():
                log.info(f"--clean: wiping {d}")
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

    if args.rerun:
        rerun_ids = [s.strip() for s in args.rerun.split(',') if s.strip()]
        for phase_id in rerun_ids:
            for side in SIDES_ALL:
                cp = CHECKPOINTS_DIR / f"{phase_id}_{side}.done"
                if cp.exists():
                    log.info(f"--rerun: removing {cp.name}")
                    cp.unlink()


def main() -> None:
    config, args = parse_cli()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_device)

    for d in (PIPELINE_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, FINAL_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime('%Y%m%d-%H%M%S')
    log_path = LOGS_DIR / f"{stamp}_pipeline.log"
    log = setup_logging(log_path)
    log.info(f"Logging to {log_path}")

    config_path = LOGS_DIR / f"{stamp}_pipeline_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    log.info(f"Config snapshot: {config_path}")

    _maybe_clean(args, log)

    try:
        run_pipeline(config, log)
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user (KeyboardInterrupt)")
        raise
    except Exception as e:
        log.error(f"Pipeline aborted: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
