import argparse
import pathlib
import time
import polars as pl

def parse_args():
    p = argparse.ArgumentParser(description="Build raw feature shards (X only) with row_id for reuse.")
    p.add_argument("--dataset", choices=["club", "tournament", "all"], default="all")
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("e:/bridge/data/acbl"))
    p.add_argument("--raw-prefix", type=str, default=None, help="Prefix for raw shard files")
    p.add_argument("--rows-per-shard", type=int, default=2_000_000)
    return p.parse_args()

def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}")

def main():
    args = parse_args()
    t0 = time.time()

    rootPath = args.root
    acblPath = rootPath if rootPath.name.lower() == 'acbl' else rootPath.joinpath('acbl')
    savedModelsPath = acblPath.joinpath('SavedModels')
    savedModelsPath.mkdir(parents=True, exist_ok=True)

    def process_one(ds: str):
        log("=== Step: Build Raw Feature Shards (X only) ===")
        log("Estimated time: ~5â€“30 min per ~10M rows (I/O-bound, writes shards)")
        log("Previous: Build working features (acbl_build_working_features.py)")
        log("Next: Train OOF per target (acbl_train_oof_target.py) or stacked (acbl_train_stacked_oof.py), then merge (acbl_merge_oof.py)")

        work_file = acblPath.joinpath(f"acbl_{ds}_working_features.parquet")
        df = pl.read_parquet(work_file)
        log(f"ðŸ“¥ Read {work_file.resolve()}: shape={df.shape} size={work_file.stat().st_size:,}")

        if 'row_id' not in df.columns:
            df = df.with_row_index(name='row_id', offset=0)

        raw_prefix = (args.raw_prefix or f"acbl_{ds}_stacked_base") if args.dataset == 'all' else (args.raw_prefix or f"acbl_{args.dataset}_stacked_base")

        try:
            from mlBridge.mlBridgeAiLib import create_torch_shards_raw  # type: ignore
        except OSError as e:
            log("âŒ Failed to import ML library (torch). Common fixes on Windows:")
            log("  - Activate the intended environment and install a compatible PyTorch build")
            log("    CPU:   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            log("    CUDA:  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
            log("  - Or for conda: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
            log(f"  - Original error: {e}")
            raise

        manifest_path = create_torch_shards_raw(
            df=df,
            saved_models_path=savedModelsPath,
            raw_prefix=raw_prefix,
            cont_cols=None,
            cat_cols=None,
            row_id_col='row_id',
            shard_rows_count=args.rows_per_shard,
        )

        log(f"âœ… Raw feature shards built: manifest={manifest_path.resolve()} size={manifest_path.stat().st_size:,}")

    if args.dataset == 'all':
        for ds in ["club", "tournament"]:
            process_one(ds)
    else:
        process_one(args.dataset)

    log(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

