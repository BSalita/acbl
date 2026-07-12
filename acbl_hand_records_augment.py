#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_results_hand_records_augment.py

Takes 1h/5m minimum (no SD calculations) for 17m->7m/17m->17m rows by /5923 columns. Uses 300GB (1.2TB?) memory/cache for processing.
Recommend 2TB PCIe 5.0 SSD for pagefile.

Augment hand records df with DD/SD analysis and probabilities.

Previous steps:
    acbl_sql_to_hand_records_clean.py

Next steps:
    acbl_board_results_augment_step1.py
    acbl_board_results_augment_step2.py
    acbl_model_data.py

TODO:
    - Rename acbl_club_hand_records_cache_df.parquet to acbl_hand_records_cache_df.parquet
    - Obsolete most logic with calls to mlBridgeAugmentLib
    - Re-implement acbl_cgd
    - Implement matchpoint dict
"""

import polars as pl
import pathlib
from collections import defaultdict
import sys
import time

sys.path.append(str(pathlib.Path.cwd().parent.joinpath('mlBridgeLib')))
import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')


def augment_hand_records(club_or_tournament='club'):
    """Augment hand records with DD analysis and probabilities."""
    print(f"Processing {club_or_tournament} hand records augmentation...")
    
    # Load cleaned hand records
    acbl_hand_records_cleaned_filename = f'acbl_{club_or_tournament}_hand_records_cleaned.parquet'
    acbl_hand_records_cleaned_file = acblPath.joinpath(acbl_hand_records_cleaned_filename)
    hrs_df = pl.read_parquet(acbl_hand_records_cleaned_file)
    print(f"Loaded {acbl_hand_records_cleaned_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_cleaned_file.stat().st_size}")
    
    # Show columns with nulls
    null_counts = [(s.name, s[0]) for s in hrs_df.null_count() if s[0] > 0]
    if null_counts:
        print(f"Columns with nulls: {null_counts}")
    
    # Load or create cache
    acbl_club_hand_records_cache_df_filename = 'acbl_club_hand_records_cache_df.parquet'
    acbl_club_hand_records_cache_df_file = acblPath.joinpath(acbl_club_hand_records_cache_df_filename)
    hrs_cache_df = pl.read_parquet(acbl_club_hand_records_cache_df_file)
    print(f"Loaded {acbl_club_hand_records_cache_df_filename}: shape:{hrs_cache_df.shape} size:{acbl_club_hand_records_cache_df_file.stat().st_size}")
    
    assert hrs_df['PBN'].null_count() == 0, hrs_df['PBN'].null_count()
    assert hrs_df['Dealer'].null_count() == 0, hrs_df['Dealer'].null_count()
    assert hrs_df['Vul'].null_count() == 0, hrs_df['Vul'].null_count()
    assert hrs_cache_df['PBN'].null_count() == 0, hrs_cache_df['PBN'].null_count()
    #assert hrs_cache_df['Dealer'].null_count() == 0, hrs_cache_df['Dealer'].null_count()
    #assert hrs_cache_df['Vul'].null_count() == 0, hrs_cache_df['Vul'].null_count()
    assert hrs_cache_df.height == hrs_cache_df.unique(subset=['PBN','Dealer','Vul']).height, "hrs_cache_df has duplicate combinations of PBN+Dealer+Vul"
    assert hrs_cache_df.select(pl.col(pl.Float64)).width == 0, hrs_cache_df.select(pl.col(pl.Float64)).columns

    # takes 1h per 10000 sd calculations using 10 trials per pbn.
    acbl_club_hand_records_cache_df_filename = 'acbl_club_hand_records_cache_df.parquet' # note: not tournament option. we want to combine into one file.
    acbl_club_hand_records_cache_df_file = acblPath.joinpath(acbl_club_hand_records_cache_df_filename)

    if False: # only needed if > 50000 sd calculations
        max_sd_adds = 50000 # len(hrs_df)
        for i in range(0, len(hrs_df), max_sd_adds):
            batch_df = hrs_df.slice(i, max_sd_adds)
            
            hand_record_augmenter = mlBridgeAugmentLib.AllHandRecordAugmentations(
                batch_df, 
                hrs_cache_df, 
                sd_productions=10, 
                max_dd_adds=None, 
                max_sd_adds=max_sd_adds
            )
            _, hrs_cache_df = hand_record_augmenter.perform_all_hand_record_augmentations()
        
        # Save cache between batches
        hrs_cache_df.write_parquet(acbl_club_hand_records_cache_df_file)
        print(f"Batch {i//max_sd_adds + 1} of {len(hrs_df)//max_sd_adds} completed")
        print(f"Saved {acbl_club_hand_records_cache_df_filename}: shape:{hrs_cache_df.shape} size:{acbl_club_hand_records_cache_df_file.stat().st_size}")

    # takes 64m/8m minimum (2m for 100 new SD productions, 3m for EVs, ?m for joins, 2m for best contracts, 11m total for DD/SD).
    # todo: should max_(dd|sd)_adds of None be 0 or all?
    # todo: is progress bar missing in  create sd calculations?
    all_augmentations = mlBridgeAugmentLib.AllHandRecordAugmentations(hrs_df,hrs_cache_df,sd_productions=10,max_dd_adds=None,max_sd_adds=None,cache_file_path=acbl_club_hand_records_cache_df_file)
    hrs_df, hrs_cache_df = all_augmentations.perform_all_hand_record_augmentations()

    assert hrs_df['PBN'].null_count() == 0, hrs_df['PBN'].null_count()
    assert hrs_df['Dealer'].null_count() == 0, hrs_df['Dealer'].null_count()
    assert hrs_df['Vul'].null_count() == 0, hrs_df['Vul'].null_count()
    assert hrs_cache_df['PBN'].null_count() == 0, hrs_cache_df['PBN'].null_count()
    #assert hrs_cache_df['Dealer'].null_count() == 0, hrs_cache_df['Dealer'].null_count() # doesn't work if dd and sd do not track each other????
    #assert hrs_cache_df['Vul'].null_count() == 0, hrs_cache_df['Vul'].null_count()
    assert hrs_cache_df.height == hrs_cache_df.unique(subset=['PBN','Dealer','Vul']).height, "hrs_cache_df has duplicate combinations of PBN+Dealer+Vul"
    assert hrs_cache_df.select(pl.col(pl.Float64)).width == 0, hrs_cache_df.select(pl.col(pl.Float64)).columns

    # takes 1m to write 9m rows by 587 columns. 3.1GB file.
    # todo: actually only want to write if cache is updated.
    acbl_club_hand_records_cache_df_filename = 'acbl_club_hand_records_cache_df.parquet' # note: not tournament
    acbl_club_hand_records_cache_df_file = acblPath.joinpath(acbl_club_hand_records_cache_df_filename)
    hrs_cache_df.write_parquet(acbl_club_hand_records_cache_df_file)
    print(f"Saved {acbl_club_hand_records_cache_df_filename}: shape:{hrs_cache_df.shape} size:{acbl_club_hand_records_cache_df_file.stat().st_size}")
    del hrs_cache_df

    assert all([col for col, dtype in zip(hrs_df.columns, hrs_df.dtypes) if dtype == pl.Object])

    # takes 20m?/45s for 17m/1.65m rows x 5904/5893 columns. 55GB/3.5GB file.
    acbl_hand_records_augmented_filename = f'acbl_{club_or_tournament}_hand_records_augmented.parquet'
    acbl_hand_records_augmented_file = acblPath.joinpath(acbl_hand_records_augmented_filename)
    hrs_df.write_parquet(acbl_hand_records_augmented_file)
    print(f"Saved {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")    

    # takes 16m?/5s for 16m/7m->100K rows x 5904/5893 columns. 375MB file.
    # save hrs_df after sampling rows
    sample_size = 100000
    acbl_hand_records_augmented_filename = f'acbl_{club_or_tournament}_hand_records_augmented_small.parquet'
    acbl_hand_records_augmented_file = acblPath.joinpath(acbl_hand_records_augmented_filename)
    hrs_df = hrs_df if sample_size >= len(hrs_df) else hrs_df.sample(sample_size)
    hrs_df.write_parquet(acbl_hand_records_augmented_file)
    print(f"Saved {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    # takes 10s/5s for 100K rows x 5664/5546 columns. 372MB/280MB file.
    # saving hrs_df after filtering out column names and using sampled rows
    acbl_hand_records_augmented_filename = f'acbl_{club_or_tournament}_hand_records_augmented_narrow.parquet'
    acbl_hand_records_augmented_file = acblPath.joinpath(acbl_hand_records_augmented_filename)
    # filter out columns starting with C_ or CT_.*_.*_ or HB_
    hrs_df = hrs_df.select(~pl.selectors.matches(r'^(C_|CT_.*_.*_|HB_)')) # todo: probably should be done using the more commonly use pl.exclude()
    hrs_df.write_parquet(acbl_hand_records_augmented_file) # note: regex is not including CT_.*_.* e.g. CT_NS_S. Should it?
    print(f"Saved {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    return

def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Augment ACBL hand records with DD/SD analysis.")
    parser.add_argument("--club", action="store_true", help="Process club data only")
    parser.add_argument("--tournament", action="store_true", help="Process tournament data only")
    args = parser.parse_args()
    if not args.club and not args.tournament:
        return ["club", "tournament"]
    modes = []
    if args.club:
        modes.append("club")
    if args.tournament:
        modes.append("tournament")
    return modes


if __name__ == "__main__":

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    for club_or_tournament in _parse_club_tournament_args():
        t = time.time()
        augment_hand_records(club_or_tournament)
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)


