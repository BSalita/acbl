@echo off
setlocal
echo ======================================================================
echo  ACBL Full Pipeline
echo  Produces data files consumed by:
echo    ..\Elo_Ratings          (player/pair Elo parquets)
echo    ..\Bridge_Game_Postmortem_Chatbot (SavedModels, Elo parquets)
echo.
echo  Approximate end-to-end wall time on the dev box
echo  (192 GB RAM, ~40-core CPU, NVMe E:, RTX 5080):
echo    Cold start (no caches, fresh DD/SD work): ~2 days dominated by 3a (~38 h)
echo    Warm rerun  (3a cache hits, just incremental work): ~12-14 h + 5c
echo    Stage 5c (train all 6 models) measured 30.4 h on 2026-07-10 -^> 2026-07-11.
echo    Empirical bottlenecks per stage are noted as "TIME:" tags below.
echo    Each step prints its own measured elapsed time as "TIME[step]: ..." lines.
echo  Last full-pipeline timing baseline: 2026-07-07 -^> 2026-07-11 (see logs/).
echo  Model training results history: RESULTS.md (append an entry after each 5c run).
echo ======================================================================
echo.
echo Start: %date% %time%
echo.
call :now PIPE_T0

:: ====================================================================
:: STAGE 1: DATA INGESTION -- download raw data from ACBL
:: ====================================================================
echo [Stage 1] Data ingestion...

:: ---- 1a ----
:: READS:  (web scrape from my.acbl.org)
:: WRITES: acbl/club-results/{club_id}/details/{session_id}.data.json
:: TIME:   incremental; only fetches sessions not already on disk.
::         Cold start (full ACBL backlog): many hours, network-bound.
::         Daily incremental: ~minutes.
echo   [1a] Downloading club results to JSON...
call :now STEP_T0
python acbl_club_download_to_json.py --sleep 2
if errorlevel 1 goto :error
call :toc 1a

:: ---- 1b ----
:: READS:  acbl/club-results/*/details/*.data.json
:: WRITES: acbl/club-results/*/details/*.data.sql
::         acbl/acbl_club_results.sqlite
:: TIME:   ~minutes incremental; ~1-2 h on a full rebuild from scratch.
echo   [1b] Loading club JSON into SQLite...
call :now STEP_T0
python acbl_club_json_to_sql.py
if errorlevel 1 goto :error
call :toc 1b

:: ---- 1c ----
:: READS:  (ACBL API via ACBL_API_KEY)
:: WRITES: acbl/tournaments/events/{sanction_id}.sanction.json
:: TIME:   incremental; ~minutes daily, ~1-2 h cold start (API-rate-limited).
echo   [1c] Downloading tournament sanctioned events...
call :now STEP_T0
python acbl_tournament_download_sanctioned_events.py
if errorlevel 1 goto :error
call :toc 1c

:: ---- 1d ----
:: READS:  acbl/tournaments/events/*.sanction.json
:: WRITES: acbl/tournaments/sessions/{session_id}.session.json
:: TIME:   incremental; ~minutes daily, several hours cold start (API-bound).
echo   [1d] Downloading tournament sessions...
call :now STEP_T0
python acbl_tournament_download_sessions_using_sanctioned_events.py
if errorlevel 1 goto :error
call :toc 1d

:: ---- 1e ----
:: READS:  acbl/tournaments/sessions/*.session.json
:: WRITES: acbl/tournaments/sessions/*.session.sql
::         acbl/acbl_tournament_results.sqlite
:: TIME:   ~minutes incremental; ~30-60 min on a full rebuild.
echo   [1e] Loading tournament sessions into SQLite...
call :now STEP_T0
python acbl_tournament_sessions_json_to_sql.py
if errorlevel 1 goto :error
call :toc 1e

:: ====================================================================
:: STAGE 2: CLEANING -- SQLite -> cleaned parquets
:: ====================================================================
echo.
echo [Stage 2] Cleaning...

:: ---- 2a ----
:: READS:  acbl/acbl_club_results.sqlite        (tables: hand_records, sessions)
::         acbl/acbl_tournament_results.sqlite   (tables: handrecord, session)
:: WRITES: acbl/acbl_club_hand_records_cleaned.parquet
::         acbl/acbl_tournament_hand_records_cleaned.parquet
:: TIME:   ~5-10 min total (both club + tournament). SQLite read-bound.
echo   [2a] Cleaning hand records...
call :now STEP_T0
python acbl_sql_to_hand_records_clean.py
if errorlevel 1 goto :error
call :toc 2a

:: ---- 2b ----
:: READS:  acbl/acbl_club_results.sqlite        (tables: events, board_results, boards, ...)
::         acbl/acbl_tournament_results.sqlite
::         acbl/acbl_club_board_results_cleaned.parquet  (for tournament enrichment)
:: WRITES: acbl/acbl_club_board_results_cleaned.parquet
::         acbl/acbl_tournament_board_results_cleaned.parquet
:: TIME:   ~15-30 min total. Wider tables, more joins than 2a.
echo   [2b] Cleaning board results...
call :now STEP_T0
python acbl_sql_to_board_results_clean.py
if errorlevel 1 goto :error
call :toc 2b

:: ====================================================================
:: STAGE 3: AUGMENTATION -- DD/SD/Par analysis + board result enrichment
:: ====================================================================
echo.
echo [Stage 3] Augmentation...

:: ---- 3a ----
:: READS:  acbl/acbl_{club,tournament}_hand_records_cleaned.parquet
::         acbl/acbl_club_hand_records_cache_df.parquet  (DD+SD cache, shared by club+tournament)
:: WRITES: acbl/acbl_club_hand_records_cache_df.parquet  (updated incrementally every 50K PBNs)
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented_small.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented_narrow.parquet
:: TIME:   COLD START ~38 h for 747K novel PBNs (batched SD pipeline; CPU-bound).
::         WARM (cache hits): ~minutes incremental for daily new PBNs.
::         By far the longest single step in a cold pipeline rebuild.
echo   [3a] Augmenting hand records (DD + SD + Par)...
call :now STEP_T0
python acbl_hand_records_augment.py
if errorlevel 1 goto :error
call :toc 3a

:: ---- 3b ----
:: READS:  acbl/acbl_{club,tournament}_board_results_cleaned.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
:: TIME:   ~5-15 min total (club is the bigger half).
echo   [3b] Augmenting board results (step 1: contracts + vulnerability)...
call :now STEP_T0
python acbl_board_results_augment_step1.py
if errorlevel 1 goto :error
call :toc 3b

:: ---- 3c ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: TIME:   ~30-60 min total. Joins ~6k-col hand-record features into board results.
echo   [3c] Augmenting board results (step 2: join hand records + full augmentation)...
call :now STEP_T0
python acbl_board_results_augment_step2.py
if errorlevel 1 goto :error
call :toc 3c

:: ====================================================================
:: STAGE 4: ELO RATINGS
:: ====================================================================
echo.
echo [Stage 4] Elo ratings...

:: ---- 4 ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_elo_ratings.parquet
::         acbl/acbl_{club,tournament}_player_elo_ratings.parquet  -> Elo_Ratings, Chatbot
::         acbl/acbl_{club,tournament}_pair_elo_ratings.parquet    -> Elo_Ratings, Chatbot
:: TIME:   ~30-60 min total (tournament ~10 min, club ~30-45 min).
::         Walks games chronologically; mostly single-threaded.
echo   [4] Computing Elo ratings (player + pair)...
call :now STEP_T0
python acbl_elo_ratings_create.py
if errorlevel 1 goto :error
call :toc 4

:: ====================================================================
:: STAGE 5: ML MODEL DATA
:: ====================================================================
echo.
echo [Stage 5] ML model pipeline...

:: ---- 5a ----
:: READS:  acbl/acbl_{club,tournament}_hand_records_augmented.parquet
::         acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_model_data_d.pkl
::         acbl/acbl_{club,tournament}_model_data.parquet
:: TIME:   tournament ~40 min (15.9M rows x 6780 cols, 16.7 GB parquet).
::         club      shards ~60 min + final merge ~80 min (59.7M rows x 6772
::                   cols, 60.6 GB). The merge streams row groups via pyarrow
::                   (measured 4822 s on 2026-07-08; polars streaming concat
::                   aborted on this width -- see _stream_concat_shards).
::         Total ~3 h. Wall clock measured 2026-07-08.
echo   [5a] Building model data...
call :now STEP_T0
python acbl_model_data.py
if errorlevel 1 goto :error
call :toc 5a

:: ---- 5b ----
:: READS:  acbl/acbl_{club,tournament}_model_data_d.pkl
::         acbl/acbl_{club,tournament}_model_data.parquet
::         acbl/acbl_{club,tournament}_player_elo_ratings.parquet
::         acbl/acbl_{club,tournament}_pair_elo_ratings.parquet
:: WRITES: acbl/acbl_{club,tournament}_prediction_data_train.parquet
::         acbl/acbl_{club,tournament}_prediction_data_test.parquet
:: TIME:   tournament ~30-60 min (~16M rows, train+test ~42 GB).
::         club      ~7.3 h (55.2M train + 4.3M test rows; 166 GB train,
::                   12 GB test parquet). One year (2022) sank in 3 h on its
::                   own; see script docstring "KNOWN ISSUE (2026-04-21)".
::         Total ~8 h. Wall clock measured 2026-04-20 -> 2026-04-21.
echo   [5b] Preparing prediction data (train/test split)...
call :now STEP_T0
python acbl_prediction_data.py
if errorlevel 1 goto :error
call :toc 5b

:: ---- 5c ----
:: READS:  acbl/acbl_{club,tournament}_prediction_data_train.parquet
::         acbl/acbl_{club,tournament}_prediction_data_test.parquet
:: WRITES: acbl/SavedModels/{model_name}_schema.json                -> Chatbot
::         acbl/SavedModels/*model_shard_*.pt                       -> Chatbot
::         acbl/SavedModels/{model_name}_importance.csv
::         acbl/debug_input_{y_name}.parquet
::         acbl/debug_predictions_{y_name}.parquet
:: TIME:   30.4 h TOTAL for all 6 models (both modes x 3 targets), 20 epochs
::         each on RTX 5080. Wall clock measured 2026-07-10 14:42 -> 2026-07-11 21:05.
::         club (84716 s = 23.5 h):
::           Declarer_Direction: shards 2h15m + train 6h10m (~1110 s/epoch)
::           Contract:           shards 2h35m + train 6h33m (~1178 s/epoch)
::           Pct_NS (pruned):    shards   13m + train 1h28m (~262 s/epoch)
::         tournament (24244 s = 6.7 h):
::           Declarer_Direction: shards   41m + train 1h35m (~284 s/epoch)
::           Contract:           shards   40m + train 1h48m (~324 s/epoch)
::           Pct_NS (pruned):    shards    4m + train   14m (early stop @15)
::         DISK: club shard sets peak at ~1.25 TB on E: (28 x ~45 GB
::         uncompressed float32); ensure ~1.3 TB free before this step
::         (a full E: caused a torch.save iostream crash on 2026-07-09).
::         Run this step alone (not via acbl_all.bat) when iterating on models.
echo   [5c] Training prediction models...
call :now STEP_T0
python acbl_prediction_train.py
if errorlevel 1 goto :error
call :toc 5c

:: ====================================================================
:: DONE
:: ====================================================================
echo.
echo ======================================================================
echo  Pipeline complete: %date% %time%
call :now PIPE_T1
set /a PIPE_ELAPSED=PIPE_T1-PIPE_T0
set /a PIPE_H=PIPE_ELAPSED/3600
set /a PIPE_M=(PIPE_ELAPSED %% 3600)/60
set /a PIPE_S=PIPE_ELAPSED %% 60
echo  TIME[total]: %PIPE_ELAPSED%s (%PIPE_H%h %PIPE_M%m %PIPE_S%s)
echo.
echo  Downstream consumers and their required files:
echo.
echo  ..\Elo_Ratings:
echo    acbl_{club,tournament}_elo_ratings.parquet         (full Elo history)
echo    acbl_{club,tournament}_player_elo_ratings.parquet  (player lookup)
echo    acbl_{club,tournament}_pair_elo_ratings.parquet    (pair lookup)
echo.
echo  ..\Bridge_Game_Postmortem_Chatbot:
echo    acbl_{club,tournament}_player_elo_ratings.parquet  (player lookup)
echo    acbl_{club,tournament}_pair_elo_ratings.parquet    (pair lookup)
echo    SavedModels\*_schema.json                         (model schemas)
echo    SavedModels\*model_shard_*.pt                     (trained weights)
echo ======================================================================
goto :eof

:: --------------------------------------------------------------------
:: Timing helpers. Epoch seconds via PowerShell so steps longer than
:: 24 h (e.g. 5c) are measured correctly; %time% arithmetic wraps at
:: midnight and cannot represent multi-day steps.
:: --------------------------------------------------------------------

:: usage: call :now VARNAME  -- store current unix epoch seconds in VARNAME
:now
for /f %%t in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "%~1=%%t"
goto :eof

:: usage: call :toc LABEL  -- print elapsed time since STEP_T0
:toc
call :now STEP_T1
set /a STEP_ELAPSED=STEP_T1-STEP_T0
set /a STEP_H=STEP_ELAPSED/3600
set /a STEP_M=(STEP_ELAPSED %% 3600)/60
set /a STEP_S=STEP_ELAPSED %% 60
echo   TIME[%~1]: %STEP_ELAPSED%s (%STEP_H%h %STEP_M%m %STEP_S%s) ended %date% %time%
echo.
goto :eof

:error
echo.
echo *** FAILED at %date% %time% (errorlevel %errorlevel%) ***
exit /b 1
