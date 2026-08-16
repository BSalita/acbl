@echo off
setlocal EnableExtensions
:: Use project venv (has requests, polars, torch, ...). Bare `python` on PATH
:: is often the system install and will fail with ModuleNotFoundError.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo *** FAILED: project venv not found: %PY%
  echo Create it with: python -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
:: Success marker for :pyrun. Ctrl+C then "Terminate batch job? N" clears
:: ERRORLEVEL to 0, so `if errorlevel 1` alone lets the next step run.
:: Only a successful python exit creates this file (via &&); missing file = abort.
set "STEP_OK=%TEMP%\acbl_all_step.ok"
echo ======================================================================
echo  ACBL Full Pipeline
echo  Produces data files consumed by:
echo    ..\Elo_Ratings          (player/pair Elo parquets)
echo    ..\Bridge_Game_Postmortem_Chatbot (SavedModels, Elo parquets)
echo.
echo  Approximate end-to-end wall time on the dev box
echo  (192 GB RAM, ~40-core CPU, NVMe E:, RTX 5080):
echo    Cold start (no caches, fresh DD/SD work): ~2 days dominated by 3a (~38 h)
echo    Warm rerun  (3a cache hits, just incremental work): ~10-12 h + 5c
echo      (was ~12-14 h; 5a shard-output change of 2026-08-16 removed the
echo       single-file merge and cut 5a to ~15 min tournament / ~1 min resume.
echo       5b now reads shards too -- remeasure on next full run.)
echo    Stage 5c (train all 6 models) measured 30.4 h on 2026-07-10 -^> 2026-07-11.
echo    Empirical bottlenecks per stage are noted as "TIME:" tags below.
echo    Each step prints its own measured elapsed time as "TIME[step]: ..." lines.
echo  Last full-pipeline timing baseline: 2026-07-07 -^> 2026-07-11 (see logs/).
echo  Model training results history: RESULTS.md (append an entry after each 5c run).
echo ======================================================================
echo.
echo Using: %PY%
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
call :pyrun 1a acbl_club_download_to_json.py --sleep 2
if errorlevel 1 goto :error

:: ---- 1b ----
:: READS:  acbl/club-results/*/details/*.data.json
:: WRITES: acbl/club_results_parquet/*.parquet
::         acbl/acbl_club_results.sqlite   (same schema as legacy)
:: TIME:   parallel JSON->Parquet->SQLite (option F). Cold rebuild target
::         ~0.5-1.5 h on 64-core/512GB vs ~4 h legacy .data.sql path.
::         Use: python acbl_club_json_to_sql.py --legacy-sql-scripts
echo   [1b] Loading club JSON into SQLite (via Parquet)...
call :pyrun 1b acbl_club_json_to_sql.py
if errorlevel 1 goto :error

:: ---- 1c ----
:: READS:  (ACBL API via ACBL_API_KEY)
:: WRITES: acbl/tournaments/events/{sanction_id}.sanction.json
:: TIME:   incremental; ~minutes daily, ~1-2 h cold start (API-rate-limited).
echo   [1c] Downloading tournament sanctioned events...
call :pyrun 1c acbl_tournament_download_sanctioned_events.py
if errorlevel 1 goto :error

:: ---- 1d ----
:: READS:  acbl/tournaments/events/*.sanction.json
:: WRITES: acbl/tournaments/sessions/{session_id}.session.json
:: TIME:   incremental; ~minutes daily, several hours cold start (API-bound).
echo   [1d] Downloading tournament sessions...
call :pyrun 1d acbl_tournament_download_sessions_using_sanctioned_events.py
if errorlevel 1 goto :error

:: ---- 1e ----
:: READS:  acbl/tournaments/sessions/*.session.json
:: WRITES: acbl/tournaments/sessions/*.session.sql
::         acbl/acbl_tournament_results.sqlite
:: TIME:   ~minutes incremental; ~30-60 min on a full rebuild.
echo   [1e] Loading tournament sessions into SQLite...
call :pyrun 1e acbl_tournament_sessions_json_to_sql.py
if errorlevel 1 goto :error

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
call :pyrun 2a acbl_sql_to_hand_records_clean.py
if errorlevel 1 goto :error

:: ---- 2b ----
:: READS:  acbl/acbl_club_results.sqlite        (tables: events, board_results, boards, ...)
::         acbl/acbl_tournament_results.sqlite
::         acbl/acbl_club_board_results_cleaned.parquet  (for tournament enrichment)
:: WRITES: acbl/acbl_club_board_results_cleaned.parquet
::         acbl/acbl_tournament_board_results_cleaned.parquet
:: TIME:   ~15-30 min total. Wider tables, more joins than 2a.
echo   [2b] Cleaning board results...
call :pyrun 2b acbl_sql_to_board_results_clean.py
if errorlevel 1 goto :error

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
call :pyrun 3a acbl_hand_records_augment.py
if errorlevel 1 goto :error

:: ---- 3b ----
:: READS:  acbl/acbl_{club,tournament}_board_results_cleaned.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
:: TIME:   ~5-15 min total (club is the bigger half).
echo   [3b] Augmenting board results (step 1: contracts + vulnerability)...
call :pyrun 3b acbl_board_results_augment_step1.py
if errorlevel 1 goto :error

:: ---- 3c ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: TIME:   ~30-60 min total. Joins ~6k-col hand-record features into board results.
echo   [3c] Augmenting board results (step 2: join hand records + full augmentation)...
call :pyrun 3c acbl_board_results_augment_step2.py
if errorlevel 1 goto :error

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
call :pyrun 4 acbl_elo_ratings_create.py
if errorlevel 1 goto :error

:: ====================================================================
:: STAGE 5: ML MODEL DATA
:: ====================================================================
echo.
echo [Stage 5] ML model pipeline...

:: ---- 5a ----
:: READS:  acbl/acbl_{club,tournament}_hand_records_augmented.parquet
::         acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_model_data_d.pkl
::         acbl/shards_{club,tournament}_model_data/*.parquet + manifest.json
::         (default --no-merge-shards since 2026-08-16: the single-file merge
::         took ~12 h for club and made downstream reads SLOWER; consumers
::         now scan the shard glob with file-level Date pruning instead)
:: TIME:   tournament ~15 min (16.7M rows x 6786 cols -> 132 monthly shards,
::                   15.3 GB; measured 889 s on 2026-08-16).
::         club      ~60-75 min cold (69.4M rows x 6778 cols -> 96 monthly
::                   shards, 86 GB; ~60 min measured 2026-04 at 59.7M rows).
::         Resume (all shards valid): ~1 min per mode (club measured 54 s
::         on 2026-08-16). Logs: logs/05a_model_data_noshardmerge_*.log.
echo   [5a] Building model data...
call :pyrun 5a acbl_model_data.py
if errorlevel 1 goto :error

:: ---- 5b ----
:: READS:  acbl/acbl_{club,tournament}_model_data_d.pkl
::         acbl/shards_{club,tournament}_model_data/*.parquet (or legacy
::         acbl_{club,tournament}_model_data.parquet if no shard dir)
::         acbl/acbl_{club,tournament}_player_elo_ratings.parquet
::         acbl/acbl_{club,tournament}_pair_elo_ratings.parquet
:: WRITES: acbl/acbl_{club,tournament}_prediction_data_train.parquet
::         acbl/acbl_{club,tournament}_prediction_data_test.parquet
:: TIME:   MEASURED 2026-04-20 -> 2026-04-21 (at 59.5M club rows, reading
::         the single merged model_data file):
::           tournament ~30-60 min (~16M rows, train+test ~42 GB).
::           club       ~7.3 h (55.2M train + 4.3M test rows; 166 GB train,
::                      12 GB test), of which 3 h was one anomalous year
::                      (2022) -- see script docstring "KNOWN ISSUE".
::           Non-anomalous sink rate: ~0.26 ms/row.
::         ESTIMATE at current volume (69.4M club rows, 2026-08-16): club
::         ~5 h (+2.5 h if the 2022 anomaly repeats), tournament ~0.5-1 h,
::         total ~6 h expected / ~8.5 h worst case. Source is now the
::         monthly shard dir (per-year scans prune to ~12 shard files vs
::         scanning the whole 86 GB merged file), which may shave another
::         10-30% off the scan side. Remeasure on next full run.
echo   [5b] Preparing prediction data (train/test split)...
call :pyrun 5b acbl_prediction_data.py
if errorlevel 1 goto :error

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
call :pyrun 5c acbl_prediction_train.py
if errorlevel 1 goto :error

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
del /q "%STEP_OK%" 2>nul
goto :eof

:: --------------------------------------------------------------------
:: Run one python step. Returns exit /b 1 on failure OR Ctrl+C.
:: Callers MUST check:  if errorlevel 1 goto :error
::
:: Two Windows quirks this defends against:
::  1) Ctrl+C then "Terminate batch job? N" clears ERRORLEVEL to 0.
::     STEP_OK is only created via `&&` after a successful python exit,
::     so a missing marker still fails the step.
::  2) `goto :error` + `exit /b` from inside a CALLed subroutine only
::     returns to the caller -- the pipeline would continue. So :pyrun
::     returns /b 1 and the top-level caller jumps to :error.
::
:: usage: call :pyrun LABEL script.py [args...]
:: --------------------------------------------------------------------
:pyrun
set "STEP_LABEL=%~1"
shift
del /q "%STEP_OK%" 2>nul
call :now STEP_T0
"%PY%" %1 %2 %3 %4 %5 %6 %7 %8 %9 && echo.>"%STEP_OK%"
if not exist "%STEP_OK%" exit /b 1
call :toc %STEP_LABEL%
exit /b 0

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
echo *** FAILED/INTERRUPTED at step %STEP_LABEL% %date% %time% ***
del /q "%STEP_OK%" 2>nul
exit /b 1
