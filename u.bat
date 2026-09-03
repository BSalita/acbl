rem Update the stage-1b parquets the ACBL club API needs. OneDrive syncs them
rem to the production host, where postmortem_start.ps1 stages them into the
rem postmortem container. Five small relationship tables support listings and
rem lookups. The Stage 3c augmented monolith supplies complete historical
rem club postmortems without rebuilding or augmenting individual sessions.
rem The tournament monolith provides the same API/MCP path for tournaments.
xcopy e:\bridge\data\acbl\club_results_parquet\events.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\players.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\pair_summaries.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\sections.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\sessions.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\acbl_club_board_results_augmented.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\acbl_tournament_board_results_augmented.parquet club_results_parquet\ /D /Y
if errorlevel 1 exit /b 1

rem Obsolete Stage 1b reconstruction files. Recent sessions missing from the
rem monolith use the bounded JSON archive/live fallback instead.
del /q club_results_parquet\club.parquet 2>nul
del /q club_results_parquet\boards.parquet 2>nul
del /q club_results_parquet\board_results.parquet 2>nul
del /q club_results_parquet\hand_records.parquet 2>nul

rem Recent slice of the session-details JSON archive (last 30 days, ~13k files,
rem ~3.5 GB). Production mounts this read-only as its archive tier so session
rem results are served without live Cloudflare scrapes. Files older than 45
rem days are pruned to keep the OneDrive folder bounded.
robocopy e:\bridge\data\acbl\club-results club-results-recent *.data.json /S /MAXAGE:30 /NDL /NFL /NJH /NJS /R:2 /W:2
forfiles /P club-results-recent /S /M *.data.json /D -45 /C "cmd /c del @path" 2>nul

rem Publish to prod acbl-pipeline, then update only acbl-stage under
rem _wslc_host. Do not /MIR the whole _wslc_host tree (SavedModels, Chrome
rem profile, and postmortem caches live there).
set "prod_acbl=\\X1-pro-470-1tb\c\sw\bridge\ML-Contract-Bridge\src\acbl-pipeline"
set "prod_stage=\\X1-pro-470-1tb\c\sw\bridge\ML-Contract-Bridge\src\elo\data\_wslc_host\acbl-stage"
if not exist "%prod_acbl%\" exit /b 1
robocopy club_results_parquet "%prod_acbl%\club_results_parquet" /E /XO /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 exit /b 1
if not exist "%prod_stage%\club_results_parquet\" (
    mkdir "%prod_stage%\club_results_parquet"
    if errorlevel 1 exit /b 1
)
robocopy club_results_parquet "%prod_stage%\club_results_parquet" /E /XO /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 exit /b 1
for %%F in (club.parquet boards.parquet board_results.parquet hand_records.parquet) do (
    del /q "%prod_acbl%\club_results_parquet\%%F" 2>nul
    del /q "%prod_stage%\club_results_parquet\%%F" 2>nul
)
if exist "club-results-recent\" (
    robocopy club-results-recent "%prod_acbl%\club-results-recent" /E /XO /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
    if errorlevel 8 exit /b 1
    robocopy club-results-recent "%prod_stage%\club-results-recent" /E /XO /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
    if errorlevel 8 exit /b 1
)

exit /b 0
