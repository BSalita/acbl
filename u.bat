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
exit /b 0
