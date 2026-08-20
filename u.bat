rem Update the stage-1b parquets the ACBL club API needs (acbl_club_api_service.py
rem tier 1: listings/lookups). OneDrive syncs them to the production host, where
rem postmortem_start.ps1 bind-mounts club_results_parquet into the acbl-club-api
rem container. Only these five tables are needed; the rest of
rem e:\bridge\data\acbl\club_results_parquet (and the club-results JSON archive)
rem stays on e: -- production live-scrapes session details it doesn't have.
xcopy e:\bridge\data\acbl\club_results_parquet\events.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\players.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\pair_summaries.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\sections.parquet club_results_parquet\ /D /Y
xcopy e:\bridge\data\acbl\club_results_parquet\sessions.parquet club_results_parquet\ /D /Y
