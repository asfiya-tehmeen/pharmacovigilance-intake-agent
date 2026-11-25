# Works through the experiment queue within each day's free-tier quota. Safe to run every day: finished work
# is served from the cache, and each command stops cleanly when the daily token budget is reached.
# Usage (PowerShell, from the project folder):  .\run_daily.ps1
$ErrorActionPreference = "Continue"
$pv = ".\.venv\Scripts\pv.exe"
$log = "results\daily_$(Get-Date -Format yyyyMMdd).log"
function Step($name, $args_) { "== $name  $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append $log; & $pv @args_ 2>&1 | Select-String -Pattern '^  \d+/|agent_|no cases' | Tee-Object -Append $log }

# 1. Finish the 150-case dev set (paired v2 comparisons)
Step "dev small"  @("agent","--split","dev","--limit","150")
# 2. MAIN RUN: 300 stratified test cases, small model, retrieval-constrained coding (resumes day to day)
Step "test main"  @("agent","--split","test")
# 3. Ablations on the SAME test cases as the main run (paired): first 100 ids of the main run
$main = $null
foreach ($d in (Get-ChildItem results -Directory | Where-Object { $_.Name -like "*agent_small_retrieval" } | Sort-Object Name -Descending)) {
  $cj = Join-Path $d.FullName "cases.jsonl"; $rj = Join-Path $d.FullName "run.json"
  if ((Test-Path $cj) -and (Test-Path $rj) -and ((Get-Item $cj).Length -gt 0) -and (Select-String -Path $rj -Pattern '"split": "test"' -Quiet)) { $main = $d.FullName; break }
}
"main run for ablations: $main" | Tee-Object -Append $log
if ($main) {
  Step "ablation prompt_only" @("agent","--split","test","--like",$main,"--limit","100","--mode","prompt_only")
  Step "large model"          @("agent","--split","test","--like",$main,"--limit","100","--model","large")
  Step "self-consistency k=2" @("agent","--split","test","--like",$main,"--limit","60","--consistency-k","2")
}
& $pv budget | Tee-Object -Append $log
