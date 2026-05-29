#!/usr/bin/env bash
# Overnight runner: refresh full S&P 500 universe + bars, then 300-iter stub research loop.
# Designed to run detached via nohup so it survives terminal/session close.
#
# Launch:   nohup bash overnight.sh > overnight.log 2>&1 & echo $! > overnight.pid && disown
# Monitor:  tail -f overnight.log
# Status:   uv run python -m scripts.inspect_runs
# Abort:    kill $(cat overnight.pid)

set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
LOG_PREFIX="[overnight $(date -u +%Y-%m-%dT%H:%M:%SZ)]"

echo "$LOG_PREFIX === Phase 1: refresh pinned universe (re-fetch S&P 500) ==="
# Force re-fetch by removing the existing pinned file. Wikipedia fetch uses
# requests with a real UA. Fallback to 100-name hardcoded list on failure.
rm -f data/sp500_universe.txt

$PY -m data.prepare --pause 0.15
PREP_STATUS=$?
echo "$LOG_PREFIX Phase 1 exit=$PREP_STATUS"

if [ $PREP_STATUS -ne 0 ]; then
    echo "$LOG_PREFIX data.prepare failed; aborting before launching loop."
    exit 1
fi

CACHED=$(ls data/cache/*.parquet 2>/dev/null | wc -l)
PINNED=$(wc -l < data/sp500_universe.txt)
echo "$LOG_PREFIX universe pinned=$PINNED tickers, cached=$CACHED parquets"

if [ "$CACHED" -lt 20 ]; then
    echo "$LOG_PREFIX too few tickers cached ($CACHED); aborting before loop."
    exit 1
fi

echo ""
echo "$LOG_PREFIX === Phase 2: 300-iter stub baseline research loop ==="
$PY -m scripts.run_research \
    --proposer stub \
    --iterations 300 \
    --seeds 1,2,3 \
    --proposer-seed 42 \
    --notes "stub baseline, 300 iter, refreshed universe $(date -u +%Y-%m-%d)"
LOOP_STATUS=$?

echo ""
echo "$LOG_PREFIX === Done. Phase 2 exit=$LOOP_STATUS ==="
date -u
