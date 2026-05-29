#!/usr/bin/env bash
# Local-LLM baseline run: 100 iter, 3 seeds, opus4.7-gods.ghost.codex-4b.gguf via LM Studio.
# Designed to run detached via nohup. Survives terminal close.
#
# Launch:   nohup bash local_baseline.sh > local_baseline.log 2>&1 & echo $! > local_baseline.pid && disown
# Monitor:  tail -f local_baseline.log
# Status:   uv run python -m scripts.inspect_runs --session N
# Abort:    kill $(cat local_baseline.pid)

set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
LOG_PREFIX="[local-baseline $(date -u +%Y-%m-%dT%H:%M:%SZ)]"

echo "$LOG_PREFIX === Starting local baseline run ==="
echo "$LOG_PREFIX model: opus4.7-gods.ghost.codex-4b.gguf @ http://192.168.50.223:1234/v1"
echo "$LOG_PREFIX 100 iterations x 3 seeds = up to 300 backtests + 100 LLM calls"
echo "$LOG_PREFIX expected wall: ~3 hours (estimate 100-150s per iteration)"
echo ""

$PY -m scripts.run_research \
    --proposer local \
    --base-url http://192.168.50.223:1234/v1 \
    --model opus4.7-gods.ghost.codex-4b.gguf \
    --max-tokens 12000 \
    --iterations 100 \
    --seeds 1,2,3 \
    --timeout 600 \
    --notes "local baseline; opus4.7-gods.ghost.codex 4B via LM Studio; deepenings #2 traces + #6 PIT membership + #7 typed signals"
LOOP_STATUS=$?

echo ""
echo "$LOG_PREFIX === Done. Exit=$LOOP_STATUS ==="
date -u
