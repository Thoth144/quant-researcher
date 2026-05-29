#!/usr/bin/env bash
# Toy-domain overnight: local LLM proposer on sklearn hyperparameter tuning.
# Domain: toy_sklearn. Expected wall: ~1.5-2.5 hours for 100 iterations.
#
# Launch:   nohup bash toy_baseline.sh > toy_baseline.log 2>&1 & echo $! > toy_baseline.pid && disown
# Monitor:  tail -f toy_baseline.log
# Status:   uv run python -m scripts.inspect_runs --session N
# Abort:    kill $(cat toy_baseline.pid)

set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
LOG_PREFIX="[toy-baseline $(date -u +%Y-%m-%dT%H:%M:%SZ)]"

echo "$LOG_PREFIX === Toy domain overnight ==="
echo "$LOG_PREFIX domain: toy_sklearn (GradientBoostingClassifier on digits)"
echo "$LOG_PREFIX model: opus4.7-gods.ghost.codex-4b.gguf @ http://192.168.50.223:1234/v1"
echo "$LOG_PREFIX 100 iter x 3 seeds, ~30-60s per iter (LLM + 3 evals)"
echo ""

$PY -m scripts.run_research \
    --domain toy_sklearn \
    --proposer local \
    --base-url http://192.168.50.223:1234/v1 \
    --model opus4.7-gods.ghost.codex-4b.gguf \
    --max-tokens 8000 \
    --iterations 100 \
    --seeds 1,2,3 \
    --timeout 300 \
    --notes "toy_sklearn baseline; opus4.7-gods.ghost.codex 4B via LM Studio; multi-domain validation run"
LOOP_STATUS=$?

echo ""
echo "$LOG_PREFIX === Done. Exit=$LOOP_STATUS ==="
date -u
