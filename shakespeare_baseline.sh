#!/usr/bin/env bash
# Shakespeare-domain overnight runner. Trains tiny GPT on tiny-shakespeare
# with the chosen proposer driving mutations to shakespeare_strategy.py.
#
# === HARDWARE TRADEOFF (READ FIRST) ===
# This machine is RTX 4070 Laptop with 8 GB VRAM. LM Studio (LocalProposer)
# claims ~5 GB when a 4B model is loaded. Tiny GPT + AdamW fits in the
# remaining ~400 MB, but Hyperball + GramNewtonSchulz ULMO does NOT — its
# spectral-iteration intermediate tensors OOM. The agent will sometimes
# propose Hyperball variants and they will crash on this hardware setup.
#
# Three valid launch configs:
#
#   1) LOCAL PROPOSER + CPU TRAINING  (safe; ~30s/seed vs ~4s on GPU)
#      LM Studio runs on GPU; training runs on CPU. No OOM risk.
#         CUDA_VISIBLE_DEVICES="" PROPOSER=local bash shakespeare_baseline.sh
#
#   2) ANTHROPIC PROPOSER + GPU TRAINING  (fast; needs API key + budget)
#      Stop LM Studio entirely; ANTHROPIC_API_KEY in ~/.anthropic_key.
#         PROPOSER=anthropic bash shakespeare_baseline.sh
#
#   3) STUB PROPOSER + GPU TRAINING  (free; deterministic-random baseline)
#      No LLM, no API. Use as a noise-floor baseline.
#         PROPOSER=stub bash shakespeare_baseline.sh
#
# === Configurable via env vars ===
#   PROPOSER       = stub | local | anthropic       (default: local)
#   ITERATIONS     = number of proposer iterations  (default: 50)
#   SEEDS          = comma-separated replicate seeds (default: 1,2,3)
#   BASE_URL       = (local only) OpenAI-compatible endpoint
#   MODEL          = (local/anthropic only) model id
#   TIMEOUT        = per-backtest subprocess timeout in seconds (default: 600)
#   ATTRIBUTE      = '--attribute' to run per-signal attribution on KEEPs
#   CUDA_VISIBLE_DEVICES="" = forces CPU training (use with PROPOSER=local)
#
# Launch (detached, survives terminal/session close):
#   nohup bash shakespeare_baseline.sh > shakespeare_baseline.log 2>&1 & \
#       echo $! > shakespeare_baseline.pid && disown
#
# Monitor:
#   tail -f shakespeare_baseline.log
#   uv run python -m scripts.dashboard --session N --refresh 2
#   uv run python -m scripts.inspect_runs --session N --leaderboard 20
#
# Abort:
#   kill $(cat shakespeare_baseline.pid)

set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
PROPOSER="${PROPOSER:-local}"
ITERATIONS="${ITERATIONS:-50}"
SEEDS="${SEEDS:-1,2,3}"
BASE_URL="${BASE_URL:-http://192.168.50.223:1234/v1}"
MODEL="${MODEL:-opus4.7-gods.ghost.codex-4b.gguf}"
TIMEOUT="${TIMEOUT:-600}"
ATTRIBUTE="${ATTRIBUTE:-}"

LOG_PREFIX="[shakespeare-baseline $(date -u +%Y-%m-%dT%H:%M:%SZ)]"

echo "$LOG_PREFIX === Shakespeare domain training run ==="
echo "$LOG_PREFIX proposer: $PROPOSER"
echo "$LOG_PREFIX iterations: $ITERATIONS (seeds=$SEEDS)"
echo "$LOG_PREFIX domain: shakespeare (tiny GPT on tiny-shakespeare, val_bpb metric)"
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "$LOG_PREFIX device: CPU (CUDA_VISIBLE_DEVICES is unset, GPU off-limits to training)"
else
    echo "$LOG_PREFIX device: CUDA if available (set CUDA_VISIBLE_DEVICES='' to force CPU)"
fi
if [ "$PROPOSER" = "local" ]; then
    echo "$LOG_PREFIX proposer model: $MODEL @ $BASE_URL"
elif [ "$PROPOSER" = "anthropic" ]; then
    echo "$LOG_PREFIX proposer model: ${MODEL:-claude-sonnet-4-5-20251022}"
fi

# Pre-flight checks
echo "$LOG_PREFIX pre-flight:"
if [ "$PROPOSER" = "local" ]; then
    if ! curl -s -m 3 "$BASE_URL/models" > /dev/null 2>&1; then
        echo "$LOG_PREFIX   WARN: LocalProposer endpoint $BASE_URL unreachable"
    else
        echo "$LOG_PREFIX   OK: LocalProposer endpoint reachable"
    fi
fi
if [ "$PROPOSER" = "anthropic" ]; then
    if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ ! -f ~/.anthropic_key ]; then
        echo "$LOG_PREFIX   ERROR: ANTHROPIC_API_KEY not set and ~/.anthropic_key missing — aborting"
        exit 1
    fi
    echo "$LOG_PREFIX   OK: Anthropic key reachable"
fi
echo ""

# Build the argv based on proposer choice
ARGS=(--domain shakespeare --proposer "$PROPOSER" --iterations "$ITERATIONS"
      --seeds "$SEEDS" --timeout "$TIMEOUT")
if [ "$PROPOSER" = "local" ]; then
    ARGS+=(--base-url "$BASE_URL" --model "$MODEL" --max-tokens 8000)
elif [ "$PROPOSER" = "anthropic" ]; then
    [ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")
fi
[ -n "$ATTRIBUTE" ] && ARGS+=("$ATTRIBUTE")
ARGS+=(--notes "shakespeare training; proposer=$PROPOSER")

$PY -m scripts.run_research "${ARGS[@]}"
LOOP_STATUS=$?

echo ""
echo "$LOG_PREFIX === Done. Exit=$LOOP_STATUS ==="
date -u
