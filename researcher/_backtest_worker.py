"""
Subprocess entrypoint: import the current strategy.py, run one backtest with a
given seed, print one JSON object to stdout. Isolation matters — a candidate
that monkey-patches global state can't corrupt sibling trials.

CLI:
    python -m researcher._backtest_worker <seed|none>
    python -m researcher._backtest_worker <seed|none> --params-overrides '{"weights": {...}}'

The optional --params-overrides JSON is shallow-merged into PARAMS via
dataclasses.replace. Used by researcher/attribution.py to isolate per-signal
contributions without editing strategy.py.
"""

import dataclasses
import json
import sys
import traceback

if __name__ == "__main__":
    args = sys.argv[1:]
    seed_arg = args[0] if args else "none"
    seed = None if seed_arg == "none" else int(seed_arg)

    overrides: dict = {}
    if "--params-overrides" in args:
        i = args.index("--params-overrides")
        overrides = json.loads(args[i + 1])

    try:
        from harness.backtest import run_backtest
        from strategy import PARAMS, generate_signals

        params = PARAMS
        if overrides:
            params = dataclasses.replace(params, **overrides)

        result = run_backtest(generate_signals, params, seed=seed)
        out = {
            "ok": True,
            "params": params.to_dict(),
            "result": result.to_dict(),
            "primary_metric": result.primary_metric,
        }
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)
