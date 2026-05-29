"""
Subprocess entrypoint for the toy_sklearn domain.

Mirrors researcher/_backtest_worker.py contract:
    python -m researcher._sklearn_worker <seed|none>
    python -m researcher._sklearn_worker <seed|none> --params-overrides '{"key":val,...}'

Prints one JSON line on stdout describing the result. Non-zero exit on crash.
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
        from toy_harness.evaluator import run_eval
        from toy_strategy import PARAMS, build_model

        params = PARAMS
        if overrides:
            params = dataclasses.replace(params, **overrides)

        result = run_eval(build_model, params, seed=seed)
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
