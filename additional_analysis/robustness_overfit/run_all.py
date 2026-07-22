"""Run all additive robustness analyses."""

from __future__ import annotations

import os
import subprocess
import sys


HERE = os.path.dirname(__file__)
SCRIPTS = [
    "01_training_diagnostics.py",
    "02_regularized_linear_benchmarks.py",
    "03_rolling_origin_sensitivity.py",
]


def main() -> None:
    mpl_cache = os.path.join(HERE, "out", "mplconfig")
    os.makedirs(mpl_cache, exist_ok=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = mpl_cache
    for script in SCRIPTS:
        path = os.path.join(HERE, script)
        print(f"\n=== running {script} ===", flush=True)
        subprocess.run([sys.executable, path], check=True, env=env)


if __name__ == "__main__":
    main()
