# gndi/run_sota_suite.py
from __future__ import annotations
import argparse, os, sys, subprocess, shlex

def run(cmd, env=None):
    print("\n▶", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, env=env)  # stream stdout/stderr live
    if proc.returncode != 0:
        raise SystemExit(f"\n❌ Command failed (exit {proc.returncode}): {' '.join(cmd)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()

    # Ensure children can import `gndi` no matter where this is launched from
    root = os.getcwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env and env["PYTHONPATH"] else "")

    for cfg_path in args.configs:
        if not os.path.isfile(cfg_path):
            raise SystemExit(f"Config not found: {cfg_path}")

        print(f"\n=== Running suite for {cfg_path} ===")

        # 1) Causal validity
        run([sys.executable, "-u", "-m", "gndi.run_casual_val",
             "--config", cfg_path, "--units", "300", "--method", "gndi"], env=env)

        # 2) Pruning curves
        run([sys.executable, "-u", "-m", "gndi.run_prune",
             "--config", cfg_path], env=env)

    print("\n✅ All configs finished.")

if __name__ == "__main__":
    main()
