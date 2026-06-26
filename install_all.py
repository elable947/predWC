"""
predWC - World Cup 2026 Knockout Predictor
Auto-installer: sets up the environment to run the stacking model.

Usage:
    python install_all.py
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, desc, shell=False, check=True):
    print(f"\n  >> {desc}...")
    result = subprocess.run(
        cmd if isinstance(cmd, list) else cmd,
        shell=shell,
        cwd=str(ROOT),
        capture_output=not shell,
        text=not shell,
    )
    if result.returncode == 0:
        print(f"  OK")
        return True
    print(f"  FAILED")
    if result.stderr and not shell:
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"  {line}")
    if check:
        sys.exit(1)
    return False


def main():
    print("=" * 60)
    print("  predWC - World Cup 2026 Knockout Predictor")
    print("  Environment Setup")
    print("=" * 60)

    # Step 1: Install uv if not present
    def tool_exists(name):
        return subprocess.run(["which", name], capture_output=True).returncode == 0

    if not tool_exists("uv"):
        print("\n[1/3] Installing uv (Python package manager)...")
        r = subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"  ERROR: Could not install uv\n{r.stderr}")
            sys.exit(1)
        uv_bin = Path.home() / ".local" / "bin"
        if uv_bin.exists():
            os.environ["PATH"] = str(uv_bin) + ":" + os.environ.get("PATH", "")
        print("  OK")
    else:
        print("\n[1/3] uv already installed")

    # Step 2: Install Python dependencies
    run(["uv", "sync"], "[2/3] Installing Python dependencies (uv sync)")

    # Step 3: Install Playwright browsers (needed for data update scripts)
    print("\n  Note: Playwright browsers are only needed if you run update_data.py.")
    print("  For the stacking model alone they are not required.")
    run(
        ["uv", "run", "playwright", "install", "chromium"],
        "[3/3] Installing Playwright browsers",
        check=False,
    )

    print("\n" + "=" * 60)
    print("  Environment ready!")
    print()
    print("  Run the stacking model:")
    print("    uv run python stacking_model.py")
    print()
    print("  Run the NLP-enhanced model:")
    print("    uv run python stacking_model_nlp.py")
    print()
    print("  View results:")
    print("    uv run python show_results.py")
    print("    uv run python show_results.py --nlp")
    print()
    print("  Regenerate all data:")
    print("    uv run python scripts/update_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
