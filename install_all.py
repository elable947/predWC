"""
predWC - World Cup 2026 Knockout Predictor
Auto-installer: sets up the environment to run the stacking model.

Usage:
    python install_all.py
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def color(text, code):
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def run(cmd, desc, shell=False, check=True, capture=True):
    print(f"\n  >> {desc}...", flush=True)
    kw = dict(cwd=str(ROOT), text=True)
    if shell:
        kw["shell"] = True
        kw["capture_output"] = False
        result = subprocess.run(cmd if isinstance(cmd, list) else cmd, **kw)
    elif capture:
        kw["capture_output"] = True
        result = subprocess.run(
            cmd if isinstance(cmd, list) else cmd.split(),
            **kw,
        )
        if result.stdout and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
    else:
        result = subprocess.run(
            cmd if isinstance(cmd, list) else cmd.split(),
            **kw,
        )
    if result.returncode == 0:
        print(f"  {color('OK', '92')}")
        return True
    print(f"  {color('FAILED', '91')}")
    if result.stderr and not shell:
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"  {line}")
    if check:
        sys.exit(1)
    return False


def tool_exists(name):
    return shutil.which(name) is not None


def main():
    print()
    print(color("=" * 60, "94"))
    print(color("  predWC - World Cup 2026 Knockout Predictor", "94"))
    print(color("  Environment Setup", "94"))
    print(color("=" * 60, "94"))

    # ------------------------------------------------------------------
    # Step 0: Check Python version
    # ------------------------------------------------------------------
    print(f"\n  Python: {sys.version}")
    if sys.version_info < (3, 12):
        print(f"  {color('ERROR: Python >= 3.12 required. You have ' + sys.version.split()[0], '91')}")
        print("  Manjaro: sudo pacman -S python")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Install uv
    # ------------------------------------------------------------------
    if not tool_exists("uv"):
        print(f"\n{color('[1/4]', '96')} Installing uv (Python package manager)...")
        if not tool_exists("curl"):
            print("  curl not found. Installing it...")
            run(["sudo", "pacman", "-S", "--noconfirm", "curl"],
                "Installing curl via pacman")
        r = subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  {color('ERROR: Could not install uv', '91')}")
            print(f"  {r.stderr}")
            sys.exit(1)
        uv_bin = Path.home() / ".local" / "bin"
        if uv_bin.exists():
            os.environ["PATH"] = str(uv_bin) + ":" + os.environ.get("PATH", "")
        print("  uv installed. If 'uv' is not found, run:")
        print('    export PATH="$HOME/.local/bin:$PATH"')
        print("  Or close and reopen your terminal.")
    else:
        print(f"\n{color('[1/4]', '96')} uv already installed")

    # ------------------------------------------------------------------
    # Step 2: Install Python dependencies
    # ------------------------------------------------------------------
    print(f"\n{color('[2/4]', '96')} Installing Python dependencies (uv sync)...")
    run("uv sync", "uv sync", check=True)

    # ------------------------------------------------------------------
    # Step 3: Install Playwright browsers
    # ------------------------------------------------------------------
    print(f"\n{color('[3/4]', '96')} Installing Playwright browsers...")
    print("  (Needed for scripts/update_data.py and data scraping)")
    playwright_ok = run(
        ["uv", "run", "playwright", "install", "--with-deps", "chromium"],
        "Installing Chromium + system dependencies",
        check=False, capture=True,
    )
    if not playwright_ok:
        print(f"\n  {color('WARNING:', '93')} Playwright install failed.")
        print("  On Manjaro you may need system deps first:")
        print("    sudo pacman -S atk at-spi2-atk cups libdrm")
        print("    sudo pacman -S libxkbcommon libxcomposite libxdamage libxrandr")
        print("    sudo pacman -S mesa nss pango cairo gtk3")
        print("  Then re-run: uv run playwright install chromium")
        print("  (Not needed for the stacking model, only for data scraping)")

    # ------------------------------------------------------------------
    # Step 4: Check data files
    # ------------------------------------------------------------------
    print(f"\n{color('[4/4]', '96')} Checking data files...")
    required = [
        "data/elo_rankings.json",
        "data/elo_history.parquet",
        "data/knockout_matches.json",
    ]
    missing = [f for f in required if not (ROOT / f).exists()]
    if missing:
        print(f"  {color('Missing files:', '93')}")
        for f in missing:
            print(f"    {f}")
        print("  Run: uv run python scripts/update_data.py")
    else:
        print("  All required data files present")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print(color("=" * 60, "92"))
    print(color("  Environment ready!", "92"))
    print(color("=" * 60, "92"))
    print()
    print("  Commands:")
    print(f"    {color('uv run python stacking_model.py', '96')}        # Base model")
    print(f"    {color('uv run python stacking_model.py --nlp', '96')}             # NLP-enhanced model")
    print(f"    {color('uv run python show_results.py', '96')}           # View results")
    print(f"    {color('uv run python show_results.py --nlp', '96')}     # View NLP results")
    print(f"    {color('uv run python scripts/update_data.py', '96')}    # Update all data")
    print(f"    {color('uv run python scripts/fetch_team_news.py', '96')}    # Update news")
    print(f"    {color('uv run python scripts/compute_team_embeddings.py', '96')}  # Recompute NLP")
    print()


if __name__ == "__main__":
    main()
