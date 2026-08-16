#!/usr/bin/env bash
# Fetches the PaySim mobile-money dataset for use as a legitimate-transaction
# base distribution (see README.md data provenance section, and the team
# brief's data plan). Not run automatically by any Makefile target.
#
# License: CC BY-SA 4.0 as listed on Kaggle at the time of writing —
# RE-VERIFY the license on the source page before use; the research report
# flags this as worth double-checking.
#
# PaySim is not committed to this repository. Running this script is a
# manual, one-time step you take under your own Kaggle account and license
# acceptance.
#
# Usage:
#   1. Install the Kaggle CLI and configure your API credentials:
#        pip install kaggle
#        # place kaggle.json (from https://www.kaggle.com/settings) at ~/.kaggle/kaggle.json
#   2. Run this script from the repo root:
#        bash scripts/download_paysim.sh

set -euo pipefail

OUT_DIR="data/raw/paysim"
mkdir -p "$OUT_DIR"

echo "Downloading PaySim into $OUT_DIR ..."
kaggle datasets download -d ealaxi/paysim1 -p "$OUT_DIR" --unzip

echo "Done. Remember: PaySim's 'oldbalance'/'newbalance' columns leak the"
echo "label (per the team brief) — drop them before using this as a"
echo "training feature source. This dataset is gitignored (data/); it is"
echo "never committed to the repository."
