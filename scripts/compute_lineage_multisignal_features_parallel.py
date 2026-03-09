#!/usr/bin/env python3
"""
Legacy wrapper for compute_lineage_multisignal_features.py.

Multiprocessing is now controlled by --n-workers on the primary script.
This wrapper remains for backward compatibility with existing docs/commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_lineage_multisignal_features import main  # noqa: E402


if __name__ == "__main__":
    print("WARNING: compute_lineage_multisignal_features_parallel.py is deprecated; "
          "use compute_lineage_multisignal_features.py --n-workers N instead.")
    main()
