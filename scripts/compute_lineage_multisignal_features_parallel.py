#!/usr/bin/env python3
"""
Legacy wrapper for compute_lineage_multisignal_features.py.

Multiprocessing is now controlled by --n-workers on the primary script.
This wrapper remains for backward compatibility with existing docs/commands.
"""

from __future__ import annotations

from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

from scripts.compute_lineage_multisignal_features import main  # noqa: E402

if __name__ == "__main__":
    print("WARNING: compute_lineage_multisignal_features_parallel.py is deprecated; "
          "use compute_lineage_multisignal_features.py --n-workers N instead.")
    main()
