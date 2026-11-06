"""Download the public corpus and prepare all datasets described in configs/data.yaml.

Steps: download -> normalize text -> build the CTC vocabulary -> noise-augmented copy
of the training split (augment.py) -> our recordings (if present).

Usage:
    uv run python scripts/prepare_data.py --config configs/data.yaml

Our own recordings are private and are read from recordings/ (git-ignored)
if present; without them, the public-corpus experiments still run.
"""

# TODO: implement
