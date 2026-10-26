"""Logging: INFO to console, full DEBUG trail to a timestamped file."""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir, game_name: str) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{game_name.lower()}_{stamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    logging.getLogger("nla").info("Logging to %s", path)
    return path
