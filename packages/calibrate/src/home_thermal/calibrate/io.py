"""Saving and loading fitted parameters (JSON: {"params": {...}, "meta": {...}, ...})."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from home_thermal.core import complete


def save_params(path: str | Path, params: dict, **extra) -> None:
    """Write parameters with a timestamp and any extra JSON-serialisable sections."""
    meta = {"saved_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            **extra.pop("meta", {})}
    Path(path).write_text(json.dumps({"params": params, "meta": meta, **extra}, indent=2, default=float),
                          encoding="utf-8")


def load_params(path: str | Path, cfg: dict, strict: bool = False) -> dict:
    """Parameters from a saved file for house `cfg`, completed with priors for anything missing.
    Names that are not parameters of this house are dropped (strict=False) or raise (strict=True)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return complete(cfg, d.get("params", d), strict=strict)
