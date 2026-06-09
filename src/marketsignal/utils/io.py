from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class JsonEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return super().default(value)


def write_json(values: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(values, indent=2, sort_keys=True, cls=JsonEncoder),
        encoding="utf-8",
    )
    return destination
