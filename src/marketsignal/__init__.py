"""MarketSignal Forecast."""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

__version__ = "2.0.0"
