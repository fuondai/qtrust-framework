from __future__ import annotations

import io
from pathlib import Path


class JsonLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, obj):
        # Only JSON serialization; no pickle or eval
        try:
            import orjson  # type: ignore
            data = orjson.dumps(obj).decode("utf-8")
        except Exception:
            import json
            data = json.dumps(obj)
        self._fh.write(data)
        self._fh.write("\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


