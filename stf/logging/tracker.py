from __future__ import annotations


class Tracker:
    def __init__(self, *keys: str):
        self.keys = list(keys)
        self.reset()

    def reset(self) -> None:
        self.total = {k: 0.0 for k in self.keys}
        self.counts = {k: 0 for k in self.keys}
        self.current = {k: 0.0 for k in self.keys}

    def update(self, key: str, value: float, n: int = 1) -> None:
        if key not in self.total:
            self.total[key] = 0.0
            self.counts[key] = 0
            self.current[key] = 0.0
            self.keys.append(key)
        self.total[key] += value * n
        self.counts[key] += n
        self.current[key] = value

    @property
    def results(self) -> dict[str, float]:
        out = {}
        for k in self.keys:
            count = self.counts[k]
            out[k] = self.total[k] / count if count > 0 else 0.0
        return out

    @property
    def now(self) -> dict[str, float]:
        return dict(self.current)
