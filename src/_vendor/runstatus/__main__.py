"""``python -m runstatus show [FILE] [--watch]``: terminal view of a status file."""
from __future__ import annotations

import json
import sys
import time

_BARS = "▁▂▃▄▅▆▇█"


def spark(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    step = max(1, len(values) // width)
    v = values[::step][-width:]
    lo, hi = min(v), max(v)
    return "".join(_BARS[0 if hi == lo else int((x - lo) / (hi - lo) * 7)] for x in v)


def render(d: dict) -> str:
    r, p, m = d["run"], d["progress"], d["metrics"]
    pct = p.get("pct") or 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    eta = p.get("eta_s")
    eta_s = f"  ETA {eta // 3600}h{eta % 3600 // 60:02d}m" if eta else ""
    lines = [f"{r.get('name') or r['id']}  [{r['kind']}]  {r['state'].upper()}  phase={r.get('phase')}",
             f"{bar} {pct:5.1f}%  step {p.get('step')}/{p.get('total_steps')}{eta_s}"]
    if d.get("eval", {}).get("active"):
        lines.append("eval in progress")
    for k, v in m["latest"].items():
        extra = []
        if k in m["best"]:
            extra.append(f"best {m['best'][k]:.4g}")
        if k in m["moving"]:
            extra.append(f"avg {m['moving'][k]:.4g}")
        s = spark([x[1] for x in d["series"].get(k, [])])
        lines.append(f"  {k:<14}{v:>10.4g}  {s}  {' '.join(extra)}")
    for e in d.get("events", [])[-3:]:
        lines.append(f"  · {e['t'][11:]} {e['msg']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    path = args[1] if len(args) > 1 else "run_status/run_status.json"
    while True:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        try:  # series live in the sidecar file
            with open(path.removesuffix(".json") + ".series.json", encoding="utf-8") as f:
                d["series"] = json.load(f)["series"]
        except (OSError, ValueError):
            d["series"] = {}
        print(render(d))
        if "--watch" not in argv:
            return 0
        time.sleep(2)
        print()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
