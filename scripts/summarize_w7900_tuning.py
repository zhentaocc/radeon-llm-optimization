#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    pattern = re.compile(r"(?P<workload>[^_]+)_isl(?P<isl>\d+)_osl(?P<osl>\d+)_c(?P<conc>\d+)\.json$")
    rows = []
    for path in sorted((root / "tuning").glob("*/*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rows.append({
            "arm": path.parent.name,
            **{k: int(v) if k != "workload" else v for k, v in match.groupdict().items()},
            "output_throughput": data.get("output_throughput"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "mean_tpot_ms": data.get("mean_tpot_ms"),
            "mean_e2el_ms": data.get("mean_e2el_ms"),
            "p99_e2el_ms": data.get("p99_e2el_ms"),
            "acceptance_rate": data.get("spec_decode_acceptance_rate"),
            "acceptance_length": data.get("spec_decode_acceptance_length"),
            "draft_tokens": data.get("spec_decode_draft_tokens"),
            "accepted_tokens": data.get("spec_decode_accepted_tokens"),
        })
    if not rows:
        raise SystemExit("no tuning JSON files found")
    out = root / "tuning_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
