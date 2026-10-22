#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    rows: list[dict[str, object]] = []
    pat = re.compile(
        r"(?P<workload>[^_]+)_isl(?P<isl>\d+)_osl(?P<osl>\d+)_c(?P<conc>\d+)_r(?P<rep>\d+)\.json$"
    )
    for path in sorted(root.glob("*/matrix/*.json")):
        match = pat.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        row: dict[str, object] = {
            "arm": path.parent.parent.name,
            **{k: int(v) if k != "workload" else v for k, v in match.groupdict().items()},
        }
        for key in (
            "completed",
            "failed",
            "duration",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p99_tpot_ms",
            "mean_e2el_ms",
            "median_e2el_ms",
            "p99_e2el_ms",
            "total_input_tokens",
            "total_output_tokens",
            "spec_decode_acceptance_rate",
            "spec_decode_acceptance_length",
            "spec_decode_num_drafts",
            "spec_decode_draft_tokens",
            "spec_decode_accepted_tokens",
            "spec_decode_per_position_acceptance_rates",
        ):
            row[key] = data.get(key)
        rows.append(row)

    if not rows:
        raise SystemExit(f"no matrix result JSON files below {root}")

    fields = list(rows[0])
    out = root / "matrix_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
