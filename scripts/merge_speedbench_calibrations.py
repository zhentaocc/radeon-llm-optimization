#!/usr/bin/env python3
"""Merge disjoint SPEED-Bench coding calibration counter deltas exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("embedded", type=Path)
parser.add_argument("complement", type=Path)
parser.add_argument("output_dir", type=Path)
args = parser.parse_args()
parts = [json.loads(path.read_text(encoding="utf-8")) for path in (args.embedded, args.complement)]

counts = [part["parameters"]["request_count"] for part in parts]
if counts != [79, 10]:
    raise SystemExit(f"Expected [79, 10] requested HTTP turns, got {counts}")
revisions = {part["dataset"]["dataset_revision"] for part in parts}
if len(revisions) != 1:
    raise SystemExit(f"Dataset revision mismatch: {revisions}")
if any(part["requests"]["failed"] for part in parts):
    raise SystemExit("Cannot form final calibration with failed requests")
successful = sum(part["requests"]["successful"] for part in parts)
if successful != 89:
    raise SystemExit(f"Final calibration requires 89 successful HTTP requests, got {successful}")


def counter(name: str) -> float:
    return sum(float(part["prometheus_delta"][name]) for part in parts)


accepted = counter("vllm:spec_decode_num_accepted_tokens")
drafts = counter("vllm:spec_decode_num_drafts")
draft_tokens = counter("vllm:spec_decode_num_draft_tokens")
local_al = accepted / drafts + 1.0
acceptance_rate = accepted / draft_tokens
if not 1.0 <= local_al <= 3.0:
    raise SystemExit(f"Merged N=2 AL out of range: {local_al}")

output = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "calibration_kind": "local_calibration",
    "official": False,
    "submission_valid": False,
    "warning": "Local W7900 calibration only; not an official InferenceX golden curve or submission.",
    "model": parts[0]["model"],
    "workload": "SPEED-Bench coding (80 rows, disjoint 70+10 source split)",
    "dataset_revision": revisions.pop(),
    "parameters": parts[0]["parameters"] | {
        "request_count": 89,
        "dataset_rows": 80,
        "expected_http_requests": 89,
    },
    "acceptance_length": {
        "raw": local_al,
        "rounded_2dp": round(local_al, 2),
        "definition": "sum(accepted_tokens) / sum(num_drafts) + 1",
    },
    "acceptance_rate": acceptance_rate,
    "prometheus_delta": {
        "vllm:spec_decode_num_accepted_tokens": accepted,
        "vllm:spec_decode_num_drafts": drafts,
        "vllm:spec_decode_num_draft_tokens": draft_tokens,
    },
    "requests": {"successful": 89, "failed": 0, "total": 89},
    "part_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.embedded, args.complement)],
}
args.output_dir.mkdir(parents=True, exist_ok=True)
manifest = args.output_dir / "local_calibration_manifest.json"
manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(args.output_dir / "local_al.txt").write_text(f"{local_al:.12g}\n", encoding="utf-8")
(args.output_dir / "calibration_sha256.txt").write_text(
    f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  {manifest.name}\n",
    encoding="utf-8",
)
print(json.dumps(output, indent=2, sort_keys=True))
