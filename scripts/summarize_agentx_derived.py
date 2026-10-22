#!/usr/bin/env python3
"""Create a compact, auditable manifest for a derived AgentX smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(report: dict, name: str) -> float | None:
    value = report.get(name)
    if isinstance(value, dict) and isinstance(value.get("avg"), (int, float)):
        return float(value["avg"])
    return None


def counter_series(server_report: dict, name: str) -> list[dict]:
    metric = server_report.get("metrics", {}).get(name, {})
    return [
        {"labels": series.get("labels", {}), "total": series.get("stats", {}).get("total"), "rate": series.get("stats", {}).get("rate")}
        for series in metric.get("series", [])
    ]


parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--concurrency", type=int, required=True)
parser.add_argument("--calibration-manifest", type=Path, required=True)
parser.add_argument("--dataset-jsonl", type=Path, required=True)
parser.add_argument("--server-config", type=Path, required=True)
args = parser.parse_args()

profile_path = args.run_dir / "profile_export_aiperf.json"
server_path = args.run_dir / "server_metrics_export.json"
records_path = args.run_dir / "profile_export.jsonl"
required = (profile_path, server_path, records_path, args.calibration_manifest, args.dataset_jsonl, args.server_config)
for path in required:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")

profile = json.loads(profile_path.read_text(encoding="utf-8"))
server = json.loads(server_path.read_text(encoding="utf-8"))
calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
profiling_records = [record for record in records if record.get("metadata", {}).get("benchmark_phase") == "profiling"]
cancelled = sum(bool(record.get("metadata", {}).get("was_cancelled")) for record in profiling_records)
errors = profile.get("error_summary") or {}
spec_metrics = {
    name: counter_series(server, name)
    for name in sorted(server.get("metrics", {}))
    if "spec_decode" in name
}

manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "workload": "agentx-derived-32k",
    "calibration_kind": "local_calibration",
    "official": False,
    "submission_valid": False,
    "warning": "Derived 32K optimization smoke only; not an official InferenceX AgentX result.",
    "model": "Qwen3.8-27B",
    "draft": "amd/Qwen3.8-27B-EAGLE3-Full-Vocab",
    "synthetic_acceptance": True,
    "synthetic_acceptance_length": float(calibration["acceptance_length"]["raw"]),
    "parameters": {"concurrency": args.concurrency, "duration_seconds": 60, "grace_period_seconds": 600, "max_context_length": 32768, "seed": 20261022, "ignore_eos": True},
    "integrity": {
        "profile_report_sha256": sha256(profile_path),
        "server_metrics_sha256": sha256(server_path),
        "records_sha256": sha256(records_path),
        "calibration_manifest_sha256": sha256(args.calibration_manifest),
        "derived_dataset_sha256": sha256(args.dataset_jsonl),
        "server_config_sha256": sha256(args.server_config),
    },
    "requests": {
        "successful": len(profiling_records) - cancelled,
        "cancelled": cancelled,
        "failed": int(errors.get("total_errors", 0)) if isinstance(errors, dict) else 0,
        "total_records": len(profiling_records),
    },
    "metrics": {
        "benchmark_duration_seconds": average(profile, "benchmark_duration"),
        "request_count": average(profile, "request_count"),
        "request_latency_ms": average(profile, "request_latency"),
        "ttft_ms": average(profile, "time_to_first_token"),
        "itl_ms": average(profile, "inter_token_latency"),
        "input_sequence_length_tokens": average(profile, "input_sequence_length"),
        "output_sequence_length_tokens": average(profile, "output_sequence_length"),
        "output_token_throughput_tokens_per_second": average(profile, "output_token_throughput"),
        "request_throughput_requests_per_second": average(profile, "request_throughput"),
    },
    "branch_stats": profile.get("branch_stats"),
    "speculative_server_metrics": spec_metrics,
}

output = args.run_dir / "derived_result_manifest.json"
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(args.run_dir / "derived_result_manifest.sha256").write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
