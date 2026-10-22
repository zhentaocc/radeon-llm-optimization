#!/usr/bin/env python3
"""Build public result JSONs without prompts, generated text, or error payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_FIELDS = {
    "date",
    "endpoint_type",
    "backend",
    "label",
    "model_id",
    "tokenizer_id",
    "num_prompts",
    "model",
    "platform",
    "tp",
    "arm",
    "dataset",
    "conc",
    "temperature",
    "thinking",
    "ignore_eos",
    "seed",
    "request_rate",
    "burstiness",
    "max_concurrency",
    "duration",
    "completed",
    "failed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "input_lens",
    "output_lens",
    "ttfts",
    "itls",
    "start_times",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    "rtfx",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p99_e2el_ms",
    "spec_decode_acceptance_length",
    "spec_decode_acceptance_rate",
    "spec_decode_accepted_tokens",
    "spec_decode_draft_tokens",
    "spec_decode_num_drafts",
    "spec_decode_per_position_acceptance_rates",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    script_sha = sha256(Path(__file__).read_bytes())
    sources = {}
    for source in sorted(args.source_root.rglob("*.json")):
        relative = source.relative_to(args.source_root)
        raw = source.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TypeError(f"expected object in {source}")
        sources[relative] = (raw, data)

    paired_text_counts = {}
    for relative, (_, data) in sources.items():
        if relative.parts[0] != "baseline":
            continue
        counterpart = Path("eagle_n2", *relative.parts[1:])
        if counterpart not in sources:
            continue
        eagle = sources[counterpart][1]
        baseline_text = data.get("generated_texts")
        eagle_text = eagle.get("generated_texts")
        if not isinstance(baseline_text, list) or not isinstance(eagle_text, list):
            continue
        total = min(len(baseline_text), len(eagle_text))
        exact = sum(
            left == right
            for left, right in zip(baseline_text[:total], eagle_text[:total])
        )
        paired_text_counts[relative] = (exact, total)
        paired_text_counts[counterpart] = (exact, total)

    entries = []
    removed_union: set[str] = set()
    for relative, (raw, data) in sources.items():
        removed = sorted(set(data) - ALLOWED_FIELDS)
        removed_union.update(removed)
        public = {key: data[key] for key in data if key in ALLOWED_FIELDS}
        if relative in paired_text_counts:
            exact, total = paired_text_counts[relative]
            public["generated_text_pair_exact"] = exact
            public["generated_text_pair_total"] = total

        destination = args.output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(public, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        destination.write_bytes(encoded)
        entries.append(
            {
                "path": relative.as_posix(),
                "source_sha256": sha256(raw),
                "sanitized_sha256": sha256(encoded),
                "source_bytes": len(raw),
                "sanitized_bytes": len(encoded),
                "removed_fields": removed,
            }
        )

    manifest = {
        "policy": "whitelist-only; values from non-whitelisted fields are never copied",
        "sanitizer": Path(__file__).name,
        "sanitizer_sha256": script_sha,
        "whitelisted_fields": sorted(ALLOWED_FIELDS),
        "derived_fields": [
            "generated_text_pair_exact",
            "generated_text_pair_total",
        ],
        "removed_fields_union": sorted(removed_union),
        "file_count": len(entries),
        "files": entries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "sanitization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
