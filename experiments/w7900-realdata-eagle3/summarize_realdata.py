#!/usr/bin/env python3
"""Aggregate W7900 real-data A/B results into CSV and Markdown."""

import argparse
import csv
import json
import math
from pathlib import Path


DATASET_ORDER = {
    "sharegpt_short": 0,
    "specbench": 1,
    "specbench_fixed256": 2,
    "longbench_gov_report_8k16k": 3,
    "longbench_gov_report_8k16k_fixed512": 4,
}
ARM_ORDER = {"baseline": 0, "eagle_n2": 1}


def pct(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def gain(new, old):
    if new is None or old in (None, 0):
        return None
    return 100.0 * (float(new) / float(old) - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []

    for path in root.glob("results/*/*/*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        arm = data.get("arm") or path.parts[-3]
        dataset = data.get("dataset") or path.parts[-2]
        input_lens = data.get("input_lens") or []
        output_lens = data.get("output_lens") or []
        row = {
            "dataset": dataset,
            "conc": int(data.get("conc") or data.get("max_concurrency") or 0),
            "arm": arm,
            "num_prompts": int(data.get("num_prompts") or 0),
            "completed": int(data.get("completed") or 0),
            "failed": int(data.get("failed") or 0),
            "duration_s": data.get("duration"),
            "request_throughput": data.get("request_throughput"),
            "output_throughput": data.get("output_throughput"),
            "total_token_throughput": data.get("total_token_throughput"),
            "input_min": min(input_lens) if input_lens else None,
            "input_p50": pct(input_lens, 0.50),
            "input_p90": pct(input_lens, 0.90),
            "input_max": max(input_lens) if input_lens else None,
            "output_min": min(output_lens) if output_lens else None,
            "output_p50": pct(output_lens, 0.50),
            "output_p90": pct(output_lens, 0.90),
            "output_max": max(output_lens) if output_lens else None,
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "p50_ttft_ms": data.get("p50_ttft_ms"),
            "p90_ttft_ms": data.get("p90_ttft_ms"),
            "p99_ttft_ms": data.get("p99_ttft_ms"),
            "mean_tpot_ms": data.get("mean_tpot_ms"),
            "p50_tpot_ms": data.get("p50_tpot_ms"),
            "p90_tpot_ms": data.get("p90_tpot_ms"),
            "p99_tpot_ms": data.get("p99_tpot_ms"),
            "mean_e2el_ms": data.get("mean_e2el_ms"),
            "p90_e2el_ms": data.get("p90_e2el_ms"),
            "spec_acceptance_rate": data.get("spec_decode_acceptance_rate"),
            "spec_acceptance_length": data.get("spec_decode_acceptance_length"),
            "spec_num_drafts": data.get("spec_decode_num_drafts"),
            "spec_draft_tokens": data.get("spec_decode_draft_tokens"),
            "spec_accepted_tokens": data.get("spec_decode_accepted_tokens"),
            "spec_position_rates": json.dumps(
                data.get("spec_decode_per_position_acceptance_rates"),
                ensure_ascii=False,
            ),
            "file": path.relative_to(root).as_posix(),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            DATASET_ORDER.get(r["dataset"], 99),
            r["conc"],
            ARM_ORDER.get(r["arm"], 99),
        )
    )
    out_csv = root / "summary.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    by_key = {(r["dataset"], r["conc"], r["arm"]): r for r in rows}
    lines = [
        "# W7900 Qwen3.8-27B EAGLE3 real-data A/B",
        "",
        "- Baseline: target TP2, GPU 2/3, no speculative decoding.",
        "- EAGLE3: target TP2 + draft TP1, N=2, GPU 4/5.",
        "- Temperature 0, thinking disabled, natural EOS; prefix caching disabled.",
        "- Same deterministic sample selection for both arms (seed 42).",
        "- Startup capacity: baseline KV cache 366,273 tokens / 2.56 GiB graph memory; EAGLE KV cache 150,241 tokens / 6.63 GiB graph memory.",
        "",
        "## Throughput and latency",
        "",
        "| Dataset | C | Arm | Done/fail | Input tokens p50/p90 [min,max] | Output tokens p50/p90 [min,max] | Output tok/s | Total tok/s | TTFT p50/p90 ms | TPOT p50/p90 ms |",
        "|---|---:|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['conc']} | {row['arm']} | "
            f"{row['completed']}/{row['failed']} | "
            f"{fmt(row['input_p50'], 0)}/{fmt(row['input_p90'], 0)} "
            f"[{fmt(row['input_min'], 0)},{fmt(row['input_max'], 0)}] | "
            f"{fmt(row['output_p50'], 0)}/{fmt(row['output_p90'], 0)} "
            f"[{fmt(row['output_min'], 0)},{fmt(row['output_max'], 0)}] | "
            f"{fmt(row['output_throughput'])} | {fmt(row['total_token_throughput'])} | "
            f"{fmt(row['p50_ttft_ms'])}/{fmt(row['p90_ttft_ms'])} | "
            f"{fmt(row['p50_tpot_ms'])}/{fmt(row['p90_tpot_ms'])} |"
        )

    lines += [
        "",
        "## EAGLE3 delta versus baseline",
        "",
        "| Dataset | C | Output tok/s gain | Total tok/s gain | p50 TTFT change | p50 TPOT change | Acceptance rate | Acceptance length | Position acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in sorted(DATASET_ORDER, key=DATASET_ORDER.get):
        for conc in (1, 4, 16):
            base = by_key.get((dataset, conc, "baseline"))
            eagle = by_key.get((dataset, conc, "eagle_n2"))
            if not base or not eagle:
                continue
            lines.append(
                f"| {dataset} | {conc} | "
                f"{fmt(gain(eagle['output_throughput'], base['output_throughput']))}% | "
                f"{fmt(gain(eagle['total_token_throughput'], base['total_token_throughput']))}% | "
                f"{fmt(gain(eagle['p50_ttft_ms'], base['p50_ttft_ms']))}% | "
                f"{fmt(gain(eagle['p50_tpot_ms'], base['p50_tpot_ms']))}% | "
                f"{fmt(eagle['spec_acceptance_rate'], 4)} | "
                f"{fmt(eagle['spec_acceptance_length'], 4)} | "
                f"`{eagle['spec_position_rates']}` |"
            )

    validation_path = root / "validation.csv"
    if validation_path.exists():
        with validation_path.open(encoding="utf-8") as handle:
            validations = list(csv.DictReader(handle))
        lines += [
            "",
            "## Matched-load validation",
            "",
            "| Dataset | C | Input lens identical | Input tokens base/EAGLE | Output tokens base/EAGLE | Requested cap total | Exact generated texts |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for item in validations:
            lines.append(
                f"| {item['dataset']} | {item['conc']} | "
                f"{item['input_lens_identical']} | "
                f"{item['baseline_total_input_tokens']}/{item['eagle_total_input_tokens']} | "
                f"{item['baseline_total_output_tokens']}/{item['eagle_total_output_tokens']} | "
                f"{item['requested_output_cap_total']} | "
                f"{item['generated_text_exact']}/{item['generated_text_total']} |"
            )

    lines += [
        "",
        "## Dataset provenance",
        "",
        "- Short: existing `sharegpt_custom_qwen35.jsonl` (82,577 real conversation prompts, SHA256 `5ab08392...24fa`); per-item real response-token caps.",
        "- Long: LongBench `gov_report` parquet pinned at `giulio98/LongBench@2683f3b`, SHA256 `184c389e...df2`; official THUDM prompt pinned at `THUDM/LongBench@2e00731`; target-tokenizer rendered input 8,192–16,384 tokens, output cap 512; 86 eligible rows, pre-shuffled with seed 42.",
        "- Speculative workload: official Spec-Bench `question.jsonl` pinned at `hemingkx/Spec-Bench@fd2c1cd`, SHA256 `4b6d33e7...fafdf04` (480 questions, 13 categories), output cap 256.",
        "- SpecBench client loader is patched only to forward seed and `enable_thinking=false`; server code is unchanged.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} rows: {out_csv} and {root / 'report.md'}")


if __name__ == "__main__":
    main()
