#!/usr/bin/env python3
"""Validate matched samples and baseline/EAGLE result comparability."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


COUNTS = {
    "sharegpt_short": {1: 16, 4: 32, 16: 64},
    "specbench": {1: 16, 4: 32, 16: 64},
    "specbench_fixed256": {1: 16, 4: 32, 16: 64},
    "longbench_gov_report_8k16k": {1: 4, 4: 8, 16: 16},
    "longbench_gov_report_8k16k_fixed512": {1: 4, 16: 16},
}

SOURCE_DATASET = {
    "specbench_fixed256": "specbench",
    "longbench_gov_report_8k16k_fixed512": "longbench_gov_report_8k16k",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_samples(root, sharegpt):
    short = read_jsonl(sharegpt)
    random.Random(42).shuffle(short)
    spec = [
        {"prompt": item["turns"][0], "output_tokens": 256}
        for item in read_jsonl(root / "datasets/specbench-question.jsonl")
    ]
    random.Random(42).shuffle(spec)
    long = read_jsonl(root / "datasets/longbench-gov_report-8k16k-custom.jsonl")
    return {
        "sharegpt_short": short,
        "specbench": spec,
        "longbench_gov_report_8k16k": long,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sharegpt")
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="rebuild sample_manifest.json from source datasets",
    )
    args = parser.parse_args()
    root = Path(args.root)
    manifest_path = root / "sample_manifest.json"
    if args.rebuild_manifest or not manifest_path.exists():
        if not args.sharegpt:
            parser.error("--sharegpt is required when rebuilding the manifest")
        datasets = load_samples(root, args.sharegpt)
        selection_manifest = {
            "seed": 42,
            "datasets": {},
        }
        for name, by_conc in COUNTS.items():
            selection_manifest["datasets"][name] = {}
            for conc, count in by_conc.items():
                selected = datasets[SOURCE_DATASET.get(name, name)][:count]
                selection_manifest["datasets"][name][str(conc)] = {
                    "num_prompts": count,
                    "prompt_sha256": [digest(item["prompt"]) for item in selected],
                    "requested_output_tokens": [
                        int(item.get("output_tokens", 256)) for item in selected
                    ],
                    "requested_output_tokens_total": sum(
                        int(item.get("output_tokens", 256)) for item in selected
                    ),
                }
        manifest_path.write_text(
            json.dumps(selection_manifest, indent=2) + "\n", encoding="utf-8"
        )
    else:
        selection_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = []
    for dataset, by_conc in COUNTS.items():
        for conc, expected_count in by_conc.items():
            stem = f"{dataset}_c{conc}_n{expected_count}.json"
            base_path = root / "results/baseline" / dataset / stem
            eagle_path = root / "results/eagle_n2" / dataset / stem
            if not base_path.exists() or not eagle_path.exists():
                continue
            base = json.loads(base_path.read_text(encoding="utf-8"))
            eagle = json.loads(eagle_path.read_text(encoding="utf-8"))
            base_text = base.get("generated_texts")
            eagle_text = eagle.get("generated_texts")
            if isinstance(base_text, list) and isinstance(eagle_text, list):
                exact_total = min(len(base_text), len(eagle_text))
                exact = sum(
                    a == b
                    for a, b in zip(
                        base_text[:exact_total], eagle_text[:exact_total]
                    )
                )
            else:
                exact = eagle.get("generated_text_pair_exact")
                exact_total = eagle.get("generated_text_pair_total")
            rows.append(
                {
                    "dataset": dataset,
                    "conc": conc,
                    "expected_prompts": expected_count,
                    "baseline_completed": base.get("completed"),
                    "eagle_completed": eagle.get("completed"),
                    "baseline_failed": base.get("failed"),
                    "eagle_failed": eagle.get("failed"),
                    "input_lens_identical": base.get("input_lens")
                    == eagle.get("input_lens"),
                    "baseline_total_input_tokens": base.get("total_input_tokens"),
                    "eagle_total_input_tokens": eagle.get("total_input_tokens"),
                    "baseline_total_output_tokens": base.get("total_output_tokens"),
                    "eagle_total_output_tokens": eagle.get("total_output_tokens"),
                    "requested_output_cap_total": selection_manifest["datasets"][
                        dataset
                    ][str(conc)]["requested_output_tokens_total"],
                    "generated_text_exact": exact,
                    "generated_text_total": exact_total,
                    "spec_acceptance_rate": eagle.get(
                        "spec_decode_acceptance_rate"
                    ),
                    "spec_acceptance_length": eagle.get(
                        "spec_decode_acceptance_length"
                    ),
                    "spec_position_rates": json.dumps(
                        eagle.get("spec_decode_per_position_acceptance_rates")
                    ),
                }
            )

    with (root / "validation.csv").open("w", newline="", encoding="utf-8") as h:
        if rows:
            writer = csv.DictWriter(h, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

