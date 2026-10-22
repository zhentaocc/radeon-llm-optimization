#!/usr/bin/env python3
"""Build a deterministic 8K-16K-token LongBench gov_report custom dataset."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


PROMPT = (
    "You are given a report by a government agency. Write a one-page summary "
    "of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary "
    "of the report.\n\nSummary:"
)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet(args.parquet)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    selected = []
    all_lengths = []

    for source_index, row in frame.iterrows():
        prompt = PROMPT.format(context=row["context"])
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Transformers 5 may return a BatchEncoding here instead of a plain
        # token-id list; count ids rather than mapping keys in that case.
        if hasattr(rendered, "input_ids"):
            rendered = rendered.input_ids
        elif hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
        if hasattr(rendered, "shape"):
            token_count = int(rendered.shape[-1])
        elif rendered and isinstance(rendered[0], (list, tuple)):
            token_count = len(rendered[0])
        else:
            token_count = len(rendered)
        all_lengths.append(token_count)
        if args.min_tokens <= token_count <= args.max_tokens:
            selected.append(
                {
                    "prompt": prompt,
                    "output_tokens": args.output_tokens,
                    "source_index": int(source_index),
                    "input_tokens": token_count,
                }
            )

    random.Random(args.seed).shuffle(selected)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    lengths = [item["input_tokens"] for item in selected]
    manifest = {
        "source": "LongBench gov_report",
        "source_rows": len(frame),
        "official_prompt": PROMPT,
        "selection": {
            "min_input_tokens": args.min_tokens,
            "max_input_tokens": args.max_tokens,
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "matched_rows": len(selected),
        },
        "rendering": {
            "tokenizer": args.tokenizer,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "matched_input_token_distribution": {
            "min": min(lengths) if lengths else None,
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "max": max(lengths) if lengths else None,
        },
        "all_input_token_distribution": {
            "min": min(all_lengths) if all_lengths else None,
            "p50": percentile(all_lengths, 0.50),
            "p90": percentile(all_lengths, 0.90),
            "max": max(all_lengths) if all_lengths else None,
        },
        "output_sha256": digest,
        "source_indices_in_order": [item["source_index"] for item in selected],
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

