# W7900 Qwen3.8-27B EAGLE3 real-data A/B

- Baseline: target TP2, GPU 2/3, no speculative decoding.
- EAGLE3: target TP2 + draft TP1, N=2, GPU 4/5.
- Temperature 0, thinking disabled, natural EOS; prefix caching disabled.
- Same deterministic sample selection for both arms (seed 42).
- Startup capacity: baseline KV cache 366,273 tokens / 2.56 GiB graph memory; EAGLE KV cache 150,241 tokens / 6.63 GiB graph memory.

## Throughput and latency

| Dataset | C | Arm | Done/fail | Input tokens p50/p90 [min,max] | Output tokens p50/p90 [min,max] | Output tok/s | Total tok/s | TTFT p50/p90 ms | TPOT p50/p90 ms |
|---|---:|---|---:|---|---|---:|---:|---:|---:|
| sharegpt_short | 1 | baseline | 16/0 | 364/628 [20,804] | 37/492 [8,804] | 18.34 | 59.01 | 258.36/450.71 | 51.94/55.20 |
| sharegpt_short | 1 | eagle_n2 | 16/0 | 364/628 [20,804] | 37/492 [8,804] | 35.08 | 112.87 | 302.92/478.55 | 28.62/36.57 |
| sharegpt_short | 4 | baseline | 32/0 | 142/573 [16,804] | 145/476 [6,804] | 60.45 | 131.03 | 293.12/672.24 | 61.21/66.96 |
| sharegpt_short | 4 | eagle_n2 | 32/0 | 142/573 [16,804] | 145/476 [6,804] | 77.66 | 168.33 | 411.92/772.63 | 47.46/59.67 |
| sharegpt_short | 16 | baseline | 64/0 | 154/554 [16,1018] | 102/439 [4,804] | 108.51 | 251.93 | 446.45/3106.09 | 102.94/116.33 |
| sharegpt_short | 16 | eagle_n2 | 64/0 | 154/554 [16,1018] | 102/439 [4,804] | 172.15 | 399.69 | 677.71/3083.54 | 66.76/88.46 |
| specbench | 1 | baseline | 16/0 | 397/823 [21,1267] | 114/256 [11,256] | 18.19 | 71.60 | 310.29/597.70 | 50.78/57.43 |
| specbench | 1 | eagle_n2 | 16/0 | 397/823 [21,1267] | 132/256 [11,256] | 33.88 | 128.49 | 359.44/612.63 | 26.29/31.19 |
| specbench | 4 | baseline | 32/0 | 45/803 [21,1450] | 123/256 [9,256] | 53.63 | 167.82 | 277.36/1158.17 | 67.56/72.69 |
| specbench | 4 | eagle_n2 | 32/0 | 45/803 [21,1450] | 125/256 [9,256] | 75.09 | 233.83 | 371.42/1135.59 | 47.73/56.45 |
| specbench | 16 | baseline | 64/0 | 66/731 [21,1450] | 146/256 [9,256] | 117.76 | 315.91 | 517.33/3834.69 | 114.01/119.81 |
| specbench | 16 | eagle_n2 | 64/0 | 66/731 [21,1450] | 142/256 [9,256] | 184.70 | 498.63 | 663.70/3927.41 | 68.59/82.60 |
| specbench_fixed256 | 1 | baseline | 16/0 | 397/823 [21,1267] | 256/256 [256,256] | 18.52 | 49.12 | 284.70/576.37 | 52.79/58.45 |
| specbench_fixed256 | 1 | eagle_n2 | 16/0 | 397/823 [21,1267] | 256/256 [256,256] | 34.44 | 91.35 | 362.80/584.93 | 27.40/31.70 |
| specbench_fixed256 | 4 | baseline | 32/0 | 45/803 [21,1450] | 256/256 [256,256] | 59.65 | 127.87 | 622.14/1459.52 | 64.15/72.01 |
| specbench_fixed256 | 4 | eagle_n2 | 32/0 | 45/803 [21,1450] | 256/256 [256,256] | 78.24 | 167.71 | 372.71/1178.59 | 47.50/53.32 |
| specbench_fixed256 | 16 | baseline | 64/0 | 66/731 [21,1450] | 256/256 [256,256] | 140.84 | 279.53 | 1915.62/4141.67 | 106.67/112.12 |
| specbench_fixed256 | 16 | eagle_n2 | 64/0 | 66/731 [21,1450] | 256/256 [256,256] | 214.16 | 425.05 | 674.14/3915.83 | 63.66/74.05 |
| longbench_gov_report_8k16k | 1 | baseline | 4/0 | 11652/14261 [8838,15070] | 512/512 [512,512] | 4.74 | 114.06 | 7145.17/8986.99 | 193.67/230.63 |
| longbench_gov_report_8k16k | 1 | eagle_n2 | 4/0 | 11652/14261 [8838,15070] | 512/512 [512,512] | 12.83 | 308.60 | 7453.77/9263.70 | 63.50/68.40 |
| longbench_gov_report_8k16k | 4 | baseline | 8/0 | 12426/15222 [8838,15577] | 512/512 [512,512] | 12.95 | 326.58 | 16731.79/28893.47 | 269.66/292.16 |
| longbench_gov_report_8k16k | 4 | eagle_n2 | 8/0 | 12426/15222 [8838,15577] | 512/512 [512,512] | 17.44 | 439.84 | 25731.63/42702.55 | 169.14/242.20 |
| longbench_gov_report_8k16k | 16 | baseline | 16/0 | 11622/14301 [8535,15577] | 512/512 [512,512] | 26.48 | 628.09 | 70824.04/130212.44 | 464.11/555.05 |
| longbench_gov_report_8k16k | 16 | eagle_n2 | 16/0 | 11622/14301 [8535,15577] | 512/512 [512,512] | 14.73 | 349.44 | 150752.31/444398.54 | 408.20/677.58 |

## EAGLE3 delta versus baseline

| Dataset | C | Output tok/s gain | Total tok/s gain | p50 TTFT change | p50 TPOT change | Acceptance rate | Acceptance length | Position acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sharegpt_short | 1 | 91.29% | 91.29% | 17.25% | -44.90% | 70.5853 | 2.4117 | `[0.7916666666666666, 0.6200396825396826]` |
| sharegpt_short | 4 | 28.47% | 28.47% | 40.53% | -22.46% | 63.3102 | 2.2662 | `[0.7445407279029462, 0.5216637781629117]` |
| sharegpt_short | 16 | 58.65% | 58.65% | 51.80% | -35.14% | 64.7693 | 2.2954 | `[0.7583383263431196, 0.5370481326143399]` |
| specbench | 1 | 86.29% | 79.46% | 15.84% | -48.24% | 68.6765 | 2.3735 | `[0.7715686274509804, 0.6019607843137255]` |
| specbench | 4 | 40.03% | 39.34% | 33.91% | -29.36% | 65.7688 | 2.3154 | `[0.7526150627615062, 0.5627615062761506]` |
| specbench | 16 | 56.84% | 57.84% | 28.29% | -39.84% | 68.9962 | 2.3799 | `[0.7844416562107904, 0.5954830614805521]` |
| specbench_fixed256 | 1 | 85.98% | 85.98% | 27.43% | -48.11% | 66.5812 | 2.3316 | `[0.7521367521367521, 0.5794871794871795]` |
| specbench_fixed256 | 4 | 31.16% | 31.16% | -40.09% | -25.96% | 62.6927 | 2.2539 | `[0.7221916299559471, 0.5316629955947136]` |
| specbench_fixed256 | 16 | 52.06% | 52.06% | -64.81% | -40.32% | 66.5362 | 2.3307 | `[0.7586452255585598, 0.5720791233812438]` |
| longbench_gov_report_8k16k | 1 | 170.55% | 170.55% | 4.32% | -67.21% | 65.9683 | 2.3194 | `[0.7814269535673839, 0.5379388448471121]` |
| longbench_gov_report_8k16k | 4 | 34.68% | 34.68% | 53.79% | -37.27% | 67.0767 | 2.3415 | `[0.7923340961098398, 0.5491990846681922]` |
| longbench_gov_report_8k16k | 16 | -44.36% | -44.36% | 112.85% | -12.05% | 66.9189 | 2.3384 | `[0.7884066247858367, 0.549971444888635]` |

## Matched-load validation

| Dataset | C | Input lens identical | Input tokens base/EAGLE | Output tokens base/EAGLE | Requested cap total | Exact generated texts |
|---|---:|---|---:|---:|---:|---:|
| sharegpt_short | 1 | True | 5399/5399 | 2435/2435 | 2435 | 10/16 |
| sharegpt_short | 4 | True | 7639/7639 | 6542/6542 | 6542 | 14/32 |
| sharegpt_short | 16 | True | 15204/15204 | 11503/11503 | 11516 | 33/64 |
| specbench | 1 | True | 6768/6768 | 2305/2424 | 4096 | 7/16 |
| specbench | 4 | True | 9369/9369 | 4400/4432 | 8192 | 16/32 |
| specbench | 16 | True | 16134/16134 | 9588/9492 | 16384 | 39/64 |
| specbench_fixed256 | 1 | True | 6768/6768 | 4096/4096 | 4096 | 6/16 |
| specbench_fixed256 | 4 | True | 9369/9369 | 8192/8192 | 8192 | 11/32 |
| specbench_fixed256 | 16 | True | 16134/16134 | 16384/16384 | 16384 | 26/64 |
| longbench_gov_report_8k16k | 1 | True | 47212/47212 | 2048/2048 | 2048 | 0/4 |
| longbench_gov_report_8k16k | 4 | True | 99194/99194 | 4096/4096 | 4096 | 2/8 |
| longbench_gov_report_8k16k | 16 | True | 186131/186131 | 8192/8192 | 8192 | 0/16 |

## Dataset provenance

- Short: existing `sharegpt_custom_qwen35.jsonl` (82,577 real conversation prompts, SHA256 `5ab08392...24fa`); per-item real response-token caps.
- Long: LongBench `gov_report` parquet pinned at `giulio98/LongBench@2683f3b`, SHA256 `184c389e...df2`; official THUDM prompt pinned at `THUDM/LongBench@2e00731`; target-tokenizer rendered input 8,192–16,384 tokens, output cap 512; 86 eligible rows, pre-shuffled with seed 42.
- Speculative workload: official Spec-Bench `question.jsonl` pinned at `hemingkx/Spec-Bench@fd2c1cd`, SHA256 `4b6d33e7...fafdf04` (480 questions, 13 categories), output cap 256.
- SpecBench client loader is patched only to forward seed and `enable_thinking=false`; server code is unchanged.
