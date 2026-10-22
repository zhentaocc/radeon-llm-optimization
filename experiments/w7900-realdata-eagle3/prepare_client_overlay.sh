#!/usr/bin/env bash
set -euo pipefail

root=${1:-/dev/shm/zhenchen-realdata-20261022}
image=${IMAGE:-specbench:vllm-0.29.0-rocm723-gfx1100}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
loader=/opt/venv/lib/python3.12/site-packages/vllm/benchmarks/datasets/datasets.py

mkdir -p "$root"
container=$(docker create "$image")
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker cp "$container:$loader" "$root/datasets.py"
patch "$root/datasets.py" <"$script_dir/vllm-specbench-loader.patch"
python3 -m py_compile "$root/datasets.py"
sha256sum "$root/datasets.py"
