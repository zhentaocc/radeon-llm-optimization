#!/usr/bin/env bash
set -euo pipefail

artifact=/dev/shm/zhenchen-wvsplitk-mlp-tune-vllm
monitor="$artifact/stage_a_gpu0_monitor.log"
run_log="$artifact/stage_a_run.log"
: >"$monitor"

record_gpu0() {
  date -Ins
  rocm-smi --showuse --showmemuse --showpids 2>/dev/null |
    awk '/GPU\[0\]/ || /KFD process information:/ || /^[0-9]+[[:space:]]/'
}

record_gpu0 >>"$monitor"
docker run --rm --name wvsplitk-mlp-stagea-20261022 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -e HIP_VISIBLE_DEVICES=0 \
  --entrypoint bash \
  -v "$artifact:/workspace" -w /workspace \
  specbench:vllm-0.29.0-rocm723-gfx1100 \
  /workspace/run_stage_a.sh >"$run_log" 2>&1 &
child=$!

set +e
while kill -0 "$child" 2>/dev/null; do
  record_gpu0 >>"$monitor"
  sleep 1
done
wait "$child"
rc=$?
set -e
record_gpu0 >>"$monitor"
cat "$run_log"
exit "$rc"
