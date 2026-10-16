#!/usr/bin/env bash
set -euo pipefail
cd /home/x/deco/doppler
systemctl --user is-active --quiet reploid-qwen-08b.service
systemctl --user is-active --quiet reploid-qwen-2b.service
systemctl --user is-active --quiet reploid-qwen-27b.service
restore() {
  test_result=$?
  trap - EXIT
  systemctl --user start reploid-qwen-27b.service || test_result=1
  systemctl --user is-active reploid-qwen-08b.service reploid-qwen-2b.service reploid-qwen-27b.service
  exit "$test_result"
}
trap restore EXIT
systemctl --user stop reploid-qwen-27b.service
python3 - <<'PY'
from pathlib import Path
m={line.split(":")[0]:int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.split()[1].isdigit()}
d=Path("/sys/class/drm/card1/device")
a=int((d/"mem_info_gtt_total").read_text())-int((d/"mem_info_gtt_used").read_text())
assert m["MemAvailable"]>48*1024**3 and a>26*1024**3,(m["MemAvailable"],a)
print({"availableBytes":m["MemAvailable"],"gttHeadroomBytes":a},flush=True)
PY
flock --nonblock /tmp/reploid-qwen27b-gpu.lock timeout --signal=TERM --kill-after=20s 600s node .tmp/qwen38-q4-qualification-20261016/qualify-quantized-chat.js .tmp/qwen38-q4-qualification-20261016/q4-quality-config.json .tmp/qwen38-q4-qualification-20261016/quality-suite.json .tmp/qwen38-q4-qualification-20261016/q4-quality-v2.json > .tmp/qwen38-q4-qualification-20261016/q4-quality-v2.log 2>&1
flock --nonblock /tmp/reploid-qwen27b-gpu.lock timeout --signal=TERM --kill-after=20s 600s node .tmp/qwen38-q4-qualification-20261016/qualify-quantized-chat.js .tmp/qwen38-q4-qualification-20261016/f16-quality-config.json .tmp/qwen38-q4-qualification-20261016/quality-suite.json .tmp/qwen38-q4-qualification-20261016/f16-quality.json > .tmp/qwen38-q4-qualification-20261016/f16-quality.log 2>&1
