#!/bin/bash
# Committed copy: the host-specific GPU lock path and gate directory are written as $GPU_LOCK and $GATE.
# DopplerX AMD gate: verify both tarballs and install each by path into its own fresh directory.
# No GPU work. usage: install.sh
set -eu
B=$HOME/dopplerx-amd-gate-20261126
cd "$B/tarballs"
echo "== tarball sha256 on the AMD host"
sha256sum dopplerx-0.1.7.tgz dopplerx-0.2.0-sha256-a97cb41f02e8957c9cc646ad6ff333972987c5062dbbf3e7301e947190dc97a2.tgz
echo "09e9ce2cc27eeacaf3702efc6320882e849d3e0ce027687ebff7f7968e30a0b0  dopplerx-0.1.7.tgz
a97cb41f02e8957c9cc646ad6ff333972987c5062dbbf3e7301e947190dc97a2  dopplerx-0.2.0-sha256-a97cb41f02e8957c9cc646ad6ff333972987c5062dbbf3e7301e947190dc97a2.tgz" | sha256sum -c -
for v in 017 020; do
  d=$B/install-$v
  [ -z "$(ls -A "$d")" ] || { echo "install dir $d not empty"; exit 1; }
  cd "$d"
  git rev-parse --show-toplevel >/dev/null 2>&1 && { echo "$d is inside a git checkout"; exit 1; }
  npm init -y >/dev/null
  if [ $v = 017 ]; then t=$B/tarballs/dopplerx-0.1.7.tgz; else t=$B/tarballs/dopplerx-0.2.0-sha256-a97cb41f02e8957c9cc646ad6ff333972987c5062dbbf3e7301e947190dc97a2.tgz; fi
  nice npm install --no-audit --no-fund "$t" 2>&1 | tail -2
  PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 nice npm install --no-audit --no-fund playwright@1.58.2 2>&1 | tail -1
  node -e "const l=require('./package-lock.json').packages; console.log('$v', l['node_modules/dopplerx'].version, l['node_modules/dopplerx'].integrity, 'webgpu', l['node_modules/webgpu']?.version, 'doe-gpu', l['node_modules/doe-gpu']?.version, 'playwright', l['node_modules/playwright']?.version)"
done
