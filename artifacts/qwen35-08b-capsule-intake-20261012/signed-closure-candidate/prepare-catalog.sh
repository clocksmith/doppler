#!/bin/sh
set -eu
cd /Users/xyz/deco/reploid
node artifacts/signed-blake3-source-20261012/prepare-catalog.js \
  /var/tmp/qwen35-capsule-closure-20261013 \
  /var/tmp/doppler-resident-preparation-candidate-20261012/consumer/node_modules/doppler-gpu \
  /Users/xyz/deco/doppler/artifacts/paired-intermediates-20261012/signed-closure-correction/fresh-qualification/direction-0/partition-1.json \
  /var/tmp/reploid-closure-qualification-v2-20261013/approved-reference.json \
  http://127.0.0.1:29670/qualification/capsule/ \
  /var/tmp/qwen35-capsule-closure-20261013/chat-models.json
