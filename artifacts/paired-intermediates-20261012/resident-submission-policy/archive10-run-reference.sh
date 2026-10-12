#!/bin/sh
set -eu
cd /var/tmp/reploid-paired-submission-fixed-20261012
export DOPPLER_CHAT_MODEL_DIR=/Users/xyz/deco/doppler/reports/local/paired-intermediates-20261012/model-9/model
export REPLOID_PAIRED_EXECUTION_BINDING=$PWD/binding.json
export REPLOID_PAIRED_EXECUTION_GRAPH=$PWD/execution.json
export REPLOID_DIAGNOSTIC_REQUESTS=$PWD/tests/fixtures/distributed-memory-requests.json
export REPLOID_REFERENCE_FILE=$PWD/approved-reference.json
export REPLOID_REFERENCE_SHA256=ad26dfbc38c5b890f48da2f1e4313dac9295e39e12f688411a10bf614d22fdd6
export REPLOID_MEMORY_PHASE=reference
export REPLOID_DIAGNOSTIC_PROMPT=2
export REPLOID_TEST_DELIVERY_BINDING=$PWD/snapshot-binding.json
export REPLOID_TEST_DELIVERY_SHA256=d9f18df505806810c963c926431795347005637485e2a3db2a6470e5a27bc0d9
export REPLOID_EXECUTOR_CDP=1
export REPLOID_EXECUTOR_WS=http://127.0.0.1:28942
export REPLOID_TEST_PEER=x@128.tail995236.ts.net
export REPLOID_E2E_BASE_URL=http://127.0.0.1:28940
export REPLOID_MODEL_BASE_URL=http://127.0.0.1:28941/
export REPLOID_PEER_MODEL_BASE_URL=http://127.0.0.1:28941/
export REPLOID_REFERENCE_REVERSE=${1:-0}
export REPLOID_CAPTURE_OUT=$PWD/split-direction-$REPLOID_REFERENCE_REVERSE.json
node tests/fixtures/browser-partition-memory-completion.js > split-direction-$REPLOID_REFERENCE_REVERSE.log 2>&1
