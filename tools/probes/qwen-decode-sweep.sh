#!/usr/bin/env bash
# Quick decode-perf sweep for Qwen 3.5 0.8B on the local Strix Halo box.
# Not shipped in the npm package — internal probe only. See AGENTS.md.

set -u

REPO=/home/x/deco/doppler
MODEL_URL="file://$REPO/models/local/qwen-3-5-0-8b-q4k-ehaf16"
MODEL_ID="qwen-3-5-0-8b-q4k-ehaf16"
PROMPT="Write a short paragraph about the ocean."
MAX_TOKENS=64

cd "$REPO"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

run_bench() {
  local label="$1"
  local cfg="$2"
  local tmpf="$TMP/run.json"
  timeout 240 node src/cli/doppler-cli.js bench --surface node --config "$cfg" --json >"$tmpf" 2>/dev/null
  python3 -c "
import json,sys,os,re
text=open('$tmpf').read()
m=re.search(r'\"path\":\s*\"(reports/[^\"]+)\"', text)
if not m:
    print(f'{\"$label\":40s} NO_REPORT_LINE'); sys.exit()
p=m.group(1)
if not os.path.isfile(os.path.join('$REPO',p)):
    print(f'{\"$label\":40s} NO_REPORT_FILE')
    sys.exit()
r=json.load(open(os.path.join('$REPO',p)))
m=r.get('metrics',{})
g=m.get('gpu',{})
tps=m.get('decodeTokensPerSec') or 0
p50=m.get('decodeMsPerTokenP50') or 0
ttft=m.get('firstTokenMs') or 0
prefill=m.get('medianPrefillTokensPerSec') or 0
sw=(g.get('decodeSubmitWaitMs') or {}).get('mean',0) or 0
print(f'{\"$label\":40s} tok/s={tps:6.2f}  p50={p50:6.2f}ms  TTFT={ttft:7.1f}ms  prefill={prefill:6.1f}t/s  sw={sw:6.0f}ms')
"
}

echo "=== Qwen 3.5 0.8B decode-perf sweep on Strix Halo ($(date -u +%FT%TZ)) ==="

# 1. Baseline (no profile)
run_bench "baseline" \
  "{\"request\":{\"workload\":\"inference\",\"modelId\":\"$MODEL_ID\",\"modelUrl\":\"$MODEL_URL\",\"inferenceInput\":{\"prompt\":\"$PROMPT\",\"maxTokens\":$MAX_TOKENS}}}"

# 2. profiles/throughput
run_bench "profiles/throughput" \
  "{\"request\":{\"workload\":\"inference\",\"modelId\":\"$MODEL_ID\",\"modelUrl\":\"$MODEL_URL\",\"runtimeProfile\":\"profiles/throughput\",\"inferenceInput\":{\"prompt\":\"$PROMPT\",\"maxTokens\":$MAX_TOKENS}}}"

# 3. Custom: ringTokens=4, stopCheckMode=per-token, bs=1
run_bench "ring4 bs1 per-token overlap" \
  "{\"request\":{\"workload\":\"inference\",\"modelId\":\"$MODEL_ID\",\"modelUrl\":\"$MODEL_URL\",\"runtimeConfig\":{\"inference\":{\"batching\":{\"batchSize\":1,\"readbackInterval\":1,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"per-token\",\"ringTokens\":4,\"ringStop\":2,\"ringStaging\":2},\"session\":{\"decodeLoop\":{\"batchSize\":1,\"readbackInterval\":1,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"per-token\",\"ringTokens\":4,\"ringStop\":2,\"ringStaging\":2}}}},\"inferenceInput\":{\"prompt\":\"$PROMPT\",\"maxTokens\":$MAX_TOKENS}}}"

# 4. Custom: bs=1 ri=2 overlapped (read every other token)
run_bench "bs1 ri2 overlap" \
  "{\"request\":{\"workload\":\"inference\",\"modelId\":\"$MODEL_ID\",\"modelUrl\":\"$MODEL_URL\",\"runtimeConfig\":{\"inference\":{\"batching\":{\"batchSize\":1,\"readbackInterval\":2,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"per-token\",\"ringTokens\":2,\"ringStop\":1,\"ringStaging\":2},\"session\":{\"decodeLoop\":{\"batchSize\":1,\"readbackInterval\":2,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"per-token\",\"ringTokens\":2,\"ringStop\":1,\"ringStaging\":2}}}},\"inferenceInput\":{\"prompt\":\"$PROMPT\",\"maxTokens\":$MAX_TOKENS}}}"

# 5. Custom: bs=2 ri=1 batch stopcheck
run_bench "bs2 ri1 batch-stop overlap" \
  "{\"request\":{\"workload\":\"inference\",\"modelId\":\"$MODEL_ID\",\"modelUrl\":\"$MODEL_URL\",\"runtimeConfig\":{\"inference\":{\"batching\":{\"batchSize\":2,\"readbackInterval\":1,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"batch\",\"ringTokens\":2,\"ringStop\":1,\"ringStaging\":2},\"session\":{\"decodeLoop\":{\"batchSize\":2,\"readbackInterval\":1,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"batch\",\"ringTokens\":2,\"ringStop\":1,\"ringStaging\":2}}}},\"inferenceInput\":{\"prompt\":\"$PROMPT\",\"maxTokens\":$MAX_TOKENS}}}"

# 6. Custom: prefillChunkSubmitMode=async + bs=1 baseline
run_bench "async-prefill bs1" \
  "{\"request\":{\"workload\":\"inference\",\"modelId\":\"$MODEL_ID\",\"modelUrl\":\"$MODEL_URL\",\"runtimeConfig\":{\"inference\":{\"session\":{\"prefillChunkSubmitMode\":\"async\",\"decodeLoop\":{\"batchSize\":1,\"readbackInterval\":1,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"per-token\"}}}},\"inferenceInput\":{\"prompt\":\"$PROMPT\",\"maxTokens\":$MAX_TOKENS}}}"

# 7. Custom: bs=4 ri=2 batch-stop overlap (compromise between baseline and qwen-throughput)
run_bench "bs4 ri2 batch-stop overlap" \
  "{\"request\":{\"workload\":\"inference\",\"modelId\":\"$MODEL_ID\",\"modelUrl\":\"$MODEL_URL\",\"runtimeConfig\":{\"inference\":{\"batching\":{\"batchSize\":4,\"readbackInterval\":2,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"batch\",\"ringTokens\":2,\"ringStop\":1,\"ringStaging\":2},\"session\":{\"decodeLoop\":{\"batchSize\":4,\"readbackInterval\":2,\"readbackMode\":\"overlapped\",\"stopCheckMode\":\"batch\",\"ringTokens\":2,\"ringStop\":1,\"ringStaging\":2}}}},\"inferenceInput\":{\"prompt\":\"$PROMPT\",\"maxTokens\":$MAX_TOKENS}}}"

echo "=== done ==="
