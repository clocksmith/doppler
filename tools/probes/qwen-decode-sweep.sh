#!/usr/bin/env bash
# Quick decode-perf sweep for Qwen 3.5 0.8B on the local Strix Halo box.
# Not shipped in the npm package — internal probe only. See AGENTS.md.

set -u

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
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
  node -e '
const fs = require("node:fs");
const path = require("node:path");
const [tmpf, repo, label] = process.argv.slice(1);
const pad = (text, width) => String(text).padEnd(width);
const fixed = (value, digits, width) => Number(value || 0).toFixed(digits).padStart(width);
const match = fs.readFileSync(tmpf, "utf8").match(/"path":\s*"(reports\/[^"]+)"/);
if (!match) { console.log(pad(label, 40) + " NO_REPORT_LINE"); process.exit(0); }
const reportPath = path.join(repo, match[1]);
if (!fs.existsSync(reportPath)) { console.log(pad(label, 40) + " NO_REPORT_FILE"); process.exit(0); }
const m = JSON.parse(fs.readFileSync(reportPath, "utf8")).metrics || {};
const g = m.gpu || {};
console.log(pad(label, 40) + " tok/s=" + fixed(m.decodeTokensPerSec, 2, 6) + "  p50=" + fixed(m.decodeMsPerTokenP50, 2, 6)
  + "ms  TTFT=" + fixed(m.firstTokenMs, 1, 7) + "ms  prefill=" + fixed(m.medianPrefillTokensPerSec, 1, 6)
  + "t/s  sw=" + fixed((g.decodeSubmitWaitMs || {}).mean, 0, 6) + "ms");
' "$tmpf" "$REPO" "$label"
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
