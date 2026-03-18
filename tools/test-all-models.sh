#!/usr/bin/env bash
# =============================================================================
# test-all-models.sh — Sequential correctness + speed sweep for all RDRR models
# =============================================================================
#
# Runs each model on the external volume through the CLI debug command with
# deterministic sampling, checks for coherent output, and logs timing.
#
# Usage:
#   bash tools/test-all-models.sh
#   bash tools/test-all-models.sh --json          # machine-readable per-model
#   bash tools/test-all-models.sh --model gemma-3-1b-it-q4k-ehf16-af32  # single model
#
# Requirements:
#   - Node.js 20+
#   - WebGPU runtime (dawn)
#   - External volume mounted at $DOPPLER_EXTERNAL_MODELS_ROOT (default: /media/x/models)
#
# Output:
#   Per-model: PASS/FAIL, output text, prefill ms, decode tok/s, tokens generated
#   Summary: total pass/fail counts
# =============================================================================

set -euo pipefail

VOLUME_ROOT="${DOPPLER_EXTERNAL_MODELS_ROOT:-/media/x/models}"
RDRR_DIR="$VOLUME_ROOT/rdrr"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
JSON_MODE=false
SINGLE_MODEL=""
MAX_TOKENS=16

for arg in "$@"; do
  case "$arg" in
    --json) JSON_MODE=true ;;
    --model=*) SINGLE_MODEL="${arg#--model=}" ;;
    --model) shift; SINGLE_MODEL="${1:-}" ;;
  esac
done

if [ ! -d "$RDRR_DIR" ]; then
  echo "ERROR: RDRR directory not found at $RDRR_DIR"
  exit 1
fi

cd "$PROJECT_DIR"

pass_count=0
fail_count=0
skip_count=0

# ---------------------------------------------------------------------------
# Model configs: modelId → expected output substring (lowercase)
# ---------------------------------------------------------------------------
# Diffusion and embedding models use different commands/checks.
# Text models use debug with deterministic sampling and check for keywords.
# ---------------------------------------------------------------------------

run_text_model() {
  local model_id="$1"
  local expected_pattern="$2"
  local model_dir="$RDRR_DIR/$model_id"
  local manifest="$model_dir/manifest.json"

  if [ ! -f "$manifest" ]; then
    echo "SKIP $model_id (no manifest)"
    skip_count=$((skip_count + 1))
    return
  fi

  local template
  template=$(node -e "const j=JSON.parse(require('fs').readFileSync('$manifest'));console.log(j.inference?.chatTemplate?.type||'standard')" 2>/dev/null)

  # Build prompt based on template type
  local prompt_json
  case "$template" in
    translategemma)
      prompt_json='{"messages":[{"role":"user","content":[{"type":"text","source_lang_code":"en","target_lang_code":"fr","text":"Hello world."}]}]}'
      ;;
    qwen)
      prompt_json='"Answer in one short sentence: What color is the sky on a clear day?"'
      ;;
    *)
      prompt_json='"The color of the sky is"'
      ;;
  esac

  local config_json
  config_json=$(cat <<ENDJSON
{
  "request": {
    "suite": "debug",
    "modelId": "$model_id",
    "modelUrl": "file://$model_dir/",
    "loadMode": "http"
  },
  "runtime": {
    "shared": { "tooling": { "intent": "investigate" } },
    "inference": {
      "prompt": $prompt_json,
      "batching": { "maxTokens": $MAX_TOKENS },
      "sampling": { "temperature": 0, "topK": 1 }
    }
  }
}
ENDJSON
)

  echo "--- $model_id ($template) ---"
  local start_time
  start_time=$(date +%s%N)

  local result
  if ! result=$(node tools/doppler-cli.js debug --config "$config_json" --json 2>&1); then
    echo "FAIL $model_id (command error)"
    if [ "$JSON_MODE" = true ]; then
      echo "$result" | tail -5
    fi
    fail_count=$((fail_count + 1))
    echo ""
    return
  fi

  local end_time
  end_time=$(date +%s%N)
  local wall_ms=$(( (end_time - start_time) / 1000000 ))

  # Extract metrics from JSON result
  local output decode_tps tokens prefill_ms
  output=$(echo "$result" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{try{const r=JSON.parse(d);console.log(r.result?.output||r.result?.metrics?.output||'')}catch{console.log('')}})" 2>/dev/null)
  decode_tps=$(echo "$result" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{try{const r=JSON.parse(d);console.log(r.result?.metrics?.decodeTokensPerSec||0)}catch{console.log(0)}})" 2>/dev/null)
  tokens=$(echo "$result" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{try{const r=JSON.parse(d);console.log(r.result?.metrics?.tokensGenerated||0)}catch{console.log(0)}})" 2>/dev/null)
  prefill_ms=$(echo "$result" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{try{const r=JSON.parse(d);console.log(Math.round(r.result?.metrics?.prefillTimeMs||0))}catch{console.log(0)}})" 2>/dev/null)

  # Check coherence
  local output_lower
  output_lower=$(echo "$output" | tr '[:upper:]' '[:lower:]')
  local status="FAIL"
  if echo "$output_lower" | grep -qiE "$expected_pattern"; then
    status="PASS"
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi

  echo "$status  output: $output"
  echo "       prefill: ${prefill_ms}ms | decode: ${decode_tps} tok/s | tokens: $tokens | wall: ${wall_ms}ms"
  echo ""
}

run_embedding_model() {
  local model_id="$1"
  local model_dir="$RDRR_DIR/$model_id"
  local manifest="$model_dir/manifest.json"

  if [ ! -f "$manifest" ]; then
    echo "SKIP $model_id (no manifest)"
    skip_count=$((skip_count + 1))
    return
  fi

  local config_json
  config_json=$(cat <<ENDJSON
{
  "request": {
    "suite": "debug",
    "modelId": "$model_id",
    "modelUrl": "file://$model_dir/",
    "loadMode": "http"
  },
  "runtime": {
    "shared": { "tooling": { "intent": "investigate" } },
    "inference": {
      "prompt": "The quick brown fox jumps over the lazy dog."
    }
  }
}
ENDJSON
)

  echo "--- $model_id (embedding) ---"

  local result
  if ! result=$(node tools/doppler-cli.js debug --config "$config_json" --json 2>&1); then
    echo "FAIL $model_id (command error)"
    fail_count=$((fail_count + 1))
    echo ""
    return
  fi

  # Embedding models: check that we got a non-empty embedding vector
  local dims
  dims=$(echo "$result" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{try{const r=JSON.parse(d);const e=r.result?.metrics?.embeddingDimensions||r.result?.embedding?.length||0;console.log(e)}catch{console.log(0)}})" 2>/dev/null)

  if [ "${dims:-0}" -gt 0 ] 2>/dev/null; then
    echo "PASS  embedding dims: $dims"
    pass_count=$((pass_count + 1))
  else
    echo "FAIL  no embedding produced"
    fail_count=$((fail_count + 1))
  fi
  echo ""
}

# ---------------------------------------------------------------------------
# Model sweep
# ---------------------------------------------------------------------------

echo "============================================="
echo "DOPPLER Model Sweep — $(date -Iseconds)"
echo "Volume: $RDRR_DIR"
echo "============================================="
echo ""

# Gemma 3 family
if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "gemma-3-270m-it-q4k-ehf16-af32" ]; then
  run_text_model "gemma-3-270m-it-q4k-ehf16-af32" "blue|sky|color"
fi

if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "gemma-3-1b-it-q4k-ehf16-af32" ]; then
  run_text_model "gemma-3-1b-it-q4k-ehf16-af32" "blue|sky|color"
fi

if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "gemma-3-1b-it-f16-af32" ]; then
  run_text_model "gemma-3-1b-it-f16-af32" "blue|sky|color"
fi

# TranslateGemma family
if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "translategemma-4b-it-q4k-ehf16-af32" ]; then
  run_text_model "translategemma-4b-it-q4k-ehf16-af32" "bonjour|monde"
fi

if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "translategemma-4b-1b-enes-q4k-ehf16-af32" ]; then
  run_text_model "translategemma-4b-1b-enes-q4k-ehf16-af32" "bonjour|monde|hola"
fi

# LFM2 (ChatML template)
if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "lfm2-5-1-2b-instruct-q4k-ehf16-af32" ]; then
  run_text_model "lfm2-5-1-2b-instruct-q4k-ehf16-af32" "blue|sky|color"
fi

# Qwen 3.5 family
if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "qwen-3-5-0-8b-q4k-ehaf16" ]; then
  run_text_model "qwen-3-5-0-8b-q4k-ehaf16" "blue|sky|color"
fi

if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "qwen-3-5-2b-q4k-ehaf16" ]; then
  run_text_model "qwen-3-5-2b-q4k-ehaf16" "blue|sky|color"
fi

# Embedding model
if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "google-embeddinggemma-300m-q4k-ehf16-af32" ]; then
  run_embedding_model "google-embeddinggemma-300m-q4k-ehf16-af32"
fi

# Diffusion (skip — needs different pipeline surface)
if [ -z "$SINGLE_MODEL" ] || [ "$SINGLE_MODEL" = "sana-sprint-0.6b-wf16-ef16-hf16-f16" ]; then
  echo "--- sana-sprint-0.6b-wf16-ef16-hf16-f16 (diffusion) ---"
  echo "SKIP  diffusion models require browser surface"
  skip_count=$((skip_count + 1))
  echo ""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "============================================="
echo "RESULTS: $pass_count passed, $fail_count failed, $skip_count skipped"
echo "============================================="

if [ "$fail_count" -gt 0 ]; then
  exit 1
fi
