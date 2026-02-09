#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The demo should consume the public API surface (via @doppler/core) rather than
# reaching into ../src/* module internals. This keeps demo=consumer honest.

if rg -n --hidden --glob '!**/node_modules/**' -S "\\.\\./src/" demo --type js --type ts; then
  echo ""
  echo "ERROR: demo contains forbidden imports of ../src/*"
  echo "Use '@doppler/core' (and only deep-import '@doppler/core/...' if truly necessary)."
  exit 1
fi

echo "OK: demo imports are core-only."
