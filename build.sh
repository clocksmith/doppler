#!/usr/bin/env bash
# DOPPLER Build Script
# Compiles TypeScript to JavaScript in dist/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building DOPPLER..."

# Clean dist
rm -rf dist

# Run TypeScript compiler
npx tsc --project tsconfig.json

echo "Build complete. Output in dist/"
echo ""
echo "To run the demo:"
echo "  npx tsx serve.ts --open"
