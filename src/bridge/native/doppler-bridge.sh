#!/bin/bash
# DOPPLER Native Bridge Launcher
# This wrapper allows Chrome to execute the Node.js native host

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Node.js native host
exec node "$SCRIPT_DIR/native-host.js"
