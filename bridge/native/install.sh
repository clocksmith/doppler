#!/bin/bash
# DOPPLER Native Bridge Installer
# Installs the Node.js native messaging host for Chrome/Chromium

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_NAME="dev.reploid.doppler"

# Detect OS and set paths
case "$(uname -s)" in
  Darwin)
    # macOS
    CHROME_NATIVE_DIR="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
    CHROMIUM_NATIVE_DIR="$HOME/Library/Application Support/Chromium/NativeMessagingHosts"
    ;;
  Linux)
    # Linux
    CHROME_NATIVE_DIR="$HOME/.config/google-chrome/NativeMessagingHosts"
    CHROMIUM_NATIVE_DIR="$HOME/.config/chromium/NativeMessagingHosts"
    ;;
  *)
    echo "Unsupported OS: $(uname -s)"
    echo "For Windows, manually copy the manifest to:"
    echo "  HKEY_CURRENT_USER\\Software\\Google\\Chrome\\NativeMessagingHosts\\$HOST_NAME"
    exit 1
    ;;
esac

# Get extension ID from argument or prompt
EXTENSION_ID="${1:-}"
if [ -z "$EXTENSION_ID" ]; then
  echo "Usage: $0 <extension-id>"
  echo ""
  echo "To find your extension ID:"
  echo "1. Open chrome://extensions/"
  echo "2. Enable 'Developer mode'"
  echo "3. Load the extension from: $SCRIPT_DIR/../extension/"
  echo "4. Copy the extension ID shown"
  exit 1
fi

# Create manifest with correct paths
MANIFEST_CONTENT=$(cat <<EOF
{
  "name": "$HOST_NAME",
  "description": "DOPPLER Native Bridge - File access for LLM inference",
  "path": "$SCRIPT_DIR/doppler-bridge.sh",
  "type": "stdio",
  "allowed_origins": [
    "chrome-extension://$EXTENSION_ID/"
  ]
}
EOF
)

# Install for Chrome
if [ -d "$(dirname "$CHROME_NATIVE_DIR")" ]; then
  mkdir -p "$CHROME_NATIVE_DIR"
  echo "$MANIFEST_CONTENT" > "$CHROME_NATIVE_DIR/$HOST_NAME.json"
  echo "Installed for Chrome: $CHROME_NATIVE_DIR/$HOST_NAME.json"
fi

# Install for Chromium
if [ -d "$(dirname "$CHROMIUM_NATIVE_DIR")" ]; then
  mkdir -p "$CHROMIUM_NATIVE_DIR"
  echo "$MANIFEST_CONTENT" > "$CHROMIUM_NATIVE_DIR/$HOST_NAME.json"
  echo "Installed for Chromium: $CHROMIUM_NATIVE_DIR/$HOST_NAME.json"
fi

echo ""
echo "Native host installed successfully!"
echo ""
echo "Next steps:"
echo "1. Reload the extension in chrome://extensions/"
echo "2. Open DOPPLER boot screen"
echo "3. Select 'DOPPLER' provider"
echo "4. The 'Local Path' field should appear"
echo ""
