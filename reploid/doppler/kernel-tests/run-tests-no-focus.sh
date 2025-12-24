#!/bin/bash
# Run GPU tests without focus stealing
# Aggressively refocuses terminal

# Start refocusing immediately in background (every 0.5s)
(
  while true; do
    osascript -e 'tell application "Terminal" to activate' 2>/dev/null || \
    osascript -e 'tell application "iTerm2" to activate' 2>/dev/null
    sleep 0.5
  done
) &
REFOCUS_PID=$!

# Run tests
playwright test tests/correctness/ "$@"
EXIT_CODE=$?

# Stop refocusing
kill $REFOCUS_PID 2>/dev/null

exit $EXIT_CODE
