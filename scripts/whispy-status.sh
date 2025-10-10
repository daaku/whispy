#!/bin/bash
# Check whispy daemon and recording status

# Check if whispy is running
if ! pgrep -x whispy > /dev/null; then
    echo "⚠️  whispy is NOT running"
    exit 1
fi

# Check if recording is active (pw-record process exists)
if pgrep -x pw-record > /dev/null; then
    echo "🔴 RECORDING - Press \$mod+grave to STOP and transcribe"
    # Check how long the recording has been running
    if [ -f /tmp/a.au ]; then
        size=$(du -h /tmp/a.au | cut -f1)
        echo "   Audio file size: $size"
    fi
else
    echo "⚪ Idle - Press \$mod+grave to START recording"
fi
