#!/bin/bash
# Start whispy daemon with proper backgrounding
#
# Usage: start-whispy.sh [path-to-model]
#
# If no model path is provided, defaults to ~/.cache/whisper/ggml-base.en.bin

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPY_BIN="${SCRIPT_DIR}/../whispy"
MODEL_PATH="${1:-${HOME}/.cache/whisper/ggml-base.en.bin}"
LOG_FILE="${WHISPY_LOG:-/tmp/whispy.log}"

if [ ! -f "$WHISPY_BIN" ]; then
    echo "Error: whispy binary not found at $WHISPY_BIN" >&2
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH" >&2
    exit 1
fi

nohup "$WHISPY_BIN" "$MODEL_PATH" >> "$LOG_FILE" 2>&1 &
disown
