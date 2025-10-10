# whispy

A daemon process that works using
[`pw-record`](https://docs.pipewire.org/page_man_pw-cat_1.html) and
[`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) to provide
speech-to-text/dictation for Linux.

Supports both **Wayland/Sway** and **X11/i3** window managers.

## Prerequisites

### Common Requirements
- `pw-record` (from PipeWire) - for audio recording
- A whisper.cpp model from [huggingface](https://huggingface.co/ggerganov/whisper.cpp)

### For Sway/Wayland (default)
- [`ydotool`](https://github.com/ReimuNotMoe/ydotool) - for typing and keyboard control
- `wl-clipboard` (provides `wl-copy`) - for clipboard operations

### For i3/X11
- `xdotool` - for typing and keyboard control
- `xclip` - for clipboard operations

## Building

### For Sway/Wayland (default)
```bash
./build-whisper  # Build whisper.cpp library first
go build
```

### For i3/X11
```bash
./build-whisper  # Build whisper.cpp library first
go build -tags=x11
```

## Setup

### Sway Setup

Once you've built the binary and downloaded a model, add this to your Sway config:

```
exec whispy /home/user/.cache/whisper/ggml-base.en.bin
bindsym $mod+grave exec 'pkill -USR2 whispy'
```

### i3 Setup

For i3, you'll want to use the included `start-whispy.sh` script for proper daemonization:

```
# i3 config (~/.config/i3/config)
exec --no-startup-id /path/to/whispy/start-whispy.sh
bindsym $mod+grave exec --no-startup-id pkill -USR2 whispy
```

The `start-whispy.sh` script (included in `scripts/`) ensures whispy runs as a proper daemon.

## Usage

Press `$mod+grave` (typically Super/Windows + backtick) to:
1. **Start recording** - Press once to begin recording audio
2. **Stop and transcribe** - Press again to stop recording and transcribe

The transcribed text will be:
- **Typed directly** into most applications
- **Pasted** (via clipboard) into Firefox (for better compatibility)

## Status Monitoring

### Command Line
Use the included `scripts/whispy-status.sh` to check if whispy is recording:

```bash
./scripts/whispy-status.sh
```

Shows:
- `⚪ Idle` - Not recording
- `🔴 RECORDING` - Currently recording (with audio file size)
- `⚠️ whispy is NOT running` - Daemon not active

### i3bar Integration
For visual feedback in your i3 status bar, use the included bumblebee-status module:

1. Copy `scripts/bumblebee-status/whispy.py` to `~/.config/bumblebee-status/modules/`
2. Add `whispy` to your i3bar status_command:

```
bar {
    status_command bumblebee-status -m \
        whispy memory date time battery \
        ...
}
```

The indicator shows:
- **⚪** - Idle (not recording)
- **🔴REC** - Recording active

## Troubleshooting

**Check if whispy is running:**
```bash
pgrep whispy
```

**View logs (if using start-whispy.sh):**
```bash
tail -f /tmp/whispy.log
```

**Test recording manually:**
```bash
# Start recording
pkill -USR2 whispy

# Speak into microphone...

# Stop and transcribe
pkill -USR2 whispy
```

**Dependencies check:**
```bash
# For Sway/Wayland
which ydotool wl-copy pw-record

# For i3/X11
which xdotool xclip pw-record
```
