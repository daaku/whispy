# whispy

A daemon process that works using
[`pw-record`](https://docs.pipewire.org/page_man_pw-cat_1.html),
[`ydotool`](https://github.com/ReimuNotMoe/ydotool) and of course
[`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) to provide
speech-to-text/dictation for Linux/Wayland.

Grab a model from [huggingface](https://huggingface.co/ggerganov/whisper.cpp).

## Sway Setup

Once you've built and installed the binary and a model, add something like this to your config:

```
exec whispy /home/naitik/.cache/whisper/ggml-large-v3-turbo-q5_0.bin
bindsym $mod+grave exec 'pkill -USR2 whispy'
```

That sets up mod+grave as your toggle.
