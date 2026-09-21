# whispy

A daemon process that works using
[`pw-record`](https://docs.pipewire.org/page_man_pw-cat_1.html),
[`wtype`](https://github.com/atx/wtype) (with `wl-copy` for browsers) and
[Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) running through
[OpenVINO](https://docs.openvino.ai) to provide speech-to-text/dictation for
Linux/Wayland. Search mode uses
[Silero VAD](https://github.com/snakers4/silero-vad) to end a recording when
speech stops.

## Setup

Grab the OpenVINO IR models and vocabulary (around 1.2GB) from
[huggingface](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-ov):

```
dir=~/.cache/whispy/parakeet-v3
base=https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-ov/resolve/main
mkdir -p "$dir"
for f in parakeet_melspectogram parakeet_encoder parakeet_decoder parakeet_joint
do
  curl -fL -o "$dir/$f.xml" "$base/$f.xml"
  curl -fL -o "$dir/$f.bin" "$base/$f.bin"
done
curl -fL -o "$dir/parakeet_vocab.json" "$base/parakeet_vocab.json"
```

Search mode also needs the Silero VAD model:

```
curl -fL -o ~/.cache/whispy/silero_vad.onnx \
  https://huggingface.co/istupakov/silero-vad-onnx/resolve/main/silero_vad_op18_ifless.onnx
```

## Sway Setup

Once you've built and installed the binary and the models, add something like
this to your config:

```
exec whispy
bindsym $mod+grave exec 'pkill -USR2 whispy'
bindsym $mod+shift+grave exec 'pkill -USR1 whispy'
```

That sets up mod+grave as your toggle and mod+shift+grave as search mode.

## Notes

- `-device` picks the OpenVINO device for the encoder, decoder and joint
  network: `CPU` (default), `GPU`, `NPU` or `AUTO`.
- `-decoder-device` picks the device for the decoder and joint networks,
  which run once per token and once per frame. It defaults to `-device`, but
  they are small and latency bound, so the encoder is usually the only part
  worth handing to an accelerator.
- `-preproc-device` picks the device for the mel spectrogram model, `CPU` by
  default. It is a small amount of arithmetic on a large buffer (15 seconds of
  16 kHz audio in, `1x128x1501` out), so the CPU is the natural home for it,
  and moving it to an accelerator would add two more transfers per chunk for
  no compute win. The v2 export also has a dynamic input there, which the NPU
  rejects. The option exists to measure the difference.
- `-properties KEY=VALUE,...` passes extra OpenVINO compile properties.
- `-replacer` points at a two column CSV of transcript replacements. The file
  is optional, and defaults to `~/.config/whispy/replacer.csv`.
- Transcription goes through a small cleanup pipeline: the `-replacer` CSV
  first, then numbers written as words become digits (`twenty three` becomes
  `23`), then clock times get their colon (`11 30 pm` becomes `11:30pm`).
- `-transcribe FILE` transcribes a 16 kHz mono WAV (or the AU written by
  `-keep-audio`) and exits, without needing a VAD model or a sway session:

  ```
  whispy -transcribe parakeet/testdata/jfk.wav -print-time -device CPU
  whispy -transcribe parakeet/testdata/jfk.wav -print-time -device NPU
  ```

  That is the way to check a model installation and compare devices on
  identical audio: the text should come out the same for `CPU` and `NPU`. The
  file is transcribed twice and only the second run is timed, since the first
  pays for lazily initialized kernels, buffers and threads.
- On Intel GPUs the plugin runs models in fp16 by default, which changes the
  transcript: an 11 second clip that reads correctly on the CPU came out as
  "And" on an Xe iGPU. `-properties EXECUTION_MODE_HINT=ACCURACY` stops the
  precision conversion (and the dynamic quantization that comes with it) and
  restores the text, at the cost of fp32 speed.
- Built and tested on CPUs, an NPU and an Intel iGPU so far.

## NPU

The [NPU plugin](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
needs models with static shapes. Dynamic shapes on the NPU are a preview
limited to bounded shapes and vision models, and the plugin README still says
they are not supported.

- The encoder, decoder and joint network are static in both the v2 and v3
  exports, so `-device NPU` is fine for them as far as shapes go.
- The mel spectrogram model is dynamic in the v2 export, and the Silero VAD
  model is dynamic in every export, so both stay on the CPU.
- If a device cannot compile one of the models, whispy says so on stderr and
  falls back to the CPU for that model instead of refusing to start.

The decoder and the joint network run once per token and once per frame, which
is hundreds of tiny inferences per chunk. Per inference overhead is higher on
the NPU than on the CPU, so the encoder is the part that pays off most. On a
Core Ultra X7 358H, all three models on the NPU ran an 11 second clip in 243ms
against 401ms on the CPU, with the same transcript either way.

On Intel Core Ultra (Panther Lake, e.g. the Core Ultra X7 358H) the NPU is
platform 5010. On Arch this needs the NPU plugin and driver, plus a kernel
with the `intel_vpu` driver:

```
pacman -S openvino-intel-npu-plugin # pulls in intel-npu-driver and compiler
ls /dev/accel/accel0                # present once the driver is loaded
dmesg | grep intel_vpu              # "Initialized intel_vpu" on a good boot
```

### If the NPU rejects a model

Startup prints one line per model that fell back, and a summary of where each
model ended up running, for example:

```
parakeet: joint: NPU, using CPU (compile ...: Compilation failed ...)
parakeet: melspectogram=CPU encoder=NPU decoder=NPU joint=CPU
```

On Panther Lake, all three models compile for the NPU once the joint network's
axis is written positively (below). If a model is still refused:

- The joint network ends in a `LogSoftmax` with `axis="-1"` on a rank 4
  tensor, which the driver compiler rejects with `Got negative index -1 for
  Dim` from its `AlignDimensionsForDPU` pass. Rewriting it to the equivalent
  positive axis avoids that pass:

  ```
  sed -i 's/axis="-1"/axis="3"/' ~/.cache/whispy/parakeet-v3/parakeet_joint.xml
  ```

  `TestJointSoftmaxAxis` checks that this does not change the logits: the two
  graphs produce identical output in OpenVINO, and decoding only uses the
  argmax anyway. With it applied, the NPU compiles all three models.

Other NPU properties are worth a try through the same flag, for example
`-properties "NPU_COMPILATION_MODE_PARAMS=optimization-level=0"`.
