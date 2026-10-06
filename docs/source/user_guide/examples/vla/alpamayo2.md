# Alpamayo 2 Super (Vision-Language-Action)

Complete workflow for running [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super), a
Qwen3-VL-32B vision-language model with a 2.3B flow-matching action expert. The language model can run in
FP16, FP8 or NVFP4; the vision encoder and action expert run in FP16.

> **Prerequisites:** Complete the
> [Installation Guide](../../getting_started/installation.md). The checkpoint requires Hugging Face login and
> license access before downloading. The FP16 language model engine is 64 GB, so it needs a GPU with more
> than 64 GB of memory; FP8 (32.8 GB) and NVFP4 (18.0 GB) fit on smaller GPUs. NVFP4 requires Blackwell-class
> hardware.

---

## Architecture Overview

Alpamayo 2 Super runs as the same chained VLM + action pipeline as [Alpamayo-R1](alpamayo.md):

1. **Vision encoder**: 24 camera images per request (6 cameras x 4 frames), each resized to
   163,840-196,608 pixels (160-192 image tokens). The PhysicalAI-AV cameras' 1920x1080 frames become
   576x320 (184,320 pixels, 180 tokens per image).
2. **Language model**: writes the chain-of-causation reasoning and stops at `<|traj_future_start|>`. Its KV
   cache is the action expert's input.
3. **Action expert**: 10 flow-matching steps over the KV cache, producing 64 waypoints of normalized
   (acceleration, curvature) at 0.1 s intervals.

All three run in one `action_inference` call or through the Python bindings. The KV cache must stay FP16
because the action expert reads it.

---

## Step 1: Export FP16 Components

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
mkdir -p $WORKSPACE_DIR && cd $WORKSPACE_DIR

hf auth login
hf download nvidia/Alpamayo2-Super --local-dir Alpamayo2-Super

tensorrt-edgellm-export Alpamayo2-Super onnx/fp16 --components action --max-kv-cache-capacity 6144
tensorrt-edgellm-export Alpamayo2-Super onnx/fp16 --components visual
tensorrt-edgellm-export Alpamayo2-Super onnx/fp16 --components thinker --max-kv-cache-capacity 6144
```

This creates `onnx/fp16/action`, `onnx/fp16/visual` and `onnx/fp16/llm`. The language model export peaks at
about 76 GB of host memory. A KV capacity of 6,144 covers the 24 images, the prompt, the 45 trajectory
history tokens and the reasoning; it must match `--maxKVCacheCapacity` in Step 3.

## Step 2 (Optional): Quantize the Language Model to FP8 or NVFP4

`tensorrt-edgellm-quantize` loads models through the transformers Auto classes, which do not know
`alpamayo2_super`. The backbone is a stock Qwen3-VL, so extract it first with
[`examples/alpamayo2/convert_to_qwen3vl.py`](source:examples/alpamayo2/convert_to_qwen3vl.py):

```bash
python examples/alpamayo2/convert_to_qwen3vl.py Alpamayo2-Super Alpamayo2-Super-qwen3vl
```

Quantize with one of:

```bash
# FP8 (34 GB checkpoint)
tensorrt-edgellm-quantize llm --model_dir Alpamayo2-Super-qwen3vl \
  --output_dir Alpamayo2-Super-qwen3vl-fp8 --quantization fp8 --num_samples 128

# NVFP4 with the LM head also in NVFP4 (20 GB checkpoint)
tensorrt-edgellm-quantize llm --model_dir Alpamayo2-Super-qwen3vl \
  --output_dir Alpamayo2-Super-qwen3vl-nvfp4 --quantization nvfp4 \
  --lm_head_quantization nvfp4 --num_samples 128
```

Export the quantized language model. Because the checkpoint is now a plain Qwen3-VL, the export writes the
Qwen3-VL chat template; replace it with the native Alpamayo 2 renderer:

```bash
PREC=nvfp4   # or fp8
tensorrt-edgellm-export Alpamayo2-Super-qwen3vl-$PREC onnx/$PREC --components thinker
rm -f onnx/$PREC/llm/chat_template.jinja onnx/$PREC/llm/processed_chat_template.json \
  onnx/$PREC/llm/chat_template.processor
echo alpamayo2 > onnx/$PREC/llm/chat_template.model
```

`--quantization nvfp4` quantizes weights and activations (W4A4). The exported graph uses TensorRT's native FP4
ops (`TRT_FP4DynamicQuantize` and `DequantizeLinear`); this path was validated on an SM120 GPU.

## Step 3: Build Engines

```bash
visual_build --onnxDir onnx/fp16/visual --engineDir engines/fp16 \
  --minImageTokens 160 --maxImageTokens 4608 --maxImageTokensPerImage 192
action_build --onnxDir onnx/fp16/action --engineDir engines/fp16 --maxBatchSize 1

PREC=fp16   # or fp8 / nvfp4
EDGELLM_SINGLE_PROFILE=1 llm_build --onnxDir onnx/$PREC/llm --engineDir engines/$PREC/llm \
  --maxInputLen 5632 --maxKVCacheCapacity 6144 --maxBatchSize 1
```

`EDGELLM_SINGLE_PROFILE=1` builds one optimization profile for prefill and decode. TensorRT keeps one copy of
the weights per profile, so without it the FP16 engine needs about 126 GB of GPU memory to build and load.

| LLM engine | Size | Build time | GPU peak during build |
|---|---|---|---|
| FP16 | 64.0 GB | 4.5 min | 61 GB |
| FP8 | 32.8 GB | 2.5 min | 31 GB |
| NVFP4 | 18.0 GB | 3 min | 17 GB |

The vision encoder (1.2 GB) and action expert (4.8 GB) engines are shared by all three. Build times are for
an RTX PRO 6000 Blackwell Max-Q.

## Step 4: Run Inference

The request mirrors the reference prompt: the cameras in the driving-profile order (front-left 120°,
front 120°, front-right 120°, rear-left 70°, rear-right 70°, front tele 30°), each camera's first frame
preceded by its label, then the 16-point ego history ending at the origin, then the instruction. Pre-resize
images to 576x320 so the runtime sees the same pixels as the reference.

```json
{
  "batch_size": 1, "temperature": 0.6, "top_p": 0.98, "top_k": 0, "max_generate_length": 256,
  "requests": [{"messages": [
    {"role": "system", "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}]},
    {"role": "user", "content": [
      {"type": "text", "text": "Front left camera: frame 0 "}, {"type": "image", "image": "/path/c0_f0.png"},
      {"type": "text", "text": "frame 1 "}, {"type": "image", "image": "/path/c0_f1.png"},
      {"type": "text", "text": "frame 2 "}, {"type": "image", "image": "/path/c0_f2.png"},
      {"type": "text", "text": "frame 3 "}, {"type": "image", "image": "/path/c0_f3.png"},
      {"type": "text", "text": "Front camera: frame 0 "}, {"type": "image", "image": "/path/c1_f0.png"},
      {"type": "trajectory", "trajectory": [[-15.0, 0.0, 0.0], [-14.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
      {"type": "text", "text": "output the chain-of-thought reasoning of the driving process, then output the future trajectory."}
    ]}
  ]}]
}
```

The example is shortened: a full request has 24 label/image pairs (cameras 0, 1, 2, 3, 5, 6 x frames 0-3)
and 16 history points. [`examples/alpamayo2/parity_harness.py`](source:examples/alpamayo2/parity_harness.py)
`prepare` writes complete requests from a PhysicalAI-AV clip. Use `"top_k": 1` for deterministic output.

```bash
action_inference --engineDir engines/$PREC/llm --multimodalEngineDir engines/fp16 \
  --inputFile input.json --outputFile output.json
```

`output_text` is the reasoning. `output_trajectory` holds 64 normalized (acceleration, curvature) pairs;
de-normalize them before use:

```text
accel     = a * 0.6810426736454882   + 0.02902694707164455     # m/s^2
curvature = k * 0.026148280660833106 + 0.0002692167976330542   # 1/m
```

Then integrate a unicycle model from the current speed, as the reference does in
`UnicycleAccelCurvatureActionSpace` (alpamayo2 repository). The constants are also written to
`engines/fp16/action/config.json`.

When decoding with special tokens kept, the runtime stops one token after `<|traj_future_start|>`; cut the
text at `<|cot_end|>` or `<|traj_future_start|>`, as `split_output()` in
[`examples/alpamayo2/rolling_inference.py`](source:examples/alpamayo2/rolling_inference.py) does.

## Rolling Inference over a Clip

[`examples/alpamayo2/rolling_inference.py`](source:examples/alpamayo2/rolling_inference.py) loads the
engines once through the Python bindings (build with `-DBUILD_PYTHON_BINDINGS=ON`) and re-plans at every t0
step across a PhysicalAI-AV clip. Measured on clip `030c760c` (t0 2-18 s every 0.5 s, 33 steps, seed 42) with
an RTX PRO 6000 Blackwell Max-Q (96 GB, PCIe):

| | FP16 | FP8 | NVFP4 |
|---|---|---|---|
| LLM engine size | 64.0 GB | 32.8 GB | 18.0 GB |
| Engine load (all three engines) | 64 s | 11 s (36 s from a cold page cache) | 12 s |
| Latency per prediction, mean / p95 | 2,601 / 2,868 ms | 1,963 / 2,236 ms | 1,293 / 1,379 ms |
| Predictions per second | 0.38 | 0.51 | 0.77 |
| ADE vs ground truth, mean / median | 1.23 / 1.33 m | 1.25 / 1.13 m | 1.21 / 1.04 m |
| Trajectory difference from FP16, mean | - | 0.41 m | 0.43 m |
| Reasoning text identical to FP16 | - | 5 / 33 steps | 7 / 33 steps |

Each prediction consumes all 24 images (6 cameras x 4 frames, about 1.6 s of history) plus the ego history,
so the rate counts complete predictions, not camera frames. Latency covers the vision encoder, reasoning and
action expert, not dataset decoding.

## Validation Status and Limitations

- **One clip:** the numbers above come from a single 33-step clip. The ADE differences between precisions
  are within the step-to-step spread of one clip, so they show no measurable loss from FP8 or NVFP4, but
  they don't prove equal accuracy. The engines have not yet been compared with the PyTorch BF16 reference
  on the official validation clips; [`parity_harness.py`](source:examples/alpamayo2/parity_harness.py)
  runs that comparison for one sample at a time.
- **Reasoning text:** with sampling, a small numeric difference changes one sampled token and the wording
  diverges from there, so the text rarely matches FP16 word for word. The driving decisions agree (keep
  distance to a slow lead vehicle, nudge around cones and a stopped truck). Each precision reproduces
  itself exactly with a fixed seed; use `"top_k": 1` (or `--greedy`) to compare text without sampling.
- **Calibration:** FP8 and NVFP4 were calibrated on 128 generic image-question samples. Calibrating on
  driving data may tighten the scales.
- **Vision encoder and action expert:** both run in FP16 in all results above. FP8 vision encoder weights
  (`tensorrt-edgellm-quantize llm --visual_quantization fp8`) have not been tested with this model. The
  action expert must stay FP16 because it reads the FP16 KV cache.
- **Speed:** the fastest configuration (NVFP4) produces 0.77 predictions per second on this GPU. That suits
  offline evaluation and other non-real-time uses; a 10 Hz control loop would need a much smaller or
  distilled model.

## Notes

- **Content-item limit:** a full driving request has 50 content items per user message; the per-message
  limit is 64.
- **Discrete GPUs behind PCIe switches:** a multi-GB engine upload in one transfer can trip PCIe completion
  timeouts on marginal links. Set `EDGELLM_PACED_LOAD_MB=256` and `EDGELLM_PACED_LOAD_PAUSE_MS=30` to upload
  in chunks.
- **Engine build writes:** the builder streams the serialized plan to disk, so host memory needs to hold
  only the ONNX weights, not a second copy of the plan.
