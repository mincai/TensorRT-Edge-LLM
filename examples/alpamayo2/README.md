# Alpamayo 2 Super examples

Helper scripts for [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super). The
end-to-end workflow (export, quantization, engine build, inference) is in the
[Alpamayo 2 Super guide](../../docs/source/user_guide/examples/vla/alpamayo2.md).

| Script | Purpose |
|---|---|
| `convert_to_qwen3vl.py` | Extract the Qwen3-VL backbone into a stock checkpoint so `tensorrt-edgellm-quantize` can produce FP8 or NVFP4 weights |
| `download_clip.py` | Download egomotion and the 7-camera ring of one PhysicalAI-AV clip |
| `rolling_inference.py` | Re-plan at every t0 step across a clip through the Python bindings; prints latency, ADE and reasoning, and can write a GIF and a JSON dump |
| `compare_rollouts.py` | Compare rolling-inference dumps from several precisions: latency, ADE, drift from a reference run, reasoning agreement |
| `parity_harness.py` | Prepare a clip sample and `action_inference` request, run the PyTorch reference, and compare trajectories |

## Requirements

- The Edge-LLM Python bindings for `rolling_inference.py`: the `tensorrt_edgellm` wheel, or a source
  build with `-DBUILD_PYTHON_BINDINGS=ON` (pass `--edgellm-build <build dir>`).
- The [Alpamayo 2 reference package](https://github.com/NVlabs/alpamayo2) and `physical-ai-av` for the
  data loader and action space. The reference pins `transformers` 4.57, while Edge-LLM export pins a
  newer version, so use a separate environment for these scripts.
- Access to the gated
  [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
  dataset (`hf auth login`).

## Compare precisions on a clip

```bash
for p in fp16 fp8 nvfp4; do
  python rolling_inference.py \
    --model-dir $WORKSPACE_DIR/Alpamayo2-Super \
    --llm-engine-dir $WORKSPACE_DIR/engines/$p/llm \
    --multimodal-engine-dir $WORKSPACE_DIR/engines/fp16 \
    --label $p --dump rollout_$p.json --gif rollout_$p.gif
done
python compare_rollouts.py rollout_fp16.json rollout_fp8.json rollout_nvfp4.json \
  --engine-gb fp16=64.0 fp8=32.8 nvfp4=18.0
```

With the default sampling (seed 42), each precision reproduces itself exactly, but the reasoning text
of different precisions diverges after the first differing token. Use `--greedy` to compare text
without sampling.
