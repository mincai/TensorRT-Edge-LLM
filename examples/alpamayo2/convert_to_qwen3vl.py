# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Derive a stock Qwen3-VL checkpoint from Alpamayo 2 Super for FP8/NVFP4 quantization.

Alpamayo 2 Super wraps a Qwen3VLForConditionalGeneration under ``vlm.`` next to its action expert
(``expert.*``). ``tensorrt-edgellm-quantize`` loads models through the transformers Auto classes,
which do not know ``alpamayo2_super``, but the backbone itself is plain Qwen3-VL. This writes:

  * ``model-*.safetensors`` with ``vlm.`` stripped and ``expert.*`` dropped, one shard at a time,
  * ``config.json`` from the embedded ``vlm_config`` (vocabulary already includes the trajectory tokens),
  * the Alpamayo 2 tokenizer, a processor config with the model's pixel budget, and the chat and
    generation configs.

Only the language model is quantized. The vision tower stays in the output because Qwen3-VL needs it to
load; ``tensorrt-edgellm-quantize`` leaves it unquantized (``model.visual*`` is excluded). Export only the
language model (``--components thinker``) from the quantized checkpoint, and the vision encoder and action
expert from the original checkpoint in FP16.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    _build_alpamayo2_tokenizer


def convert(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    root = json.loads((src / "config.json").read_text())
    index = json.loads((src / "model.safetensors.index.json").read_text())

    weight_map = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(src / shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith("vlm."):
                    tensors[key[len("vlm."):]] = f.get_tensor(key)
        if tensors:
            save_file(tensors, dst / shard, metadata={"format": "pt"})
            weight_map.update({k: shard for k in tensors})
        print(f"{shard}: {len(tensors)} backbone tensors", flush=True)
    total = sum((dst / s).stat().st_size for s in set(weight_map.values()))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": total
                },
                "weight_map": weight_map
            },
            indent=1))

    config = dict(root["vlm_config"])
    config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    config["torch_dtype"] = root.get("dtype", "bfloat16")
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    for name in ("chat_template.jinja", "generation_config.json",
                 "video_preprocessor_config.json", "merges.txt", "vocab.json",
                 "LICENSE"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    pre = json.loads((src / "preprocessor_config.json").read_text())
    pre["size"] = {
        "shortest_edge": root["min_pixels"],
        "longest_edge": root["max_pixels"]
    }
    pre["min_pixels"], pre["max_pixels"] = root["min_pixels"], root[
        "max_pixels"]
    (dst / "preprocessor_config.json").write_text(json.dumps(pre, indent=2))

    # Same tokenizer as the Alpamayo 2 exporter writes, so token IDs match the exported engines.
    _build_alpamayo2_tokenizer(root, str(src), str(dst))
    print(f"wrote {dst} ({total / 1e9:.1f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir",
                        type=Path,
                        help="Alpamayo 2 Super checkpoint directory")
    parser.add_argument("output_dir",
                        type=Path,
                        help="where to write the Qwen3-VL checkpoint")
    args = parser.parse_args()
    convert(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
