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
"""Alpamayo 2 Super parity harness: PyTorch reference vs TensorRT Edge-LLM.

Needs the alpamayo2 reference package; the reference subcommand needs about 72 GB of GPU memory.

Subcommands:
    prepare    Load one PhysicalAI-AV clip (gated dataset, needs `hf auth login`), select the
               driving profile (6 cameras x 4 frames), pre-resize every image to the processor's
               exact target size, and write a self-contained sample directory plus the matching
               Edge-LLM `action_inference` request.
    reference  Run the PyTorch model on a sample directory and save CoT text, prompt token ids
               and trajectories.
    compare    Convert an `action_inference` output to ego-frame XYZ and report ADE/FDE
               against the reference and the ground-truth future.

Pre-resizing makes both pipelines consume identical pixels: the HF processor and the Edge-LLM
C++ preprocessor then both see images already at the smart_resize target, so resampling
differences cannot masquerade as model differences.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DRIVING_PROMPT = (
    "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
)


def _smart_resize_hw(height: int, width: int, min_pixels: int,
                     max_pixels: int) -> tuple[int, int]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import \
        smart_resize

    # Qwen3-VL: patch 16 x spatial merge 2 -> 32 px per LLM token.
    return smart_resize(height,
                        width,
                        factor=32,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels)


def _edge_request(sample_dir: Path, meta: dict, greedy: bool) -> dict:
    """Build the Edge-LLM request mirroring alpamayo2_super.helper.create_messages."""
    from alpamayo2_super.common import constants

    user = []
    for cam_id in meta["camera_ids"]:
        label = f"{constants.CAMERA_INDICES_TO_DISPLAY_NAMES[cam_id]}: "
        for frame in range(meta["num_frames"]):
            # Adjacent text items concatenate in the template; merging keeps the item count low.
            user.append({
                "type":
                "text",
                "text": (label if frame == 0 else "") + f"frame {frame} "
            })
            user.append({
                "type":
                "image",
                "image":
                str(sample_dir / "images" / f"c{cam_id}_f{frame}.png")
            })
    user.append({"type": "trajectory", "trajectory": meta["history_xyz"]})
    user.append({"type": "text", "text": DRIVING_PROMPT})
    return {
        "batch_size":
        1,
        "temperature":
        0.6,
        "top_p":
        0.98,
        "top_k":
        1 if greedy else 0,
        "max_generate_length":
        256,
        "requests": [{
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }]
                },
                {
                    "role": "user",
                    "content": user
                },
            ]
        }],
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import \
        load_physical_aiavdataset
    from PIL import Image

    config = json.loads((Path(args.model) / "config.json").read_text())
    data = select_task_input(
        load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us),
        "trajectory")

    frames = data["image_frames"]  # (cams, frames, 3, H, W) uint8
    n_cams, n_frames, _, height, width = frames.shape
    new_h, new_w = _smart_resize_hw(height, width, config["min_pixels"],
                                    config["max_pixels"])
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    camera_ids = [int(c) for c in data["camera_indices"].tolist()]
    for ci, cam_id in enumerate(camera_ids):
        for fi in range(n_frames):
            img = Image.fromarray(frames[ci, fi].permute(1, 2, 0).numpy())
            img.resize((new_w, new_h), Image.Resampling.BICUBIC).save(
                out / "images" / f"c{cam_id}_f{fi}.png")

    meta = {
        "clip_id":
        args.clip_id,
        "t0_us":
        args.t0_us,
        "camera_ids":
        camera_ids,
        "num_frames":
        n_frames,
        "image_hw": [new_h, new_w],
        "history_xyz":
        data["ego_history_xyz"][0, 0].tolist(),
        "history_rot":
        data["ego_history_rot"][0, 0].tolist(),
        "future_xyz":
        data["ego_future_xyz"][0, 0].tolist()
        if "ego_future_xyz" in data else None,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    for greedy in (True, False):
        name = "edge_input_greedy.json" if greedy else "edge_input_sampled.json"
        (out / name).write_text(
            json.dumps(_edge_request(out, meta, greedy), indent=1))
    print(f"wrote {out}: {n_cams} cams x {n_frames} frames at {new_w}x{new_h}")


def _load_sample(sample_dir: Path) -> tuple[dict, dict]:
    from PIL import Image

    meta = json.loads((sample_dir / "meta.json").read_text())
    frames = torch.stack([
        torch.stack([
            torch.from_numpy(
                np.array(
                    Image.open(
                        sample_dir / "images" /
                        f"c{cam_id}_f{fi}.png").convert("RGB"))).permute(
                            2, 0, 1) for fi in range(meta["num_frames"])
        ]) for cam_id in meta["camera_ids"]
    ])
    data = {
        "image_frames":
        frames,
        "camera_indices":
        torch.tensor(meta["camera_ids"]),
        "ego_history_xyz":
        torch.tensor(meta["history_xyz"], dtype=torch.float32)[None, None],
        "ego_history_rot":
        torch.tensor(meta["history_rot"], dtype=torch.float32)[None, None],
    }
    return meta, data


def cmd_reference(args: argparse.Namespace) -> None:
    from alpamayo2_super import helper
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.models.utils import fuse_traj_tokens

    sample_dir = Path(args.sample)
    _, data = _load_sample(sample_dir)
    model = Alpamayo2Super.from_pretrained(args.model,
                                           dtype=torch.bfloat16,
                                           device_map="cuda:0")
    model_inputs = helper.to_device(
        helper.prepare_model_inputs(data, model.config, model.tokenizer),
        "cuda")
    prompt_ids = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        model_inputs["tokenized_data"]["input_ids"],
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"]
        },
        model.config.traj_ids,
    )

    results = {"prompt_ids": prompt_ids[0].tolist(), "runs": []}
    for run in range(args.num_samples):
        torch.cuda.manual_seed_all(args.seed + run)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, _, extra = model.sample_trajectories_from_data(
                data=model_inputs,
                top_p=0.98,
                top_k=1 if args.greedy else None,
                temperature=0.6,
                num_traj_samples=1,
                diffusion_kwargs={"inference_step": 10},
                return_extra=True,
            )
        results["runs"].append({
            "cot":
            str(extra["cot"].reshape(-1)[0]),
            "pred_xyz":
            pred_xyz[0, 0, 0].float().cpu().tolist()
        })
        print(f"[run {run}] CoT: {results['runs'][-1]['cot']}")
    name = "reference_greedy.json" if args.greedy else "reference_sampled.json"
    (sample_dir / name).write_text(json.dumps(results))
    print(
        f"prompt tokens: {len(results['prompt_ids'])}; wrote {sample_dir / name}"
    )


def _ade_fde(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(a[:, :2] - b[:, :2], axis=-1)
    return float(d.mean()), float(d[-1])


def cmd_compare(args: argparse.Namespace) -> None:
    import hydra.utils as hyu

    sample_dir = Path(args.sample)
    meta, data = _load_sample(sample_dir)
    config = json.loads((Path(args.model) / "config.json").read_text())
    action_space = hyu.instantiate(config["expert_config"]["action_space_cfg"])
    edge = json.loads(Path(args.edge_output).read_text())
    responses = edge.get("responses",
                         edge if isinstance(edge, list) else [edge])

    ref = json.loads((sample_dir / args.reference).read_text())
    ref_xyz = np.array(ref["runs"][0]["pred_xyz"])
    gt = np.array(meta["future_xyz"]) if meta.get("future_xyz") else None
    print(f"reference CoT: {ref['runs'][0]['cot']}")
    for i, resp in enumerate(responses):
        action = torch.tensor(
            resp["output_trajectory"],
            dtype=torch.float32)[None]  # (1, 64, 2) normalized
        xyz, _ = action_space.action_to_traj(action,
                                             data["ego_history_xyz"][0],
                                             data["ego_history_rot"][0])
        xyz = xyz[0].numpy()
        ade, fde = _ade_fde(xyz, ref_xyz)
        line = f"[edge {i}] vs reference ADE {ade:.3f} m FDE {fde:.3f} m"
        if gt is not None:
            line += " | vs GT ADE {:.3f} m FDE {:.3f} m".format(
                *_ade_fde(xyz, gt))
        print(line)
        print(f"[edge {i}] text: {resp.get('output_text')}")
    if gt is not None:
        print("reference vs GT ADE {:.3f} m FDE {:.3f} m".format(
            *_ade_fde(ref_xyz, gt)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--clip-id", required=True)
    p.add_argument("--t0-us", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("reference")
    p.add_argument("--sample", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("compare")
    p.add_argument("--sample", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--edge-output", required=True)
    p.add_argument("--reference", default="reference_greedy.json")
    args = parser.parse_args()
    {
        "prepare": cmd_prepare,
        "reference": cmd_reference,
        "compare": cmd_compare
    }[args.cmd](args)


if __name__ == "__main__":
    main()
