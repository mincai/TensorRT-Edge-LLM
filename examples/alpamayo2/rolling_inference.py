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
"""Rolling Alpamayo 2 Super inference over one PhysicalAI-AV clip with TensorRT Edge-LLM engines.

Sweeps t0 across the clip and re-plans at every step, the way a driving stack re-plans on a rolling
window. The engines load once; each step reads the 6-camera x 4-frame driving input and the 16-point
ego history, runs vision encoder + LLM + action expert, and integrates the expert's normalized
(acceleration, curvature) waypoints to ego-frame XYZ with the checkpoint's action space.

Prints latency, ADE against the ground-truth future and the reasoning per step, then a summary.
Optionally writes a GIF (front camera next to a bird's-eye view of predicted and ground-truth paths)
and a JSON dump for compare_rollouts.py.

Latency covers only the inference call. Prep is dataset decoding, which live camera input would not pay.

Requirements: the Edge-LLM Python bindings (tensorrt_edgellm wheel, or a build with
-DBUILD_PYTHON_BINDINGS=ON), the alpamayo2 reference package (data loader and action space), and
access to the gated nvidia/PhysicalAI-Autonomous-Vehicles dataset (`hf auth login`).
"""

import argparse
import importlib
import importlib.util
import io
import json
import re
import sys
import time
from pathlib import Path

import hydra.utils as hyu
import matplotlib
import numpy as np
import physical_ai_av
import torch
from alpamayo2_super.common import constants
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DRIVING_PROMPT = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
FRONT_WIDE_CAMERA = 1


def load_edgellm_runtime(build_dir: Path | None):
    try:
        return importlib.import_module("tensorrt_edgellm._edgellm_runtime")
    except ImportError:
        if build_dir is None:
            sys.exit(
                "Edge-LLM Python bindings not installed; pass --edgellm-build <build dir>"
            )
    so = sorted(build_dir.rglob("_edgellm_runtime*.so"))
    if not so:
        sys.exit(
            f"_edgellm_runtime*.so not found under {build_dir}; build with -DBUILD_PYTHON_BINDINGS=ON"
        )
    spec = importlib.util.spec_from_file_location("_edgellm_runtime", so[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def short_text(text: str, n: int = 70) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def split_output(raw: str) -> tuple[str, str]:
    """Split a generation decoded with special tokens into (chain of causation, meta action).

    The runtime stops one token after <|traj_future_start|>; everything from that marker on is
    dropped, as extract_text_tokens does in the reference.
    """
    raw = raw.split("<|traj_future_start|>", 1)[0]
    meta = ""
    if "<|meta_action_start|>" in raw:
        raw, meta = raw.split("<|meta_action_start|>", 1)
        meta = meta.split("<|meta_action_end|>", 1)[0]
    cot = raw.split("<|cot_end|>", 1)[0]

    def strip(t: str) -> str:
        return " ".join(re.sub(r"<\|[^|]*\|>", " ", t).split())

    return strip(cot), strip(meta)


def png_bytes(frame_chw: np.ndarray, size_wh: tuple[int, int]) -> bytes:
    img = Image.fromarray(frame_chw.transpose(1, 2, 0)).resize(
        size_wh, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    return buf.getvalue()


def build_request(rt, data, size_wh, args):
    """One request mirroring alpamayo2_super.helper.create_messages for the driving profile."""
    frames = data["image_frames"].cpu().numpy()  # (cams, frames, 3, H, W)
    camera_ids = [int(c) for c in data["camera_indices"].tolist()]
    contents, images = [], []
    for ci, cam_id in enumerate(camera_ids):
        label = f"{constants.CAMERA_INDICES_TO_DISPLAY_NAMES[cam_id]}: "
        for fi in range(frames.shape[1]):
            contents.append(
                rt.MessageContent("text",
                                  (label if fi == 0 else "") + f"frame {fi} "))
            contents.append(rt.MessageContent("image", ""))
            images.append(
                rt.load_image_from_bytes(png_bytes(frames[ci, fi], size_wh)))
    contents.append(rt.MessageContent("trajectory", ""))
    contents.append(rt.MessageContent("text", DRIVING_PROMPT))

    req = rt.Request([
        rt.Message("system", [rt.MessageContent("text", SYSTEM_PROMPT)]),
        rt.Message("user", contents)
    ])
    req.image_buffers = images
    req.past_trajectory = [
        tuple(map(float, p)) for p in data["ego_history_xyz"][0, 0].tolist()
    ]
    req.sampling_seed = args.seed

    gen = rt.LLMGenerationRequest()
    gen.requests = [req]
    gen.temperature = 0.6
    gen.top_p = 0.98
    gen.top_k = 1 if args.greedy else 0  # 0 disables top-k, like top_k=None in the reference
    gen.max_generate_length = 256
    gen.apply_chat_template = True
    gen.add_generation_prompt = True
    gen.skip_special_tokens = False  # split_output() needs the markers
    return gen


def save_gif(path: str, steps: list, label: str) -> None:
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(15, 5.5))
    imshow = axc.imshow(steps[0]["frame"])
    axc.axis("off")
    (gt_line, ) = axb.plot([], [],
                           "-",
                           color="tab:green",
                           lw=2.5,
                           label="ground truth")
    (pr_line, ) = axb.plot([], [],
                           "--",
                           color="tab:red",
                           lw=2.0,
                           label=f"predicted ({label})")
    axb.scatter([0], [0], c="k", marker="s", s=60, zorder=5, label="ego")
    lat = np.concatenate(
        [np.r_[s["pred"][:, 1], s["gt"][:, 1]] for s in steps])
    fwd = np.concatenate(
        [np.r_[s["pred"][:, 0], s["gt"][:, 0]] for s in steps])
    axb.set_xlim(lat.max() + 1, lat.min() - 1)  # +left on the left
    axb.set_ylim(fwd.min() - 2, fwd.max() + 2)
    axb.set_xlabel("lateral +left (m)")
    axb.set_ylabel("forward (m)")
    axb.grid(alpha=0.3)
    axb.legend(fontsize=9, loc="upper left")

    def update(k):
        s = steps[k]
        imshow.set_data(s["frame"])
        gt_line.set_data(s["gt"][:, 1], s["gt"][:, 0])
        pr_line.set_data(s["pred"][:, 1], s["pred"][:, 0])
        axc.set_title(f"front wide @ t0={s['t0']:.1f}s", fontsize=11)
        fig.suptitle(
            f"t0={s['t0']:.1f}s  ADE={s['ade']:.3f} m  lat={s['lat_ms']:.0f} ms  |  {short_text(s['cot'])}",
            fontsize=12)
        return imshow, gt_line, pr_line

    anim = FuncAnimation(fig, update, frames=len(steps), interval=600)
    anim.save(path, writer=PillowWriter(fps=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir",
                        type=Path,
                        required=True,
                        help="Alpamayo 2 Super checkpoint (config only)")
    parser.add_argument("--llm-engine-dir", type=Path, required=True)
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        required=True,
        help="directory holding the visual/ and action/ engines")
    parser.add_argument(
        "--edgellm-build",
        type=Path,
        help="Edge-LLM build directory, if the wheel is not installed")
    parser.add_argument("--clip-id",
                        default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument(
        "--dataset-revision",
        help=
        "PhysicalAI-AV revision; pin it to the cached one to avoid a re-download"
    )
    parser.add_argument("--t0-start", type=float, default=2.0)
    parser.add_argument("--t0-end", type=float, default=18.0)
    parser.add_argument("--t0-step", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy",
                        action="store_true",
                        help="top_k=1 for deterministic text")
    parser.add_argument("--label",
                        default="TensorRT",
                        help="name for this run in the dump and GIF")
    parser.add_argument("--gif", help="write a replay GIF to this path")
    parser.add_argument(
        "--dump",
        help="write per-step results to this JSON for compare_rollouts.py")
    args = parser.parse_args()

    rt = load_edgellm_runtime(args.edgellm_build)
    config = json.loads((args.model_dir / "config.json").read_text())
    action_space = hyu.instantiate(config["expert_config"]["action_space_cfg"])
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        revision=args.dataset_revision, confirm_download_threshold_gb=1e9)

    t = time.perf_counter()
    runtime = rt.LLMRuntime(str(args.llm_engine_dir),
                            str(args.multimodal_engine_dir))
    load_s = time.perf_counter() - t
    print(f"engines loaded in {load_s:.1f}s", flush=True)

    t0s = np.arange(args.t0_start, args.t0_end + 1e-6, args.t0_step)
    print(
        f"{len(t0s)} steps, t0 {args.t0_start}..{args.t0_end}s every {args.t0_step}s, clip {args.clip_id}"
    )
    print(
        f"{'t0(s)':>6} | {'lat(ms)':>7} | {'prep(ms)':>8} | {'ADE(m)':>7} | {'meta action':<18} | reasoning"
    )

    steps = []
    for t0 in t0s:
        tp = time.perf_counter()
        data = select_task_input(
            load_physical_aiavdataset(args.clip_id,
                                      t0_us=int(t0 * 1_000_000),
                                      avdi=avdi), "trajectory")
        h, w = data["image_frames"].shape[-2:]
        # Qwen3-VL: patch 16 x spatial merge 2 = 32 px per LLM token.
        new_h, new_w = smart_resize(h,
                                    w,
                                    factor=32,
                                    min_pixels=config["min_pixels"],
                                    max_pixels=config["max_pixels"])
        request = build_request(rt, data, (new_w, new_h), args)
        prep_ms = (time.perf_counter() - tp) * 1000

        t = time.perf_counter()
        response = runtime.handle_request(request)
        lat_ms = (time.perf_counter() - t) * 1000

        action = torch.tensor(response.output_trajectories[0],
                              dtype=torch.float32)[None]
        pred_xyz, _ = action_space.action_to_traj(action,
                                                  data["ego_history_xyz"][0],
                                                  data["ego_history_rot"][0])
        pred = pred_xyz[0].numpy()
        gt = data["ego_future_xyz"].cpu().float().numpy()[0, 0]
        ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean())
        cot, meta = split_output(response.output_texts[0])

        cams = data["camera_indices"].tolist()
        cam = cams.index(FRONT_WIDE_CAMERA) if FRONT_WIDE_CAMERA in cams else 0
        frame = data["image_frames"].cpu().numpy()[cam, -1].transpose(1, 2, 0)
        steps.append({
            "t0": float(t0),
            "lat_ms": lat_ms,
            "prep_ms": prep_ms,
            "ade": ade,
            "cot": cot,
            "meta": meta,
            "pred": pred,
            "gt": gt,
            "frame": frame
        })
        print(
            f"{t0:6.1f} | {lat_ms:7.0f} | {prep_ms:8.0f} | {ade:7.3f} | {short_text(meta, 18):<18} | {short_text(cot)}",
            flush=True)

    lat = np.array([s["lat_ms"] for s in steps])
    ade = np.array([s["ade"] for s in steps])
    print(
        f"\nsummary: steps={len(steps)}  latency mean={lat.mean():.0f}ms p50={np.median(lat):.0f} "
        f"max={lat.max():.0f} ({1000 / lat.mean():.2f} Hz)  "
        f"ADE mean={ade.mean():.3f} min={ade.min():.3f} max={ade.max():.3f} m")

    if args.dump:
        rows = [{
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in s.items() if k != "frame"
        } for s in steps]
        with open(args.dump, "w") as f:
            json.dump(
                {
                    "label": args.label,
                    "clip": args.clip_id,
                    "seed": args.seed,
                    "greedy": args.greedy,
                    "load_s": load_s,
                    "steps": rows
                }, f)
        print("wrote", args.dump)
    if args.gif:
        save_gif(args.gif, steps, args.label)
        print("wrote", args.gif)


if __name__ == "__main__":
    main()
