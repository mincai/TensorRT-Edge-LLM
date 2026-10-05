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
"""Compare Alpamayo 2 rolling-replay dumps across precisions (FP16 / FP8 / NVFP4).

Each input is a JSON written by rolling_inference.py --dump over the same clip and t0 steps. Prints
one table: latency, ADE against ground truth, and how far each run's trajectories and reasoning drift
from the reference run (the first file, normally FP16).

Usage: python compare_rollouts.py fp16.json fp8.json nvfp4.json [--engine-gb fp16=64.0 fp8=32.8 nvfp4=18.0]
"""

import argparse
import json

import numpy as np


def load(path):
    d = json.load(open(path))
    steps = d["steps"]
    return {
        "name": d["label"],
        "load_s": d.get("load_s", float("nan")),
        "t0": np.array([s["t0"] for s in steps]),
        "lat": np.array([s["lat_ms"] for s in steps]),
        "ade": np.array([s["ade"] for s in steps]),
        "pred": np.array([s["pred"] for s in steps]),  # (steps, 64, 3)
        "gt": np.array([s["gt"] for s in steps]),
        "cot": [s["cot"] for s in steps],
    }


def xy_err(a, b):
    """Per-step mean and final displacement between two trajectory stacks, in metres."""
    d = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)  # (steps, 64)
    return d.mean(axis=1), d[:, -1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps",
                    nargs="+",
                    help="DUMP json files; the first is the reference")
    ap.add_argument("--engine-gb",
                    nargs="*",
                    default=[],
                    help="name=GB pairs for the LLM engine sizes")
    args = ap.parse_args()
    runs = [load(p) for p in args.dumps]
    sizes = dict(kv.split("=") for kv in args.engine_gb)
    ref = runs[0]
    for r in runs[1:]:
        if len(r["t0"]) != len(
                ref["t0"]) or not np.allclose(r["t0"], ref["t0"]):
            raise SystemExit(
                f"{r['name']} was not run over the same t0 steps as {ref['name']}"
            )

    rows = [
        ("LLM engine size (GB)", lambda r: f"{float(sizes[r['name']]):.1f}"
         if r["name"] in sizes else "-"),
        ("Engine load (s)", lambda r: f"{r['load_s']:.0f}"),
        ("Latency mean (ms)", lambda r: f"{r['lat'].mean():.0f}"),
        ("Latency median (ms)", lambda r: f"{np.median(r['lat']):.0f}"),
        ("Latency p95 (ms)", lambda r: f"{np.percentile(r['lat'], 95):.0f}"),
        ("Latency max (ms)", lambda r: f"{r['lat'].max():.0f}"),
        ("Update rate (Hz)", lambda r: f"{1000 / r['lat'].mean():.2f}"),
        ("Speed-up vs " + ref["name"],
         lambda r: f"{ref['lat'].mean() / r['lat'].mean():.2f}x"),
        ("ADE vs ground truth, mean (m)", lambda r: f"{r['ade'].mean():.3f}"),
        ("ADE vs ground truth, median (m)",
         lambda r: f"{np.median(r['ade']):.3f}"),
        ("ADE vs ground truth, worst (m)", lambda r: f"{r['ade'].max():.3f}"),
        ("Final disp. vs ground truth, mean (m)",
         lambda r: f"{xy_err(r['pred'], r['gt'])[1].mean():.3f}"),
        ("Trajectory drift from " + ref["name"] + ", mean ADE (m)",
         lambda r: f"{xy_err(r['pred'], ref['pred'])[0].mean():.3f}"),
        ("Trajectory drift from " + ref["name"] + ", worst step (m)",
         lambda r: f"{xy_err(r['pred'], ref['pred'])[0].max():.3f}"),
        ("Reasoning identical to " + ref["name"] + " (steps)", lambda r:
         f"{sum(a == b for a, b in zip(r['cot'], ref['cot']))}/{len(r['cot'])}"
         ),
    ]
    width = max(len(label) for label, _ in rows)
    head = f"{'':<{width}}  " + "  ".join(f"{r['name']:>10}" for r in runs)
    print(head)
    print("-" * len(head))
    for label, fn in rows:
        print(f"{label:<{width}}  " + "  ".join(f"{fn(r):>10}" for r in runs))

    print("\nper-step ADE vs ground truth (m)")
    print(f"{'t0':>5}  " + "  ".join(f"{r['name']:>8}" for r in runs) +
          "   reasoning (first run)")
    for i, t0 in enumerate(ref["t0"]):
        print(f"{t0:5.1f}  " + "  ".join(f"{r['ade'][i]:8.3f}" for r in runs) +
              f"   {ref['cot'][i][:60]}")


if __name__ == "__main__":
    main()
