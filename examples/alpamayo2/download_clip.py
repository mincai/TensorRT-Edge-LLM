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
"""Download the PhysicalAI-AV features Alpamayo 2 needs for one clip: egomotion and the 7-camera ring.

Features already in the Hugging Face cache are skipped. The dataset is gated; run `hf auth login` first.
"""

import argparse

import physical_ai_av


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_id",
                        nargs="?",
                        default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument(
        "--dataset-revision",
        help="pin to the revision already in the cache to avoid a re-download")
    args = parser.parse_args()

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        revision=args.dataset_revision, confirm_download_threshold_gb=1e9)
    cam = avdi.features.CAMERA
    features = [
        avdi.features.LABELS.EGOMOTION, cam.CAMERA_CROSS_LEFT_120FOV,
        cam.CAMERA_FRONT_WIDE_120FOV, cam.CAMERA_CROSS_RIGHT_120FOV,
        cam.CAMERA_REAR_LEFT_70FOV, cam.CAMERA_REAR_TELE_30FOV,
        cam.CAMERA_REAR_RIGHT_70FOV, cam.CAMERA_FRONT_TELE_30FOV
    ]
    print("downloading clip features:", args.clip_id, flush=True)
    avdi.download_clip_features(args.clip_id, features)


if __name__ == "__main__":
    main()
