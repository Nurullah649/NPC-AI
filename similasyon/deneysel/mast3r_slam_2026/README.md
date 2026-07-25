# MASt3R-SLAM 2026 competition experiment

This directory runs upstream MASt3R-SLAM on the same 7.5 FPS RGB route and
health protocol used by the DPVO/DPV-SLAM evaluation, then reports directly
comparable `E_3d`, `RMSE_3d`, axis RMSE, median, P95, maximum error and
10/50/100/250 m relative drift.

## Leakage and causality contract

- The SLAM runner never opens the ground-truth CSV.
- All 7.5 FPS frames are processed in order (`subsample: 1`).
- Retrieval, relocalization and graph optimization can use only current/past
  images. Previously emitted pose rows are never revised.
- At sample 450, the then-current raw positions of health=1 keyframes are
  frozen as visual gauge anchors. Later graph gauge changes are repaired with
  those raw visual pairs, without GT.
- The separate evaluator maps those anchors to GT using only anchor frame IDs
  `< 450`. Health=0 GT is opened only for final offline scoring.
- As in the production positioner, health=1 output is GT passthrough and all
  reported metrics cover health=0 only.

This makes the result a competition replay, not MASt3R's usual offline final
keyframe trajectory (which is allowed to rewrite history).

## Source and environment

Upstream source is cloned recursively under `third_party/MASt3R-SLAM`. The
tested revision is `e6f4e3d474fad0e11f561482012be864ba8c3f17`.

```bash
git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git \
  similasyon/deneysel/mast3r_slam_2026/third_party/MASt3R-SLAM
similasyon/deneysel/mast3r_slam_2026/setup_env.sh
similasyon/deneysel/mast3r_slam_2026/download_weights.sh
```

The current machine uses Python 3.11, PyTorch 2.5.1 CUDA 12.4 and
`opencv-python==4.10.0.84` in the `mast3r-slam-npc` Conda environment.

## Run

The full replay, scoring and comparison are one command:

```bash
similasyon/deneysel/mast3r_slam_2026/run_full.sh
```

The default keyframe buffer is 96 to fit the RTX 3060 Laptop's 6 GB VRAM. A
112-slot run reached frame 2130 but OOMed while calibrated GN requested another
136 MiB; the observed route projects to about 84 keyframes. PyTorch expandable
CUDA segments are intentionally not enabled because CUDA IPC to the spawned
backend then fails with `pidfd_getfd` on this host.
If the route exhausts it, rerun with a larger value if VRAM permits:

```bash
MAST3R_KEYFRAME_BUFFER=104 similasyon/deneysel/mast3r_slam_2026/run_full.sh
```

Outputs are written under `results/competition_full/`:

- `raw/raw_trajectory.csv`: immutable per-frame causal snapshots
- `raw/calibration_anchors.csv`: raw health=1 visual gauge anchors
- `raw/runtime.json`: provenance, timing, memory and GT isolation flags
- `evaluated/metrics.json`: health=0 E/RMSE and drift metrics
- `evaluated/alignment.json`: affine matrix diagnostics
- `comparison/comparison.md`: MASt3R-SLAM vs DPV-SLAM and production DPVO

## Tests

```bash
python -m unittest discover \
  -s similasyon/deneysel/mast3r_slam_2026/tests -v
```

## Measured full-route result

The verified run processed all 2258 frames, used the first 450 only for
calibration, scored 1808 health=0 frames, finished with 83 keyframes and took
677.94 seconds on the RTX 3060 Laptop GPU.

| Method | E (m) | RMSE (m) | Median (m) | P95 (m) | Max (m) |
|---|---:|---:|---:|---:|---:|
| Production DPVO | 18.758 | 21.277 | 21.478 | 31.569 | 36.581 |
| MASt3R-SLAM causal replay | 18.969 | 24.156 | 12.299 | 52.739 | 61.548 |
| DPV-SLAM learned-loop run | 45.614 | 60.838 | 24.196 | 134.806 | 143.870 |

MASt3R's mean error is close to production DPVO (1.1% higher) and its median
is substantially lower, but its P95/maximum tail is worse. It therefore remains
an experimental alternative rather than a drop-in production replacement.
