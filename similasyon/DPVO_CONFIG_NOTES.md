# DPVO Configuration Differences

## Old (Manual) vs New (Official GitHub) DPVO Config Differences

### config/npc.yaml (Custom, High-Accuracy) — ACTIVELY USED
| Parameter | Old Custom Value | Official Default | Notes |
|-----------|-----------------|------------------|-------|
| PATCHES_PER_FRAME | 128 | 96 | More patches = higher accuracy |
| REMOVAL_WINDOW | 30 | 22 | Larger window |
| OPTIMIZATION_WINDOW | 25 | 10 | Much larger = slower but better |
| PATCH_LIFETIME | 26 | 13 | Patches live longer |
| KEYFRAME_THRESH | 15.0 | 15.0 | Same |
| MOTION_MODEL | CONSTANT_VELOCITY | DAMPED_LINEAR | Different motion model |
| MOTION_DAMPING | 0.5 | 0.5 | Same |
| MIXED_PRECISION | False | True | Full precision (slower, more accurate) |
| CENTROID_SEL_STRAT | EDGE_BIAS | RANDOM | Edge-biased patch selection |

### dpvo/config.py (Code Defaults)
| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| MIXED_PRECISION | False | True |

### Data Files (Old Only, Runtime Outputs)
- points.csv (163KB) — Previous run trajectory points
- points.txt (200KB) — Previous run output
- saved_trajectories/result.txt (462KB) — Previous trajectory
- result.ply (1.5MB) — Point cloud from previous run
- vicon1_easy.csv (447KB) — Test data

These are runtime artifacts, not source code changes.

## Decision
The old custom `config/npc.yaml` values are kept in `similasyon/config/dpvo/npc.yaml` 
as the active configuration. The old DPVO directory at `Class/DPVO/` is preserved 
as-is with the custom configs. The fresh GitHub clone was discarded since it 
would overwrite the customizations.
