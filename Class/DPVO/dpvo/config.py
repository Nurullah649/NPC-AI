from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 4096

# bias patch selection towards high gradient regions?
_C.CENTROID_SEL_STRAT = 'RANDOM'

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = False

# Loop closure
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15

# Experimental numerical gauge conditioning. A non-zero value calls
# PatchGraph.normalize() every N input frames after tracking has initialized.
# It is disabled in the live config because an external DPVO->NED alignment
# must consume the emitted gauge event before it can safely be used.
_C.PERIODIC_NORMALIZE_FREQ = 0
_C.PERIODIC_NORMALIZE_START_FRAME = 0

# Classic loop closure
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3
_C.LOOP_RETR_THRESH = 0.04
_C.CLASSIC_LOOP_SYNCHRONOUS = False
_C.CLASSIC_LOOP_PGO_ITERS = 30
_C.CLASSIC_LOOP_PGO_TIMEOUT_SECONDS = 180

cfg = _C
