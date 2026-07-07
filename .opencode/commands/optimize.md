---
description: Council optimization flow with measurement, worker execution, and dual review
agent: council
---
Load the `council-orchestration` and `council-optimization` skills first, then handle this task through the full council workflow:

$ARGUMENTS

Requirements:
- Start from a measured or strongly evidenced bottleneck.
- Delegate broad search, profiling, or repetitive optimization work to `worker-minimax`; use `worker-kimi` for targeted improvements.
- Require both `council-review-deepseek` and `council-review-kimi` before the final answer if any worker-backed change or conclusion is involved. Add `council-review-glm` for critical-path optimizations.
- Final answer must state the bottleneck, evidence used, changes made, verification run, and tradeoffs.
