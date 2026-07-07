---
description: Council security-analysis flow with worker investigation and dual review
agent: council
---
Load the `council-orchestration` and `council-security-analysis` skills first, then handle this task through the full council workflow:

$ARGUMENTS

Requirements:
- Treat findings as severity-ordered and evidence-driven.
- Delegate broad trust-boundary mapping or large code sweeps to `worker-minimax`; use `worker-kimi` for narrower security fixes.
- Require both `council-review-deepseek` and `council-review-kimi` before the final answer if any worker-backed finding or fix is involved. Add `council-review-glm` for high-severity findings.
- Final answer must present findings first, then fixes or recommendations, then verification limits and residual risk.
