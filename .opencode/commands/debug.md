---
description: Council debugging flow with reproduction, worker delegation, and dual review
agent: council
---
Load the `council-orchestration` and `council-debug` skills first, then handle this task through the full council workflow:

$ARGUMENTS

Requirements:
- Reproduce the issue or gather the strongest available evidence first.
- Delegate repo work to `worker-kimi` unless the debugging surface is broad or log-heavy enough for `worker-minimax`.
- Require both `council-review-deepseek` and `council-review-kimi` before the final answer if any worker-backed conclusion or fix is involved. Add `council-review-glm` for high-risk debugging.
- Final answer must separate reproduction status, root cause, changes made, verification run, and remaining unknowns.
