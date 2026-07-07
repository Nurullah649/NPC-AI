---
description: Council implementation flow with worker delegation and triple review sign-off
agent: council
---
Load the `council-orchestration` skill first, then use the council workflow for this task:

$ARGUMENTS

Requirements:
- Decompose the task before acting.
- If the task involves backend architecture decisions, consult `council-consultant-opus` first.
- Delegate implementation, editing, and repo-search work to the appropriate worker: `worker-kimi`, `worker-deepseek` for normal work, or `worker-minimax` for heavy/broad work.
- Do not present raw worker output to the user.
- Run the default reviewers (`council-review-deepseek`, `council-review-gpt5x`, `council-review-kimi`) before the final answer. Only add `council-review-glm` for notably complex tasks.
- If any reviewer blocks, iterate until all are clear.
- Reply only with the council-approved outcome.
