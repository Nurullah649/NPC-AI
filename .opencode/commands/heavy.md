---
description: Minimax-first council flow for broad or heavy repo tasks
agent: council
---
Load the `council-orchestration` skill first, then use the council workflow for this task, but start with `worker-minimax` unless there is a strong reason not to:

$ARGUMENTS

Requirements:
- Treat this as a broad, repetitive, cross-surface, or log-heavy assignment.
- Keep the triple-review gate mandatory with `council-review-deepseek`, `council-review-gpt5x`, and `council-review-kimi`; add `council-review-glm` only for notably complex tasks.
- Do not present raw worker output to the user.
- Reply only with the council-approved outcome.
