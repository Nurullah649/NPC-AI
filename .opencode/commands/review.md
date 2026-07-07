---
description: Read-only triple council review of the current worktree or requested scope
agent: council
---
Load the `council-orchestration` skill first, then perform a read-only council review for this scope:

$ARGUMENTS

Requirements:
- Do not edit files unless the user explicitly asks for fixes.
- Collect independent reviews from `council-review-deepseek`, `council-review-gpt5x`, and `council-review-kimi`. Include `council-review-glm` only for notably complex scope.
- Merge them into one severity-ordered council verdict.
- Call out residual risks and missing verification clearly.
