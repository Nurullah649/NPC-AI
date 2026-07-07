---
description: Consult Claude Opus 4 for backend architecture guidance before implementation
agent: council
---
Load the `council-orchestration` skill first, then consult `council-consultant-opus` on this backend topic:

$ARGUMENTS

Requirements:
- This is a consultation-only flow. Do not delegate to workers yet.
- Send the question to `council-consultant-opus` with all relevant context (file paths, schema details, current architecture).
- Review the consultant's analysis and recommendations yourself.
- If the consultant identifies actions that require implementation, propose a follow-up plan using the normal council -> worker -> triple-review flow.
- Present the architectural guidance to the user, clearly labeled as Opus 4 consultation output.
