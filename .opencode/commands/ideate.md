---
description: Council ideation flow — GPT-5.5 generates ideas, DeepSeek V4 Pro refines them into implementable plans
agent: council
---
Load the `council-orchestration` skill first, then use the council ideation flow for this task:

$ARGUMENTS

Requirements:
- This is the ideation flow: idea generation → implementation refinement → optional worker execution.
- Step 1: Collect context about the problem domain (relevant files, constraints, architecture).
- Step 2: Send the problem + context to `council-ideator` (GPT-5.5) to generate multiple diverse ideas.
- Step 3: Send the ideator's output + repo context to `council-impl-refiner` (DeepSeek V4 Pro) to assess feasibility and produce an implementable plan.
- Step 4: Present the refined plan to the user with the best candidate highlighted.
- Step 5: If the user approves, proceed to implementation using the normal council → planner → worker → triple-review flow.
- Do NOT skip the refinement step — raw ideas are not actionable.
- Do not delegate to workers until the user has approved the refined plan.
