---
name: council-orchestration
description: Enforce the repository's council, worker, and multi-review approval workflow for non-trivial tasks.
compatibility: opencode
metadata:
  area: multi-agent
  workflow: council-worker-review
---

## Purpose
- Use this skill for any non-trivial repo-backed task in this repository.
- This is the primary skill for the multi-agent architecture itself.
- It exists to make the council workflow explicit and repeatable instead of relying only on prompt memory.

## Required roles
- `council`: the only user-facing agent and final decision maker.
- `council-planner`: GPT-5.5 medium-reasoning planner — receives the council's collected findings and produces a structured implementation plan with scoped work packets, worker assignments, and verification strategy.
- `worker-kimi`: focused coding, file reading, targeted edits, and normal implementation work.
- `worker-minimax`: broad sweeps, repetitive refactors, cross-surface changes, and long-output work.
- `worker-deepseek`: focused implementation, file reading, and targeted edits — alternative to worker-kimi.
- `council-review-deepseek`: blocking DeepSeek V4 Pro review — adversarial second pass, edge cases, inconsistencies (default).
- `council-review-gpt5x`: blocking GPT-5.5 executive review — strategic alignment, long-term maintainability, product-level correctness (default, medium reasoning).
- `council-review-kimi`: blocking Kimi K2.6 review — executive oversight, architecture judgment, product-level correctness (default).
- `council-review-glm`: blocking GLM-5.1 review — deep adversarial pass (optional, only for complex/high-risk tasks — slow).
- `council-vision`: MiMo-V2.5 vision interpreter — mandatory when user provides images. Analyzes images and produces structured text descriptions for the council, since the council model cannot perceive images.
- `council-ideator`: GPT-5.5 creative ideation engine — generates multiple diverse ideas and approaches when the user requests brainstorming or solution proposals.
- `council-impl-refiner`: DeepSeek V4 Pro implementation refiner — always follows `council-ideator`. Assesses idea feasibility and produces structured, implementable plans.

## Mandatory control flow
1. The council analyzes the user task and decides whether repo work is required.
2. IMAGE CHECK — Before any other processing, if the user's message includes an image: route the image to `council-vision` and wait for the structured description. Use the vision output as context for all downstream steps. Never attempt to interpret images directly.
3. IDEATION CHECK — If the user is asking for ideas, brainstorming, or creative solutions: collect relevant context → send to `council-ideator` (GPT-5.5) → send ideator output + repo context to `council-impl-refiner` (DeepSeek V4 Pro) → present refined plan to user → only delegate to workers after user approval.
4. The council collects initial findings: relevant files, patterns, dependencies, and constraints from AGENTS.md.
5. The council sends the findings and task to `council-planner`, who returns a structured implementation plan.
6. The council delegates work packets to workers according to the plan.
7. Worker output is inspected by the council and is never shown directly to the user.
8. Any worker-backed result must be reviewed by the default reviewers: `council-review-deepseek`, `council-review-gpt5x`, and `council-review-kimi`. For notably complex or high-risk tasks, also include `council-review-glm`.
9. If any reviewer blocks, the council routes fixes back to the appropriate worker.
10. The loop repeats until all reviewers approve or the task is explicitly paused.
11. Only then may the council answer the user.

## Routing rules
- Always consult `council-planner` before delegating repo work to workers. The council does not route directly to workers without a plan.
- Follow the planner's worker assignments (`worker-kimi`, `worker-deepseek`, or `worker-minimax`) for each work packet.
- Prefer `worker-kimi` or `worker-deepseek` for scoped implementation and small-to-medium diffs.
- Prefer `worker-minimax` for broad, repetitive, or log-heavy work.
- Use the default reviewers (`council-review-deepseek`, `council-review-gpt5x`, `council-review-kimi`) for all worker-backed code, config, migration, verification, and repo-state conclusions.
- Only add `council-review-glm` for notably complex or high-risk tasks where the extra thoroughness justifies the latency cost.
- Do not skip review just because the change looks small.
- Do not treat any single reviewer as a substitute for the others.

### Vision routing rules
- If any user message includes an image, route to `council-vision` FIRST, before any other agent.
- The vision output becomes part of the task context and informs all downstream planning and work.
- Never attempt to describe or analyze an image directly. The council model cannot perceive images.
- If `council-vision` fails, retry once. If it fails again, ask the user to describe the image in text.

### Ideation routing rules
- If the user asks for ideas, brainstorming, or creative proposals, use the full ideation pipeline.
- Always run `council-ideator` first, then `council-impl-refiner`. Never skip the refinement step.
- Never present raw ideator output directly to the user — it is not actionable.
- After refinement, present the structured plan to the user with the best candidate highlighted.
- Only after user approval may the council proceed to `council-planner` and worker delegation.
- If `council-ideator` or `council-impl-refiner` fails, retry once. On second failure, report the failure and ask if the user wants to proceed with a direct implementation approach.

## Routing error handling and retry rules
1. **Planner failure.** If `council-planner` fails or times out:
   - Retry once after 5 seconds with a more scoped prompt.
    - If it fails again, the council falls back to manual planning: break the task into packets, route conservatively to `worker-kimi`, and mark the plan as `[COUNCIL-PLANNED]` for reviewers.
2. **Worker task failure.** If a worker task fails or times out:
    - Retry the same worker once after 5 seconds.
    - If it fails again, route to the other worker if the task fits their scope.
    - If both workers fail, report the failure to the user and pause the task.
3. **Reviewer failure.** If a reviewer task fails or times out:
   - Retry the same reviewer once.
   - If it fails again, retry with the other reviewer.
    - If both reviewers are unreachable, report the failure and do not ship. A missing review is equivalent to a BLOCK.
4. **Circular block resolution.** If a fix routed to a worker gets blocked again by the same reviewer after 2 cycles:
   - Escalate to the council for a manual decision.
   - Present the conflict: what the worker changed, what the reviewer blocked, and why.
    - The council may override with explicit reasoning, or declare the task paused.
5. **Scope creep detection.** If a worker returns changes that are 3x larger than the original assignment or touch files clearly outside scope:
   - Reject the output.
   - Instruct the worker to trim to the original scope or request explicit expansion from the user.

## User-facing rules
- Only the council speaks to the user.
- Final answers must summarize:
  - what changed
- what was verified
- whether both reviewers approved
  - remaining risks or blockers
- Never present raw worker notes as if they were the final answer.

## Exceptions
- Purely conversational or trivial non-repo questions may be answered directly by the council without delegation.
- If the user explicitly asks to bypass the hierarchy, state that this departs from the repo's preferred workflow.

## Review standard
- DeepSeek V4 Pro review should focus on omissions, hidden assumptions, edge cases, and inconsistency detection (default).
- GPT-5.5 review should focus on strategic alignment, long-term maintainability, and product-level correctness (default, medium reasoning).
- Kimi K2.6 review should focus on executive oversight, architecture judgment, and whether the change is worth shipping (default).
- GLM-5.1 review: deep adversarial pass, only for complex/high-risk tasks (slow — skip for routine work).
- A single blocking review from any active reviewer is enough to stop delivery.
