---
name: council-debug
description: Run debugging tasks through the council workflow with reproduction, root-cause isolation, and fix verification discipline.
compatibility: opencode
metadata:
  area: debugging
  workflow: council-worker-review
---

## Purpose
- Use this for bug reports, runtime errors, flaky behavior, broken commands, lifecycle inconsistencies, and "why is this happening?" tasks.
- This skill assumes the repo's multi-agent architecture is active. Load `council-orchestration` first or alongside this skill.

## Required workflow
1. Reproduce or gather the strongest available evidence first.
2. Delegate evidence collection and code-path tracing to a worker.
3. Separate symptoms, root cause, and fix proposal.
4. After any code change, require both council reviewers before answering.

## Reproduction protocol
1. **Check existing tests.** If a test already catches the bug, run it first and capture the full failure output (stdout + stderr + traceback).
2. **Write a minimal reproducer.** If no test exists, create the smallest possible script or command that demonstrates the bug before touching source code.
3. **Gather multiple samples for intermittent bugs.** If the bug is flaky, collect at least 3 failure samples or run the suspect code in a loop until the pattern is clear.
4. **Confirm environment assumptions.** Check versions, env vars, DB state, and config files. A bug that "only happens locally" is still a real bug.
5. **Do not propose a fix until reproduction is confirmed.** If reproduction fails after 2 honest attempts, report the strongest narrowing evidence and the missing prerequisite.

## Root-cause isolation protocol
1. **Trace from symptom to source.** Start at the error message or failing test, then trace backward through the call stack.
2. **Use `grep` to find entrypoints.** Search for the error string, function name, or relevant log message to narrow the surface.
3. **Check recent changes.** Use `git log --oneline -20` and `git diff HEAD~N` to see what changed recently around the failure.
4. **Validate invariants.** Assert the assumptions the code relies on (e.g., "this list is never empty", "this function always returns a string"). If an invariant is violated, that is the root cause.
5. **Distinguish types of bugs:**
   - **Deterministic bugs:** Reproduce 100% of the time. Fix the logic.
   - **Environment/setup bugs:** Reproduce only in certain configs. Fix the setup or guard against the bad state.
   - **Race conditions / timing bugs:** Flaky or load-dependent. Add synchronization, retries, or logging.
   - **Data-dependent bugs:** Triggered by specific input. Add validation and targeted tests.

## Fix verification protocol
1. **Run the reproducer after the fix.** The reproducer must pass or the error must disappear.
2. **Run the full relevant test suite.** `pytest <module>` or `npm test -- <pattern>`. Do not rely on a single test.
3. **Check for regressions.** If the fix touches shared code, run broader tests that exercise the shared surface.
4. **Verify edge cases.** If the bug was an off-by-one or null handling, add explicit assertions for the boundary.
5. **If verification fails, stop.** Do not ship a partial fix. Report the new failure and iterate.

## Debugging priorities
- Prefer the smallest reproducible path over broad speculation.
- Validate entrypoints, environment assumptions, and lifecycle transitions.
- Distinguish between deterministic bugs, environment/setup issues, and unverified hypotheses.
- If the bug cannot be reproduced, return the strongest narrowing evidence and the missing prerequisite.

## Expected final answer
- reproduction status
- root cause or best-supported hypothesis
- files changed, if any
- verification run
- remaining unknowns
