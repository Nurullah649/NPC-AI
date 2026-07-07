---
name: council-optimization
description: Run performance and complexity optimization tasks through the council workflow with measurement, minimal diffs, and regression review.
compatibility: opencode
metadata:
  area: optimization
  workflow: council-worker-review
---

## Purpose
- Use this for performance work, render slowness, heavy queries, build inefficiency, repeated work, memory waste, or reducing unnecessary complexity.
- This skill assumes the repo's multi-agent architecture is active. Load `council-orchestration` first or alongside this skill.

## Required workflow
1. Measure or inspect the current bottleneck before changing code.
2. Delegate profiling, broad search, or repetitive cleanup to the appropriate worker.
3. Prefer the smallest change with measurable benefit.
4. Require both council reviewers before presenting optimization results.

## Measurement protocol
1. **Identify the metric.** Define what "slow" means in numbers: response time (ms), build time (s), query duration (ms), memory usage (MB), FPS, bundle size (KB).
2. **Capture a baseline.** Run the same measurement 3 times and take the median. Record the exact command, environment, and data size used.
3. **Use the right tool for the stack:**
   - **Python backend:** `pytest-benchmark`, `cProfile`, `line_profiler`, `time` module.
   - **React / web:** Lighthouse, Chrome DevTools Performance tab, `webpack-bundle-analyzer`.
   - **Database:** Query `EXPLAIN ANALYZE`, slow query logs, `pg_stat_statements`.
   - **Build / CI:** `time npm run build`, `time docker build`, build logs.
   - **Mobile:** Metro bundler profiling, Flipper, Xcode Instruments (on macOS).
4. **Locate the bottleneck before optimizing.** Do not optimize by guesswork. If profiling is blocked (e.g., no test data, no access to production), report the blocker and do not proceed.
5. **Hypothesize one bottleneck at a time.** Do not change 5 things at once. Change one thing, measure again.

## Optimization rules
- Do not optimize by guesswork alone when evidence can be collected.
- Avoid readability regressions unless the gain is clear.
- Distinguish hot-path fixes from speculative cleanup.
- Report tradeoffs honestly, including when an optimization was not worth shipping.
- **Complexity budget:** If an optimization adds significant complexity (new dependencies, async refactoring, caching layers), the improvement must be at least 20% on the target metric or it is likely not worth it.

## Verification protocol
1. **Re-run the same measurement after the change.** Use the identical command, data, and environment as the baseline.
2. **Compute the delta.** Report before / after / percentage improvement.
3. **Run regression tests.** Optimizations must not break existing tests. Run the full module test suite.
4. **Check for secondary regressions.**
   - If you added caching, check memory usage.
   - If you parallelized work, check for race conditions.
   - If you changed a data structure, check for correctness on edge cases.
5. **If improvement is <10%, question shipping.** The complexity may outweigh the benefit. Present the tradeoff to the council.
6. **Document the optimization.** Add a comment or note explaining why the change was made and what metric it improves. Future maintainers need context.

## Expected final answer
- bottleneck identified
- evidence or measurement used
- changes made
- verification run
- tradeoffs and residual risks
