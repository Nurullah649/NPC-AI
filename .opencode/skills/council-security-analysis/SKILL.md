---
name: council-security-analysis
description: Run security-focused analysis through the council workflow with blocking review for auth, data exposure, validation, and unsafe operations.
compatibility: opencode
metadata:
  area: security
  workflow: council-worker-review
---

## Purpose
- Use this for security review, auth checks, permission boundaries, secret handling, input validation, unsafe file or shell operations, and data exposure risks.
- This skill assumes the repo's multi-agent architecture is active. Load `council-orchestration` first or alongside this skill.

## Required workflow
1. Start read-heavy and evidence-driven.
2. Delegate broad search and trust-boundary mapping to a worker.
3. Treat security findings as severity-ordered and blocking until disproven.
4. If fixes are made, require both council reviewers before shipping.

## Trust-boundary mapping protocol
1. **Identify all entrypoints.** List every API endpoint, route, WebSocket handler, webhook, cron job, CLI command, and mobile screen that accepts external input.
2. **Map data flow.** Trace how user input travels from the entrypoint through validation, business logic, database, and back to the response.
3. **Mark trust boundaries.** Every time data crosses a boundary (user → API → DB → third-party service), check:
   - Is the input validated?
   - Is the caller authenticated?
   - Is the caller authorized for this action?
   - Is sensitive data encrypted or masked in transit and at rest?
4. **Check authentication coverage.** Use `grep` to find all route definitions. For each route, confirm whether it has an auth decorator or middleware. Unauthenticated routes are higher risk and must be explicitly justified.

## Input validation checklist
For every entrypoint that accepts user input, verify:
- [ ] **Type validation:** Is the input checked against expected types (string, int, enum, UUID)?
- [ ] **Length / range limits:** Are there upper and lower bounds?
- [ ] **Format validation:** Are emails, URLs, phone numbers, or regex patterns validated?
- [ ] **Sanitization:** Is user input escaped before rendering or insertion into SQL / NoSQL queries?
- [ ] **File upload checks:** Are file types, sizes, and contents validated? Is the upload directory outside the web root?
- [ ] **Rate limiting:** Are brute-force or abuse paths protected by rate limits?

## Secret and exposure scanning
1. **Search for hardcoded secrets.** Use `grep` for patterns like `password=`, `secret=`, `api_key`, `token=`, `private_key`, `BEGIN RSA PRIVATE KEY`.
2. **Check logging and error responses.** Verify that logs do not print tokens, passwords, or PII. Verify that error responses do not leak stack traces or internal paths in production.
3. **Check environment handling.** Ensure `.env` files and `secrets/` directories are in `.gitignore`. Ensure production secrets are not checked into source control.
4. **Check database queries.** Look for raw string interpolation in SQL or NoSQL queries. Prefer parameterized queries or ORM methods.

## Severity classification
- **CRITICAL:** Authentication bypass, remote code execution, SQL injection, secret leakage. Block immediately.
- **HIGH:** Missing authorization checks, insecure defaults, XSS, CSRF, file path traversal. Block unless disproven.
- **MEDIUM:** Missing input validation, verbose error messages, weak rate limiting. Fix before shipping if practical.
- **LOW:** Missing security headers, informational findings, defense-in-depth gaps. Document and schedule.

## Fix verification protocol
1. **Write a test that reproduces the vulnerability.** The test must fail before the fix and pass after.
2. **Verify the fix does not break legitimate use cases.** Run the full auth and API test suites.
3. **Re-scan the surface.** After fixing, re-run the same `grep` patterns or manual checks to ensure no similar issue exists nearby.
4. **If the fix is complex (e.g., new auth middleware), get both reviewers.** Security changes are high-stakes and must be reviewed by both GLM and GPT-5.5.

## Review focus
- authentication and authorization gaps
- insecure defaults and missing validation
- accidental secret exposure or unsafe logging
- replay, duplication, or tampering paths in networked flows
- dangerous bash, git, file, or deploy behaviors

## Expected final answer
- severity-ordered findings first
- exploitability or risk level
- fixes made or recommended
- verification limits and residual risk
