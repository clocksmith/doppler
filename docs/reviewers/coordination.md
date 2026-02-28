# Reviewer Coordination Playbook

## Purpose
Define how Reviewers A/B/C/D work in parallel without overlap and with deterministic progress tracking.

## Global rules
1. One reviewer owns one file at a time from their shard.
2. A file is only reassigned after explicit owner confirmation.
3. Reviewers do not edit overlapping files, including tests or shared tracking files, unless explicitly coordinated.
4. The owning reviewer is responsible for all required tracking updates for each file they finish.
5. No runtime behavior change without explicit config/policy justification.

## Review sequence
1. Reviewer opens assigned file and determines:
   - whether a code change is required,
   - whether config-based replacement exists,
   - whether risk reduction is achieved without changes.
2. Reviewer applies change or records explicit no-code decision.
3. Reviewer updates:
   - status files in `docs/tracking/`
   - queue rows (`state`, `decision`, optional `plan`)
4. Reviewer logs a short handoff note in this format:
   - File: `<path>`
   - Decision: `<decision label>`
   - Rationale: `<short>`
   - Follow-up: `<none | deferred / requires coordination>`

## Tracking updates (required)
- `docs/tracking/execution-plane-review-queue.md`
- `docs/tracking/execution-plane-review-queue.js-runtime.md`
- `docs/tracking/execution-plane-audit.md`
- `docs/tracking/execution-plane-audit.json`
- `docs/tracking/execution-plane-audit.csv`
- `docs/tracking/execution-plane-review-queue.json`
- `docs/tracking/execution-plane-review-queue.js-runtime.json`

## Handoff and conflict protocol
- If a reviewer suspects another reviewer is already editing a file, pause and request confirmation before touching it.
- If behavior crosses shard boundaries, route dependency to primary owner first.
- For unresolved conflicts:
  1. pause change,
  2. send a concise note with file + finding + why overlap,
  3. wait for explicit reassignment.

## Completion criteria
- Reviewer stops only when all files in their assigned set are `reviewed` in both queue files.
- Final merge safety check: no tracked file left as `in-review` for more than one reviewer without handoff note.
