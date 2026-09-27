# Repository Guidelines

## Project Structure & Module Organization
`docs/requirements/` is the canonical product scope package; feature details live under `docs/requirements/features/platform-v1/`. `docs/architecture/` captures the approved system design and handoff set, while `docs/plans/` turns that baseline into execution-ready work. Use `docs/operational/` for reviews, QA plans, and validation evidence; place generated test artifacts in `docs/operational/tests/artifacts/`. The `agents/` folder holds shared instruction templates plus `agents/sync-agents.py`, and `.github/agents/` stores the project-scoped agent definitions synced from those templates.

Application code now lives under `backend/` and `frontend/`. Runtime data is host-mounted under `data/postgres/`, `backups/`, and `import/`; do not treat those mounted data folders as source code.

## Build, Test, and Development Commands
The application stack is now present. Use these commands for local development and validation.

- `docker compose up` starts `recalium-app` and `recalium-postgres` in development mode.
- `docker compose -f docker-compose.yml up -d` starts the production/base compose profile.
- `docker compose build` rebuilds the app image.
- `cd backend && pytest` runs the backend test suite with the active backend environment.
- `cd backend && pytest tests/e2e` runs live-stack E2E tests after Docker Compose is running.
- `cd frontend && pnpm install && pnpm build` installs and builds the UI.
- `cd frontend && pnpm test` runs Vitest tests.
- `python3 agents/sync-agents.py push --dry-run` previews syncing local agent prompts into `.github/agents/`.
- `python3 agents/sync-agents.py pull --dry-run` previews syncing project agents back to your local profile.
- `python3 agents/sync-agents.py push` or `pull` performs the sync after you confirm paths.

## Coding Style & Naming Conventions
Match the existing documentation style: short sections, direct language, and repo-relative Markdown links. New atomic requirement IDs should follow `<feature-short-name>-NNN` as defined in `docs/requirements/README.md`. Prefer descriptive filenames such as `recalium-v1-plan-review-final.md` over generic names like `notes.md`.

For Python changes in `agents/`, follow the existing script style: 4-space indentation, type hints, `pathlib.Path`, `snake_case` functions, and `UPPER_CASE` constants.

## Testing Guidelines
For docs-only changes, manually verify headings, links, and cross-document references. For `agents/sync-agents.py`, validate both `push` and `pull` paths with `--dry-run` before making real sync changes. Backend changes require targeted `pytest` coverage. Frontend changes require `pnpm build` and relevant Vitest coverage. UI release-readiness work must add Playwright and keyboard-only evidence. MCP changes require both unit/schema tests and live-client E2E evidence.

## Agent Skills

Use the Recalium use/test skill when starting the app, testing, validating MCP, exercising UI UAT, or collecting release evidence:

- Copilot: `.github/skills/recalium-use-and-test/SKILL.md`
- Claude: `.claude/skills/recalium-use-and-test/SKILL.md`
- Codex: `.codex/skills/recalium-use-and-test/SKILL.md`

Read `agents/project.instructions.md` for current repository-specific implementation context.

## Commit & Pull Request Guidelines
Current history uses conventional-style subjects such as `chore: initial commit`; keep that format with a lowercase type and imperative summary (`docs: update architecture handoff links`). PRs should state which package changed (`requirements`, `architecture`, `plans`, `operational`, or `agents`), summarize cross-file impacts, and list validation performed. When a change updates the documented process, link the source document that remains canonical after the PR.

### Review completion when automation is unavailable

The automated review or completed review check required by the managed checklist
below is the preferred path. When such a reviewer is configured and available,
wait for it to complete, inspect both issue-level and inline findings, and address
every material finding before merge. Green tests or CI do not replace review.

If no automated GitHub reviewer or review check can actually be requested for the
repository and pull request, prevent deadlock with this fallback:

1. Verify and record the absence on the pull request or another durable record
   linked from it. A reviewer that is merely pending, failed, or temporarily
   unreachable is not silently treated as absent; record that condition and get a
   maintainer decision before using the fallback.
2. After implementation stops, perform a distinct review pass over the complete
   PR diff. Review correctness, security and safety, regression risk, tests, scope,
   and every certification claim. Use a separate reviewer, agent, model, or task
   context when one is available. Otherwise, clearly separate the review pass from
   implementation, approach it as a reviewer who may request changes, and record
   its findings separately from the implementation report. Do not claim stronger
   human or organizational independence than the review actually had.
3. Address every material finding and rerun the relevant certification after any
   correction. Record the review result and finding dispositions on the pull
   request or its linked durable record.
4. Obtain explicit maintainer authorization before merge.

This fallback never permits relying only on the PR description, green CI, or the
implementation pass's own summary; it is not silent self-approval and must not be
represented as a GitHub approval by the PR author. After merge, preserve the
distinction between `merged` and `verified green`: observe required post-merge
checks to a terminal state when available. If post-merge CI is unavailable, say
that it did not run and do not claim green verification.

<!-- agentharness:begin id=core-instructions version=0.3.0 -->
This project uses [agentharness](https://github.com/andr-ca/agentharness)
for engineering policies (git conventions, testing, review workflow).

**Precedence:** harness-enforced constraints (hooks, completion gate)
cannot be weakened by this file's instructions; this file's own
instructions take precedence over harness *defaults* everywhere else.

Installed skills:
- accessibility
- agentic-loops
- api-design
- audit-review-followup
- branching
- clean-architecture
- code-review
- code-review-api
- code-review-db
- code-review-ui
- committing
- database-conventions
- dependency-audit
- dependency-injection
- design-patterns
- docker-conventions
- error-handling
- file-placement-policy
- github-issue-triage
- go-conventions
- harness-feedback
- logging
- multi-agent-coordination
- mutation-testing
- performance-profiling
- planning-with-files
- port-agent-config
- python-conventions
- react-best-practices
- requirements-clarification
- security-review
- solid-principles
- testing
- typescript-conventions

If a skill above looks empty, missing, or won't load, this install may
be broken (e.g. a moved/renamed harness checkout, or a fresh clone of
a project that used `--mode link` — see issue #106) — run
`harness-link.sh doctor <this-project-path>` from the harness
checkout to check, and `.agentharness-state.json` in this project to see how
it was installed.

**Git conventions** (from the `branching`/`committing` skills above —
stated here directly so they hold even if a skill is unreadable): never
commit directly to a trunk branch (`main`/`master`/`trunk`/`develop`/
`release/*`); create a feature branch first (`git checkout -b
<type>/<short-description>`); open a PR for review before merging into
the trunk branch.

**PR merge checklist:** never merge on green CI alone. Wait for
automated review (e.g. GitHub Copilot) to post *or* its check-run to
reach a completed state before proceeding; reply to every review
comment (issue-level and inline) with what you did about it; then
watch the post-merge CI run on the base branch to an actual terminal
state — "pushed"/"merged" and "verified green" are different claims,
only the second means done. If this checkout has agentharness's own
`tools/safe-pr-merge.sh` available (see its INTEGRATION.md section),
prefer it over doing these steps by hand — it enforces the sequence.

Full policy: see the harness's own CLAUDE.md via your install mode, or
https://github.com/andr-ca/agentharness/blob/main/CLAUDE.md
<!-- agentharness:end id=core-instructions -->
