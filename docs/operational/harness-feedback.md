# Harness Feedback Log

Status: living document — append dated entries as issues are found while using
agentharness in this repo. Do not delete old entries; mark them resolved
in-place if a later harness version fixes them.

Purpose: this repo installed `agentharness-toolkit` (npm mode) on 2026-07-16
with `--with-hook` (trunk-protection). This log tracks real incidents where
the harness's guidance, hooks, or bootstrap either failed to prevent a mistake
or wasn't surfaced clearly enough to prevent one. Feed findings back upstream
(agentharness repo issues) when a pattern repeats or looks structural rather
than a one-off operator error — record the issue number back in this file
once filed, so the two stay linked.

**Upstream issues filed from this log:**
- [andr-ca/agentharness#76](https://github.com/andr-ca/agentharness/issues/76) — trunk-protection hook doesn't fire on merge commits (first entry below)
- [andr-ca/agentharness#77](https://github.com/andr-ca/agentharness/issues/77) — "give review time to post" mandate has no concrete threshold (second + third entries below)
- [andr-ca/agentharness#78](https://github.com/andr-ca/agentharness/issues/78) — no mechanism surfaces stale unaddressed review comments on pre-existing open PRs (fifth entry below)
- [andr-ca/agentharness#79](https://github.com/andr-ca/agentharness/issues/79) — feature request: an optional harness mechanism that *enforces* this exact monitor-log-file loop, instead of it happening only when a user asks (sixth entry below)
- [andr-ca/agentharness#88](https://github.com/andr-ca/agentharness/issues/88) — npm-mode installer leaves 32+ skill files and `.agentharness-pkg/` uncommitted with no signal (seventh entry below)
- [andr-ca/agentharness#149](https://github.com/andr-ca/agentharness/issues/149) — 0.3.0 `update` leaves `doctor` red: the pre-merge-commit hook is checked for but not shipped/installed (eighth entry below)
- [andr-ca/agentharness#154](https://github.com/andr-ca/agentharness/issues/154) — `multi-agent-coordination` skill's lock tool is never installed into consumer projects, so its protocol is entirely inert here (ninth entry below)

---

## 2026-07-17: Trunk-protection hook doesn't fire on merge commits

**What happened:** Three feature branches (built by parallel subagents in
worktrees) were integrated with `git merge --no-ff <branch>` run directly
while checked out on `main`. All three merge commits succeeded — no hook
fired, no warning, no block.

Minutes later, a plain `git commit` attempted directly on `main` (unrelated,
while testing something else) **was** correctly blocked by
`.agentharness-pkg/.github/hooks/prevent-trunk-commit` via the `pre-commit`
dispatcher. So the hook works — it just doesn't cover the merge path.

**Root cause:** git only invokes the plain `pre-commit` hook as a *fallback*
for merge commits when no `pre-merge-commit` hook file exists in
`core.hooksPath`. The harness's hook directory
(`.github/hooks/pre-commit`, `pre-push`, `prevent-trunk-commit`) has no
`pre-merge-commit` file, and in this environment the fallback did not
trigger for `git merge --no-ff` merge commits the way it does for `git
commit`. Net effect: the exact operation the hook exists to prevent
("don't commit directly to trunk — go through a PR") is fully reachable via
one extra git subcommand (`merge` instead of `commit`), with zero signal to
the agent that anything was bypassed.

**Impact this time:** Three merge commits landed on local `main` with no
CI run and no PR. Caught before anything was pushed to `origin` (a separate,
unrelated `git reset --soft HEAD~1` mistake while investigating a different
hook question surfaced the need to look closely at `main`'s history, which is
when this was noticed). Recovered by resetting local `main` back to its
pre-merge tip, pushing the three branches, and opening/merging real PRs
(#6, #7, #8) — see the second entry below for what still went wrong on the
"real" attempt.

**What agentharness should do:**
1. Ship a `pre-merge-commit` hook file (can just delegate to the same
   `prevent-trunk-commit` script) alongside the existing `pre-commit` in
   `.github/hooks/`, so `harness-link.sh init --with-hook` installs coverage
   for both commit paths, not just plain commits.
2. `tools/setup/harness-link.sh doctor` could detect this class of gap
   directly: check whether `core.hooksPath` contains a `pre-merge-commit`
   file, and warn if only `pre-commit` exists but the repo's real git version
   is one where the fallback isn't reliable (this seems to vary — worth
   testing across git versions rather than assuming the documented fallback
   always applies).
3. The CLAUDE.md router's "Agent Workflow Completion" section already has a
   strong, explicit mandate for the commit→push→PR path. It should say
   in as many words that **merging a branch into a trunk branch is a form of
   committing to it and is covered by the same stop-before-publish rule** —
   an agent skimming the router text could plausibly read "commit" narrowly
   and miss that a local `git merge` onto `main` is the exact thing the
   mandate is about.

---

## 2026-07-17: Merged 3 PRs on green CI alone, without checking for review comments

**What happened:** After redoing the above properly (pushed 3 branches,
opened PRs #6/#7/#8, watched each PR's pre-merge CI go green), I called
`gh pr merge --merge` on all three within seconds of CI passing — without
fetching or waiting for review comments (issue-level or inline), and without
polling the *post-merge* CI run on `main` before reporting anything.

This directly contradicts an explicit, already-installed mandate in this
repo's own `.agentharness-pkg/CLAUDE.md` (§"Agent Workflow Completion"):

> **Never merge a PR on CI status alone — wait for review comments, then
> address them, before merging.**

and

> **Never report a push/merge as done while CI is still running or red...
> A merge to `main` is not finished until `main`'s own resulting CI run (the
> run the merge commit itself triggers, not just the PR's pre-merge run) is
> confirmed green.**

**Root cause — this one is not the harness's fault.** The router file with
this exact guidance was installed in `.agentharness-pkg/CLAUDE.md` the day
before and was available the whole time; it was simply never read before
performing the merge. This is an operator/agent-behavior gap, not a tooling
gap: the harness gave the right instruction and I didn't consult it before
taking the action it governs.

**Actual outcome (checked after the fact, not before):** No harm resulted —
`gh api repos/.../pulls/{6,7,8}/comments` and `gh pr view --json comments`
both came back empty for all three PRs (no automated reviewer like GitHub
Copilot Code Review is configured on this repo, and no human had time to
comment), and `main`'s post-merge CI run (the one triggered by the #8 merge
commit, not #6/#7's — those got auto-cancelled by the next push, exactly as
the mandate's point 5 warns) came back `success` when polled retroactively.
But "no harm resulted" was luck, not verification — the mandate exists
precisely so this isn't left to luck.

**What agentharness should do:**
1. Nothing new needed in the router text itself — the existing mandate
   already says exactly the right thing. The gap is discovery, not content.
2. Consider whether `tools/check-completion.sh` (the Stop-hook completion
   gate) could be extended with a *pre-merge* check specifically for
   `gh pr merge` calls — e.g. a thin wrapper script
   (`tools/safe-pr-merge.sh <n>`) that fetches both comment types, checks
   the PR's own CI status, and only then calls `gh pr merge`, refusing with
   a clear message if comments haven't been fetched yet or CI isn't green.
   Baking the checklist into a script closes the "I forgot to read the
   router" failure mode structurally instead of relying on the agent
   re-deriving the mandate from memory every time.
3. `harness-link.sh init` (or `doctor`) could print a one-line pointer to
   the completion-mandate section the first time it detects a session is
   about to do multi-PR work (hard to detect generically, but worth a
   thought) — or at minimum, this reinforces that the router file should be
   *re-read*, not just read once at session start, before any push/PR/merge
   action, since its content is exactly the kind of thing that's easy to
   skim past days after installation.

**Corrective action taken this session:** confirmed (retroactively) zero
review comments existed on #6/#7/#8, and confirmed `main`'s post-merge CI run
(triggered by the #8 merge commit) completed with `conclusion: success`
before reporting the work as done.

---

## 2026-07-17: "Give automated review time to post" has no concrete threshold — repeated the mistake on the very next PR

**What happened:** Immediately after documenting the incident above, I opened
PR #9 (to add this very document) and — within roughly 10-20 seconds of its
CI turning green — checked for review comments once and merged. Same gap,
same session, on the PR *about* the gap. The user caught it: "no comments
because you didn't give enough time for reviewer."

**Root cause:** the mandate's step 1 ("give automated review time to post...
don't merge the instant CI turns green") names the right principle but gives
no concrete threshold — no wait duration, no poll loop, no signal to check
for *whether a reviewer is even configured* before deciding how long "enough
time" is. Checking `gh pr view --json comments` (issue-level) plus
`gh api repos/<owner>/<repo>/pulls/<n>/comments` (inline) once, immediately,
satisfies the letter of "fetch both comment types" (step 2) while
completely missing the intent of step 1. The two steps read as sequential in the mandate but
nothing enforces that step 1 actually elapsed before step 2 runs — an agent
under time pressure will naturally collapse them into one check.

**What agentharness should do:**
1. Give step 1 a concrete default (e.g. "poll for new checks/comments every
   30s for up to 5 minutes after CI goes green, or until a bot/reviewer
   check-run appears in the PR's check suite — whichever comes first") instead of
   the qualitative "give it time." A number an agent can literally implement
   closes this gap; a principle it can rationalize around does not.
2. Distinguish two cases the router currently conflates: "no automated
   reviewer is configured on this repo at all" (verifiable up front via
   `gh api repos/<owner>/<repo>/branches/<default>/protection` or by checking
   whether a Copilot/other review check-run ever appears in `gh pr checks`
   history) vs. "a reviewer is configured but hasn't run yet." The wait only
   matters in the second case — an agent that confirms the first case can
   reasonably skip the wait, but must say so explicitly rather than silently
   checking once and moving on.
3. This is a good candidate for the same `tools/safe-pr-merge.sh` wrapper
   idea from the entry above — bake the poll loop and the
   reviewer-configured check into the script so "did I wait long enough" stops
   being a judgment call made under time pressure.

**Corrective action taken this session:** none yet — PR #9 (this file's own
addition) was already merged before the user's correction landed. No comments
appeared on any of #6-#9 even after the fact, and post-merge CI was confirmed
green on each, so no undetected regression is suspected — but the *process*
of checking too early is the finding, independent of that outcome.

Filed upstream as
[andr-ca/agentharness#77](https://github.com/andr-ca/agentharness/issues/77)
(combined with the "merge on CI alone" entry above — same mandate, same
missing threshold).

**Update (2026-09-27) — repository-local fallback proposed:** PR #1 later
reproduced the unresolved case in a fork with no available automated reviewer
or review check. The implementation needed an actual diff review, a corrective
commit, updated certification, and a one-off maintainer exception before merge;
the managed instruction still supplied no durable path for future PRs. A paid
reviewer was unnecessary for the project's needs, while the evaluated third-party
review app requested fixed repository-code and workflow write permissions broader
than a review-only gate required. The proposed local policy in `AGENTS.md` keeps
configured automated review as the preferred path and adds a provider-neutral
fallback: verify and record the absence, conduct and record a distinct review of
the complete diff, address material findings, rerun relevant certification, and
obtain explicit maintainer authorization. It also preserves the distinction
between a completed merge and post-merge verification that actually ran. This is
a local resolution linked to existing issue #77, not a claim that the upstream
agentharness default has changed.

---

## 2026-07-17: Request — a built-in, optional "harness feedback" mechanism

**What happened:** the three entries above were all produced by hand, ad hoc,
because the user asked for a feedback doc and then asked for upstream issues
— there is no harness-provided command, skill, or script that does either
step. Filing the two issues above required manually gathering the writeup,
re-deriving the right level of detail, linking back to this repo's commits,
and remembering to record the resulting issue numbers back in this file
(easy to skip, since nothing enforces it).

**Request (not yet a fix, no code proposed here):** a built-in, **opt-in**
mechanism — a `tools/harness-feedback.sh` script, a skill, or a
`harness-link.sh feedback` subcommand — that does two things on request:
1. Appends a dated entry to the consuming repo's own feedback log
   (`docs/operational/harness-feedback.md` or equivalent), using a standard
   template (what happened → root cause → impact → what agentharness should
   do → corrective action taken) instead of an agent re-deriving that
   structure from scratch each time.
2. Optionally (must default off — not every consumer wants an agent auto-filing
   issues on a repo it doesn't own) files the same finding as a GitHub issue
   on `andr-ca/agentharness`, automatically including: a link back to the
   consuming repo, the relevant commit/PR, and the full explanation — then
   writes the resulting issue number back into the local log entry so the
   two stay linked without a manual round-trip.

**Why this matters:** without it, every consuming project reinvents this
loop from scratch (or, more likely, never does it at all) — the exact
"N drifted copies" problem agentharness exists to solve for conventions
applies just as much to *feedback about the conventions themselves*.

**Corrective action taken this session:** none (this is a feature request,
not an incident). Filed as part of
[andr-ca/agentharness#77](https://github.com/andr-ca/agentharness/issues/77)'s
"separate, smaller suggestion" section rather than its own issue, since it's
a small addendum to the same discussion rather than a standalone bug.

---

## 2026-07-17: A pre-existing PR sat 5 days with unaddressed, security-relevant review comments — nothing surfaced it

**What happened:** while auditing PR #5 (`fix/gpt5.6-p0-backlog-closure`, a
large pre-existing 21-commit PR, open since 2026-07-12) before merging it,
`gh api repos/<owner>/<repo>/pulls/5/comments` turned up 8 inline Copilot
review comments dated 2026-07-12/13 — several substantive, including a real
gap in the PR's own flagship deletion-safety fix (the bundle-tombstone path
doesn't crypto-erase plaintext the way the regular delete path does,
undermining the PR's stated guarantee) and a transaction-scoping bug. None
of these had been read, replied to, or acted on in the five days the PR sat
open. Nothing about the harness, CI, or GitHub surfaced this proactively —
it was found only because a session happened to audit the PR for an
unrelated reason (chasing down why two *other* PRs' code assumed a bundle
format this PR was the one actually implementing).

**Root cause:** the completion-gate mandate ("never merge on CI alone —
check review comments") only fires at the moment of merging a PR *you just
opened*. There is no mechanism — hook, doctor check, or otherwise — that
periodically surfaces PRs with stale unaddressed review comments sitting on
the repo, especially ones opened days or weeks earlier by a previous
session. An agent that never happens to revisit an old PR has no signal
that security-relevant findings are waiting on it.

**Impact:** a real gap in deletion-safety guarantees sat undiscovered and
unfixed for 5 days on a PR whose entire purpose was closing deletion-safety
findings — the opposite of what the PR was for.

**What agentharness should do:**
1. `harness-link.sh doctor` (or a new `harness-link.sh audit-prs`) could
   list open PRs with review comments newer than the PR's last commit —
   a simple, mechanical staleness signal independent of remembering to
   check any individual PR.
2. More generally: the router's "never merge on CI alone" mandate is framed
   entirely around PRs an agent is actively finishing. It should also say
   that starting or resuming work in a repo is a good trigger to check
   `gh pr list` for any open PR with unaddressed review comments older than
   a day or two, not just the PR currently being merged.

**Corrective action taken this session:** the 3 substantive findings (crypto-erase gap, transaction scoping, backup-path edge case) are being triaged before merging PR #5, per the user's direction — see the PR itself for the resolution. Filed upstream as
[andr-ca/agentharness#78](https://github.com/andr-ca/agentharness/issues/78).

---

## 2026-07-17: Request — an optional harness mechanism that *enforces* this monitor-log-file loop

**What happened:** this file's own "Standing instruction" section (below) tells
future sessions to log harness friction locally *and* file it upstream, every
time, without being asked. But that instruction only exists in this repo's
`CLAUDE.md` because the user asked for it explicitly, after already having to
ask twice: once for the local doc (after the trunk-hook/merge-on-CI incidents),
and again as a correction when the doc-only version missed the "also file it
upstream" half of the loop. Nothing in the harness itself required or
scaffolded any of this — every step so far has been reactive to a direct user
ask, not something the harness made happen on its own.

**Root cause:** the harness currently treats "give agents the right
instructions" as sufficient. It isn't, on its own — an instruction living in a
CLAUDE.md router competes with everything else the agent is doing under time
pressure (see the entries above: an agent can read the exact right mandate and
still not apply it in the moment). There is no *mechanism* — hook, completion
check, or skill — that enforces the monitor→log→file loop the way, say,
`prevent-trunk-commit` enforces trunk protection or `check-completion.sh`
enforces the lint/test/coverage gate. Feedback-about-the-harness is currently
the one category of "important standing behavior" left entirely to prose.

**What agentharness should do:** filed as a standalone feature request,
[andr-ca/agentharness#79](https://github.com/andr-ca/agentharness/issues/79)
(a fuller version of the smaller suggestion already in #77) — an **opt-in**
mechanism that: (1) makes noticing harness friction the default behavior for
qualifying events rather than something only a direct ask triggers, (2) logs
it locally with a standard template, (3) files it upstream with the
originating repo/project, full context, and a concrete recommendation always
included structurally (not left to the filing agent's judgment each time),
and (4) keeps the local entry and the upstream issue linked in both
directions.

**Corrective action taken this session:** filed
[andr-ca/agentharness#79](https://github.com/andr-ca/agentharness/issues/79)
per the user's explicit request, since the earlier informal mention inside
#77 wasn't enough — it needed to be its own tracked, standalone item.

---

## 2026-07-17: npm-mode installer left 32+ skill files and `.agentharness-pkg/` uncommitted, with no signal anything was wrong

**What happened:** while auditing `.claude/skills/` in an unrelated session, `git ls-files .claude/skills/` returned only 2 tracked file paths (`.claude/skills/recalium-memory/SKILL.md` and `.claude/skills/recalium-use-and-test/SKILL.md` — this project's own hand-authored skills, added separately), meaning only those 2 of the 34 skill directories had any tracked content at all. All 32 of the other skills the harness
installer wrote (`accessibility`, `agentic-loops`, `api-design`, ... `testing`,
`typescript-conventions`) exist on disk — real, substantial content, not stubs — but were
never committed. `git log --all -- .claude/skills/` shows no commit ever added them.
`.gitignore` does not exclude the directory; this isn't an intentional exclusion, the files
are just sitting untracked. The harness's own bootstrap package, `.agentharness-pkg/`
(`AGENTS.md`, `bin/`), is in the identical state — completely untracked, zero files in
`git ls-files .agentharness-pkg/`.

`.agentharness-state.json` confirms the install ran in npm mode on 2026-07-17T03:57:51Z. This
went unnoticed through multiple sessions of actively using several of the installed skills —
discovered only by accident, running `git ls-files` for an unrelated reason.

**Root cause:** the npm-mode installer writes its skill bundle and package directory directly
to the working tree, but nothing in the install flow stages, commits, or warns that a fresh
install leaves 30+ new files outside version control. Unlike `node_modules/`-style installed
content (expected and gitignored), these files are meant to be part of the consuming repo —
they're read and used by agents working in it — so leaving them untracked means they're
invisible to any other contributor, invisible in PR/blame history, and at risk of silent loss
if the local checkout is ever wiped, with nothing marking them as repo content worth
protecting.

**Impact:** 32 real, working skill files existed only on one machine's local disk — not in
repository history, not visible to any other contributor or CI run, not part of any
reviewable PR — for the entire life of the install until this session found it by chance.

**What agentharness should do:**
1. After a fresh npm-mode install, either auto-stage the new files and print a clear "N new
   files staged — commit these to make the skills part of your repo" message, or explicitly
   prompt to commit.
2. Decide and document the intended status of `.agentharness-pkg/` — if it's meant to behave
   like a local-only cache, the installer should gitignore it automatically so the untracked
   state reads as intentional rather than indistinguishable from "forgot to commit." If it's
   meant to travel with the repo (so the pinned harness revision is reproducible), include it
   in the same staging/prompt step as the skills.
3. `harness-link.sh doctor` could detect this class of drift directly: for each skill listed
   in `.agentharness-state.json`, check `git ls-files --error-unmatch <path>` and warn if an
   installed skill exists only in the working tree.

**Corrective action taken this session:** filed upstream as
[andr-ca/agentharness#88](https://github.com/andr-ca/agentharness/issues/88). Not yet
decided/committed: whether to commit the 32 skill directories as-is in this repo, and how to
handle `.agentharness-pkg/` — deferred pending that issue's resolution, since committing
`.agentharness-pkg/` without knowing its intended status could itself be wrong.

**Update (2026-07-18) — maintainer verified and partially corrected the report:**
[andr-ca/agentharness#88](https://github.com/andr-ca/agentharness/issues/88#issuecomment)
confirmed the core finding by actually running `harness-link.sh init --mode npm` +
`doctor` in a scratch repo (not just reading code), and found it's **worse and broader**
than filed: `cmd_init` in npm mode leaves everything it writes untracked — not just
`.claude/skills/`, but also `.agents/skills/`, `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`,
`.github/`, and both state-json files — and `cmd_doctor` reports "all checks passed" on
that exact untracked state, since it never checks git-tracked status. That's a false-green
signal on a repo that would lose every installed skill on a fresh clone.

However, **one half of this report was wrong on current `main`**: the `.agentharness-pkg/`
gitignore ambiguity (ask #2) is already fixed upstream —
`.github/.gitignore.template:212` has carried `.agentharness-pkg/` since commit `8ab1478`,
the same commit that introduced npm mode. A live test confirmed a fresh npm-mode init does
not show `.agentharness-pkg/` as untracked. This repo's `.agentharness-state.json` records
harness revision `0.2.1` and our own `.gitignore` has no `.agentharness-pkg/` entry at
all — we were on a stale pin relative to that fix, not observing a live upstream bug.
Added the missing `.gitignore` entry directly in this repo (no need to wait for a harness
version bump to fix our own state) rather than leaving the report's now-inaccurate half
uncorrected here.

**Second correction (2026-07-19) — the "32 files" were never independent content:** while
deciding whether to commit the 32 untracked skill directories per the plan above, `readlink`
on all 32 showed every one is a **symlink** into `.agentharness-pkg/.claude/skills/<name>` —
only this project's own 2 hand-authored skills (`recalium-memory`, `recalium-use-and-test`)
are real, independent directories. This narrows the original report: nothing unique is at
risk of loss from these 32 staying untracked, since re-running `harness-link.sh init
--mode npm` on a fresh clone recreates identical (or newer-pinned) symlinks, as long as
`init` places `.agentharness-pkg/` at the same repo-relative path every time. The real gap
is narrower than "32 real files could be silently lost" — it's "a fresh clone has zero
working skills until someone remembers to re-run `init`, with no signal that anything is
missing," which the maintainer's broader `cmd_init`/`doctor` finding above already
substantially covers. Posted this correction, plus an open design question (should these
symlinks be committed, so `doctor` can detect a dangling-symlink state specifically instead
of the current total silence?), as a follow-up comment on
[andr-ca/agentharness#88](https://github.com/andr-ca/agentharness/issues/88). **Decided:
do not commit `.claude/skills/*` in this repo** — there's nothing there to commit that
isn't already either tracked (the 2 real skills) or regenerable via `init`.

---

## 2026-07-22: 0.3.0 `update` leaves `doctor` red — pre-merge-commit hook checked but not shipped/installed

**What happened:** Updated `agentharness-toolkit` 0.2.1 → 0.3.0 via
`npx agentharness-toolkit@latest update <project>` (npm mode). The update
succeeded (added `github-issue-triage` + `harness-feedback` skills, re-synced
34 skills, merged `.gitignore`), but the post-update `harness-link.sh doctor`
**exited 1**:

    ✗ pre-commit hook exists but pre-merge-commit is missing — merge commits to
      trunk branches may bypass protection (see issue #76 for details)

**Root cause:** 0.3.0 implemented the *detection* half of the closed #76 (doctor
now checks for a `pre-merge-commit` hook) but not the *fix* half — the package
still ships only `pre-commit`/`pre-push`/`prevent-trunk-commit` in
`.github/hooks/`, and `update` "never touches hooks". Since `core.hooksPath`
points straight at the shipped dir, an npm-mode upgrade lands on a doctor check
for a hook the harness neither ships nor installs.

**Impact:** `doctor` (documented as a CI-usable gate) fails immediately after a
clean upgrade, with no CLI-performed remediation. The 5 `✗ MISSING: tools/*`
lines under `audit` are unrelated and expected (the npm package intentionally
ships only `tools/setup/harness-link.sh`); `audit` itself exited 0.

**What agentharness should change:** Ship `.github/hooks/pre-merge-commit`
(delegating to `prevent-trunk-commit`, mirroring the `pre-commit` dispatcher) so
fresh installs pass doctor, and make `update`/`init --with-hook` self-heal the
hook on upgrade — or have doctor's message name the exact remediation command.

**Corrective action taken:** Hand-created
`.agentharness-pkg/.github/hooks/pre-merge-commit` delegating to
`prevent-trunk-commit`; `doctor` then passed (`✓ both pre-commit and
pre-merge-commit hooks present`, exit 0). That file lives in the gitignored
durable copy and will be wiped on the next `update`, so the durable fix must
land upstream. Logged upstream as
[andr-ca/agentharness#149](https://github.com/andr-ca/agentharness/issues/149)
(references the closed #76).

---

## Standing instruction (updated 2026-07-17)

Per user request, this repo's `CLAUDE.md` now includes a standing instruction
to treat harness friction as a first-class finding, **and to always do both
of the following, together, not as separate optional steps**:

1. Add a dated entry to this file **before** ending the session, using the
   entries above as the template (what happened → root cause → impact →
   what agentharness should do → corrective action taken).
2. File the same finding as a GitHub issue on
   [andr-ca/agentharness](https://github.com/andr-ca/agentharness/issues),
   linking back to this file's entry, the consuming repo, and the relevant
   commit/PR — and record the resulting issue number back in this file's
   entry so the two stay linked (see the "Upstream issues filed from this
   log" list at the top of this file).

Do not wait to be asked, and do not do only one of the two — a doc entry
with no upstream issue never reaches the harness maintainers; an issue with
no local entry loses the project-specific context of what happened here.
This applies whenever a session hits a gap, ambiguity, or near-miss
involving `agentharness` (hooks not firing as expected, unclear router
guidance, a mandate that was hard to find or apply, a bootstrap step that
didn't do what its output claimed, or a process gap like the one just above
this section).

---

## 2026-07-23: multi-agent-coordination lock tool is never installed here, so two sessions collided undetected

**What happened:** Mid-session, a routine `git status` on the primary checkout
(`/home/andrey/projects/recalium`, not a worktree) showed it wasn't in the
state this session had left it: a different branch checked out
(`docs/harness-feedback-0.3.0-doctor`, later `fix/theme-tokens-readability`),
new merged PRs (#28 through #32) this session never made, and uncommitted
edits to `.gitignore`/`AGENTS.md`/`CLAUDE.md`/`GEMINI.md`/
`frontend/src/tailwind.css` sitting in the working tree. Checking again minutes
later showed further movement (more merged PRs, more uncommitted changes) —
confirming a second, independent agent session was actively working the same
checkout concurrently. Nothing from the harness surfaced this; it was found by
accident via routine `git status`, not by any coordination mechanism.

**Root cause:** This checkout has `.claude/skills/multi-agent-coordination`
present on disk (part of the same untracked skill-sync pile documented in the
#88 entry above — it is not committed to the repo; `git ls-files` shows only
`recalium-memory` and `recalium-use-and-test` as tracked skills), and it
documents a lock protocol — `tools/agent-lock.sh check` / `acquire` /
`release`, a `.agentharness-locks/` directory, and a CLAUDE.md-level mandate
describing `pre-push` enforcement via `check-branch`. None of that exists
here: `tools/agent-lock.sh` is not present in this repo, and
`.agentharness-locks/` was never created. The skill file itself carries a
warning banner acknowledging `harness-link.sh` never installs the tool into
consumer projects — but that caveat is easy to miss (it precedes the
protocol's own "Enforcement" section, which reads as a hard guarantee), and
nothing at bootstrap time (`doctor`, `audit`) flags that the coordination
skill is present without its supporting tool.

**Impact:** For every consumer install (not just this repo), the entire
multi-agent-coordination protocol is decorative. Two sessions can work the
same primary checkout — not even separate worktrees — with zero lock check,
zero `pre-push` block, and zero warning, exactly as happened here. The only
reason this was caught was an incidental `git status`; nothing would have
surfaced it otherwise, and a less careful pass could have raced a `git
checkout`/`pull`/commit against the other session's in-progress, uncommitted
work.

**What agentharness should do:** See
[andr-ca/agentharness#154](https://github.com/andr-ca/agentharness/issues/154)
for the full writeup and recommended options: ship the lock tool into
consumer installs, or clearly downgrade the CLAUDE.md enforcement language to
match reality, or (minimum) have `doctor` flag a repo that has the
coordination skill installed but not the tool it depends on.

**Corrective action taken:** Did not touch the shared primary checkout at all
once the collision was suspected — no `checkout`, `pull`, `commit`, or `push`
against it. All follow-up work (this log entry included) was done in a
separate `git worktree` off `origin/main`, per the skill's own "Option B" for
when a feature is contended. Asked the user directly whether the other
session was expected/theirs before resuming any shared-checkout work.
