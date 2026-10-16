# Generation baseline tooling and staged cutover

PR 1 provides operator tooling only. It does not enable R2, completed-job
reprocessing, restore, rebuild, or Slice 2. Publication and deployment of this
tooling do not authorize its production execution. No new schema is required.

## Commands and evidence

From `backend`, using the existing environment and separately authorized database
connection, invoke `python -m app.domain.derivations.cutover_cli`:

```text
plan                 # read-only planning; missing eligible baselines permitted
prepare --plan FILE  # OFF-only mutation; repeat same command to resume
check                # read-only full-population coverage
check --cutover      # coverage + this must be the only PG client in this database
```

### Running app, R2 OFF: plan / prepare / ordinary check

From the correct Compose project directory, with `recalium-app` already running
and only under the relevant separate production authorization:

```text
docker compose exec recalium-app python -m app.domain.derivations.cutover_cli plan
docker compose exec recalium-app python -m app.domain.derivations.cutover_cli prepare --plan /tmp/recalium-baseline-plan.json
docker compose exec recalium-app python -m app.domain.derivations.cutover_cli check
```

`exec` runs only the one-shot tool inside the existing container, not another app
entrypoint/worker. Prepare still needs its separate mutation authorization and
the reviewed plan file must already be readable at the specified container path.
For machine-readable capture, disable the pseudo-TTY so JSON stdout stays separate
from human-readable stderr; stage the artifact only in the authorized preparation
gate, for example:

```text
docker compose exec -T recalium-app python -m app.domain.derivations.cutover_cli plan > baseline-plan.json
docker compose cp baseline-plan.json recalium-app:/tmp/recalium-baseline-plan.json
```

Use a new output filename; never overwrite source/configuration or prior evidence.
An ordinary `check` while the app is running is coverage evidence, not the stopped
app's quiescent cutover certificate.

### STOPPED app: cutover certificate only

After separately authorized maintenance/STOP, with Postgres already running,
use exactly this one-shot invocation:

```text
docker compose run --rm --no-deps --entrypoint python recalium-app -m app.domain.derivations.cutover_cli check --cutover
```

- `docker compose exec` cannot execute against a stopped service/container.
- Plain `docker compose run recalium-app ...` is unsafe: it uses the service
  entrypoint, which runs Alembic and starts Uvicorn; app lifespan starts the
  in-process worker (and other app tasks). That would break the barrier.
- `--entrypoint python` deliberately bypasses that entire startup path. It runs
  only the CLI: no Alembic execution, Uvicorn, app lifespan or worker.
- `--no-deps` must not start dependencies. It requires the existing authorized
  Postgres service to be up; it does not authorize starting/recreating it.
- Run with the **same DB role as the application**, using the same reviewed
  Compose environment/connection, so `pg_stat_activity` visibility is adequate.
  Do not substitute a restricted observer role or override credentials to make
  the certificate pass.

The Postgres `pg_isready` healthcheck may transiently appear as another DB client
and cause `other_database_clients`. This is a **false fail, never a false pass**.
Rerun the full certificate after that transient connection disappears. Do not
weaken the quiescence predicate, ignore clients or accept the failed certificate
to accommodate the healthcheck. Keep the app/normal clients stopped/quiet.

These commands are documentation only in the implementation/review gate; do not
execute them in production without the corresponding separate authorization.

JSON is written to stdout, human-readable evidence to stderr. Save stdout to a
new plan/certificate file, never over an existing source/configuration file. Exit
0 means the requested operation passed, 2 means blocked/failed. Unexpected errors
produce a static code only, never a database error/URL/credential dump.

Reports contain IDs, source/control hashes, static classes/reasons, counts,
software hash, deployed/current Alembic revision, PostgreSQL version and snapshot
metadata. No source text, prompts, provider responses, job errors, or secrets.
Evidence is capped at 10,000 archives / 16 MiB; exceeding a cap fails closed and
requires a separately reviewed capacity extension, not cursor-only certification.
The ID is a content digest, not a signature or authorization. Keep artifacts local.

The job contract is deliberately strict: **exactly one completed job per archive**,
including deleted archives. Failed/claimed/pending/retryable_failed/pending_provider
or unsupported states block; no implicit terminal-failure waiver is provided.
Resolve unexpected jobs only under separate authorization, then replan.

Preparation locks the archive first, then existing head, sorted jobs and
generations, and rereads tombstones/projections. It never creates heads or
generations, attributes legacy rows, clears retirement, calls a provider, or writes
legacy `fts_entries`. Eligible live, unretired, unmanaged sources get only a
NULL-generation `raw_fallback` projection of exactly `raw_content[:10000]`, using
schema 0013's generated English vector. Suppressed/stale rows are not repaired or
reactivated. Source-ID and content-hash tombstones both exclude preparation.

Per-archive commits preserve partial progress. On lost acknowledgement, rerun the
same plan: valid existing baseline identities are reused. Fresh source/control
hashes must still match; otherwise stop and replan. No giant transaction, run
ledger, cursor authority, or exactly-once batch guarantee. Population changes during
preparation can only be accepted by a fresh full-population certificate afterward.

## Certificate contract

Every archive appears exactly once; class counts sum to the full population.
One `REPEATABLE READ READ ONLY` transaction covers the entire check:

- A: live/unretired/no managed active head; exactly one active matching raw baseline
  with valid generated vector.
- B: live managed-active; head/owner/job/generation binding checked and the real
  shared publication validator must pass. No baseline replacement.
- C: live retired/headless; **initial-bootstrap anomaly/blocker**, never accepted.
- D: deleted/tombstoned; no active head/candidate/open generation/projection,
  ordinary derived rows suppressed and plaintext erased across all generations.
  Canonical/curated retention remains separately governed by PRIV-02.

Hash-only tombstones with NULL deleted_at block until tombstone reapplication is
separately resolved. Other blockers include suppressed/stale/missing baselines,
open candidates, duplicate/missing/orphan jobs, invalid publication/bindings,
orphan derivation state, partition mismatch, unsafe job states, schema mismatch,
and (with `--cutover`) other PostgreSQL clients. Empty stop-word-only vectors are
valid if non-NULL and equal to PostgreSQL's generated English vector.

## Future production sequence — separate authorization gates

None of these operations is authorized merely by adding this runbook.

1. Keep R2 OFF. Review software/version, plan and current state read-only.
2. Obtain explicit production preparation authorization; prepare/resume baselines.
3. Enter an authorized maintenance window. Keep clients/ingestion quiet.
4. Under separate stop authorization, STOP `recalium-app` (including its worker,
   watcher and cache clients). Do not use a running app to execute the certificate.
5. Start only the one-shot certifier against Postgres using the deployed tooling
   and existing connection environment; ensure **only its client connection** is
   present in that database. Do not start lifespan, worker or migrations.
6. Use the exact stopped-app `docker compose run --rm --no-deps --entrypoint python`
   certificate invocation above, never plain `run` or stopped-service `exec`.
   It rescans the whole population in one RR read-only
   transaction, checks deployed Alembic head/current, ownership/jobs/privacy and
   coverage. Require exit 0 and inspect the bounded evidence. Do not reuse an old
   certificate after any source/control/schema/software change. Remain quiescent
   between certificate and activation; the tool does not fence out new clients.
7. Obtain separate R2 config authorization. Only then deliver the root Compose
   interpolation variable as true. Base Compose explicitly maps
   `GENERATION_IDEMPOTENCY_ENABLED: ${GENERATION_IDEMPOTENCY_ENABLED:-false}`.
   No backend dotenv is a substitute. Explicit process environment wins dotenv.
8. Verify resolved Compose configuration privately. Do **not** print or save the
   full `docker compose config` (it contains secrets); parse JSON and emit only the
   app's R2 key and the intended topology/identity predicates. A shell environment
   can override root interpolation: verify the effective value, not just the file.
9. Obtain separate recreate authorization. Recreate the app; do not assume a
   WatchFiles reload delivers a changed container environment. Container startup
   runs Alembic: expected deployed/current head must already match.
10. Verify bounded startup INFO `generation_idempotency_enabled=True`, in-container
    effective setting, health/DB OK, and separately authorized keyword/semantic
    smoke tests. Query embeddings/models/providers require their own authorization.
    Keep clients quiet until checks pass. Observe generation creation carefully.

## Rollback windows

Window A: returning OFF is permitted only after separately verifying ALL:

- `derivation_generations = 0`;
- no head has `legacy_retired_at`;
- facts/summaries/embeddings with non-NULL generation IDs = 0;
- no claimed job.

Blank heads/baselines alone are harmless OFF. OFF readers ignore new baselines.
This tooling does not perform rollback or change config.

Window B: once **any** managed generation exists, use forward repair only. No R2
OFF, old-code revert, retirement clearing or destructive cleanup. Stop and obtain
explicit repair authorization. Completed jobs remain blocked by PR #18 until a
separate PR 2 design/implementation/activation gate authorizes new generations.
