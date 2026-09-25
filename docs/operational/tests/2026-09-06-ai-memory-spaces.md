# AI memory spaces — implementation and validation checkpoint

Date: 2026-09-06. Repository: Recalium working tree.

## Outcome

Implemented, validated and installed the local AI-memory-spaces milestone. Exactly three AI-facing tools remain: retrieve_memory, ingest_memory, get_ingest_status, through the existing REST and MCP paths. No real clients, grants or destination bindings were provisioned. No remote connectivity, OAuth, tunnel, public listener, routing or export policy was configured.

Health after installation: HTTP 200, status=ok, db=ok, api_version=1. REST and MCP bridge requests without credentials returned HTTP 401. Schema revision is 0010. The validation database contained zero bridge clients, grants, archive assignments, destination bindings and alias receipts.

The read-only before/after preservation check reported matching counts and SHA-256 fingerprints for the monitored memory tables: raw archives, jobs, summaries, facts, embeddings, FTS entries, canonical memory, links, tags and fact/tag associations. No monitored memory was changed, moved, shared or reprocessed.

## Exact behavior

### Spaces and destinations

Projects now carry explicit shared/private kind and private ownership. Private spaces require an owner; shared spaces have no private owner. Runtime authorization requires both the relevant grant and matching ownership for private spaces. Ordinary operator provisioning cannot grant another client access to a private space, even under a misleading name, and does not perform implicit kind conversion. Existing trusted owner interfaces and infrastructure access remain intact.

Server-managed bindings map each client's private/default-shared aliases to concrete spaces. A binding supplies a destination, never authority. A matching kind and current write permission are required. Missing/unauthorized bindings fail, with no fallback. New optional CLI flags are --space (alias of --project), --kind and --bind; existing credential rotation and revocation remain supported.

### Natural retrieval

`{"query":"What did we decide about the Recalium bridge?"}` resolves and searches the authenticated client's readable union. Optional space_ids narrows that union. An empty list or no readable grants yields no results, never unrestricted retrieval. Explicit foreign and nonexistent selectors produce the same generic permission error. Legacy project_id remains an exclusive single-space selector.

Candidate SQL scopes keyword, semantic, canonical and linked results before limits. Both endpoints of cross-space links must be authorized. Canonical/source consistency and orphan exclusion remain enforced. Serialization checks actual source assignment/deletion and linked-source permission. One global ranking/fusion path and one budget/result limit cover the union; bridge cache bypass remains enabled.

Result maximum remains 1–50 (default 20); budget remains 1–20,000 content characters. Scoped hybrid fusion no longer imposes an unconditional 20-direct-result cap. Existing relevance thresholds/candidate caps may still return fewer than requested. Existing type-priority budget selection remains unchanged. A new trimming reason, result_limit, accurately distinguishes the count limit from character-budget exhaustion.

Each result includes its actual memory_space {id, kind}; the response lists searched_space_ids. Legacy top-level project_id is echoed only when supplied, otherwise null.

### Natural ingestion and retries

Use destination=private or destination=shared; shared may additionally specify space_id for a particular shared project. Private may not specify another explicit space. Legacy project_id ingestion remains supported, mutually exclusive with the new destination fields. Identity and permission fields cannot be asserted through request metadata.

A committed alias receipt pins a client's private/shared idempotency key to its first resolved concrete space. Rebinding affects new keys, not retries. A replay rechecks permission to the original space; revoked access fails without writing elsewhere. Explicit-space receipt isolation and old receipt fingerprints are preserved. Alias pin, archive, pending job, assignment, receipt and audit commit together. Concurrent matching writes create one archive/job. Failed submission leaves no partial source or alias receipt.

Bridge writes retain existing local_only processing and its local-Qwen authorization. A useful raw notebook entry remains durable and searchable after indexing even if extraction produces zero facts. Source-span/fact validation and canonical promotion rules are unchanged.

### Provenance and audit

Returned provenance distinguishes authenticated source client, claimed source metadata, stored processing {method, model}, and retrieval {method, model}. FTS, embedding retrieval and memory_links are no longer presented as the original content-producing model. Unknown processing provenance for raw/indexed excerpts is null, not inferred. Stored summary selection is deterministic so processing provenance matches the chosen content. Canonical fact references must agree with their source.

Audit records contain resolved searched space IDs, resolved write destination, authenticated actor, outcome, result count and replay information; no query text, memory bodies or credentials. Source/model metadata remains attributed claims, not human-authored evidence or instructions to another client.

Lock order is authenticated client, spaces sorted by ID, then grants sorted by ID. Client/grant revocation is ordered against in-flight calls and applies to subsequent calls after commit. Official binding/grant provisioning locks the client first. Status now accepts archive_id alone and resolves/checks its space; the legacy project_id remains an optional restriction.

## Files changed

Production files modified:

- `backend/app/domain/bridge/models.py`: space kind/owner constraint; binding and alias-receipt models.
- `backend/app/domain/bridge/contracts.py`: optional retrieval selectors, natural destinations, optional status project and ambiguity validation.
- `backend/app/domain/bridge/admin.py`: ownership-safe provisioning and server-managed bindings.
- `backend/app/domain/bridge/service.py`: resolved authorization, union operations, pinned retries, current permission checks and separate provenance/audit.
- `backend/app/domain/retrieval/service.py`: authorized scope lists, cross-space traversal, global fusion/limits, truthful limit reporting and deterministic summary selection.
- `backend/app/api/bridge.py`: tool descriptions for natural memory use; no new tools or routes.

Production files added:

- `backend/app/domain/bridge/access.py`: ordered permission/ownership resolution and locking.
- `backend/migrations/versions/0010_memory_spaces.py`: kind/owner fields plus bridge_bindings and bridge_alias_receipts. Existing projects default to shared; names do not imply ownership. No archive backfill or reassignment.

Tests and records added:

- `backend/tests/test_memory_spaces.py`: 15 focused cases, including parameterized retrieval modes and in-flight revocation.
- `docs/guides/ai-memory-spaces.md`: exact contracts, operator examples and boundaries.
- `.planning/quick/260906-ai-memory-spaces/PLAN.md`: bounded implementation record.
- `docs/operational/tests/2026-09-06-ai-memory-spaces.md`: this report.
- `docs/operational/tests/artifacts/2026-09-06-ai-memory-spaces/`: captured results, preservation fingerprints, installed source hashes and isolated launchers.

Verified byte-identical to the prior implementation: worker dispatcher, policy gate, policy resolver, derived-memory processing service, ingest service, app startup and DB configuration. No .env or Docker configuration changed. Replaced files are backed up in a task-local backup workspace. No commit or PR was created.

## Validation performed

All database tests used disposable databases, never the application database:

- a disposable memory-spaces test database: pytest fixtures may reset schema/data only here.
- a separate disposable memory-spaces MCP database: real MCP test and full migration from empty schema through 0010.

The launcher derives a distinct test URL and asserts it differs from the application URL before loading the app. External provider variables were cleared. Synthetic vectors exercised semantic/hybrid authorization deterministically. The full-app MCP test worker was replaced by an idle coroutine in that disposable process only; no model calls or job reprocessing occurred there.

Results:

- Host-side contract validation and compilation passed while Docker was initially unavailable.
- First focused run: **15 passed in 15.62s**.
- Relevant combined regression run: **190 passed in 136.53s**.
- Real MCP HTTP/full-app mount and migration test: **passed**.
- Ruff checks on changed bridge files, migration and new tests: **passed** after formatting/import cleanup.
- Installed source hashes match the tested staging files.
- Health and preservation checks: **passed**.

The 190-test command included:

```text
the isolated memory-spaces test runner
  tests/test_memory_spaces.py tests/test_memory_bridge.py tests/test_ingest.py
  tests/domain/test_retrieval.py tests/integration/test_retrieval_filters.py
  tests/mcp/test_mcp_server.py tests/api/test_auth_middleware.py
  tests/worker/test_dispatcher.py tests/worker/test_ollama_responses.py
  tests/worker/test_ollama_response_pipeline.py tests/worker/test_local_llm_policy.py
  tests/worker/test_policy_egress_gate.py tests/domain/test_policy_resolver.py
  tests/domain/test_policy_gate.py -q --tb=short
```

Coverage includes private isolation in both directions; shared grants; private ownership in provisioning and runtime despite a corrupt grant; automatic union/narrowing/empty scope; indistinguishable unauthorized/nonexistent selectors; global ranking, budget and >20 hybrid results; permitted/forbidden cross-space links; canonical mismatches/orphans; private/shared/explicit-shared destinations; missing binding and cross-private selection; replay after rebinding; five simultaneous writes/replays; atomic rollback; private-owner schema constraints; client and grant revocation blocked behind an in-flight read then enforced afterward; separated processing/retrieval/source provenance; zero-fact notebook retrieval; query/body-free audit; cache bypass; prior bridge identity/idempotency/legacy-record tests and core privacy/Qwen regressions.

The real MCP test exercised the actual three-tool catalog, shared and private alias ingestion, pinned replay, project-free status, automatic union retrieval, a second reader unable to see the first client's private note, denied writes and revocation removing shared results. The disposable server was stopped after testing.

The disposable database environment was temporarily unavailable, so database testing waited. Once it was restored, all validation completed. That startup issue did not require changing Recalium configuration or widening this milestone.

## Remaining limitations and clean checkpoint

No failing validation remains for this local milestone. The bridge remains closed and unprovisioned, as requested. Actual client integration/provisioning is separate work.

Indexed excerpts do not always retain an unambiguous original processing-model reference; the bridge reports null rather than inventing one. Stored facts and summaries expose their recorded models. This is a provenance limitation, not a reason to weaken validation or reprocess existing memories.

Existing ranking thresholds and type priorities remain; requesting 50 is a maximum, not a guarantee. Topic organization, ownership transfers, kind conversion, legacy-memory sharing and corpus reorganization are not added. The existing owner API is still a trusted local owner interface, not an OS sandbox against unrestricted local software.

Remote Sol connectivity and its trust boundary are untouched and remain the next separate stage. Existing allow_external processing semantics were not reinterpreted. No automatic per-memory approval ceremony was introduced. One persistent memory system now supports independently owned AI notebooks and deliberately shared memory within explicit grants.
