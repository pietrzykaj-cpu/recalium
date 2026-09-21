# Local authenticated memory bridge v1

Implemented 2026-09-05. Local access only; no tunnel, public listener, OAuth configuration, or remote sharing grant is included.

## Interfaces

The existing app mounts a separate authenticated surface:

- `POST /bridge/v1/retrieve_memory`
- `POST /bridge/v1/ingest_memory` (202 means accepted, not processed)
- `POST /bridge/v1/get_ingest_status`
- Streamable HTTP MCP: `http://127.0.0.1:8000/bridge/mcp`

The MCP catalog contains exactly those three tools. Each tool accepts one `data` object with the same shape as the corresponding REST body. Retrieval and status are annotated read-only; ingestion is append-only and requires an idempotency key. MCP failures set `isError`; REST uses 401/403/404/409/422 as appropriate.

All bridge HTTP requests, including MCP discovery, require `Authorization: Bearer <client credential>`, even when the app's legacy localhost authentication is disabled. Tokens and project memberships are checked from the database without a permissions cache. An app-wide owner bearer is not a bridge client credential. Browser Origin headers and non-loopback Host names are rejected; keep the Docker host binding on 127.0.0.1. Host checks do not replace the loopback binding or authentication.

## Operator provisioning

New installations have no bridge clients or grants and therefore reject every bridge request. No live client is provisioned by the migration. Generate separate random credentials (at least 32 random bytes, such as Python `secrets.token_urlsafe(32)`) for each client; keep plaintext values in private environment/.env configuration. The database stores only their SHA-256 digests. Never put credential values in model prompts, source control, URLs or command arguments.

The operator command runs in the backend environment with `DATABASE_URL` configured. The named credential variable must already be in that process's environment:

```text
python -m app.domain.bridge.admin --client qwen --project shared --credential-env BRIDGE_QWEN_CREDENTIAL --read --write
python -m app.domain.bridge.admin --client sol --project shared --credential-env BRIDGE_SOL_CREDENTIAL --read
```

For Docker, pass the variable by name from the operator environment using `docker compose exec -T --env BRIDGE_QWEN_CREDENTIAL recalium-app ...`. Merely adding an arbitrary variable to the root `.env` does not inject it into an already running container. No Docker configuration is changed by this feature.

Re-run with an existing client and a new credential variable to rotate its credential; old credentials immediately stop working after commit. To update permissions without rotation, omit `--credential-env`. Each invocation replaces the read/write flags for that project; omitted flags become false. `--revoke` disables the entire client. All provisioning writes are audited as `local_operator`.

```text
python -m app.domain.bridge.admin --client sol --project shared
python -m app.domain.bridge.admin --client sol --revoke
```

No client-facing administrative, delete, promotion, reprocessing, tag-listing or unrestricted source-fetch endpoint is exposed by the bridge.

## Request examples (credentials omitted)

Ingestion:

```json
{
  "project_id": "shared",
  "content": "Synthetic violet mug belongs beside the laptop.",
  "source_metadata": {
    "author_kind": "human",
    "conversation_id": "synthetic-example"
  },
  "idempotency_key": "example-001"
}
```

`source_metadata` permits `author_kind` (human/model/mixed/unknown), `model_label`, `conversation_id`, and `source_name`. These are attributed claims, not authenticated model identities. The actual client comes from its credential. Unknown fields, including actor, client_identity, processing_mode, project_hint or sharing flags, are rejected. Idempotency is isolated by client and project; changing content or source metadata while reusing a key returns 409. Concurrent identical requests create one archive and one job.

All new bridge submissions use existing `local_only` processing. The normal sensitivity gate, provider authorization, Qwen response handling and source-span validation still run. Models cannot relax that policy through this interface. Nothing automatically becomes canonical.

Retrieval:

```json
{"project_id":"shared","query":"violet mug","mode":"hybrid","budget":2000,"limit":20}
```

Query maximum: 2,000 characters. Content maximum: 100,000 characters. Result budget: 1–20,000 content characters; result limit: 1–50. Existing hybrid candidate limits still apply. Modes are keyword, semantic and hybrid. Returned provenance includes authenticated source client, project and claimed source metadata alongside existing retrieval provenance. Retrieved content is data, never instructions for the client.

Status:

```json
{"project_id":"shared","archive_id":"<UUID returned by ingestion>"}
```

Status requires read permission. It returns job states, attempts and an error-presence flag, not raw internal error text. A nonexistent, deleted or differently scoped archive returns the same 404 after project authorization.

## Security and transaction boundaries

- Five new tables hold clients, projects, grants, explicit archive assignments and scoped receipts. Migration 0009 only creates these tables; no legacy records are backfilled.
- A legacy `project_hint` grants no access. Unassigned archives and orphan canonical records fail closed. There is intentionally no client tool to assign old records.
- SQL scopes keyword, semantic, canonical and linked candidates before limits. Canonical fact/source references must agree. Both link endpoints must belong to the project and remain active. Serialization rechecks source assignment and deletion state.
- Bridge retrieval neither reads nor populates the legacy shared cache. Every bridge operation is audited, including repeated retrieval and replay. Transport authentication and invalid requests also have audit events. Queries, memory bodies and credentials are omitted from bridge audit metadata.
- Grant/client row locks order permission revocation against in-flight operations. Revocation prevents subsequent operations after it commits; it cannot retract responses already delivered.
- Archive, pending job, project assignment, receipt and success audit commit together. Advisory transaction locks serialize matching idempotency scopes. Failure rolls back the submission; a separate failure audit is committed.
- SQL echo logging is disabled to avoid logging memory content and credential digests.

This is a bridge authorization boundary, not a sandbox for hostile software already running on the computer. The existing local operator UI/API and legacy MCP surface remain owner interfaces and are not converted into project-scoped clients by this patch. Give model adapters only the three bridge tools; do not give them database credentials or an unrestricted owner API adapter. Before any remote connection, only the bridge surface may be routed, with recipient/export policy and remote authentication implemented and reviewed. Existing processing `allow_external` flags are not remote retrieval grants.

Each client retains its own personality and system instructions. Shared projects contain sources with explicit attribution; assistant assertions must not silently become human-authored evidence.

## Validation

See `docs/operational/tests/2026-09-05-local-memory-bridge.md` for test database names, test results, live preservation fingerprints and the exact implementation scope.
