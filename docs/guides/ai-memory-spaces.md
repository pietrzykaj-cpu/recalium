# AI memory spaces and natural access

Local milestone implemented 2026-09-06. One persistent memory system, independent AI clients, private working notebooks and deliberately shared knowledge. No remote transport or export behavior is configured by this milestone.

## The three tools

The existing REST paths and Streamable HTTP MCP endpoint remain unchanged:

- `POST /bridge/v1/retrieve_memory`
- `POST /bridge/v1/ingest_memory`
- `POST /bridge/v1/get_ingest_status`
- MCP: `http://127.0.0.1:8000/bridge/mcp`

MCP tools accept their request object under `data`. All requests require the existing client bearer credential. No extra AI-facing tools or per-memory approval step are introduced. Client integrations retain control over when they permit tool calls and any integration-specific confirmations.

## Retrieval

Ordinary use:

```json
{"query":"What did we decide about the Recalium bridge?"}
```

The authenticated client's readable spaces are resolved server-side. Private ownership and grants both apply. One pipeline searches the authorized union; candidate SQL, canonical and link queries are restricted before ranking and limits. Both endpoints of a surfaced link must be permitted. Source checks run before serialization. Empty permissions return an empty result without querying unrestricted memory.

Optional narrowing:

```json
{"query":"Recalium bridge","space_ids":["shared","recalium-shared"],"mode":"hybrid","limit":20,"budget":2000}
```

Omitted `space_ids` means all currently readable spaces. An explicit empty list means no spaces. Explicit unauthorized and nonexistent spaces return the same generic permission error. Maximum explicit selectors: 100. The legacy `project_id` remains an exact single-space selector, but cannot be combined with `space_ids`.

The response includes `searched_space_ids`; every result includes `memory_space: {id, kind}`. Legacy `project_id` is echoed when explicitly supplied; otherwise that top-level field is null. Each result's provenance contains its actual project identifier.

The query limit remains 2,000 characters. The content-character budget is 1–20,000; the result maximum is 1–50, default 20. One budget and result limit apply across all permitted spaces. Scoped hybrid fusion now accepts up to 50 direct candidates instead of an unconditional cap of 20. The existing relevance threshold, deduplication and candidate caps can still yield fewer results than requested. The existing canonical/fact/summary/excerpt budget priority is retained; there is no private-space-first preference. `trimming_reason` can now also be `result_limit`, distinct from `budget_met` and `result_exhausted`.

Bridge cache bypass remains in force. Grant checks are not cached.

## Ingestion

```json
{
  "destination":"private",
  "content":"A useful working note, observation or hypothesis.",
  "source_metadata":{"author_kind":"model","model_label":"Sol","conversation_id":"conversation-reference"},
  "idempotency_key":"client-generated-stable-request-id"
}
```

Use `destination: "shared"` to select the client's default shared destination. For a specific shared space, supply `destination: "shared", space_id: "recalium-shared"`. The server resolves the binding and checks the existing write grant. Missing bindings or permissions fail; there is no fallback and no automatic grant creation. A private destination cannot take an explicit space_id. An explicit shared destination cannot target a private space.

Legacy ingestion with `project_id` remains supported, exclusively of `destination`/`space_id`; ownership is still enforced. The destination chooses existing authority and never creates it. Unknown identity/permission fields remain rejected. Claimed model/author metadata cannot override the authenticated identity.

A new alias receipt pins each client's private/shared idempotency key to its first committed space. Later rebinding does not redirect a replay: it returns the original archive after current authorization is checked. If that original space is no longer writable, replay fails rather than writing elsewhere. New keys use the new binding. Explicit-space receipts remain isolated by client, concrete space and key. Content or source-metadata changes conflict. Receipt pinning, archive, job, assignment and audits commit atomically; failed submissions create no partial memory.

The content and source metadata limits are unchanged. Every archive belongs to one authoritative memory space. Bridge ingestion continues to request existing local_only processing, with the verified privacy gate and local-Ollama behavior. A zero-fact extraction does not invalidate the raw notebook entry or its later FTS/semantic retrieval. No automatic canonical promotion occurs.

## Status

```json
{"archive_id":"<archive UUID returned by ingestion>"}
```

The server resolves and checks the archive's space. The optional legacy project_id narrows authorization. Unknown, deleted and inaccessible archives do not reveal their status. Acceptance means durably saved/enqueued; status distinguishes pending and completed processing. Raw internal error text is not returned.

## Provenance and audit

Per-item provenance separates:

- `authenticated_client`: the client that submitted the source, not a caller-supplied author label.
- `source_metadata`: attributed author_kind/model_label/conversation_id/source_name claims.
- `processing: {method, model}`: recorded fact derivation or summary processing information; canonical records retain source consistency. Unknown processing provenance for raw/indexed excerpts is null rather than guessed.
- `retrieval: {method, model}`: FTS, embedding retrieval or link traversal information.

The bridge no longer presents retrieval machinery as the original content-producing model in a top-level provenance.derivation_model field. Legacy owner retrieval interfaces retain their established format. Selection among stored summaries is deterministic so returned content and recorded summary provenance agree.

Audit records contain resolved searched space IDs, resolved write destination, authenticated actor, outcome and result/replay information. They omit memory bodies, query text and credentials. Source labels remain claims, not human-authored evidence or client instructions.

## Operator provisioning

No real clients, bindings or grants are provisioned by this milestone. Plaintext credentials remain in operator-controlled environment configuration; only digests are stored. The existing administrative command adds `--space` as an alias for `--project`, plus `--kind` and `--bind`:

```text
python -m app.domain.bridge.admin --client sol --space sol-private --kind private --bind private --credential-env BRIDGE_SOL_CREDENTIAL --read --write
python -m app.domain.bridge.admin --client sol --space shared --kind shared --bind shared --read --write
python -m app.domain.bridge.admin --client qwen --space qwen-private --kind private --bind private --credential-env BRIDGE_QWEN_CREDENTIAL --read --write
python -m app.domain.bridge.admin --client qwen --space shared --kind shared --bind shared --read --write
```

These are examples, not commands executed on live data. The named credential variable must already be present in the process environment; never paste a credential value into a model prompt or command argument.

Creating a private space assigns ownership to that client. Ordinary provisioning refuses another client's private space and refuses in-place kind conversion. Runtime authorization also rejects a foreign-private grant, even if a corrupt grant row was inserted outside provisioning. The database enforces private-owner presence and shared-owner absence. Bindings require the matching kind and a write grant; they do not grant access by themselves.

Existing credential rotation and revocation commands are unchanged. Permission flags remain replacement flags: omitted read/write flags are false. For in-flight ordering, operations lock the authenticated client, then spaces sorted by ID, then grants sorted by ID. Official provisioning locks the client before modifying its permissions/bindings. Committed revocation controls subsequent operations; already delivered responses cannot be retracted.

Anna retains owner access through the existing trusted local owner interfaces and infrastructure. Space ownership is an AI-client isolation boundary, not secrecy from the operator. There is intentionally no client tool to transfer ownership, reclassify spaces, assign legacy archives or share existing sources automatically.

## Migration and boundaries

Migration 0010 adds kind/owner fields to bridge_projects and adds bridge_bindings and bridge_alias_receipts. Existing projects default to shared, preserving their existing grant meaning; names are not interpreted as privacy policy. It does not assign, move, share or reprocess any existing archive. The live bridge was empty before installation.

Private means private between clients. An AI permitted to read private memory and write shared memory may deliberately preserve a useful insight in shared memory. Clients keep their own personality and instructions. Retrieved notes remain untrusted context.

No remote connectivity, OAuth, tunnel, listener, routing, real credentials or remote sharing policy was added. Local processing allow_external semantics were not reinterpreted. Remote Sol connectivity and its trust boundary remain a separate stage.
