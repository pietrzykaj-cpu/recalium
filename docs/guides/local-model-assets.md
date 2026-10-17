# Offline local MiniLM contract

**DO NOT recreate production `recalium-app` before the existing real MiniLM cache
has been deliberately captured to durable host storage under separate explicit
authorization.** The only known real copy may be in the running container's
ephemeral Hugging Face cache from the first trial. Recreation can delete it and
force a future separately authorized re-download. This implementation/follow-up
does not authorize capture, copying, download, provisioning or recreation.

Local generation must not acquire assets from Hugging Face. Both sensitivity-gate
and archive-embedding loaders use one contract in
`backend/app/infrastructure/local_model.py`: model `all-MiniLM-L6-v2`, exact revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The public embedding identity remains
`all-MiniLM-L6-v2`; dimensions 384, max sequence length 256, tokenizer, mean pooling
and normalization are unchanged. Existing generations/vectors/provenance are not
rewritten or invalidated. No schema/migration or R2 lifecycle change.

## Admission and failure

All managed admissions require local assets, including externally permitted policy
paths: the sensitivity gate itself uses local MiniLM. Completed PR23 reprocess,
automatic managed admission, and managed failure-state manual reprocess share the
same static readiness prerequisite. It runs off-loop after the lookup transaction
ends, before the archive/head/job/generation locking transaction, blank-head
creation, candidate creation, or job reset. Existing CAS/lock order remains intact.
If readiness fails it takes precedence over later locked admission checks; unknown
job lookup still returns missing. No model is loaded during readiness.

Readiness checks absolute revision-named snapshot root, exact required file set,
size and content digest, contained symlinks only, and approved installed loader
versions. Small asset identities are Hub Git-blob SHA-1 (including `blob <size>\0`
prefix), not plain SHA-1; safetensors uses LFS SHA-256. Hashes and sizes come from
the verified public snapshot recorded in the investigation. No external manifest
or floating `main` is trusted. Extra files, including adapter/code/alternate
weights, fail closed. Normalize is parameter-free: its data folder may be absent.
Unexpected files may not be “fixed” by relaxing the manifest without review.

Missing/incomplete/corrupt/wrong-revision/incompatible assets return the static
`local_model_not_ready`, without filesystem paths, URLs, tokens or exception text.
Completed/manual API reprocess returns HTTP 503 with that exact detail and no
reset/generation. Worker automatic failure occurs after its existing claim: the
current claim is fenced and marked failed with that marker, without a generation,
attempt refund, cooldown, or claim-predicate change. Fix assets only under a new
operator authorization; do not auto-download or loop. PR14 unexpected-crash and
M4 coordination handling remain separate.

Both lazy constructors revalidate and load the absolute local snapshot with
`local_files_only=True`, `trust_remote_code=False`, pinned `revision`; no remote
shortname fallback remains. HF/Transformers offline flags are set before lazy ML
imports and explicitly delivered by Compose; they are not an OS egress sandbox.
Readiness validates bytes/config identity, not hardware or all later runtime
failures. Keep assets immutable/read-only: validation is not a concurrent-writer
lock. No filesystem/model work spans database locks. Existing history/readers
remain available when future admission is blocked.

Query embeddings catch `LocalModelNotReady` at both managed and legacy retrieval
degradation boundaries. Missing/invalid assets do not block existing reads:
default/hybrid retrieval returns keyword results with degraded mode; explicit
semantic-only mode keeps its existing degraded response semantics. API/MCP use
the same boundaries. Admission readiness remains strict.

A gate load-time `LocalModelNotReady` is re-raised by `classify_async`, not converted
to an unclassified policy decision. Existing managed/OFF dispatcher gate failure
handling marks the job retryable (or terminal if its ordinary attempts are
exhausted). No policy-skipped stage/publication/legacy retirement is caused by
that model failure; an admitted candidate stays unpublished/building. Existing
active authority remains unchanged. Other classifier exceptions retain their
existing privacy-first unclassified fallback. No dispatcher lifecycle redesign.

## Dormant deployment plumbing — not deployment authorization

Defaults:

- Host `LOCAL_MODEL_ASSETS_HOST_DIR=./data/local-models`.
- Read-only bind at `/app/local-models`, `create_host_path: false`.
- App `LOCAL_MODEL_ASSETS_DIR=/app/local-models/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`.

Use an ordinary existing host bind, not new storage architecture. Container
recreation preserves host assets. App has read-only access. The app never creates
the host root or copies/downloads model assets. Native deployments must set an
equivalent absolute immutable snapshot path and deliver offline environment flags
before imports. Pydantic reading a dotenv is not third-party environment export.

Future deployment requires separate explicit authorization for each bounded step:

1. Prepare durable assets from an approved already-local source; materialize
   cache symlinks into this snapshot (do not copy dangling external blob links).
   No production cache access/copy or download was performed by implementation.
2. Verify all 11 files against the built-in manifest and verify loader package
   versions. Current contract pins ST 5.3.0, Transformers 5.16.1, Hub 1.30.0,
   tokenizers 0.23.2, safetensors 0.8.0, CPU Torch 2.14.0+cpu. This repair certifies
   the current CPU contract, not a GPU deployment. Version drift blocks readiness; do not
   update packages or pins silently. Validate with the intended image/runtime.
3. Deliver path/config mapping. A root `.env` edit alone does not update a running
   container. Source sync before assets are provisioned blocks future admissions.
4. Controlled app recreate only when authorized. Recreate is required to deliver
   new bind/environment; restart alone does not deliver them. Existing entrypoint
   runs Alembic: schema/restart authorization must not be inferred from merge.
5. Authorized no-network readiness/model smoke and preservation verification.
   No new generation/reprocess is authorized implicitly by a smoke or recreate.

Window B: preserve G1 and managed history; never use R2 OFF/old code or history
deletion as asset-failure recovery. The operational ingestion/reprocess freeze
remains until Anna authorizes otherwise. Restore, Slice2, model upgrades and
re-embedding are outside this repair.
