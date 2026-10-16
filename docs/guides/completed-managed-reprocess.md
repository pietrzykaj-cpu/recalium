# Explicit completed-job reprocess (1C PR 2)

This capability requires effective `GENERATION_IDEMPOTENCY_ENABLED=true`.
Deploying it does not authorize a production reprocess or provider/model call.
It schedules work for the existing worker; it does not call providers in the
request transaction. Never test this endpoint on live data without authorization.

`POST /api/jobs/{job_id}/reprocess` accepts an optional JSON body:

```json
{"expected_generation": "legacy"}
```

Use literal `legacy` only when the observed current job binding is NULL. For a
managed completed result, supply the exact current generation UUID instead.
Omitted body or JSON null does not mean legacy: completed jobs still return 409.
An object with an omitted/null/invalid expected generation, or unknown fields,
returns 422. Any explicit expected-generation fence while R2 OFF returns bounded
409 `Generation-aware reprocess is disabled`, with no job mutation or new
generation/head state. This also applies to explicit requests for unknown IDs;
ordinary no-body unknown IDs remain 404.

Successful explicit requests return bounded `status`, `job_id`, `generation_id`.
The transaction creates one building `manual_reprocess` generation, binds the job
and resets status to pending, attempts to zero, error/completed timestamp to NULL.
Other job metadata is preserved. Legacy rows receive no invented provenance.
Managed successors preserve the previous published generation and predecessor link.

The request uses the established archive/head phase to create a blank head if
absent and atomically sets `head.candidate_generation_id` to the new generation.
Every open managed candidate is represented by that pointer. For legacy -> G1,
active stays NULL, predecessor is NULL and legacy remains current. For G1 -> G2,
active stays G1, candidate is G2 and predecessor is G1; G1 is not superseded.
There is no request-time retirement or reader switch. Normal fenced worker
admission consumes the already-attached candidate without a special lifecycle.
Successful publication alone switches active head/retirement/supersession.
Privacy withdrawal can abandon this work before processing. No restore is added.

Fresh locks follow archives -> heads (blank-head creation here if needed) ->
sorted jobs -> generations.
Binding comparison and status/source/publication checks happen under those locks.
Deletion, either tombstone form, retired/abandoned/corrupt binding or invalid current
publication fail closed. Schema 0013 suffices; no new migration is introduced.

On binding mismatch the endpoint returns 409 with only:

```json
{"detail": {"reason": "expected_generation_mismatch", "current_generation": "<UUID or legacy>", "current_status": "pending"}}
```

Current status is bounded; unsupported durable values become `unsupported`. Never
blindly resubmit with a newly observed binding: that would be a new deliberate
request, not a retry. After acknowledgement loss, retry the original precondition.
It conflicts without creating another generation or resetting the job again.
This is a binding fence, not a generic request-ID/exactly-once framework.

Existing PR #18 no-body calls still accept failed/retryable_failed/pending_provider
and reject claimed/pending/unsupported states. Unknown IDs remain 404. Supplying
an explicit precondition on these failure-state paths also checks the binding
when R2 ON, and returns disabled 409 when OFF. No-body PR18 eligibility is unchanged.
Old no-body successes keep their
two-field response. Source text, provider data and job errors are never returned.
If dispatch fails, the next eligible manual failure-state reprocess retires the
attached candidate through the existing failed/abandoned lifecycle before creating
its successor. Automatic retry reuses the same candidate/generation.

## Commitment point: Window A -> Window B

Before the first accepted production request, R2 ON + prepared baselines + zero
managed generations is the final Window-A state. Anna reports production currently
has R2 effective true, completed cutover, 10 baselines, 10 completed jobs and zero
generations/heads/stages; this implementation follow-up does not probe production.

**Window A ends at the first successfully committed completed-job managed-reprocess
admission, not at publication.** The commitment point is the first transaction
that creates a `derivation_generations` row. Window B applies immediately even
while that generation is only building and has never published: forward repair
only. Do not flip R2 OFF, revert to old code as recovery, clear retirement/history
or destructively clean managed state. Failed/rolled-back admission creates no
durable generation and does not cross that boundary.

M3/M4, PR #14 crash recovery, claim predicates, restore and Slice 2 conflict/review
replay identity remain unchanged. No user resolutions or canonical promotions are
transferred automatically to newly derived facts.
