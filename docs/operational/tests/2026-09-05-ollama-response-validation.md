# Ollama response handling — validation, 2026-09-05

## Established causes

The installed `qwen3:4b` checkpoint reports `general.finetune=Thinking`,
`general.version=2507`, and a Qwen3-4B-Thinking-2507 license link. Its template
prefills `<think>`. The [official model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
confirms thinking-only behavior and explains why generated output can contain
only a closing `</think>` tag. Recalium requests `think=false` and previously
persisted the entire returned `message.content`; the reproduced summary contains
reasoning followed by the closing marker and the final summary.

For the supplied purple-mug input, the original extraction request returned a
valid bare fact object (`fact_text`, `source_span`, `confidence_tier`, `entities`,
`tags`), not the required `facts` array wrapper. `_parse_json_object` accepted it;
`.get("facts", [])` then discarded it as an empty result. The observed stage counts
were zero parsed facts and zero facts after validation/deduplication. Span
validation did not cause that loss.

Historical job logs show both Ollama calls
returned HTTP 200, without malformed-JSON or span warnings. The historical raw
extraction response was not retained, so its exact bytes cannot be proven. The
same-input reproduction establishes the concrete loss mechanism and is consistent
with that historical evidence. No claim is made that the historical response was
captured retrospectively.

## Final behavior

- Preserve model, endpoint, `think=false`, temperature, locality/proxy restrictions,
  and provider routing. The no-thinking prompt did not work for this checkpoint;
  a native-thinking structured-extraction probe exceeded the existing 300-second
  timeout, so neither change is included.
- Separate explicit leading `<think>...</think>` blocks and template-prefilled
  closing tags on their own lines. Preserve ordinary prose and literal tags in
  valid JSON; do not strip words such as “Okay”. Reject empty final answers and
  unfinished explicit thinking blocks.
- The shared response boundary covers Ollama summaries, extraction and link
  classification. Separate `message.thinking` is never combined into the answer.
- Request a JSON schema with a required `facts` array and typed fact fields.
  Validate the full returned object (allowing an explicit Markdown JSON fence).
  Missing wrappers, invalid JSON, wrong types and trailing junk raise sanitized
  errors instead of becoming successful empty extraction. An explicit valid
  `{"facts": []}` remains a valid no-facts result.
- The existing dispatcher turns extraction failures into retryable jobs. Remove
  the blanket HTTP-400 retry that silently discarded the `think` setting.
- Log extraction counts without logging the raw response or memory text.
- Sensitivity gate, policy dispatcher, locality checks, provider resolution,
  span validation and deduplication are unchanged; code comparisons verified this.

## Scope and data protection

Changed files: `backend/app/worker/dispatcher.py`,
`backend/tests/worker/test_ollama_responses.py`,
`backend/tests/worker/test_ollama_response_pipeline.py`, and this report.

The original dispatcher (including the privacy patch) was backed up in the Codex
workspace. Production source was not changed until staged checks completed.
Repeatable database tests use only a disposable response-validation database.
Pre-existing archives under observation were neither reprocessed nor repaired by this change.

## Automated validation

- Original response handling: 18 new regression cases failed and 5 passed,
  demonstrating the reported defects and missing safeguards.
- Final combined suite: **112 passed in 108.06 seconds**.
- Includes the new response tests and database pipeline-failure test, existing
  dispatcher tests, all local-Ollama authorization cases, external-egress tests,
  sensitivity gate tests, and policy resolver tests.
- Tested valid/invalid wrappers and types; malformed/trailing JSON; explicit JSON
  fences; thinking blocks and template-prefilled closing tags; empty/unfinished
  answers; literal tags; link classification; unchanged source-span downgrading;
  HTTP errors; and preservation of local-only proxy/redirect settings.
- Invalid extraction produces `retryable_failed` and zero persisted facts, with a
  sanitized error. A legitimate explicit empty facts array succeeds.

## Real model and database validation

The final staged dispatcher processed the supplied purple-mug text in a new archive
in a disposable response-validation database using the real local `qwen3:4b`:

- Gate: unclassified, blocked=true; policy: allow_external=false, local_only.
- Both operations audited as local Ollama and allowed.
- Completed successfully with **one clean summary and two persisted facts**.
- Neither thinking tag appears in the saved summary; the returned answer was
  separated from the reasoning trace using the explicit protocol boundary.
- Both persisted source spans are exact substrings of the original mug text.
- Extraction diagnostics: returned=2, retained=2.
- Both model requests returned HTTP 200. Local embedding also completed.

No existing monitored memory was reprocessed. A before/after fingerprint check of the
original mug archive and its job, summary, facts, embedding, FTS and audit records
is performed after applying the verified source.

## Post-apply verification

The verified patch was installed in the application under validation. The app health endpoint
returned HTTP 200, and the running application instance exposes the extraction schema and response
boundary functions. The original purple-mug archive, job, summary, facts, embedding,
FTS and audit record counts and SHA-256 fingerprints all match their pre-change
values. Existing unrelated startup/configuration changes were preserved.

No repair or reprocessing of the original completed archive was performed. Its
previous summary and zero-fact state remain unchanged by design. The only historical
uncertainty is the unavailable original extraction response; the loss path was
reproduced using the same text and original request. No further fix is needed for
the verified response-handling defects. Correcting existing derived memories would
be a separate, explicitly authorized operation.
