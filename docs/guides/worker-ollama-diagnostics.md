# Worker Ollama completion diagnostics

The ordinary worker Ollama response path emits one INFO record from
`app.worker.dispatcher`, with message `Ollama completion diagnostics: %s`, when
the response contains at least one valid completion diagnostic. Availability
in operator logs depends on the configured logging level and handlers.

The worker reuses `parse_ollama_completion_diagnostics` from
`backend/app/domain/model_context/ollama.py`. This shared policy allows `done`,
`done_reason`, `prompt_eval_count`, `eval_count`, and four durations: total,
load, prompt evaluation, and evaluation. Duration keys in the log end in `_ns`.
Null or invalid values are omitted; absent or wholly invalid diagnostics produce
no record. Zero counts and `done=False` remain observable. Termination reasons
use the shared parser's bounded, single-token validation.

The record is emitted after HTTP success and JSON decoding, before final-content
parsing, so it remains available when the existing content parser raises. It
does not report success of the processing job, extraction quality, or absence
of truncation. A `length` reason does not change the response's success/failure
behavior. HTTP failures do not emit completion diagnostics.

This record includes no prompt, response content, reasoning, credential,
provider context/token array, tool call, model label, endpoint, or arbitrary
provider field. No database record or provider payload archive is added.
The worker's string return, final-answer parser, request controls, policy gate,
coordinator, retries, and job transitions keep their existing behavior.

Socket-free regression coverage is in
`backend/tests/worker/test_ollama_responses.py`. Synthetic `stop` responses test
metadata handling; they are not evidence of a naturally terminating model run.
