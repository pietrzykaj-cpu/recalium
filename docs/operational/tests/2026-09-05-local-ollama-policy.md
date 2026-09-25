# Local Ollama privacy policy validation — 2026-09-05

## Change

Sensitive and unclassified archives, including `local_only`, may use local Ollama
for summarization and extraction. Each operation checks its actual provider.
OpenAI, Anthropic, and remote Ollama remain subject to `allow_external`.
The sensitivity gate and resolver are unchanged. Local-only HTTP calls disable
environment proxies and redirects. Policy audits record each operation's provider,
locality, and authorization; audit failure prevents LLM processing. Summary model
provenance now follows the summarization provider.

Link classification retains its existing external-policy restriction.
Provider selection order and existing completed archives are unchanged.

## Automated evidence

- Original dispatcher: the initial 51 new regression cases failed as expected.
- Final staged dispatcher: **84 tests passed** (48.22 seconds).
- Suites: `test_local_llm_policy.py` (53 cases), existing dispatcher tests,
  external-egress tests, policy resolver tests, and sensitivity gate tests.
- Includes all four classifications, `local_only`, both directions of mixed local
  and remote providers, sensitive hints, remote Ollama, endpoint lookalikes and
  malformed URLs, audit failure, and local HTTP proxy/redirect isolation.
- The existing invalid-key fixture now selects a provider as well as reporting
  availability, so it continues to exercise retryable authentication failures.

Tests ran in the isolated test environment with temporary pytest dependencies under
a task-local temporary dependency location. Database fixtures were explicitly redirected to
a disposable policy test database; the application memory database was not used.
The source was loaded into a separate process before the working repository changed.

## Backup and scope

Original dispatcher, existing dispatcher test, and privacy document were copied
into the Codex task workspace before editing. Applying the patch verifies their
hashes to avoid overwriting intervening changes. Existing startup-script edits
and local configuration files are preserved. No migrations or environment changes
are needed.

## Real local inference evidence

A fictional personal memory was processed in the isolated test database through
the real sensitivity gate, worker, local `qwen3:4b`, and local embeddings:

- Gate: `personal_profile`, blocked=true, confidence=1.0.
- Intent: `local_only`, hint=`private`.
- Effective policy: `allow_external=false`.
- Result: completed, **1 summary and 3 facts**.
- Audit: provider=`ollama`; summarize and extract each local=true, allowed=true.
- Both native Ollama requests returned HTTP 200.

The local embedding library also checked model metadata on Hugging Face, as in
its existing behavior; these were model-file requests, not memory inference calls.
The patch does not change model download/cache behavior.
