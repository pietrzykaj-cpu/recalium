# Privacy and Policy

## Default posture
- local-first storage
- localhost-first exposure
- explicit opt-in for external providers
- conservative extraction behavior

## Sensitive-content policy
Sensitive categories include at least:
- personal profile
- relationships

Sensitive detection uses:
1. user-declared sensitivity
2. local rule-based pre-classification

If content is unknown or low-confidence, external processing is blocked by default.

## Policy hooks needed in v1
Even though v1 is single-user, the architecture should keep explicit policy decision points for:
- external provider eligibility
- retrieval suppression
- deletion/redaction propagation
- network exposure mode
- future service/tenant policies
- configured exclusion from embedding and indexing by category or source

## Deletion and redaction policy
- derived summaries, facts, embeddings, and search visibility are immediately suppressed
- canonical entries with removed sources remain but require review and source-removed marking
- future backups/exports exclude removed data
- old backups/exports must be flagged as potentially containing removed data

The enforceable mechanism for this behavior is a durable tombstone/deletion-ledger model. See [deletion-and-tombstones.md](deletion-and-tombstones.md).

## Local LLM processing

The sensitivity gate and effective policy continue to govern external egress. A blocked
classification, a sensitive hint, or `local_only` does not prevent summarization and
fact extraction through a local Ollama server. Each operation is authorized separately
using its selected provider; a local summarizer never authorizes a remote extractor,
or vice versa. Provider auto-selection order is unchanged; a blocked selected remote
provider is not silently replaced by Ollama.

Local Ollama endpoints are HTTP(S) origins on loopback addresses, `localhost`, or
Docker Desktop's `host.docker.internal` host gateway. LAN addresses, remote hosts,
URLs containing credentials, and proxy paths do not receive local permission.
This trusts the configured local Ollama service to execute inference locally.
Local-only calls disable environment HTTP proxies and do not follow redirects.

The `policy_decision` audit's `operations` map records the selected provider, locality,
and permission for summarization and extraction. These are authorization decisions,
not claims that a call completed (an existing summary may be reused). The legacy
`provider` field names the sole permitted provider, or is null when none or multiple
providers are permitted; the operation map supplies the detail. LLM processing retries
without calling a provider if the policy audit cannot be persisted. Summary provenance
records the summarization model independently of the extraction model.

Link classification Pass B retains its existing external-policy gate. Local FTS,
embeddings, and other non-LLM processing retain their existing behavior.
