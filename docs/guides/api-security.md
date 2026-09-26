# API and security boundaries

## Authoritative OpenAPI source

The running FastAPI application at `/openapi.json` is authoritative. Use
`backend/scripts/export_openapi.py OUTPUT` to produce a deterministic local
JSON artifact for review, tests, or downstream tooling. The generated artifact
is not the hand-maintained `recallium-openapi.json` snapshot and should not
silently overwrite it.

The exporter builds the schema from the application factory, so routes are not
duplicated in a second source of truth. It does not start a server, call a
model provider, require API keys, or contact external services.

## Main application authentication

The default bind host is loopback (`127.0.0.1`). When the application is
bound to a non-local address, middleware requires a bearer token for
`/api/*` and `/mcp/*` requests. The token is compared with the configured
runtime bearer value; provider API keys are separate and are not persisted.

The current middleware exemption list contains `/health`, `/api/docs`,
`/api/redoc`, and `/openapi.json`. The health router is mounted under the
`/api` prefix, so the implemented `/api/health` route matches the
`/api/*` protection rule when bound non-locally. This document records the
current behavior; changing that boundary requires a separate, explicit design
decision and auth-matrix tests.

## Bridge and MCP boundaries

`/bridge` is a separate application with its own loopback/origin checks and
BridgeClient bearer validation. It deliberately does not contribute paths to
the parent FastAPI OpenAPI document.

`/mcp` is mounted as an MCP transport application rather than ordinary
FastAPI routes. Its protocol surface is therefore documented and tested
separately from the parent schema.

## Publication guidance

Do not treat the old `recallium-openapi.json` file as a current authority.
Retain it only as a historical local snapshot until a provenance-backed
replacement or archive decision is made.
