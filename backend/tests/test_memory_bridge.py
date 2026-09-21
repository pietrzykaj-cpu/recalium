"""Authorization tests: disposable PostgreSQL only; never run against live DB."""

import hashlib
import secrets

import pytest
from sqlalchemy import select

from app.domain.bridge.models import BridgeClient, BridgeGrant, BridgeProject


@pytest.fixture
async def identities(db_session):
    tokens = {name: secrets.token_urlsafe(32) for name in ("qwen", "sol", "reader", "writer")}
    db_session.add_all([BridgeProject(id=p) for p in ("shared", "private")])
    await db_session.flush()
    for name, token in tokens.items():
        db_session.add(
            BridgeClient(id=name, credential_digest=hashlib.sha256(token.encode()).hexdigest())
        )
    await db_session.flush()
    for name in tokens:
        db_session.add(
            BridgeGrant(
                client_id=name,
                project_id="shared",
                can_read=name != "writer",
                can_write=name != "reader",
            )
        )
    db_session.add(
        BridgeGrant(client_id="qwen", project_id="private", can_read=True, can_write=True)
    )
    await db_session.commit()
    return {name: {"Authorization": f"Bearer {token}"} for name, token in tokens.items()}


@pytest.fixture(autouse=True)
def local_bridge_host(client):
    client.base_url = "http://localhost"


def payload(project="shared", key="one"):
    return {
        "project_id": project,
        "content": "Synthetic violet mug belongs beside the laptop.",
        "source_metadata": {"author_kind": "human", "conversation_id": "synthetic"},
        "idempotency_key": key,
    }


@pytest.mark.asyncio
async def test_auth_and_permissions(client, identities):
    url = "/bridge/v1/ingest_memory"
    assert (await client.post(url, json=payload())).status_code == 401
    assert (await client.post(url, json=payload(), headers=identities["reader"])).status_code == 403
    assert (
        await client.post(url, json=payload("private"), headers=identities["sol"])
    ).status_code == 403
    assert (await client.post(url, json=payload(), headers=identities["writer"])).status_code == 202
    assert (
        await client.post(
            "/bridge/v1/retrieve_memory",
            json={"project_id": "shared", "query": "mug"},
            headers=identities["writer"],
        )
    ).status_code == 403


@pytest.mark.asyncio
async def test_identity_spoofing_and_metadata(client, identities, db_session):
    from app.domain.archive.models import RawArchiveItem
    from app.domain.audit.models import AuditEvent

    for field in ("actor", "client_identity", "allow_external", "processing_mode", "project_hint"):
        assert (
            await client.post(
                "/bridge/v1/ingest_memory",
                json={**payload(), field: "sol"},
                headers=identities["qwen"],
            )
        ).status_code == 422
    bad = payload()
    bad["source_metadata"]["actor"] = "sol"
    assert (
        await client.post("/bridge/v1/ingest_memory", json=bad, headers=identities["qwen"])
    ).status_code == 422
    r = await client.post(
        "/bridge/v1/ingest_memory",
        json=payload(),
        headers={**identities["qwen"], "X-Client-Identity": "sol"},
    )
    assert r.status_code == 202
    import uuid

    archive = await db_session.get(RawArchiveItem, uuid.UUID(r.json()["archive_id"]))
    assert archive.metadata_json["client_identity"] == "qwen"
    assert archive.metadata_json["processing_mode"] == "local_only"
    audit = (
        await db_session.execute(select(AuditEvent).where(AuditEvent.event_type == "ingest"))
    ).scalar_one()
    assert audit.actor == "qwen"


@pytest.mark.asyncio
async def test_idempotency_isolation_conflict_and_status(client, identities, db_session):
    from sqlalchemy import func

    from app.domain.jobs.models import Job

    results = []
    for who, project in [("qwen", "shared"), ("sol", "shared"), ("qwen", "private")]:
        response = await client.post(
            "/bridge/v1/ingest_memory", json=payload(project), headers=identities[who]
        )
        assert response.status_code == 202
        results.append(response.json()["archive_id"])
    assert len(set(results)) == 3
    replay = await client.post(
        "/bridge/v1/ingest_memory", json=payload(), headers=identities["qwen"]
    )
    assert replay.json()["idempotent_replay"] and replay.json()["archive_id"] == results[0]
    changed = payload()
    changed["source_metadata"]["author_kind"] = "model"
    assert (
        await client.post("/bridge/v1/ingest_memory", json=changed, headers=identities["qwen"])
    ).status_code == 409
    assert (await db_session.scalar(select(func.count()).select_from(Job))) == 3
    for who, project, aid, expected in [
        ("reader", "shared", results[0], 200),
        ("writer", "shared", results[0], 403),
        ("sol", "shared", results[2], 404),
        ("sol", "private", results[2], 403),
    ]:
        response = await client.post(
            "/bridge/v1/get_ingest_status",
            json={"project_id": project, "archive_id": aid},
            headers=identities[who],
        )
        assert response.status_code == expected
        if expected == 200:
            assert response.json()["jobs"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_concurrent_idempotency(client, identities, db_session):
    import asyncio

    from sqlalchemy import func

    from app.domain.jobs.models import Job

    responses = await asyncio.gather(
        *[
            client.post("/bridge/v1/ingest_memory", json=payload(), headers=identities["qwen"])
            for _ in range(5)
        ]
    )
    assert all(r.status_code == 202 for r in responses)
    assert len({r.json()["archive_id"] for r in responses}) == 1
    assert sum(not r.json()["idempotent_replay"] for r in responses) == 1
    assert (await db_session.scalar(select(func.count()).select_from(Job))) == 1


@pytest.mark.asyncio
async def test_scope_links_legacy_cache_audit_revocation(client, identities, db_session):
    import uuid

    from sqlalchemy import text, update

    from app.domain.archive.models import RawArchiveItem
    from app.domain.audit.models import AuditEvent
    from app.domain.derived_memory.models import Fact
    from app.domain.retrieval import service as retrieval

    ids = []
    for project in ("shared", "private"):
        r = await client.post(
            "/bridge/v1/ingest_memory", json=payload(project), headers=identities["qwen"]
        )
        ids.append(uuid.UUID(r.json()["archive_id"]))
    legacy = RawArchiveItem(
        source_type="test",
        raw_content="violet legacy secret",
        content_hash="legacy",
        metadata_json={"project_hint": "shared", "client_identity": "sol"},
    )
    db_session.add(legacy)
    await db_session.flush()
    ids.append(legacy.id)
    facts = []
    for aid, label in zip(ids, ("shared", "private", "legacy")):
        f = Fact(
            raw_archive_id=aid,
            fact_text=f"violet {label} secret",
            source_span="violet",
            confidence_tier="high",
            derivation_method="test",
            derivation_model="synthetic",
        )
        db_session.add(f)
        facts.append(f)
    await db_session.flush()
    # Higher-ranked forbidden links must not suppress the allowed target at LIMIT.
    for target in facts[1:]:
        await db_session.execute(
            text(
                "INSERT INTO memory_links(id,source_fact_id,target_fact_id,link_type,confidence,created_by,created_at) VALUES (:id,:s,:t,'related',0.99,'test',now())"
            ),
            {"id": uuid.uuid4(), "s": facts[0].id, "t": target.id},
        )
    await db_session.commit()
    req = {"project_id": "shared", "query": "violet", "mode": "keyword"}
    retrieval._cache.clear()
    unrestricted = await retrieval.retrieve(
        db_session, retrieval.RetrievalRequest(query="violet", mode="keyword")
    )
    await db_session.commit()
    assert str(ids[2]) in {item.source_id for item in unrestricted.items}
    before = dict(retrieval._cache)
    for who, project, expected in [
        ("qwen", "shared", ids[0]),
        ("qwen", "private", ids[1]),
        ("sol", "shared", ids[0]),
        ("sol", "shared", ids[0]),
    ]:
        r = await client.post(
            "/bridge/v1/retrieve_memory",
            json={**req, "project_id": project},
            headers=identities[who],
        )
        assert r.status_code == 200, r.text
        assert r.json()["items"]
        assert {x["source_id"] for x in r.json()["items"]} == {str(expected)}
        assert all(x["provenance"]["authenticated_client"] == "qwen" for x in r.json()["items"])
    assert dict(retrieval._cache) == before
    events = (
        (
            await db_session.execute(
                select(AuditEvent).where(AuditEvent.event_type == "bridge_access")
            )
        )
        .scalars()
        .all()
    )
    assert sum(e.operation_metadata["operation"] == "retrieve_memory" for e in events) == 4
    assert "violet" not in str([e.operation_metadata for e in events])
    await db_session.execute(
        update(BridgeGrant).where(BridgeGrant.client_id == "sol").values(can_read=False)
    )
    await db_session.commit()
    assert (
        await client.post("/bridge/v1/retrieve_memory", json=req, headers=identities["sol"])
    ).status_code == 403
    await db_session.execute(
        update(BridgeClient).where(BridgeClient.id == "qwen").values(active=False)
    )
    await db_session.commit()
    assert (
        await client.post("/bridge/v1/retrieve_memory", json=req, headers=identities["qwen"])
    ).status_code == 401
    denied = (
        (
            await db_session.execute(
                select(AuditEvent).where(AuditEvent.event_type == "bridge_access")
            )
        )
        .scalars()
        .all()
    )
    assert sum(e.operation_metadata["outcome"] == "denied" for e in denied) == 2


@pytest.mark.asyncio
async def test_transport_host_origin_and_bounds(client, identities):
    req = {"project_id": "shared", "query": "violet"}
    for header in ({"Origin": "https://evil.example"}, {"Host": "evil.example"}):
        assert (
            await client.post(
                "/bridge/v1/retrieve_memory", json=req, headers={**identities["qwen"], **header}
            )
        ).status_code == 403
    for bad in ({"budget": 1000000}, {"limit": 0}, {"project_id": ""}, {"query": ""}):
        assert (
            await client.post(
                "/bridge/v1/retrieve_memory", json={**req, **bad}, headers=identities["qwen"]
            )
        ).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
async def test_all_candidate_paths_and_deleted_sources(
    client, identities, db_session, monkeypatch, mode
):
    import uuid

    from sqlalchemy import text, update

    from app.domain.archive.models import RawArchiveItem
    from app.domain.canonical_memory.models import CanonicalMemoryItem
    from app.domain.derived_memory import service as derived
    from app.domain.derived_memory.models import Embedding, Fact, FtsEntry, Summary

    async def embed(_):
        return [1.0] + [0.0] * 383

    monkeypatch.setattr(derived, "embed_text", embed)
    archives = []
    for project in ("shared", "private"):
        r = await client.post(
            "/bridge/v1/ingest_memory", json=payload(project), headers=identities["qwen"]
        )
        archives.append(uuid.UUID(r.json()["archive_id"]))
    legacy = RawArchiveItem(source_type="test", raw_content="violet legacy", content_hash="legacy")
    db_session.add(legacy)
    await db_session.flush()
    archives.append(legacy.id)
    facts = []
    for aid in archives:
        f = Fact(
            raw_archive_id=aid,
            fact_text="violet fact",
            source_span="violet",
            confidence_tier="high",
            derivation_method="test",
            derivation_model="synthetic",
        )
        db_session.add(f)
        facts.append(f)
        db_session.add(
            Summary(
                raw_archive_id=aid,
                summary_text="violet summary",
                model_used="synthetic",
                derivation_method="test",
            )
        )
        db_session.add(
            Embedding(
                raw_archive_id=aid,
                embedding=await embed(""),
                embedding_model=derived.ACTIVE_EMBEDDING_MODEL,
            )
        )
        db_session.add(FtsEntry(raw_archive_id=aid, text_content="violet excerpt"))
    await db_session.flush()
    # Allowed canonical, cross-project fact reference, and orphan canonical.
    canonical = []
    for aid, fid in [
        (archives[0], facts[0].id),
        (archives[0], facts[1].id),
        (None, None),
        (archives[1], facts[1].id),
    ]:
        cm = CanonicalMemoryItem(
            raw_archive_id=aid, fact_id=fid, content="violet canonical", promoted_from="fact"
        )
        db_session.add(cm)
        canonical.append(cm)
    await db_session.flush()
    # ORM fixture has a fetched rather than generated canonical/FTS column.
    await db_session.execute(
        text("UPDATE canonical_memory SET search_vector = to_tsvector('english', content)")
    )
    await db_session.execute(
        text("UPDATE fts_entries SET search_vector = to_tsvector('english', text_content)")
    )
    await db_session.commit()
    request = {"project_id": "shared", "query": "violet", "mode": mode, "budget": 20000}
    r = await client.post("/bridge/v1/retrieve_memory", json=request, headers=identities["sol"])
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert items
    assert {x["source_id"] for x in items} == {str(archives[0])}
    assert not {str(c.id) for c in canonical[1:]} & {x["id"] for x in items}
    if mode == "keyword":
        assert {"canonical", "fact", "excerpt"} <= {x["type"] for x in items}
    if mode == "semantic":
        assert "summary" in {x["type"] for x in items}
    await db_session.execute(
        update(RawArchiveItem)
        .where(RawArchiveItem.id == archives[0])
        .values(deleted_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    )
    await db_session.commit()
    assert (
        await client.post("/bridge/v1/retrieve_memory", json=request, headers=identities["sol"])
    ).json()["items"] == []
    assert (
        await client.post(
            "/bridge/v1/get_ingest_status",
            json={"project_id": "shared", "archive_id": str(archives[0])},
            headers=identities["sol"],
        )
    ).status_code == 404
    assert (
        await client.post("/bridge/v1/ingest_memory", json=payload(), headers=identities["qwen"])
    ).status_code == 409


@pytest.mark.asyncio
async def test_transaction_failure_does_not_leave_archive(
    client, identities, db_session, monkeypatch
):
    from sqlalchemy import func

    from app.domain.archive.models import RawArchiveItem
    from app.domain.bridge import service

    original = service.ingest_text_content

    async def fail_after_ingest(*args, **kwargs):
        await original(*args, **kwargs)
        raise service.BridgeError(409, "synthetic_failure")

    monkeypatch.setattr(service, "ingest_text_content", fail_after_ingest)
    assert (
        await client.post("/bridge/v1/ingest_memory", json=payload(), headers=identities["qwen"])
    ).status_code == 409
    assert await db_session.scalar(select(func.count()).select_from(RawArchiveItem)) == 0


@pytest.mark.asyncio
async def test_catalog_only_three_tools():
    from app.api.bridge import bridge_mcp

    tools = await bridge_mcp.list_tools()
    assert {t.name for t in tools} == {
        "retrieve_memory",
        "build_context_packet",
        "ingest_memory",
        "get_ingest_status",
    }
    for tool in tools:
        assert tool.annotations.readOnlyHint == (tool.name != "ingest_memory")
        assert tool.inputSchema["properties"].keys() == {"data"}


@pytest.mark.asyncio
async def test_link_scope_before_limit_and_allowed_link(client, identities, db_session):
    import uuid

    from app.domain.derived_memory.models import Fact, MemoryLink
    from app.domain.retrieval.service import _traverse_links

    archives = []
    for project, key in [("shared", "source"), ("shared", "target"), ("private", "private")]:
        r = await client.post(
            "/bridge/v1/ingest_memory", json=payload(project, key), headers=identities["qwen"]
        )
        archives.append(uuid.UUID(r.json()["archive_id"]))
    facts = []
    for i in range(14):
        aid = archives[0] if i == 0 else archives[1] if i == 1 else archives[2]
        f = Fact(
            raw_archive_id=aid,
            fact_text="anchor" if i == 0 else "linked context",
            source_span="context",
            confidence_tier="high",
            derivation_method="test",
            derivation_model="synthetic",
        )
        db_session.add(f)
        facts.append(f)
    await db_session.flush()
    for index, target in enumerate(facts[1:]):
        db_session.add(
            MemoryLink(
                source_fact_id=facts[0].id,
                target_fact_id=target.id,
                link_type="related",
                confidence=0.1 if index == 0 else 0.99,
                created_by="test",
            )
        )
    await db_session.commit()
    result = await _traverse_links(
        db_session, [str(archives[0])], max_links=1, bridge_project_id="shared"
    )
    assert [item["id"] for item in result] == [str(facts[1].id)]
    assert await _traverse_links(db_session, [str(archives[0])], bridge_project_id="private") == []
    r = await client.post(
        "/bridge/v1/retrieve_memory",
        json={"project_id": "shared", "query": "anchor", "mode": "keyword"},
        headers=identities["sol"],
    )
    assert str(facts[1].id) in {i["id"] for i in r.json()["items"]}
    assert all(i["source_id"] != str(archives[2]) for i in r.json()["items"])


@pytest.mark.asyncio
async def test_scope_before_candidate_limit(client, identities, db_session):
    import uuid

    from app.domain.derived_memory.models import Fact

    ids = []
    for project in ("shared", "private"):
        r = await client.post(
            "/bridge/v1/ingest_memory", json=payload(project), headers=identities["qwen"]
        )
        ids.append(uuid.UUID(r.json()["archive_id"]))
    for i in range(70):
        db_session.add(
            Fact(
                raw_archive_id=ids[1],
                fact_text="violet " * 20,
                source_span="violet",
                confidence_tier="high",
                derivation_method="test",
                derivation_model="synthetic",
            )
        )
    db_session.add(
        Fact(
            raw_archive_id=ids[0],
            fact_text="violet",
            source_span="violet",
            confidence_tier="high",
            derivation_method="test",
            derivation_model="synthetic",
        )
    )
    await db_session.commit()
    r = await client.post(
        "/bridge/v1/retrieve_memory",
        json={"project_id": "shared", "query": "violet", "mode": "keyword"},
        headers=identities["sol"],
    )
    assert r.status_code == 200
    assert len(r.json()["items"]) == 1
    assert r.json()["items"][0]["source_id"] == str(ids[0])


@pytest.mark.asyncio
async def test_operator_grants_rotation_and_revoke(client, db_session, monkeypatch):
    from argparse import Namespace

    from app.domain.bridge.admin import provision

    first, second = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    monkeypatch.setenv("SYNTHETIC_BRIDGE_CREDENTIAL", first)
    args = Namespace(
        client="operator_test",
        project="operator_project",
        read=True,
        write=False,
        revoke=False,
        credential_env="SYNTHETIC_BRIDGE_CREDENTIAL",
    )
    await provision(args)
    query = {"project_id": args.project, "query": "synthetic", "mode": "keyword"}

    async def get(token):
        return await client.post(
            "/bridge/v1/retrieve_memory", json=query, headers={"Authorization": "Bearer " + token}
        )

    assert (await get(first)).status_code == 200
    monkeypatch.setenv("SYNTHETIC_BRIDGE_CREDENTIAL", second)
    await provision(args)
    assert (await get(first)).status_code == 401
    assert (await get(second)).status_code == 200
    args.credential_env = None
    args.read = False
    await provision(args)
    assert (await get(second)).status_code == 403
    args.revoke = True
    await provision(args)
    assert (await get(second)).status_code == 401
    stored = await db_session.get(BridgeClient, args.client)
    assert stored.credential_digest == hashlib.sha256(second.encode()).hexdigest()
    assert stored.credential_digest not in (first, second)
