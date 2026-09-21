"""Synthetic memory-space contracts, isolation, union retrieval and stable destinations."""

import asyncio
import hashlib
import secrets
from argparse import Namespace
from uuid import UUID

import pytest
from sqlalchemy import func, select, text, update

from app.domain.archive.models import RawArchiveItem
from app.domain.audit.models import AuditEvent
from app.domain.bridge.models import (
    BridgeAliasReceipt,
    BridgeArchive,
    BridgeBinding,
    BridgeClient,
    BridgeGrant,
    BridgeProject,
)
from app.domain.canonical_memory.models import CanonicalMemoryItem
from app.domain.derived_memory.models import Embedding, Fact, FtsEntry, MemoryLink, Summary
from app.infrastructure.db import get_session_factory


@pytest.fixture(autouse=True)
def local_host(client):
    client.base_url = "http://localhost"


@pytest.fixture
async def spaces(db_session):
    credentials = {name: secrets.token_urlsafe(32) for name in ("sol", "qwen", "empty")}
    for name, token in credentials.items():
        db_session.add(
            BridgeClient(id=name, credential_digest=hashlib.sha256(token.encode()).hexdigest())
        )
    await db_session.flush()
    for name in ("sol", "qwen"):
        db_session.add(BridgeProject(id=name + "-private", kind="private", owner_client_id=name))
    for name in ("shared", "topic", "replacement"):
        db_session.add(BridgeProject(id=name, kind="shared"))
    await db_session.flush()
    for name in ("sol", "qwen"):
        for project in (name + "-private", "shared", "topic", "replacement"):
            db_session.add(
                BridgeGrant(client_id=name, project_id=project, can_read=True, can_write=True)
            )
        db_session.add(
            BridgeBinding(client_id=name, destination="private", project_id=name + "-private")
        )
        db_session.add(BridgeBinding(client_id=name, destination="shared", project_id="shared"))
    await db_session.commit()
    return {name: {"Authorization": "Bearer " + token} for name, token in credentials.items()}


def note(**overrides):
    return {
        "destination": "private",
        "content": "Synthetic violet notebook remembers a failed approach.",
        "source_metadata": {
            "author_kind": "model",
            "model_label": "claimed-model",
            "conversation_id": "synthetic",
        },
        "idempotency_key": secrets.token_hex(10),
        **overrides,
    }


async def ingest(client, headers, **overrides):
    result = await client.post("/bridge/v1/ingest_memory", json=note(**overrides), headers=headers)
    assert result.status_code == 202, result.text
    return result.json()


async def search(client, headers, **overrides):
    return await client.post(
        "/bridge/v1/retrieve_memory",
        json={"query": "violet", "mode": "keyword", **overrides},
        headers=headers,
    )


async def index_note(db, archive_id, content="violet notebook", model="qwen-stored"):
    aid = UUID(archive_id)
    fact = Fact(
        raw_archive_id=aid,
        fact_text=content,
        source_span="violet",
        confidence_tier="high",
        derivation_method="llm_extraction",
        derivation_model=model,
    )
    db.add(fact)
    await db.flush()
    return fact


@pytest.mark.asyncio
async def test_private_shared_and_union(client, db_session, spaces):
    sol = await ingest(client, spaces["sol"])
    qwen = await ingest(client, spaces["qwen"])
    shared = await ingest(client, spaces["sol"], destination="shared")
    topic = await ingest(client, spaces["sol"], destination="shared", space_id="topic")
    assert sol["memory_space"] == {"id": "sol-private", "kind": "private"}
    assert shared["memory_space"]["id"] == "shared"
    for row in (sol, qwen, shared, topic):
        await index_note(db_session, row["archive_id"])
    await db_session.commit()
    for actor, private in [("sol", sol), ("qwen", qwen)]:
        response = await search(client, spaces[actor])
        assert response.status_code == 200
        assert {i["source_id"] for i in response.json()["items"]} == {
            private["archive_id"],
            shared["archive_id"],
            topic["archive_id"],
        }
        assert all(
            i["memory_space"]["id"] != ("qwen-private" if actor == "sol" else "sol-private")
            for i in response.json()["items"]
        )
    narrow = await search(client, spaces["sol"], space_ids=["shared", "topic"])
    assert {i["source_id"] for i in narrow.json()["items"]} == {
        shared["archive_id"],
        topic["archive_id"],
    }
    assert (await search(client, spaces["sol"], project_id="shared")).json()[
        "searched_space_ids"
    ] == ["shared"]
    a = await search(client, spaces["sol"], space_ids=["qwen-private"])
    b = await search(client, spaces["sol"], space_ids=["does-not-exist"])
    assert a.status_code == b.status_code == 403 and a.json() == b.json()
    assert (await search(client, spaces["empty"])).json()["items"] == []
    assert (await search(client, spaces["sol"], space_ids=[])).json()["items"] == []
    assert (
        await search(client, spaces["sol"], project_id="shared", space_ids=["shared"])
    ).status_code == 422
    assert (
        await client.post(
            "/bridge/v1/get_ingest_status",
            json={"archive_id": qwen["archive_id"]},
            headers=spaces["sol"],
        )
    ).status_code == 404
    assert (
        await client.post(
            "/bridge/v1/get_ingest_status",
            json={"archive_id": sol["archive_id"]},
            headers=spaces["sol"],
        )
    ).status_code == 200


@pytest.mark.asyncio
async def test_private_owner_runtime_and_operator_invariant(client, db_session, spaces):
    from app.domain.bridge.admin import provision

    args = Namespace(
        client="qwen",
        project="sol-private",
        read=True,
        write=True,
        credential_env=None,
        revoke=False,
        kind=None,
        bind=None,
    )
    with pytest.raises(ValueError, match="Another client"):
        await provision(args)
    # Even a corrupt direct grant cannot defeat the read/write ownership predicate.
    db_session.add(
        BridgeGrant(client_id="qwen", project_id="sol-private", can_read=True, can_write=True)
    )
    await db_session.commit()
    assert (await search(client, spaces["qwen"], project_id="sol-private")).status_code == 403
    request = note(destination=None, project_id="sol-private")
    assert (
        await client.post("/bridge/v1/ingest_memory", json=request, headers=spaces["qwen"])
    ).status_code == 403
    args.project = "topic"
    args.kind = "private"
    with pytest.raises(ValueError, match="Changing"):
        await provision(args)


@pytest.mark.asyncio
async def test_destination_errors_and_provisioning(client, db_session, spaces):
    from app.domain.bridge.admin import provision

    args = Namespace(
        client="sol",
        project="second-notebook",
        read=True,
        write=True,
        credential_env=None,
        revoke=False,
        kind="private",
        bind="private",
    )
    await provision(args)
    assert (await ingest(client, spaces["sol"]))["project_id"] == "second-notebook"
    for extras, expected in [
        ({"destination": "private", "space_id": "qwen-private"}, 422),
        ({"destination": "shared", "space_id": "sol-private"}, 403),
        ({"destination": "private", "project_id": "shared"}, 422),
        ({"destination": None}, 422),
    ]:
        assert (
            await client.post(
                "/bridge/v1/ingest_memory", json=note(**extras), headers=spaces["sol"]
            )
        ).status_code == expected
    assert (
        await client.post(
            "/bridge/v1/ingest_memory", json=note(destination="shared"), headers=spaces["empty"]
        )
    ).status_code == 403
    await db_session.execute(
        update(BridgeGrant)
        .where(BridgeGrant.client_id == "sol", BridgeGrant.project_id == "shared")
        .values(can_write=False)
    )
    await db_session.commit()
    assert (
        await client.post(
            "/bridge/v1/ingest_memory", json=note(destination="shared"), headers=spaces["sol"]
        )
    ).status_code == 403


@pytest.mark.asyncio
async def test_alias_replay_binding_change_and_concurrency(client, db_session, spaces):
    original = note(destination="shared", idempotency_key="stable")
    first = (
        await client.post("/bridge/v1/ingest_memory", json=original, headers=spaces["sol"])
    ).json()
    await db_session.execute(
        update(BridgeBinding)
        .where(BridgeBinding.client_id == "sol", BridgeBinding.destination == "shared")
        .values(project_id="replacement")
    )
    await db_session.commit()
    responses = await asyncio.gather(
        *[
            client.post("/bridge/v1/ingest_memory", json=original, headers=spaces["sol"])
            for _ in range(5)
        ]
    )
    assert all(
        r.status_code == 202
        and r.json()["archive_id"] == first["archive_id"]
        and r.json()["project_id"] == "shared"
        and r.json()["idempotent_replay"]
        for r in responses
    )
    second = await ingest(client, spaces["sol"], destination="shared", idempotency_key="new")
    assert second["project_id"] == "replacement"
    same_new = note(destination="private", idempotency_key="concurrent")
    responses = await asyncio.gather(
        *[
            client.post("/bridge/v1/ingest_memory", json=same_new, headers=spaces["sol"])
            for _ in range(5)
        ]
    )
    assert len({r.json()["archive_id"] for r in responses}) == 1
    assert sum(not r.json()["idempotent_replay"] for r in responses) == 1
    changed = {**original, "content": "A different synthetic note has different content."}
    assert (
        await client.post("/bridge/v1/ingest_memory", json=changed, headers=spaces["sol"])
    ).status_code == 409
    await db_session.execute(
        update(BridgeGrant)
        .where(BridgeGrant.client_id == "sol", BridgeGrant.project_id == "shared")
        .values(can_write=False)
    )
    await db_session.commit()
    assert (
        await client.post("/bridge/v1/ingest_memory", json=original, headers=spaces["sol"])
    ).status_code == 403
    assert await db_session.scalar(select(func.count()).select_from(RawArchiveItem)) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
async def test_union_paths_provenance_and_limits(client, db_session, spaces, monkeypatch, mode):
    from app.domain.derived_memory import service as derived

    async def embed(_):
        return [1.0] + [0.0] * 383

    monkeypatch.setattr(derived, "embed_text", embed)
    created = []
    for who, destination in [("sol", "private"), ("sol", "shared"), ("qwen", "private")]:
        row = await ingest(client, spaces[who], destination=destination)
        created.append(row)
        await index_note(db_session, row["archive_id"])
        db_session.add(
            Summary(
                raw_archive_id=UUID(row["archive_id"]),
                summary_text="violet summary",
                model_used="qwen-summary",
                derivation_method="llm_summarization",
            )
        )
        db_session.add(
            Embedding(
                raw_archive_id=UUID(row["archive_id"]),
                embedding=await embed(""),
                embedding_model=derived.ACTIVE_EMBEDDING_MODEL,
            )
        )
    await db_session.commit()
    r = await search(client, spaces["sol"], mode=mode, budget=20000)
    assert r.status_code == 200, r.text
    assert {i["source_id"] for i in r.json()["items"]} == {x["archive_id"] for x in created[:2]}
    for item in r.json()["items"]:
        p = item["provenance"]
        assert (
            p["authenticated_client"] == "sol"
            and p["source_metadata"]["model_label"] == "claimed-model"
        )
        assert p["processing"]["model"] in ("qwen-stored", "qwen-summary")
        assert p["retrieval"]["model"] in ("postgresql_fts", derived.ACTIVE_EMBEDDING_MODEL)
        assert "derivation_model" not in p
    r = await search(client, spaces["sol"], mode=mode, limit=1, budget=20000)
    assert len(r.json()["items"]) == 1 and r.json()["budget_used"] == len(
        r.json()["items"][0]["content"]
    )


@pytest.mark.asyncio
async def test_links_and_canonical_across_spaces(client, db_session, spaces):
    rows = [
        await ingest(client, spaces["sol"]),
        await ingest(client, spaces["sol"], destination="shared"),
        await ingest(client, spaces["qwen"]),
    ]
    facts = []
    for i, row in enumerate(rows):
        facts.append(
            await index_note(
                db_session, row["archive_id"], content="anchor" if i == 0 else "linked context"
            )
        )
    for f in facts[1:]:
        db_session.add(
            MemoryLink(
                source_fact_id=facts[0].id,
                target_fact_id=f.id,
                link_type="related",
                confidence=0.9,
                created_by="synthetic",
            )
        )
    canonical = []
    for aid, fid in [
        (UUID(rows[1]["archive_id"]), facts[1].id),
        (UUID(rows[1]["archive_id"]), facts[2].id),
        (None, None),
    ]:
        cm = CanonicalMemoryItem(
            raw_archive_id=aid, fact_id=fid, content="anchor canonical", promoted_from="fact"
        )
        db_session.add(cm)
        canonical.append(cm)
    await db_session.flush()
    await db_session.execute(
        text("UPDATE canonical_memory SET search_vector=to_tsvector('english',content)")
    )
    await db_session.commit()
    r = await search(client, spaces["sol"], query="anchor")
    items = r.json()["items"]
    assert str(facts[1].id) in {x["id"] for x in items}
    assert str(canonical[0].id) in {x["id"] for x in items}
    assert not {str(facts[2].id), str(canonical[1].id), str(canonical[2].id)} & {
        x["id"] for x in items
    }
    narrow = await search(client, spaces["sol"], query="anchor", space_ids=["sol-private"])
    assert {x["source_id"] for x in narrow.json()["items"]} == {rows[0]["archive_id"]}


@pytest.mark.asyncio
async def test_notes_without_facts_and_audit(client, db_session, spaces):
    row = await ingest(client, spaces["sol"])
    entry = FtsEntry(
        raw_archive_id=UUID(row["archive_id"]),
        text_content="violet hypothesis without validated facts",
    )
    db_session.add(entry)
    await db_session.flush()
    await db_session.execute(
        text("UPDATE fts_entries SET search_vector=to_tsvector('english',text_content)")
    )
    await db_session.commit()
    assert await db_session.scalar(select(func.count()).select_from(Fact)) == 0
    from app.domain.retrieval import service as retrieval

    retrieval._cache.clear()
    for _ in range(2):
        r = await search(client, spaces["sol"])
        assert r.json()["items"][0]["source_id"] == row["archive_id"]
        assert r.json()["items"][0]["provenance"]["processing"]["model"] is None
    assert not retrieval._cache
    events = (
        (
            await db_session.execute(
                select(AuditEvent).where(AuditEvent.event_type == "bridge_access")
            )
        )
        .scalars()
        .all()
    )
    searches = [e for e in events if e.operation_metadata["operation"] == "retrieve_memory"]
    assert len(searches) == 2 and searches[0].operation_metadata["searched_space_ids"] == [
        "replacement",
        "shared",
        "sol-private",
        "topic",
    ]
    assert "violet" not in str([e.operation_metadata for e in events])
    assert any(e.operation_metadata.get("resolved_destination") == "sol-private" for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["grant", "client"])
async def test_revocation_orders_against_inflight_read(
    client, db_session, spaces, monkeypatch, kind
):
    from app.domain.bridge import service

    entered, release, updating = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = service._retrieve
    once = True

    async def paused(*args, **kwargs):
        nonlocal once
        if once:
            once = False
            entered.set()
            await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, "_retrieve", paused)
    read = asyncio.create_task(search(client, spaces["sol"], space_ids=["shared"]))
    await asyncio.wait_for(entered.wait(), 5)

    async def revoke():
        async with get_session_factory()() as db:
            await db.execute(
                text("SELECT set_config('application_name','spaces-revocation-test',false)")
            )
            updating.set()
            if kind == "grant":
                await db.execute(
                    update(BridgeGrant)
                    .where(BridgeGrant.client_id == "sol", BridgeGrant.project_id == "shared")
                    .values(can_read=False)
                )
            else:
                await db.execute(
                    update(BridgeClient).where(BridgeClient.id == "sol").values(active=False)
                )
            await db.commit()

    revocation = asyncio.create_task(revoke())
    try:
        await asyncio.wait_for(updating.wait(), 5)
        for _ in range(100):
            blocked = await db_session.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity WHERE application_name='spaces-revocation-test' AND wait_event_type='Lock'"
                )
            )
            if blocked:
                break
            await asyncio.sleep(0.01)
        assert blocked and not revocation.done()
    finally:
        release.set()
    assert (await asyncio.wait_for(read, 5)).status_code == 200
    await asyncio.wait_for(revocation, 5)
    assert (await search(client, spaces["sol"], space_ids=["shared"])).status_code == (
        403 if kind == "grant" else 401
    )


@pytest.mark.asyncio
async def test_global_rank_budget_and_hybrid_above_twenty(client, db_session, spaces, monkeypatch):
    from app.domain.derived_memory import service as derived

    async def embed(_):
        return [1.0] + [0.0] * 383

    monkeypatch.setattr(derived, "embed_text", embed)
    for i in range(30):
        row = await ingest(client, spaces["sol"], destination="private" if i % 2 else "shared")
        aid = UUID(row["archive_id"])
        entry = FtsEntry(raw_archive_id=aid, text_content="violet " + str(i))
        db_session.add(entry)
        db_session.add(
            Embedding(
                raw_archive_id=aid,
                embedding=await embed(""),
                embedding_model=derived.ACTIVE_EMBEDDING_MODEL,
            )
        )
    await db_session.flush()
    await db_session.execute(
        text("UPDATE fts_entries SET search_vector=to_tsvector('english',text_content)")
    )
    await db_session.commit()
    r = await search(client, spaces["sol"], mode="hybrid", limit=30, budget=20000)
    assert r.status_code == 200, r.text
    assert len(r.json()["items"]) == 30
    assert {x["memory_space"]["id"] for x in r.json()["items"]} == {"sol-private", "shared"}
    limited = await search(client, spaces["sol"], mode="hybrid", limit=3, budget=100)
    assert len(limited.json()["items"]) <= 3
    assert (
        limited.json()["budget_used"]
        == sum(len(i["content"]) for i in limited.json()["items"])
        <= 100
    )


@pytest.mark.asyncio
async def test_alias_failure_atomicity(client, db_session, spaces, monkeypatch):
    from app.domain.bridge import service

    original = service.ingest_text_content

    async def broken(*args, **kwargs):
        await original(*args, **kwargs)
        raise service.BridgeError(409, "synthetic_failure")

    monkeypatch.setattr(service, "ingest_text_content", broken)
    result = await client.post("/bridge/v1/ingest_memory", json=note(), headers=spaces["sol"])
    assert result.status_code == 409
    for model in (RawArchiveItem, BridgeArchive, BridgeAliasReceipt):
        assert await db_session.scalar(select(func.count()).select_from(model)) == 0


@pytest.mark.asyncio
async def test_global_keyword_rank_has_no_private_first_bias(client, db_session, spaces):
    private = await ingest(client, spaces["sol"])
    shared = await ingest(client, spaces["sol"], destination="shared")
    await index_note(db_session, private["archive_id"], content="violet")
    await index_note(db_session, shared["archive_id"], content="violet " * 20)
    await db_session.commit()
    result = await search(client, spaces["sol"], limit=1)
    assert result.json()["items"][0]["source_id"] == shared["archive_id"]


@pytest.mark.asyncio
async def test_schema_requires_private_owner(db_session, spaces):
    from sqlalchemy.exc import IntegrityError

    for project in (
        BridgeProject(id="bad-private", kind="private"),
        BridgeProject(id="bad-shared", kind="shared", owner_client_id="sol"),
    ):
        with pytest.raises(IntegrityError):
            async with db_session.begin_nested():
                db_session.add(project)
                await db_session.flush()
