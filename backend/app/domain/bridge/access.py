"""Memory-space authorization. Lock order: client, spaces by id, grants by id."""

from sqlalchemy import or_, select

from app.domain.bridge.models import BridgeGrant, BridgeProject


async def permitted_spaces(session, actor, requested=None, *, write=False):
    permission = BridgeGrant.can_write if write else BridgeGrant.can_read
    query = (
        select(BridgeProject)
        .join(BridgeGrant, BridgeGrant.project_id == BridgeProject.id)
        .where(
            BridgeGrant.client_id == actor,
            permission.is_(True),
            or_(BridgeProject.kind == "shared", BridgeProject.owner_client_id == actor),
        )
    )
    if requested is not None:
        query = query.where(BridgeProject.id.in_(requested))
    projects = (
        (
            await session.execute(
                query.order_by(BridgeProject.id).with_for_update(read=True, of=BridgeProject)
            )
        )
        .scalars()
        .all()
    )
    # Recheck grants while acquiring their locks. A concurrent direct revocation
    # may have committed between the first query and this one.
    grants = (
        (
            await session.execute(
                select(BridgeGrant)
                .where(
                    BridgeGrant.client_id == actor,
                    BridgeGrant.project_id.in_([p.id for p in projects]),
                    permission.is_(True),
                )
                .order_by(BridgeGrant.project_id)
                .with_for_update(read=True)
            )
        )
        .scalars()
        .all()
    )
    ids = {g.project_id for g in grants}
    return {p.id: p for p in projects if p.id in ids}
