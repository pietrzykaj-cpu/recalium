"""Operator-only provisioning. No administrative HTTP/MCP endpoint is exposed.

Credentials are read from the named environment variable, never printed or stored
in plaintext. Generate at least 32 random bytes and keep the value in .env.
"""

import argparse
import asyncio
import os
import re

from sqlalchemy import select

from app.domain.bridge.models import BridgeBinding, BridgeClient, BridgeGrant, BridgeProject
from app.domain.bridge.service import audit, digest
from app.infrastructure.db import get_session_factory


async def provision(args):
    for value in (args.client, args.project):
        if value is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
            raise ValueError("Invalid client/project identifier")
    async with get_session_factory()() as session:
        client = (
            await session.execute(
                select(BridgeClient).where(BridgeClient.id == args.client).with_for_update()
            )
        ).scalar_one_or_none()
        if args.revoke:
            if client is None:
                raise ValueError("Unknown client")
            client.active = False
        else:
            if not args.project:
                raise ValueError("--project is required")
            credential = os.environ.get(args.credential_env, "") if args.credential_env else ""
            if client is None and not credential:
                raise ValueError("New clients require --credential-env")
            if credential and (not 43 <= len(credential) <= 256 or len(set(credential)) < 16):
                raise ValueError("Use a randomly generated credential of at least 32 bytes")
            if client is None:
                client = BridgeClient(id=args.client, credential_digest=digest(credential))
                session.add(client)
            elif credential:
                client.credential_digest = digest(credential)
                client.active = True
            await session.flush()
            space = (
                await session.execute(
                    select(BridgeProject).where(BridgeProject.id == args.project).with_for_update()
                )
            ).scalar_one_or_none()
            requested_kind = getattr(args, "kind", None)
            if space is None:
                kind = requested_kind or "shared"
                space = BridgeProject(
                    id=args.project,
                    kind=kind,
                    owner_client_id=args.client if kind == "private" else None,
                )
                session.add(space)
                await session.flush()
            elif requested_kind is not None and requested_kind != space.kind:
                raise ValueError("Changing a memory-space kind requires a separate owner operation")
            if space.kind == "private" and space.owner_client_id != args.client:
                raise ValueError("Another client owns this private memory space")
            grant = await session.get(BridgeGrant, (args.client, args.project))
            if grant is None:
                grant = BridgeGrant(client_id=args.client, project_id=args.project)
                session.add(grant)
            grant.can_read, grant.can_write = args.read, args.write
            destination = getattr(args, "bind", None)
            if destination:
                if destination != space.kind or not args.write:
                    raise ValueError("Binding requires a matching space kind and write grant")
                binding = await session.get(BridgeBinding, (args.client, destination))
                if binding is None:
                    session.add(
                        BridgeBinding(
                            client_id=args.client, destination=destination, project_id=space.id
                        )
                    )
                else:
                    binding.project_id = space.id
        audit(
            session,
            "local_operator",
            "provision",
            args.project,
            "allowed",
            client_id=args.client,
            revoked=args.revoke,
            destination_binding=getattr(args, "bind", None),
        )
        await session.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True)
    parser.add_argument("--project", "--space", dest="project")
    parser.add_argument("--kind", choices=["private", "shared"])
    parser.add_argument("--bind", choices=["private", "shared"])
    parser.add_argument("--credential-env")
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--revoke", action="store_true")
    args = parser.parse_args()
    asyncio.run(provision(args))
    print("Bridge permissions updated.")


if __name__ == "__main__":
    main()
