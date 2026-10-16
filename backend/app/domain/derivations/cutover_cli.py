"""Operator entry point; no app lifespan, worker, provider or migration execution."""
import argparse
import asyncio
import json
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.infrastructure.settings import get_settings
from app.domain.derivations.cutover import (
    CutoverError, MAX_ARTIFACT_BYTES, bounded_json, human_report, inspect_population, prepare,
)


async def run(args):
    engine = create_async_engine(get_settings().database_url, pool_size=1, max_overflow=0)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            if args.action == 'prepare':
                with open(args.plan, 'rb') as source:
                    payload = source.read(MAX_ARTIFACT_BYTES + 1)
                if len(payload) > MAX_ARTIFACT_BYTES:
                    raise CutoverError('artifact_limit_exceeded')
                try:
                    plan = json.loads(payload)
                except (ValueError, UnicodeError):
                    raise CutoverError('invalid_plan') from None
                result = await prepare(session, plan)
            else:
                result = await inspect_population(session, kind=args.action, cutover=args.cutover)
            print(bounded_json(result))
            print(human_report(result), file=sys.stderr, end='')
            return 0 if result.get('ok', True) else 2
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('plan', 'check'):
        command = commands.add_parser(name, help='Read-only full-population snapshot')
        command.add_argument('--cutover', action='store_true', help='Also require this to be the only database client')
    command = commands.add_parser('prepare', help='OFF-only mutation; rerun same plan to resume')
    command.add_argument('--plan', required=True)
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except CutoverError as error:
        print(json.dumps({'ok': False, 'reason': str(error)}), file=sys.stderr)
        return 2
    except Exception:
        # Database connection messages can contain passwords/URLs/source values.
        print('{"ok":false,"reason":"operator_tool_failed"}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
