"""python -m app.ingest <path> [--force]

Indexes every Markdown file under <path>. Re-running is cheap: a document
whose text has not changed is skipped without re-embedding it.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from itertools import chain

from app import db
from app.ingest.connectors import DirConnector, iter_files
from app.ingest.pdf import PDFConnector
from app.ingest.pipeline import ingest


async def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingest")
    parser.add_argument("path", type=Path, help="file or directory to index")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed even when the content hash is unchanged",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.path.exists():
        print(f"no such path: {args.path}", file=sys.stderr)
        return 2

    pdfs, root = iter_files(args.path, {".pdf"})

    await db.bootstrap()
    await db.open_pool()
    try:
        docs = chain(
            DirConnector(args.path).load(),
            PDFConnector(pdfs, root=root).load(),
        )
        report = await ingest(docs, force=args.force)
    finally:
        await db.close_pool()

    if not args.quiet:
        for r in report.results:
            mark = {"created": "+", "updated": "~", "unchanged": "="}[r.status]
            suffix = f"{r.chunks} chunks" if r.chunks else "no change"
            print(f"  {mark} {r.path:<48} {suffix}")

    print(
        f"{report.documents} documents "
        f"({report.written} written, {report.unchanged} unchanged) · "
        f"{report.chunks} chunks · budget {report.chunk_budget} tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
