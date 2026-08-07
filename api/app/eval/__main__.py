"""python -m app.eval [--k 5] [--out ../eval-results.md]

Runs the golden set through every retrieval configuration and prints the table.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app import db
from app.config import get_settings
from app.eval.runner import as_markdown, load_golden, run, validate


async def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.eval")
    parser.add_argument("--k", type=int, default=5, help="how many results to score")
    parser.add_argument("--out", type=Path, help="also write the table to this file")
    parser.add_argument(
        "--misses", action="store_true", help="list the questions nothing answered"
    )
    args = parser.parse_args()

    questions = load_golden()

    await db.open_pool()
    try:
        rows = await db.fetch_all("SELECT path, text FROM document")
        corpus = {path: text for path, text in rows}
        if not corpus:
            print(
                "the corpus is empty — run `python -m app.ingest ../docs` first",
                file=sys.stderr,
            )
            return 2

        problems = validate(questions, corpus)
        if problems:
            print("the golden set does not match the corpus:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 2

        results = await run(questions, k=args.k)
    finally:
        await db.close_pool()

    settings = get_settings()
    header = (
        f"Corpus: {len(corpus)} documents · embeddings {settings.embed_model} · "
        f"re-ranker {settings.rerank_model}"
    )
    table = as_markdown(results, len(questions))
    print(header)
    print()
    print(table)

    if args.misses:
        for name, score in results.items():
            if score.misses:
                print(f"\nnot found in top {args.k} by {name}:")
                for question in score.misses:
                    print(f"  - {question}")

    if args.out:
        args.out.write_text(f"{header}\n\n{table}\n")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
