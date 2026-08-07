"""Generate docs/handbook.pdf — the PDF half of the demo corpus.

Committing a binary nobody can diff is worse than committing the script that
produces it. Run this to regenerate:

    uv run --directory api python scripts/make_handbook_pdf.py
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parents[2] / "docs" / "handbook.pdf"

PAGES: list[tuple[str, str]] = [
    (
        "Access and environments",
        """New engineers get read access to staging on day one.

Production access is granted after the first on-call shadow rotation and
requires a second approver from the platform team.

Credentials live in the shared vault. Nothing is ever pasted into a ticket, a
chat message or a pull request description, including short-lived tokens.""",
    ),
    (
        "Running services locally",
        """Every service in the monorepo ships a compose file.

The convention is that a single compose command must be enough to get a working
local environment with seeded data. Anything that needs a manual step after that
is considered a bug in the service, not in the onboarding.

If a service cannot be run locally at all, its README has to say so on the first
screen and point at the staging environment instead.""",
    ),
    (
        "Support rotation",
        """The weekday rotation is one primary and one secondary.

Handover happens at 10:00 in the support channel, with open incidents listed
explicitly rather than left implied by the channel history.

The primary is not expected to fix everything. The primary is expected to make
sure everything is either fixed, assigned, or written down.""",
    ),
]


def main() -> None:
    doc = pymupdf.open()
    for index, (heading, body) in enumerate(PAGES, start=1):
        page = doc.new_page()
        page.insert_text((72, 96), heading, fontsize=16)
        page.insert_text((72, 130), body, fontsize=11)
        page.insert_text((72, 760), f"Engineering Onboarding Handbook · {index}", fontsize=8)

    doc.set_metadata({"title": "Engineering Onboarding Handbook"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    doc.close()
    print(f"wrote {OUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()
