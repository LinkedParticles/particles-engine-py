"""Source trust rules table.

Revision ID: 004
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None

_NOW = datetime(2026, 5, 11, tzinfo=UTC)
_SYSTEM = "system"


def _row(
    scope: str, pattern: str, score: float | None, modifier: float | None, rationale: str
) -> dict:  # type: ignore[type-arg]
    return {
        "id": str(uuid.uuid4()),
        "scope": scope,
        "pattern": pattern,
        "score": score,
        "modifier": modifier,
        "rationale": rationale,
        "created_at": _NOW,
        "asserted_by": _SYSTEM,
    }


def upgrade() -> None:
    source_trust = op.create_table(
        "source_trust",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("scope", sa.String, nullable=False),
        sa.Column("pattern", sa.String, nullable=False, index=True),
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("modifier", sa.Float, nullable=True),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("asserted_by", sa.String, nullable=False),
    )

    op.bulk_insert(
        source_trust,
        [
            # Domain baseline scores
            _row(
                "domain",
                "api.numista.com",
                0.90,
                None,
                "Curated numismatic database; structured API",
            ),
            _row(
                "domain",
                "www.wikidata.org",
                0.85,
                None,
                "Community-curated structured knowledge graph",
            ),
            _row(
                "domain",
                "en.wikipedia.org",
                0.70,
                None,
                "High-quality encyclopedia; LLM-inferred particles",
            ),
            _row(
                "domain",
                "en.numista.com",
                0.65,
                None,
                "Numista HTML pages; LLM-inferred from richer HTML",
            ),
            _row("domain", "*", 0.50, None, "Default score for unknown domains"),
            # URL-pattern modifiers
            _row(
                "url_pattern",
                r"en\.numista\.com/(?:catalogue/pieces\d+\.html|\d+/?$)",
                None,
                +0.10,
                "Individual coin detail page; richer than search listing",
            ),
            _row(
                "url_pattern",
                r"en\.numista\.com/catalogue/index\.php",
                None,
                -0.15,
                "Search listing; lower fidelity than individual coin pages",
            ),
            _row(
                "url_pattern",
                r"en\.wikipedia\.org/wiki/Special:",
                None,
                -0.20,
                "Wikipedia Special pages; rarely contain substantive claims",
            ),
        ],
    )


def downgrade() -> None:
    op.drop_table("source_trust")
