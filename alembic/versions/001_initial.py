"""Initial schema — v0.2 Core tables.

Revision ID: 001
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "corpus_entries",
        sa.Column("entry_id", sa.String, primary_key=True),
        sa.Column("uri_r", sa.String, nullable=True),
        sa.Column("source_type", sa.String, nullable=False),
        sa.Column("mutability", sa.String, nullable=False),
        sa.Column("fetch_policy", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deposited_by", sa.String, nullable=False),
        sa.Column("tags_json", sa.Text, nullable=False, default="[]"),
    )
    op.create_index("ix_corpus_entries_uri_r", "corpus_entries", ["uri_r"])

    op.create_table(
        "snapshots",
        sa.Column("snapshot_id", sa.String, primary_key=True),
        sa.Column("entry_id", sa.String, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String, nullable=False),
        sa.Column("etag", sa.String, nullable=True),
        sa.Column("last_modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("warc_record_type", sa.String, nullable=False),
        sa.Column("archive_path", sa.String, nullable=True),
        sa.Column("refers_to", sa.String, nullable=True),
        sa.Column("extraction_status", sa.String, nullable=False),
        sa.Column("author_id", sa.String, nullable=True),
        sa.Column("author_role", sa.String, nullable=True),
    )
    op.create_index("ix_snapshots_entry_id", "snapshots", ["entry_id"])
    op.create_index("ix_snapshots_content_hash", "snapshots", ["content_hash"])
    op.create_index("ix_snapshots_extraction_status", "snapshots", ["extraction_status"])
    op.create_index("ix_snapshots_entry_extraction", "snapshots", ["entry_id", "extraction_status"])

    op.create_table(
        "particles",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("confidence_value", sa.Float, nullable=False),
        sa.Column("confidence_variance", sa.Float, nullable=True),
        sa.Column("confidence_calibration_source", sa.String, nullable=False),
        sa.Column("confidence_calibration_method", sa.String, nullable=True),
        sa.Column("confidence_calibration_ref", sa.String, nullable=True),
        sa.Column("uncertainty_nature", sa.String, nullable=False),
        sa.Column("asserted_by", sa.String, nullable=False),
        sa.Column("asserted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("status_reason", sa.String, nullable=True),
        sa.Column("schema_version", sa.String, nullable=False),
        sa.Column("particle_type", sa.String, nullable=False, default="CLAIM"),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes", sa.String, nullable=True),
        sa.Column("provenance_json", sa.Text, nullable=False, default="[]"),
        sa.Column("extractor_ref_json", sa.Text, nullable=True),
        sa.Column("embedding_json", sa.Text, nullable=True),
    )
    op.create_index("ix_particles_status", "particles", ["status"])
    op.create_index("ix_particles_confidence_value", "particles", ["confidence_value"])
    op.create_index("ix_particles_status_confidence", "particles", ["status", "confidence_value"])

    op.create_table(
        "particle_provenance_edges",
        sa.Column("particle_id", sa.String, primary_key=True),
        sa.Column("corpus_entry_id", sa.String, primary_key=True),
        sa.Column("snapshot_id", sa.String, nullable=True),
    )
    op.create_index("ix_prov_edges_entry", "particle_provenance_edges", ["corpus_entry_id"])

    op.create_table(
        "trust_statements",
        sa.Column("statement_id", sa.String, primary_key=True),
        sa.Column("domain", sa.String, nullable=False),
        sa.Column("source_ref_type", sa.String, nullable=False),
        sa.Column("source_ref_value", sa.String, nullable=False),
        sa.Column("trust_rank", sa.Float, nullable=False),
        sa.Column("policy_provenance", sa.String, nullable=False),
        sa.Column("asserted_by", sa.String, nullable=False),
        sa.Column("asserted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("basis", sa.Text, nullable=True),
        sa.Column("review_id", sa.String, nullable=True),
    )
    op.create_index("ix_trust_domain", "trust_statements", ["domain"])
    op.create_index(
        "ix_trust_domain_ref",
        "trust_statements",
        ["domain", "source_ref_type", "source_ref_value"],
    )


def downgrade() -> None:
    op.drop_table("trust_statements")
    op.drop_table("particle_provenance_edges")
    op.drop_table("particles")
    op.drop_table("snapshots")
    op.drop_table("corpus_entries")
