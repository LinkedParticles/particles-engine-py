"""Per-provider calibration table (1.54.0).

Move calibration off ``extractor_records`` (one row per extractor) into a new
``extractor_calibrations`` table keyed by ``(extractor_id, provider_model)``, so
several models' calibrations coexist and the pipeline selects the one matching
the configured extraction model. A swap back to a previously
benchmarked model restores its calibration without re-fitting. Existing single
calibrations migrate keyed by their recorded ``provider_model`` (or the
historical default ``anthropic:claude-sonnet-4-6`` when null). No SCHEMA_VERSION
change — calibration is Extension A, not part of the particle interchange.

Revision ID: 025
Revises: 024
"""

import json

import sqlalchemy as sa

from alembic import op

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None

# A calibration fitted before the provider_model key existed is keyed
# under the historical default model (kept in sync with
# particles.store.extractor_store.LEGACY_PROVIDER_MODEL).
_LEGACY_PROVIDER_MODEL = "anthropic:claude-sonnet-4-6"


def upgrade() -> None:
    op.create_table(
        "extractor_calibrations",
        sa.Column("extractor_id", sa.String(), nullable=False),
        sa.Column("provider_model", sa.String(), nullable=False),
        sa.Column("calibration_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("extractor_id", "provider_model"),
    )
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT extractor_id, calibration_json FROM extractor_records "
            "WHERE calibration_json IS NOT NULL"
        )
    ).fetchall()
    for extractor_id, cal_json in rows:
        try:
            provider_model = json.loads(cal_json).get("provider_model") or _LEGACY_PROVIDER_MODEL
        except (ValueError, TypeError):
            provider_model = _LEGACY_PROVIDER_MODEL
        bind.execute(
            sa.text(
                "INSERT INTO extractor_calibrations "
                "(extractor_id, provider_model, calibration_json) "
                "VALUES (:e, :p, :c)"
            ),
            {"e": extractor_id, "p": provider_model, "c": cal_json},
        )
    with op.batch_alter_table("extractor_records") as batch:
        batch.drop_column("calibration_json")


def downgrade() -> None:
    with op.batch_alter_table("extractor_records") as batch:
        batch.add_column(sa.Column("calibration_json", sa.Text(), nullable=True))
    bind = op.get_bind()
    # The old column held at most one calibration per extractor; restore the
    # first row per extractor and drop the rest.
    rows = bind.execute(
        sa.text("SELECT extractor_id, calibration_json FROM extractor_calibrations")
    ).fetchall()
    seen: set[str] = set()
    for extractor_id, cal_json in rows:
        if extractor_id in seen:
            continue
        seen.add(extractor_id)
        bind.execute(
            sa.text("UPDATE extractor_records SET calibration_json = :c WHERE extractor_id = :e"),
            {"c": cal_json, "e": extractor_id},
        )
    op.drop_table("extractor_calibrations")
