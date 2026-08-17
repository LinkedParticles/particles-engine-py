"""${message}.

REVIEW ME: autogenerate is not magic — check for wrong drops, missing
server defaults, and constraint-name churn, then replace this paragraph
with real prose (which tables/columns, which ADR, why; see 018/019/020
for the house docstring style) before committing.

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""

import sqlalchemy as sa

from alembic import op
${imports if imports else ""}
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
