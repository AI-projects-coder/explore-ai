"""runs table: one row per agent run, with its status and final report

Revision ID: 0001
Revises:
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE runs (
            run_id          text PRIMARY KEY,
            request         text NOT NULL,
            status          text NOT NULL CHECK (status IN
                                ('running', 'awaiting_review', 'approved', 'rejected', 'partial', 'failed')),
            ticker          text,
            report_markdown text,
            spend           jsonb,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now(),
            finished_at     timestamptz
        )
    """)
    # `list` sorts by newest; a v3 alert looks for runs stuck in 'running'
    op.execute("CREATE INDEX runs_created_at_idx ON runs (created_at DESC)")
    op.execute("CREATE INDEX runs_open_idx ON runs (updated_at) WHERE status IN ('running', 'awaiting_review')")


def downgrade() -> None:
    op.execute("DROP TABLE runs")
