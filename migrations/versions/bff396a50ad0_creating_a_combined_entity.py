"""creating a combined entity

Revision ID: bff396a50ad0
Revises: 8e95b58bb593
Create Date: 2025-12-09 20:19:15.250451

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "bff396a50ad0"
down_revision: Union[str, Sequence[str], None] = "8e95b58bb593"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "combined_entities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    with op.batch_alter_table("raw_entities", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("combined_entity_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_raw_entities_combined_entity_id",
            "combined_entities",
            ["combined_entity_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("raw_entities", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_raw_entities_combined_entity_id", type_="foreignkey"
        )
        batch_op.drop_column("combined_entity_id")

    op.drop_table("combined_entities")
