from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "8e95b58bb593"
down_revision: Union[str, Sequence[str], None] = "a684db4fd1ff"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table("entities", "raw_entities")
    with op.batch_alter_table("raw_entities") as batch_op:
        batch_op.drop_column("embedding")


def downgrade() -> None:
    with op.batch_alter_table("raw_entities") as batch_op:
        batch_op.add_column(sa.Column("embedding", sa.LargeBinary(), nullable=True))
    op.rename_table("raw_entities", "entities")
