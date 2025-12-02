"""CLI helper for stamping an existing SQLite database to the current Alembic head.

This script updates the Alembic version state of a target SQLite database
without running any migrations. It is intended for adopting pre-existing
databases into the Alembic migration history by marking them as being at
the current head revision.
"""

from pathlib import Path
import argparse

from alembic.config import Config
from alembic import command


def stamp_db(db_path: Path):
    """Stamp the given SQLite database to the current Alembic head revision.

    This updates the Alembic version table in the target database to match
    the latest migration (head) without applying any schema changes.

    Args:
        db_path: Path to the SQLite database file to stamp.
    """
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.stamp(cfg, "head")


def main():
    """Parse CLI arguments and stamp the specified SQLite database.

    Expects a single positional argument pointing to the SQLite database file
    that should be stamped to the current Alembic head revision.
    """
    parser = argparse.ArgumentParser(
        description="Stamp an existing SQLite database to the current Alembic head."
    )
    parser.add_argument(
        "db_path", type=Path, help="Path to the SQLite database file"
    )
    args = parser.parse_args()
    stamp_db(args.db_path)


if __name__ == "__main__":
    main()
