"""CLI helper for running Alembic migrations against a specific SQLite database."""

from pathlib import Path
import argparse

from alembic.config import Config
from alembic import command


def upgrade_db(db_path: Path):
    """Run Alembic migrations up to head on the given SQLite database.

    Args:
        db_path: Path to the SQLite database file to migrate.
    """
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


def main():
    """Parse CLI arguments and run migrations on the specified database.

    Expects a single positional argument pointing to the SQLite database file.
    """
    parser = argparse.ArgumentParser(
        description="Run Alembic migrations on a SQLite database."
    )
    parser.add_argument(
        "db_path", type=Path, help="Path to the SQLite database file"
    )
    args = parser.parse_args()
    upgrade_db(args.db_path)


if __name__ == "__main__":
    main()
