## Database migrations

We use Alembic to manage schema changes for the `Page` and `Entity` models.

### Changing models

When you change `crawler.models`:

    alembic revision --autogenerate -m "describe the change"

Then:

1. Open the generated file in `migrations/versions/` and verify the operations.
2. Commit the new migration file to the repo.
3. Apply the migration to any database that should be updated:

       python migrate_db.py path/to/db.sqlite

### Adopting an existing database

`stamp_db.py` is used to put an existing database (created outside Alembic) under Alembic version control without changing its schema:

    python stamp_db.py path/to/existing.db

This writes the current Alembic head revision into the database’s `alembic_version` table but does not run any DDL.

### Creating a new database

To create a brand new database with the current schema:

    python migrate_db.py app_new/data/new_crawl.sqlite

This will apply the full Alembic migration history to the new SQLite file and create all required tables.
