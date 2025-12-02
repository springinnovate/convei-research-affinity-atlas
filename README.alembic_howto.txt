If you change your models, run:

alembic revision --autogenerate -m "describe the change"

That will make a file that you should open and check out if the commands make sense.
If so, commit that file to the repo, then apply to databases:

python migrate_db.py path/to/db.sqlite


The `stamp_db.py` script is used to version databses that we made outside of almebic.
