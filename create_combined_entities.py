import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from crawler.models import RawEntity, CombinedEntity


def backfill_combined_entities(session, batch_size=10000):
    existing = {}
    for ce in session.query(CombinedEntity):
        existing[(ce.type, ce.name)] = ce

    groups = {}
    query = session.query(RawEntity).filter(RawEntity.combined_entity_id.is_(None))
    for raw in tqdm(query, desc="Grouping raw entities"):
        key = (raw.type, raw.name)
        groups.setdefault(key, []).append(raw)

    batch = 0
    for (etype, name), raws in tqdm(groups.items(), desc="Creating combined entities"):
        combined = existing.get((etype, name))
        if combined is None:
            combined = CombinedEntity(type=etype, name=name, text=None, embedding=None)
            session.add(combined)
            existing[(etype, name)] = combined
        for raw in raws:
            raw.combined_entity = combined
            batch += 1
            if batch >= batch_size:
                session.commit()
                batch = 0
    if batch:
        session.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    engine = create_engine(args.db_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    backfill_combined_entities(session, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
