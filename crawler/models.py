"""ORM models for storing crawled web pages and LLM-extracted entities.

A Page record represents a single crawled URL and the subset of HTML that was
kept as relevant for downstream processing. An Entity record represents a
single entity mention extracted from one Page, including a natural-language
description, optional vector embedding, and arbitrary structured attributes.

These tables are intended to support workflows where an LLM extracts entities
from crawled content, the results are stored here, and later queries can search
over Entity.text / embeddings and link back to the originating Page.url.
"""

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.mutable import MutableList
import datetime as dt

Base = declarative_base()


class Page(Base):
    """Crawled web page containing source content for entity extraction.

    Each Page corresponds to a single URL and stores the subset of HTML that
    was retained as relevant, along with crawl metadata and processing status.
    Related Entity rows reference this page as their source of evidence.
    """

    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False, unique=True, index=True)
    html = Column(Text, nullable=True)
    crawled_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    status = Column(Text, nullable=False)
    # Use MutableList so in-place edits (append/remove) to this JSON list
    # are tracked by SQLAlchemy and persisted without needing to reassign the
    # whole list.
    entities_analyzed = Column(
        MutableList.as_mutable(JSON),
        nullable=False,
        default=list,
    )

    raw_entities = relationship("RawEntity", back_populates="page")
    combined_entities = relationship(
        "CombinedEntity",
        secondary="raw_entities",
        viewonly=True,
        back_populates="pages",
    )
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class RawEntity(Base):
    __tablename__ = "raw_entities"

    id = Column(Integer, primary_key=True)
    type = Column(String, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    text = Column(Text)
    attributes = Column(JSON)

    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    page = relationship("Page", back_populates="raw_entities")

    combined_entity_id = Column(
        Integer, ForeignKey("combined_entities.id"), nullable=True
    )
    combined_entity = relationship(
        "CombinedEntity", back_populates="raw_entities"
    )


class CombinedEntity(Base):
    __tablename__ = "combined_entities"

    id = Column(Integer, primary_key=True)
    type = Column(String, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    last_name_norm = Column(String, index=True, nullable=True)
    text = Column(Text)
    embedding = Column(LargeBinary, nullable=True)

    raw_entities = relationship("RawEntity", back_populates="combined_entity")
    pages = relationship(
        "Page",
        secondary="raw_entities",
        viewonly=True,
        back_populates="combined_entities",
    )


class EntityBio(Base):
    """Canonical bio text aggregated per logical entity.

    This table stores a single synthesized bio for each unique (type, name)
    pair found in the entities table, such as a person or session. Bios are
    intended to be stable, human-readable summaries built from one or more
    underlying Entity rows and used for display and search.

    Attributes:
        id: Surrogate primary key.
        type: Logical category of the entity (for example, 'Person', 'Session').
        name: Human-readable label of the entity.
        bio: Aggregated, free-form bio text describing the entity.

    The (type, name) pair is enforced to be unique so that each logical
    entity has at most one canonical bio.
    """

    __tablename__ = "entity_bios"

    id = Column(Integer, primary_key=True)
    type = Column(String, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    bio = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("type", "name", name="uq_entity_bios_type_name"),
    )
