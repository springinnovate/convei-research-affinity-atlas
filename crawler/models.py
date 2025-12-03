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
from sqlalchemy.orm import declarative_base, relationship
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
    entities_analyzed = Column(JSON, nullable=True, default=list)
    entities = relationship("Entity", back_populates="page")
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class Entity(Base):
    """LLM-extracted entity tied to a single source page.

    An Entity represents one extracted entity mention, including:
    - type: logical category such as 'Person', 'Session', etc.
    - name: human-readable label for the entity.
    - text: free-form description or summary used as LLM context.
    - embedding: binary-encoded vector of the text for similarity search.
    - attributes: JSON payload for structured fields specific to the type.

    The page relation links the entity back to the Page whose content was used
    to derive this description and attributes.
    """

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True)
    type = Column(String, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    text = Column(Text)
    embedding = Column(LargeBinary, nullable=True)
    attributes = Column(JSON)

    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    page = relationship("Page", back_populates="entities")
