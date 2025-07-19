from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from .database import Base


class URLContent(Base):
    """A fetched webpage or other URL addressable resource."""

    __tablename__ = "url_contents"

    url_content_id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String)
    html_content = Column(Text)
    text_content = Column(Text)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    analyzed = Column(Boolean, default=False)

    contexts = relationship(
        "EntityContextPage",
        back_populates="url_content",
        cascade="all, delete-orphan",
    )
    entities = relationship(
        "EntityPage", back_populates="page", cascade="all, delete-orphan"
    )


class Entity(Base):
    """A canonical real‑world or conceptual entity (e.g., a person)."""

    __tablename__ = "entities"

    entities_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True, nullable=False)
    type = Column(String)  # optional free‑form label (person, org, place)

    contexts = relationship(
        "EntityContext",
        back_populates="entity",
        cascade="all, delete-orphan",
    )
    analyses = relationship(
        "EntityAnalysis",
        back_populates="entity",
        order_by="EntityAnalysis.version",
        cascade="all, delete-orphan",
    )
    pages = relationship(
        "EntityPage", back_populates="entity", cascade="all, delete-orphan"
    )


class EntityPage(Base):
    __tablename__ = "entity_pages"

    entity_id = Column(
        Integer, ForeignKey("entities.entities_id"), primary_key=True
    )
    url_content_id = Column(
        Integer, ForeignKey("url_contents.url_content_id"), primary_key=True
    )

    # two‑way helpers (optional)
    entity = relationship("Entity", back_populates="pages")
    page = relationship("URLContent", back_populates="entities")


class EntityContext(Base):
    """
    A unique text snippet describing an entity.

    The same snippet can appear on multiple webpages; deduplicated by
    (`entity_id`, `context_hash`).
    """

    __tablename__ = "entity_contexts"

    entity_contexts_id = Column(Integer, primary_key=True)
    entity_id = Column(
        Integer, ForeignKey("entities.entities_id"), nullable=False
    )
    context_text = Column(Text, nullable=False)
    context_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    entity = relationship("Entity", back_populates="contexts")
    pages = relationship(
        "EntityContextPage",
        back_populates="entity_context",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("entity_id", "context_hash", name="uq_entity_context"),
    )


class EntityContextPage(Base):
    """
    Association table linking a context snippet to each page where it occurs.

    Composite primary key avoids duplicate links.
    """

    __tablename__ = "entity_context_pages"

    entity_context_id = Column(
        Integer,
        ForeignKey("entity_contexts.entity_contexts_id"),
        primary_key=True,
    )
    url_content_id = Column(
        Integer,
        ForeignKey("url_contents.url_content_id"),
        primary_key=True,
    )

    entity_context = relationship("EntityContext", back_populates="pages")
    url_content = relationship("URLContent", back_populates="contexts")


class EntityAnalysis(Base):
    """
    Cached LLM summary of an entity’s contexts.

    * `contexts_hash` is a stable hash of the concatenated context hashes
      used to produce the summary.
    * `version` bumps whenever the set of contexts changes.
    """

    __tablename__ = "entity_analyses"

    entity_analyses_id = Column(Integer, primary_key=True)
    entity_id = Column(
        Integer, ForeignKey("entities.entities_id"), nullable=False
    )
    version = Column(Integer, nullable=False)
    contexts_hash = Column(String(64), nullable=False)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    entity = relationship("Entity", back_populates="analyses")

    __table_args__ = (
        UniqueConstraint(
            "entity_id", "contexts_hash", name="uq_entity_analysis"
        ),
        UniqueConstraint(
            "entity_id", "version", name="uq_entity_analysis_version"
        ),
    )
