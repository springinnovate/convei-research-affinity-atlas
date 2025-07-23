"""SQL Alchemy models for the CONVEI research atlas."""

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
    Table,
)
from sqlalchemy.orm import relationship
from .database import Base


webpage_snippet_association = Table(
    "webpage_snippet_association",
    Base.metadata,
    Column(
        "snippet_id",
        Integer,
        ForeignKey("entity_webpage_snippet.webpage_snippet_id"),
        primary_key=True,
    ),
    Column(
        "webpage_content_id",
        Integer,
        ForeignKey("webpage_content.webpage_content_id"),
        primary_key=True,
    ),
)


class WebpageContent(Base):
    """A fetched webpage plus any entities/snippets extracted from it."""

    __tablename__ = "webpage_content"

    webpage_content_id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String)
    html_content = Column(Text)
    text_content = Column(Text)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    analyzed = Column(Boolean, default=False)

    # Entity via association table
    related_entities = relationship(
        "EntityWebpageContentAssociation",
        back_populates="webpage_content",
        cascade="all, delete-orphan",
    )

    # Snippets (many-to-many)
    related_snippets = relationship(
        "EntityWebpageSnippet",
        secondary=webpage_snippet_association,
        back_populates="related_webpages",
        cascade="all, delete",
    )


class Entity(Base):
    """A conceptual entity (person, organisation, place, etc.)."""

    __tablename__ = "entity"

    entity_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True, nullable=False)

    analyses = relationship(
        "EntityLLMAnalysis",
        back_populates="entity",
        order_by="EntityLLMAnalysis.version",
        cascade="all, delete-orphan",
    )

    related_webpages = relationship(
        "EntityWebpageContentAssociation",
        back_populates="entity",
        cascade="all, delete-orphan",
    )

    # ↔ Snippets
    snippets = relationship(
        "EntityWebpageSnippet",
        back_populates="entity",
        cascade="all, delete-orphan",
    )


class EntityWebpageContentAssociation(Base):
    """Links an Entity to the WebpageContent where it was detected."""

    __tablename__ = "entity_webpage_content_association"

    entity_id = Column(
        Integer, ForeignKey("entity.entity_id"), primary_key=True
    )
    webpage_content_id = Column(
        Integer,
        ForeignKey("webpage_content.webpage_content_id"),
        primary_key=True,
    )

    entity = relationship("Entity", back_populates="related_webpages")
    webpage_content = relationship(
        "WebpageContent", back_populates="related_entities"
    )


class EntityWebpageSnippet(Base):
    """A unique text snippet that describes (or mentions) an entity.

    Deduplicated by (entity_id, snippet_hash) so the *same* snippet only
    exists once per entity, even if it is found on many webpages.
    """

    __tablename__ = "entity_webpage_snippet"

    webpage_snippet_id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("entity.entity_id"), nullable=False)
    snippet_text = Column(Text, nullable=False)
    snippet_hash = Column(String(64), nullable=False, index=True)

    entity = relationship("Entity", back_populates="snippets")

    # ↔ Webpages (many-to-many)
    related_webpages = relationship(
        "WebpageContent",
        secondary=webpage_snippet_association,
        back_populates="related_snippets",
    )

    __table_args__ = (
        UniqueConstraint("entity_id", "snippet_hash", name="uq_entity_context"),
    )


class EntityLLMAnalysis(Base):
    """Cached LLM summaries of an entity’s snippets."""

    __tablename__ = "entity_llm_analysis"

    entity_analysis_id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("entity.entity_id"), nullable=False)
    version = Column(Integer, nullable=False)
    context_hash = Column(String(64), nullable=False)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    entity = relationship("Entity", back_populates="analyses")

    __table_args__ = (
        UniqueConstraint(
            "entity_id", "context_hash", name="uq_entity_analysis"
        ),
        UniqueConstraint(
            "entity_id", "version", name="uq_entity_analysis_version"
        ),
    )
