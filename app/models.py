"""SQL Alchemy models for the CONVEI research atlas."""

from datetime import datetime
from typing import List, Dict

from sqlalchemy import String, Text, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ProcessedFile(Base):
    __tablename__ = "processed_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), unique=True)
    processed_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(1024), unique=True)
    bio: Mapped[str] = mapped_column(Text)
    bio_source: Mapped[dict] = mapped_column(JSON)
    url_list: Mapped[List[str]] = mapped_column(JSON)
