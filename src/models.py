import pickle

from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    LargeBinary,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship

from .database import Base


class URLContent(Base):
    __tablename__ = "url_contents"
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, index=True)
    title = Column(String, nullable=True)
    html_content = Column(Text, nullable=True)
    text_content = Column(Text, nullable=True)
    analyzed = Column(Boolean, default=False)
    people_contexts = relationship(
        "PersonContext", back_populates="url_content"
    )


class PersonContext(Base):
    __tablename__ = "person_contexts"
    id = Column(Integer, primary_key=True)
    url_content_id = Column(Integer, ForeignKey("url_contents.id"))
    name = Column(String, index=True)
    context = Column(Text)

    url_content = relationship("URLContent", back_populates="people_contexts")
