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
    content = Column(Text, nullable=True)
    analyzed = Column(Boolean, default=False)
    entities = relationship("Entity", back_populates="url_content")


class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    embedding = Column(LargeBinary)  # Stored as pickled bytes
    url_content_id = Column(Integer, ForeignKey("url_contents.id"))
    url_content = relationship("URLContent", back_populates="entities")

    def get_embedding(self):
        return pickle.loads(self.embedding)

    def set_embedding(self, emb):
        self.embedding = pickle.dumps(emb)
