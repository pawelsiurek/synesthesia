from sqlalchemy import Column, Integer, String, Float, Text, DateTime, func
from pgvector.sqlalchemy import Vector
from .database import Base

class Song(Base):
    __tablename__ = "songs"

    id            = Column(Integer, primary_key=True, index=True)
    title         = Column(String(255), nullable=False)
    artist        = Column(String(255), nullable=False)
    genre         = Column(String(100))
    mood          = Column(String(100))
    energy        = Column(Float)
    valence       = Column(Float)
    tempo         = Column(Float)
    year          = Column(Integer)
    themes        = Column(Text)
    cultural_tags = Column(Text)
    spotify_id    = Column(String(100), unique=True)
    preview_url   = Column(String(500), nullable=True)
    document      = Column(Text, nullable=False)
    embedding     = Column(Vector(512), nullable=False)
    created_at    = Column(DateTime, server_default=func.now())