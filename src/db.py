import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy import select
from pgvector.sqlalchemy import Vector
from sqlalchemy import and_, or_, tuple_, text, distinct


DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autoflush=True, bind=engine)
Session = SessionLocal()
Base = declarative_base()

class Embedding(Base):
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True, index=True)
    # page_index = Column(Integer)
    section_name = Column(String(250))
    document_name = Column(String(50))
    embedding = Column(Vector(1024))
    chunk_content = Column(String)


class DocumentSection(Base):
    __tablename__ = 'document_sections'

    id = Column(Integer, primary_key=True, index=True)
    document_name = Column(String(50))
    section_name = Column(String(250))
    section_content = Column(String)


Base.metadata.create_all(bind=engine)

# DROP TABLE document_sections;
# DROP TABLE embeddings;
