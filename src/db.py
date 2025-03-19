import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy import select
from pgvector.sqlalchemy import Vector


DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autoflush=True, bind=engine)
Session = SessionLocal()
Base = declarative_base()

class Embedding(Base):
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True, index=True)
    # page_index = Column(Integer)
    page_section = Column(String(250))
    document_name = Column(String(50))
    embedding = Column(Vector(1024))
    page_content = Column(String)


Base.metadata.create_all(bind=engine)
