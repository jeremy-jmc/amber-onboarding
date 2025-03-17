from dotenv import load_dotenv
import random
import json
import boto3
import os

load_dotenv('../.env')


FILE_PATH = '../data/tdr_0.pdf'
# -----------------------------------------------------------------------------
# Read PDF
# -----------------------------------------------------------------------------


import pymupdf
# import langchain document object
from langchain_core.documents import Document

doc: pymupdf.Document = pymupdf.open(FILE_PATH)

for page in doc:
    print(page.get_text())
    break

print(doc.get_page_text(0))
print(doc.get_page_text(1))

docs = [
    Document(page_content=doc.get_page_text(i), metadata={"page_index": i + 1})
    for i in range(len(doc))
]

print(docs[0])

"""
get_text_blocks()
get_text_words()
get_text_trace()


IDEAS:
- Preprocess more the text to remove noise like '\n' in the middle of sentences.

"""


"""
import pymupdf4llm

md_text = pymupdf4llm.to_markdown(FILE_PATH)

type(md_text)

import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
"""


"""
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(FILE_PATH)

docs = loader.load()

print(docs[0].page_content)
print(docs[1].page_content)
"""


# -----------------------------------------------------------------------------
# Chunking / Splitting
# -----------------------------------------------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = []
for doc in docs:
    splits = text_splitter.split_text(doc.page_content)  # Splitting content
    for split in splits:
        all_splits.append({
            "page_content": split,
            "metadata": doc.metadata  # Preserve original metadata (page number)
        })

# all_splits = text_splitter.split_documents(docs)

all_splits = [Document(page_content=split['page_content'], metadata=split['metadata']) for split in all_splits]

print(len(all_splits))

print(all_splits[5].page_content)

"""
Study more about the Splitters algorithms/methods.
How can I split obtain meaningful chunks joining text between pages before splitting.
"""


# -----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------- 

idx = random.randint(0, len(all_splits))
sample_chunk = all_splits[idx]

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

from llm import embed_call

chunk_emb: dict = embed_call(bedrock_runtime, sample_chunk.page_content)
emb = chunk_emb['embedding']

# chunk_emb.keys()
# chunk_emb['embeddingsByType'].keys()
# chunk_emb['inputTextTokenCount']

# -----------------------------------------------------------------------------
# Save to Vector Database
# -----------------------------------------------------------------------------

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
    page_index = Column(Integer)
    document_name = Column(String(50))
    embedding = Column(Vector(1024))
    page_content = Column(String)


Base.metadata.create_all(bind=engine)

"""
https://www.datacamp.com/tutorial/pgvector-tutorial
"""

new_embedding = Embedding(
    page_index=sample_chunk.metadata['page_index'],
    document_name='tdr_0.pdf',
    page_content=sample_chunk.page_content,
    embedding=emb
)

Session.add(new_embedding)
Session.commit()


