from dotenv import load_dotenv
import random
import json
import boto3
import os
from tqdm import tqdm
import shutil
import re
from langchain_core.documents import Document

load_dotenv('../.env')


FILE_PATH = '../data/tdr_v4.pdf'
# get file name

# -----------------------------------------------------------------------------
# Read PDF
# -----------------------------------------------------------------------------

"""
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(FILE_PATH)

docs = loader.load()

print(docs[0].page_content)
print(docs[1].page_content)

os.makedirs('../data/page_extraction/pypdf', exist_ok=True)
for d in tqdm(docs, total=len(docs), desc="Saving pages"):
    with open(f'../data/page_extraction/pypdf/{os.path.basename(FILE_PATH).replace(".pdf", "")}_page_{d.metadata["page_label"]}.txt', 'w') as f:
        f.write(d.page_content)
"""

# import pymupdf

# doc: pymupdf.Document = pymupdf.open(FILE_PATH)

# for page in doc:
#     print(page.get_text())
#     break

# print(doc.get_page_text(0))
# print(doc.get_page_text(1))

# # re.sub(r'(?<!\.)\n(?!\n)', ' ', doc.get_page_text(i))
# docs = [
#     Document(page_content=doc.get_page_text(i), metadata={"page_index": i + 1})
#     for i in range(len(doc)) if i != 1 # (omitiendo el indice)
# ]

# print(docs[0])

# shutil.rmtree('../data/page_extraction/pymupdf', ignore_errors=True)
# os.makedirs('../data/page_extraction/pymupdf', exist_ok=True)
# for d in tqdm(docs, total=len(docs), desc="Saving pages"):
#     with open(f'../data/page_extraction/pymupdf/{os.path.basename(FILE_PATH).replace(".pdf", "")}_page_{d.metadata["page_index"]}.txt', 'w') as f:
#         f.write(d.page_content)

"""
get_text_blocks()
get_text_words()
get_text_trace()


IDEAS:
- Preprocess more the text to remove noise like '\n' in the middle of sentences.

"""

# # TODO: try DocLing
# from io import BytesIO
# from docling.datamodel.base_models import DocumentStream
# from docling.document_converter import DocumentConverter

# binary_stream = open(FILE_PATH, 'rb').read()
# buf = BytesIO(binary_stream)
# source = DocumentStream(name=FILE_PATH, stream=buf)
# converter = DocumentConverter()
# result = converter.convert(source)


import pymupdf4llm
import pathlib

md_text = pymupdf4llm.to_markdown(FILE_PATH)

type(md_text)

pathlib.Path(f"../data/{FILE_PATH.replace('.pdf', '')}.md").write_bytes(md_text.encode())

pages_md = md_text.split('\n-----\n')
print(f"{len(pages_md)=}")

docs = [
    Document(page_content=p, metadata={"page_index": i + 1})
    for i, p in enumerate(pages_md)
    # if i != 1 # (omitiendo el indice)
]

idx_page = random.randint(0, len(docs))
sample_page = docs[idx_page]

print(docs[idx_page].page_content)

shutil.rmtree('../data/page_extraction/pymupdf4llm', ignore_errors=True)
os.makedirs('../data/page_extraction/pymupdf4llm', exist_ok=True)
for d in tqdm(docs, total=len(docs), desc="Saving pages"):
    with open(f'../data/page_extraction/pymupdf4llm/{os.path.basename(FILE_PATH).replace(".pdf", "")}_page_{d.metadata["page_index"]}.txt', 'w', encoding='utf-8') as f:
        f.write(d.page_content)


# -----------------------------------------------------------------------------
# Chunking / Splitting
# -----------------------------------------------------------------------------

# https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
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

from llm import bedrock_runtime, embed_call

chunk_emb: dict = embed_call(bedrock_runtime, sample_chunk.page_content)
emb = chunk_emb['embedding']


all_embs = [
    embed_call(bedrock_runtime, split.page_content)
    for split in tqdm(all_splits, total=len(all_splits), desc="Get embeddings")
]

# chunk_emb.keys()
# chunk_emb['embeddingsByType'].keys()
# chunk_emb['inputTextTokenCount']

"""
How can I track the costs of my experiments?
"""

# -----------------------------------------------------------------------------
# Save to Vector Database
# -----------------------------------------------------------------------------

from db import *

"""
https://www.datacamp.com/tutorial/pgvector-tutorial
"""

# Entire document
for idx, emb in tqdm(enumerate(all_embs), total=len(all_embs), desc="Saving embeddings"):
    new_embedding = Embedding(
        page_index=all_splits[idx].metadata['page_index'],
        document_name=FILE_PATH,
        page_content=all_splits[idx].page_content,
        embedding=emb['embedding']
    )

    Session.add(new_embedding)
Session.commit()

"""
# Individual sample
new_embedding = Embedding(
    page_index=sample_chunk.metadata['page_index'],
    document_name=FILE_PATH,
    page_content=sample_chunk.page_content,
    embedding=emb['embedding']
)

Session.add(new_embedding)
Session.commit()
"""
