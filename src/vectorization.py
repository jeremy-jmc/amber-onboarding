from dotenv import load_dotenv
import random
import json
import boto3
import os
from tqdm import tqdm
import shutil
import re
from langchain_core.documents import Document
import xml.etree.ElementTree as ET
import xml.dom.minidom
from db import *

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

md_text = pymupdf4llm.to_markdown(FILE_PATH)    # , page_chunks=True

type(md_text)


pathlib.Path(f"../data/{FILE_PATH.replace('.pdf', '')}.md").write_bytes(md_text.encode())

pages_md = md_text.split('\n-----\n')
print(f"{len(pages_md)=}")

header = [
    '**Gerencia Central de Tecnologías de Información y Comunicaciones**',
    'Servicio de Infraestructura, Plataforma y Microservicios en Nube Pública para el despliegue de las Aplicaciones y Nuevos Servicios de la Gerencia',
    'Central de Tecnologías de Información y Comunicaciones de Essalud',
]

pages_md = [[v.strip() for v in p.split('\n') if len(v)] for p in pages_md]

filtered_pages_md = [s[:-1] if len(s) > 1 and s[-1].startswith('Página') else s for s in pages_md]

filtered_pages_md = [[v.strip() for v in s if v not in header] for s in filtered_pages_md]


def get_sections(page: list[str]) -> list[str]:
    """Get the sentences representing sections in the current page

    Args:
        page (list[str]): List of sentences in the current page
    
    Docs:
        A section is represented by a sentence envolved between '**' characters and started with any type of enumeration like '1.', 'a.', 'i.', 'I.', etc.

        Positive cases:
            - **3.** **ANTECEDENTES**
            - **4.2. Objetivo Especifico**

        Negative cases:
            - **Servicio Web Application Firewall**
            - b. El servicio debe permitir crear reglas que bloquean ataques comunes como la inyección de SQL, cross-site scripting, etc.
    """
    roman_pattern = re.compile(r"^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+")
    enumeration_pattern = re.compile(r"^\s*((I{1,3}|IV|V|VI{0,3}|IX|X)|\d+(\.\d+)*)\.\s+")
    
    filtered_sentences = [(s.replace("*", '').replace("#", '').strip(), s) for s in page if (s.startswith('**') and s.endswith('**')) or bool(roman_pattern.match(s)) or (s.replace('*', '').replace('#', '').strip().startswith("ANEXO"))]
    
    filtered_sentences = [tup for tup in filtered_sentences if enumeration_pattern.match(tup[0])  or (tup[0].replace('*', '').replace('#', '').strip().startswith("ANEXO"))]
    return filtered_sentences


def split_text_by_headings(full_text, headings_list):
    """
    Divide un texto completo en secciones basadas en una lista de subtítulos.
    
    Args:
        full_text (str): El texto completo a dividir
        headings_list (list): Lista de subtítulos para usar como puntos de división
        
    Returns:
        dict: Un diccionario donde las claves son los subtítulos y los valores son los contenidos
    """
    # Ordenar los encabezados por su posición en el texto
    # Esto es crucial porque necesitamos procesar los encabezados en el orden en que aparecen
    heading_positions = {}
    for heading in headings_list:
        # Escapar caracteres especiales de regex en el encabezado
        escaped_heading = re.escape(heading)
        match = re.search(escaped_heading, full_text)
        if match:
            heading_positions[heading] = match.start()
    
    # Ordenar los encabezados por posición
    sorted_headings = sorted(heading_positions.keys(), key=lambda x: heading_positions[x])
    
    # Crear los chunks de texto
    chunks = {}
    for i in range(len(sorted_headings)):
        current_heading = sorted_headings[i]
        
        # Determinar dónde termina esta sección (inicio de la siguiente sección)
        start_pos = heading_positions[current_heading]
        
        if i < len(sorted_headings) - 1:
            next_heading = sorted_headings[i + 1]
            end_pos = heading_positions[next_heading]
        else:
            end_pos = len(full_text)
        
        # Extraer el contenido de esta sección
        content = full_text[start_pos:end_pos].strip()
        chunks[current_heading] = content
    
    return chunks


def get_section_to_root_path(tree: dict, section: str) -> list[str]:
    """Get the path from the root to the current section

    Args:
        tree (dict): The tree structure
        section (str): The current section
    
    Returns:
        list[str]: The path from the root to the current section
    """
    path = []
    for k, v in tree.items():
        if k == section:
            return [k]
        elif isinstance(v, dict):
            path = get_section_to_root_path(v, section)
            if path:
                return [k] + path
    return path


def create_index_xml(path_list: list) -> str:   # , chunk_content: str
    root = ET.Element("index")

    current_node = root

    for level, section in enumerate(path_list, start=1):
        subsection = ET.SubElement(current_node, "subsection", title=section, level=str(level))
        current_node = subsection

    # current_node.text = f"\n{chunk_content}\n"

    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = xml.dom.minidom.parseString(rough_string)

    return reparsed.toprettyxml(indent="\t", newl='\n').replace('<?xml version="1.0" ?>\n', '')


sections = [get_sections(page) for idx, page in enumerate(filtered_pages_md) if idx > 1]

for tup in sections:
    if len(tup):
        print([f for f, s in tup])

cleaned_pages = ['\n'.join(ls) for ls in filtered_pages_md if len(ls) > 1]

print(f"{len(cleaned_pages)=}")
cleaned_doc: str = '\n\n\n'.join(cleaned_pages)

with open(f'../data/cleaned_doc_{os.path.basename(FILE_PATH).replace(".pdf", "")}.md', 'w', encoding='utf-8') as f:
    f.write(cleaned_doc)

original_tups = [t for tup_list in sections for t in tup_list]
section_mapping_original_to_clean = {s: f for f, s in original_tups}
original_splitters = [s for f, s in original_tups]

print(f"{len(original_splitters)=}")

# ! manually created tree, bc there are only two documents to compare
tree = json.loads(open(f'../data/tree_{os.path.basename(FILE_PATH).replace(".pdf", "")}.json', 'r').read())
print(json.dumps(tree, indent=2, ensure_ascii=False))

# ! chunks no incluyen la caratula por la naturaleza de las secciones
sectioned_chunks = split_text_by_headings(cleaned_doc, original_splitters)
print(f"{len(sectioned_chunks)=}")

for section, section_content in sectioned_chunks.items():
    record = DocumentSection(document_name=FILE_PATH, section_name=section, section_content=section_content)
    Session.add(record)
Session.commit()

# filtered empty sections
sectioned_chunks = {k: v for k, v in sectioned_chunks.items() if len(v.replace(k, ''))}
print(json.dumps(sectioned_chunks, indent=2, ensure_ascii=False))

for k, v in sectioned_chunks.items():
    print(k, len(v))

docs = [
    # Document(page_content=p, metadata={"page_index": i + 1})
    # for i, p in enumerate(pages_md)
    # # if i != 1 # (omitiendo el indice)
    Document(page_content=v, metadata={"section": get_section_to_root_path(tree, section_mapping_original_to_clean[k])})
    for idx, (k, v) in enumerate(sectioned_chunks.items())
]
for d in docs:
    d.metadata['xml_header'] = create_index_xml(d.metadata['section'])

check = True
if check:
    idx_page = random.randint(0, len(docs) - 1)
    sample_page = docs[idx_page]

    # print(sample_page.metadata)
    # print(json.dumps(sample_page.metadata, indent=2, ensure_ascii=False))

    print(f"{sample_page.metadata['xml_header']}<content>\n{sample_page.page_content}\n</content>")


shutil.rmtree('../data/page_extraction/pymupdf4llm', ignore_errors=True)
os.makedirs('../data/page_extraction/pymupdf4llm', exist_ok=True)
for idx, d in tqdm(enumerate(docs), total=len(docs), desc="Saving pages"):
    with open(f'../data/page_extraction/pymupdf4llm/{os.path.basename(FILE_PATH).replace(".pdf", "")}_chunk_{idx}.txt', 'w', encoding='utf-8') as f:
        f.write(f"{d.metadata['section']}\n\n{d.page_content}")


# -----------------------------------------------------------------------------
# Chunking / Splitting
# -----------------------------------------------------------------------------

# https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=250, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

# all_splits = text_splitter.split_documents(docs)

# all_splits = []
# for doc in docs:
#     splits = text_splitter.split_text(doc.page_content)  # Splitting content
#     for split in splits:
#         all_splits.append({
#             "page_content": split,
#             "metadata": doc.metadata  # Preserve original metadata (page number/page hierarchy)
#         })
# all_splits = [Document(page_content=split['page_content'], metadata=split['metadata']) for split in all_splits]

print(len(all_splits))
print(dir(all_splits[0]))
print(all_splits[5].page_content)
print(json.dumps(all_splits[5].metadata, indent=2, ensure_ascii=False))

# for idx, split in enumerate(all_splits):
#     print(f"{split.metadata['xml_header']}<content>\n{split.page_content}\n</content>\n\n")
#     if idx > 25:
#         break

"""
Study more about the Splitters algorithms/methods.
How can I split obtain meaningful chunks joining text between pages before splitting.
"""


# -----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------- 

from llm import bedrock_runtime, embed_call

def format_chunk_content(chunk: Document) -> str:
    # section_formatted = '\n'.join(chunk.metadata['section']).upper()
    # return f"{section_formatted}\n\n{chunk.page_content}"
    return f"{chunk.metadata['xml_header']}<content>\n{chunk.page_content}\n</content>"

# SRC: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags#example-legal-contract-analysis
all_embs = [
    embed_call(bedrock_runtime, format_chunk_content(split))
    for split in tqdm(all_splits, total=len(all_splits), desc="Get embeddings")
]

# idx = random.randint(0, len(all_splits))
# sample_chunk = all_splits[idx]
# chunk_emb: dict = embed_call(bedrock_runtime, sample_chunk.page_content)
# # chunk_emb.keys()
# # chunk_emb['embeddingsByType'].keys()
# # chunk_emb['inputTextTokenCount']
# emb = chunk_emb['embedding']

"""
How can I track the costs of my experiments?
"""

# -----------------------------------------------------------------------------
# Save to Vector Database
# -----------------------------------------------------------------------------

"""
https://www.datacamp.com/tutorial/pgvector-tutorial
"""

# Entire document
for idx, emb in tqdm(enumerate(all_embs), total=len(all_embs), desc="Saving embeddings"):
    new_embedding = Embedding(
        # page_index=all_splits[idx].metadata['page_index'],
        section_name=all_splits[idx].metadata['section'][-1],
        document_name=FILE_PATH,
        chunk_content=format_chunk_content(all_splits[idx]),
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
