from sqlalchemy import select
from llm import bedrock_runtime, embed_call, claude_call
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import shutil
from IPython.display import display
import copy
from db import *
from tqdm import tqdm
from metrics import *

np.set_printoptions(precision=3)

# https://github.com/amberpe/poc-rag-multidocs/blob/main/RAG.py#L218

def search_similar_text(query, top_k=5) -> list[tuple]:
    query_embedding = embed_call(bedrock_runtime, query)
    return Session.execute(
        select(
            Embedding.document_name,
            Embedding.section_name,
            Embedding.chunk_content,
            Embedding.embedding.cosine_distance(query_embedding['embedding']).label('cosine_distance')
        )
        .order_by(
            Embedding.embedding.cosine_distance(query_embedding['embedding'])
        ).limit(top_k)
    )


def extract_answer(xml_string):
    try:
        root = ET.fromstring(f"<root>{xml_string}</root>")  # Wrap in a root tag to handle multiple top-level elements
        answer_element = root.find("respuesta")
        return answer_element.text.strip() if answer_element is not None else None
    except Exception as e:
        raise e
    # except ET.ParseError:
    #     return None  # Handle invalid XML cases


"""
Document(
    metadata={'page_index': 6}, 
    page_content='g.\nEl servicio debe permitir asociar una o más direcciones IP elásticas a cualquier\ninstancia\nde\nla\nnube\nprivada\nvirtual,\nde modo que puedan alcanzarse\ndirectamente desde Internet.\nh.\nEl servicio debe permitir conectarse a la nube privada virtual con otras nubes\nprivadas virtuales y obtener acceso a los recursos de otras nubes privadas\nvirtuales a través de direcciones IP privadas mediante la interconexión de nube\nprivada virtual.\ni.\nEl servicio debe permitir conectarse de manera privada a los servicios del\nfabricante de la nube pública sin usar una gateway de Internet, ni una NAT ni\nun proxy de firewall mediante un punto de enlace de la nube privada virtual.\nj.\nEl servicio debe permitir conectar la nube privada virtual y la infraestructura de\nTI local con la VPN del fabricante de la nube pública de sitio a sitio.\nk.\nEl servicio debe permitir asociar grupos de seguridad de la nube privada virtual\ncon instancias en la plataforma.\nl.'
)


<Context>
<Tdr - v(6)>
Contenido asd asd asd 
<Page>2</Page>
</Tdr - v(6)>

<Tdr - v(6)>
Contenido asd asd asd 
<Page>4</Page>
</Tdr - v(6)>
</Context>
"""

# TODO: levantar BD de las secciones y traerte las secciones enteras para presentarlas en el contexto

# Prompt Improver: https://console.anthropic.com/dashboard
QA = [
# # * Chunk question
#     'g.\nEl servicio debe permitir asociar una o más direcciones IP elásticas a cualquier\ninstancia\nde\nla\nnube\nprivada\nvirtual,\nde modo que puedan alcanzarse\ndirectamente desde Internet.\nh.\nEl servicio debe permitir conectarse a la nube privada virtual con otras nubes\nprivadas virtuales y obtener acceso a los recursos de otras nubes privadas\nvirtuales a través de direcciones IP privadas mediante la interconexión de nube\nprivada virtual.\ni.\nEl servicio debe permitir conectarse de manera privada a los servicios del\nfabricante de la nube pública sin usar una gateway de Internet, ni una NAT ni\nun proxy de firewall mediante un punto de enlace de la nube privada virtual.\nj.\nEl servicio debe permitir conectar la nube privada virtual y la infraestructura de\nTI local con la VPN del fabricante de la nube pública de sitio a sitio.\nk.\nEl servicio debe permitir asociar grupos de seguridad de la nube privada virtual\ncon instancias en la plataforma.\nl.',

# * Cross document question
    # Bien
    ("¿Cual es la finalidad publica de los documentos?", ""), 

    # Bien
    ("¿Cuales son los objetivos generales y especificos de la contratacion en los documentos?", ""),

    # Bien
    ("¿Cuál es el porcentaje mínimo de disponibilidad que debe tener la infraestructura de Nube Pública?",
        "La infraestructura de Nube Pública descrita en los presentes términos de referencia deberá tener una disponibilidad mínima del 99.9%."),

    # Bien
    ("¿Cuántos centros de datos mínimos se requieren para asegurar la resiliencia y continuidad del servicio?",
        "Se requiere la implementación de como mínimo dos (2) centros de datos (zonas de disponibilidad) en una misma zona geográfica (región)."),
    
    # Bien, track: FALLA -> Mejora con el "De acuerdo ..." indep. XML en el header del chunk -> Mejora rspta
    ("De acuerdo a la sección 5. ¿Cuál es el periodo de garantía que debe tener toda la solución?",
        "Toda la solución deberá contar con una garantía por 12 meses."),
    
    # ! FALLA -> No mejora con "De acuerdo ..."
    ("De acuerdo a la sección 4. ¿Qué tipo de controles debe implementar el servicio para garantizar la seguridad de la información?",
        "El servicio debe implementar diversos controles tanto lógicos como físicos para garantizar un alto nivel de seguridad en el despliegue de las aplicaciones."),

    # ! MASO FALLA -> No mejora con "De acuerdo ..."
    ("De acuerdo a la sección 5. ¿Quién será responsable de administrar la consola de los servicios de Infraestructura?",
        "La consola será manejada por el Especialista asignado por la Sub Gerencia de Operaciones de Tecnologías de Información de la Gerencia de Producción de la GCTIC."),

    # TODO: preguntas acerca de Anexos

# * Section questions
    ("Que dice la seccion 2 'Finalidad publica'?", ""),
    ("Que dice la seccion 6.3.1.?", ""),
    ("Explicame lo que dice en la seccion 5.5?", ""),

# * Comparative questions
    ("¿Qué diferencias hay en la sección 2 'Finalidad publica'?", ""),       # ! FALLA -> Si pongo solo 2, confunde con romanos -> Si pongo el titulo de la seccion lo hace algo mejor    
    ("¿Qué diferencias hay en la sección 6.3.1.?", "")
    
    # Seccion 5.1: diferencias importantes
]

SYSTEM_PROMPT = """
Eres un asistente de IA especializado en responder preguntas basadas en contextos proporcionados. Tu tarea es analizar el contexto dado, entender la pregunta y proporcionar una respuesta precisa y relevante en español.
"""

PROMPT = """
Primero, te presentaré el contexto sobre el cual se basará la pregunta:

{context}

Ahora, te haré una pregunta relacionada con este contexto. Tu objetivo es responder a esta pregunta utilizando únicamente la información proporcionada en el contexto anterior.

<PREGUNTA>
{query}
</PREGUNTA>

Para asegurar una respuesta precisa y bien fundamentada, sigue estos pasos:

1. Analiza cuidadosamente el contexto proporcionado.
2. Extrae y cita la información relevante del contexto para responder a la pregunta.
3. Considera las posibles interpretaciones de la pregunta.
4. Evalúa la fuerza de la evidencia para cada posible respuesta.
5. Escribe tu razonamiento dentro de las etiquetas <analisis> para explicar tu proceso de pensamiento y cómo llegaste a tu respuesta.
6. Proporciona tu respuesta final dentro de las etiquetas <respuesta>.
7. Devuelve una estructura de respuesta XML bien formada, parseable. Sin etiquetas adicionales dentro y fuera de las etiquetas de respuesta. Asegúrate de que cada etiqueta esté correctamente cerrada y que la estructura sea jerárquicamente válida.

Recuerda:
- Utiliza solo la información proporcionada en el contexto.
- Si la información necesaria para responder la pregunta no está en el contexto, indica que no puedes responder basándote en la información disponible.
- Mantén tu respuesta clara, concisa y directamente relacionada con la pregunta.

Comienza tu proceso de razonamiento y respuesta ahora.
"""

CALL_CLAUDE = True

K = 10       # 5 mejor que 3, 10 mejor que 5 pero no sustancial

shutil.rmtree('../data/qa', ignore_errors=True)
os.makedirs('../data/qa', exist_ok=True)


def classify_question_type(query: str) -> str:
    CLASSIFY_QUESTION_PROMPT = f"""
You are a sophisticated question classification system. Your task is to categorize a given question into one of two categories based on whether it refers to a single document or compares multiple documents.

Here is the question you need to classify:

<question>
{query}
</question>

Please follow these steps to classify the question:

1. Analyze the question carefully, considering its content and structure.
2. Determine if the question is about a single document or if it compares two or more documents.
3. Classify the question as follows:
   - If the question is about a single document, classify it as 'general'.
   - If the question compares two or more documents, classify it as 'comparative'.

In your classification process, consider the following:
- List any keywords or phrases that might indicate a comparative question (e.g., "difference", "compare", "versus", "both").
- Explicitly state whether multiple documents are mentioned in the question.
- Consider arguments for classifying the question as 'general'.
- Consider arguments for classifying the question as 'comparative'.
- Make a final decision based on the strength of these arguments.

After your analysis, provide your classification as a single word: either 'general' or 'comparative'.

Example output:
<classification_process>
Keywords indicating comparison: None found
Multiple documents mentioned: No
Arguments for 'general': The question asks about the main theme of a specific book, without mentioning any other documents.
Arguments for 'comparative': None
Decision: The question focuses on a single document's content without any comparative elements.
</classification_process>
general

Please proceed with your classification process and final classification.
    """

    return claude_call(bedrock=bedrock_runtime, query=CLASSIFY_QUESTION_PROMPT, system_message=SYSTEM_PROMPT)


def format_pg_section(doc, pg_section, pg_cntnt) -> str:
    tag = os.path.splitext(os.path.basename(doc))[0].upper()
    return f"\n\t|{tag}_CHUNK|\n|TAG|{pg_section}|/TAG|\n{pg_cntnt}\n\t|/{tag}_CHUNK|\n"


def general_qa_tool(question: str) -> str:
    retrieved_rag = search_similar_text(question, K)
    context_format = """<CONTEXT>\n{entries}\n</CONTEXT>""".strip()

    df_rag = pd.DataFrame(copy.deepcopy(list(retrieved_rag)), columns=['doc', 'section_name', 'section_cntnt', 'cosine_distance'])
    query_df = df_rag[['doc', 'section_name']].drop_duplicates(subset=['section_name'], keep='first')
    query_pairs = list(zip(query_df['doc'], query_df['section_name']))

    retrieved_docs = Session.execute(
        select(
            DocumentSection.document_name,
            DocumentSection.section_name,
            DocumentSection.section_content
        ).where(
            or_(*(tuple_(DocumentSection.document_name, DocumentSection.section_name) == pair for pair in query_pairs))
        )   # .distinct()
    ).all()

    # Preserve the order of (doc, section_name) pairs
    retrieved_docs = list(sorted(
        retrieved_docs,
        key=lambda x: (query_df['doc'].tolist().index(x.document_name), query_df['section_name'].tolist().index(x.section_name))
    ))

    assert len(list(retrieved_docs)) > 0, f"No se encontraron documentos similares para la pregunta: {question}"

    print(f"{question=}")
        
    entries = "\n".join([
        format_pg_section(doc, section, cntnt)
        for doc, section, cntnt in retrieved_docs
    ])

    context = context_format.format(entries=entries)
    p = PROMPT.format(context=context, query=question)

    return {
        'context': context,
        'prompt': p,
        'system_prompt': SYSTEM_PROMPT
    }


def compare_documents_tool(query: str) -> str:
    return {
        # 'context': "",
        # 'prompt': "",
        # 'system_prompt': SYSTEM_PROMPT
    }


def handle_query(q_a: tuple):
    q, a = q_a
    question_response = classify_question_type(q)['content'][0]['text']
    question_type = question_response.split()[-1]

    query_to_fname = q.replace('¿', '').replace('?', '').replace(' ', '_').lower()
    query_to_fname = ''.join(e for e in query_to_fname if e.isalnum() or e == '_')[:50]

    print(f"{q=}")
    print(f"{question_response=}")
    print(f"{question_type=}")

    retrieval_dict: dict = None
    if question_type == 'general':
        retrieval_dict = general_qa_tool(q)
    elif question_type == 'comparative':
        retrieval_dict = compare_documents_tool(q)
    else:
        raise ValueError(f"Invalid question type: {question_type}")
    
    print(json.dumps(retrieval_dict, indent=2, ensure_ascii=False))

    if len(retrieval_dict) and CALL_CLAUDE:
        with open(f'../data/qa/prompt-{K}-{query_to_fname}.txt', 'w') as f:
            f.write(retrieval_dict['prompt'])

        response: dict = claude_call(bedrock_runtime, retrieval_dict['system_prompt'], retrieval_dict['prompt'])
        # print(json.dumps(response['content'], indent=2, ensure_ascii=False))

        llm_answer = extract_answer(response['content'][0]['text'])

        if a:
            b_score = bert_score.compute(predictions=[llm_answer], references=[a], lang='es')
        else:
            b_score = None

        retrieval_dict['response'] = {
            'question': q,
            'answer': a,
            'llm_response': llm_answer,
            # 'bleu_dict': bleu.compute(predictions=[llm_answer], references=[answer]),
            # 'rouge_dict': rouge.compute(predictions=[llm_answer], references=[answer]),
            # 'meteor_dict': meteor.compute(predictions=[llm_answer], references=[answer]),
            'bert_score_dict': b_score
        }

        with open(f'../data/qa/response-{K}-{query_to_fname}.json', 'w') as f:
            f.write(json.dumps(response, indent=2, ensure_ascii=False))

        with open(f'../data/qa/response-{K}-{query_to_fname}.txt', 'w') as f:
            f.write(llm_answer)

    return retrieval_dict

records = []
for q_a in tqdm(QA, total=len(QA)):
    records.append(handle_query(q_a))
print(f"{records=}")


if CALL_CLAUDE:
    df = pd.DataFrame([r['response'] for r in records if 'response' in r])
    # display(df)

    # df['bleu'] = df['bleu_dict'].apply(lambda x: x['bleu'])
    # df['rouge1'] = df['rouge_dict'].apply(lambda x: x['rouge1'])
    # df['rouge2'] = df['rouge_dict'].apply(lambda x: x['rouge2'])
    # df['rougeL'] = df['rouge_dict'].apply(lambda x: x['rougeL'])
    # df['meteor'] = df['meteor_dict'].apply(lambda x: x['meteor'])
    df['bert_score'] = df['bert_score_dict'].apply(lambda x: np.mean(x['f1']) if x else None)

    metric_list = ['bert_score']    # 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor', 

    display(df[['question', 'answer', 'llm_response'] + metric_list].sort_values(by=metric_list, ascending=True))

"""
    Preguntas de comprensión general
¿Cuál es el objetivo principal del documento?
¿Cuáles son los requisitos para la contratación?
¿Qué incluye el servicio de infraestructura en la nube?

    Preguntas de detalles específicos
¿Qué certificaciones debe tener el proveedor de servicios en la nube?
¿Cuáles son las condiciones del soporte técnico?
¿Cuáles son los requerimientos de seguridad en el servicio?

    Preguntas de inferencia y relación
¿Cómo se relacionan los requisitos de la infraestructura con los objetivos del contrato?
¿Por qué es importante contar con alta disponibilidad en la nube?
¿Cómo afectan los servicios de monitoreo a la seguridad del sistema?

¿Cual es la finalidad publica de los documentos?
¿Cuales son los objetivos generales y especificos de la contratacion en los documentos?
"""
