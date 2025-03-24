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


# def generate_response(query):

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
qa = [
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

system_prompt = """
Eres un asistente de IA especializado en responder preguntas basadas en contextos proporcionados. Tu tarea es analizar el contexto dado, entender la pregunta y proporcionar una respuesta precisa y relevante en español.
"""

prompt = """
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

shutil.rmtree('../data/qa', ignore_errors=True)
os.makedirs('../data/qa', exist_ok=True)

call_claude = True
k = 10       # 5 mejor que 3, 10 mejor que 5 pero no sustancial

df = []
for question, answer in tqdm(qa, total=len(qa)):
    print(question)
    retrieved_rag = search_similar_text(question, k)
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
    retrieved_docs = sorted(
        retrieved_docs,
        key=lambda x: (query_df['doc'].tolist().index(x.document_name), query_df['section_name'].tolist().index(x.section_name))
    )
    retrieved_docs = list(retrieved_docs)
    # print(retrieved_docs)

    assert len(list(retrieved_docs)) > 0, f"No se encontraron documentos similares para la pregunta: {question}"

    print(f"{question=}")

    def format_pg_section(doc, pg_section, pg_cntnt) -> str:
        tag = os.path.splitext(os.path.basename(doc))[0].upper()
        return f"\n\t|{tag}_CHUNK|\n|TAG|{pg_section}|/TAG|\n{pg_cntnt}\n\t|/{tag}_CHUNK|\n"
        
    entries = "\n".join([
        format_pg_section(doc, section, cntnt)
        for doc, section, cntnt in retrieved_docs
    ])
    context = context_format.format(entries=entries)

    query_to_fname = question.replace('¿', '').replace('?', '').replace(' ', '_').lower()
    query_to_fname = ''.join(e for e in query_to_fname if e.isalnum() or e == '_')[:50]
    # print(f"{query_to_fname=}")
    # print(context)
    with open(f'../data/qa/prompt-{k}-{query_to_fname}.txt', 'w') as f:
        f.write(prompt.format(context=context, query=question))
    # print(prompt)

    if call_claude:
        response: dict = claude_call(bedrock_runtime, system_prompt, prompt.format(context=context, query=question))
        # print(json.dumps(response['content'], indent=2, ensure_ascii=False))
        
        llm_answer = extract_answer(response['content'][0]['text'])
        if answer:
            b_score = bert_score.compute(predictions=[llm_answer], references=[answer], lang='es')
        else:
            b_score = None

        df.append({
            'question': question,
            'answer': answer,
            'llm_response': llm_answer,
            # 'bleu_dict': bleu.compute(predictions=[llm_answer], references=[answer]),
            # 'rouge_dict': rouge.compute(predictions=[llm_answer], references=[answer]),
            # 'meteor_dict': meteor.compute(predictions=[llm_answer], references=[answer]),
            'bert_score_dict': b_score
        })

        with open(f'../data/qa/response-{k}-{query_to_fname}.json', 'w') as f:
            f.write(json.dumps(response, indent=2, ensure_ascii=False))

        with open(f'../data/qa/response-{k}-{query_to_fname}.txt', 'w') as f:
            f.write(llm_answer)


if call_claude:
    df = pd.DataFrame(df)
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
