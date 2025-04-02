from sqlalchemy import select
from llm import bedrock_runtime, embed_call, claude_call
import json
import pandas as pd
import numpy as np
import shutil
from IPython.display import display
import copy
from tqdm import tqdm
from langchain_aws import ChatBedrock
import logging
from difflib import SequenceMatcher, HtmlDiff, ndiff

from db import *
from metrics import *
from utilities import *
import warnings
import re

warnings.filterwarnings("ignore")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 50)
np.set_printoptions(precision=3)


# https://github.com/amberpe/poc-rag-multidocs/blob/main/RAG.py#L218

# Prompt Improver: https://console.anthropic.com/dashboard
QA = [
# # * Chunk question
#     'g.\nEl servicio debe permitir asociar una o más direcciones IP elásticas a cualquier\ninstancia\nde\nla\nnube\nprivada\nvirtual,\nde modo que puedan alcanzarse\ndirectamente desde Internet.\nh.\nEl servicio debe permitir conectarse a la nube privada virtual con otras nubes\nprivadas virtuales y obtener acceso a los recursos de otras nubes privadas\nvirtuales a través de direcciones IP privadas mediante la interconexión de nube\nprivada virtual.\ni.\nEl servicio debe permitir conectarse de manera privada a los servicios del\nfabricante de la nube pública sin usar una gateway de Internet, ni una NAT ni\nun proxy de firewall mediante un punto de enlace de la nube privada virtual.\nj.\nEl servicio debe permitir conectar la nube privada virtual y la infraestructura de\nTI local con la VPN del fabricante de la nube pública de sitio a sitio.\nk.\nEl servicio debe permitir asociar grupos de seguridad de la nube privada virtual\ncon instancias en la plataforma.\nl.',

# * Cross document question
    # Bien
    ("¿Cual es la finalidad publica de los documentos?", 
     "La presente contratación pública tiene como finalidad mantener la operatividad y modernización de nuestra plataforma tecnológica, buscando elevar los niveles de eficiencia y satisfacción del personal administrativo, profesionales de la salud, usuarios internos y externos de EsSalud. Por la naturaleza del valor de los activos de información, por las mejores prácticas de seguridad y continuidad de los servicios, es necesario el fortalecimiento de las capacidades para la habilitación y validación de los niveles de transacción necesarios para despliegue de las aplicaciones, sobre el cual se brindará una atención oportuna a los asegurados y personal administrativo de EsSalud a nivel nacional, con la finalidad de asegurar la disponibilidad y confiabilidad de la documentación que generan las diferentes unidades orgánicas."), 

    # Bien
    ("¿Cuales son los objetivos generales y especificos de la contratacion en los documentos?", """
Objetivo General:
Contratar el Servicio de Infraestructura, Plataforma y Microservicios en Nube Pública 
para el despliegue de las Aplicaciones y Nuevos Servicios de la Gerencia Central de 
Tecnologías de Información y Comunicaciones de EsSalud.

Objetivos Específicos:
- Contar con un servicio que permita un alto rendimiento en capacidades de procesamiento, 
  memoria, almacenamiento, comunicaciones, seguridad y redes a través de una Infraestructura 
  Pública o nube pública.
- Garantizar un alto nivel de seguridad en el despliegue de las aplicaciones de EsSalud.
- Proporcionar un servicio garantizando el soporte técnico brindado por el fabricante. 
  Asimismo, el servicio debe tener como alta prioridad la seguridad de la información, a través 
  de diversos controles tanto lógicos como físicos.
"""),

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

    # ("¿Que se dice acerca del 'Servicio de visualización de datos en la nube'?", ""),     # GPT no la hace

    ("¿Cual es la responsabilidad del postor durante la implementacion del servicio?", """
El postor es responsable de:

- Elaborar un plan de trabajo para la implementación del proyecto.
- Diseñar la arquitectura de la solución con Infraestructura como Código (IaC) en dos ambientes: QA y PROD.
- Implementar y diseñar el flujo de despliegue y entrega continua (CI/CD) para backend, frontend y móvil Android en QA y PROD.
- Realizar hasta cinco pruebas de estrés end-to-end en QA para validar la conexión VPN Site-To-Site y la comunicación con APIs externos.
- Asegurar la observabilidad de la aplicación, APIs y base de datos mediante herramientas de OpenTelemetry o similares.
- Ejecutar hasta dos pruebas de ethical hacking y/o pentesting en una aplicación end-to-end que incluye backend, frontend y APIs.
- Entregar informes técnicos detallados con mejoras de arquitectura y vulnerabilidades detectadas en las pruebas de seguridad.
- Brindar soporte técnico 24/7 durante la duración del contrato.
- Garantizar que todos los trabajadores cuenten con Seguro Complementario de Trabajo de Riesgo (SCTR).

El proveedor deberá presentar la documentación de implementación en un plazo máximo de 30 días calendarios desde la firma del contrato.
"""),
    # TODO: preguntas acerca de Anexos

# * Section questions
    ("¿Que dice la seccion 2 'Finalidad publica'?", ""),
    ("¿Que dice la seccion 6.3.1.?", ""),
    ("Explicame lo que dice en la seccion 5.5", """
La prestación del servicio será en el (Piso 6) de la Sede Central Essalud, ubicado en el Edificio Lima - Jr. Domingo Cueto 120 – Jesús María – Lima.

La implementación del servicio será realizada en un plazo máximo de hasta treinta (30) días calendarios, contados a partir del día siguiente de la firma del contrato. La Sub Gerencia de Operaciones de Tecnologías de la Información y Sub Gerencia de Sistemas Aseguradores, Subsidios y Sociales – GCTIC, con el Proveedor suscribirán el Acta de Conformidad de Implementación (Anexo A), a más tardar a los cinco (05) días calendarios posteriores al término de la implementación, en caso no se presenten observaciones. Finalizada la implementación del servicio, se firmará el 'Acta de Inicio del Servicio' (Ver Anexo B), suscrita por el representante de la Sub Gerencia de Operaciones de TI y el Proveedor, para dar inicio al conteo de los trescientos sesenta y cinco (365) días calendarios de prestación del servicio.

El plazo de prestación del servicio será de trescientos sesenta y cinco (365) días calendario, contados a partir del día siguiente de la firma del 'Acta de Inicio de Servicio'.
     """),

# * Comparative questions
    
    # No hay diferencias
    ("¿Qué diferencias hay en la sección 2 'Finalidad publica'?", 
     "No hay diferencias en la seccion 2 'Finalidad publica' entre los 2 documentos"),
    
    # Seccion 5.1: diferencias importantes, pero contexto muy largo para la LLM
    ("Cuales son las diferencias en la sección 5.1.?", """Las diferencias principales en la sección 5.1 entre ambos documentos son las siguientes:
    1. Composición de las prestaciones:
        - En tdr_v6, se agrupan las prestaciones en dos ítems: la prestación principal (que incluye el Servicio de implementación y migración y el Servicio de infraestructura Pública o Nube Pública) y la prestación accesoria (que agrupa el Servicio de Gestión y Soporte y el Servicio de capacitación).
        - En tdr_v4, se detallan tres ítems: la prestación principal (con el Servicio de implementación y migración y el Servicio de infraestructura, soporte de la marca y adicionales), la prestación accesoria (Servicio de Gestión y Soporte) y un ítem separado para la capacitación (Servicio de capacitación).
    2. Certificaciones requeridas:
        - Ambos documentos exigen certificaciones para el servicio de nube pública, incluyendo CSA, FedRAMP, SOC 1, SOC 2, SOC 3, ISO 9001, ISO 27001, ISO 27017, ISO 27018 e ISO 22301:2019.
        - tdr_v4 incorpora adicionalmente la certificación HIPAA, que no se menciona en tdr_v6.
    3. Redacción y formato:
        - Aunque ambos textos establecen requisitos similares (por ejemplo, disponibilidad mínima del 99.99%, plataforma de administración de la nube, y la necesidad de contar con al menos dos centros de datos para garantizar la resiliencia), la estructura y el orden de presentación varían ligeramente entre ambos documentos.
    Estas diferencias reflejan ajustes en la forma de agrupar los servicios y en algunos requisitos específicos, como las certificaciones adicionales en tdr_v4."""),

    # Hay diferencias
    ("¿Qué diferencias hay en la sección 6.3.1.?", """Las diferencias en la sección 6.3.1. Entregables del servicio entre ambos documentos son las siguientes:

1. Cambio en la cláusula sobre servicios no utilizados:
   - En 'tdr_v4.pdf': Si no se utiliza el 100% de los servicios solicitados, la entidad podrá utilizar la diferencia hasta la finalización del contrato.
   - En 'tdr_v6.pdf': En lugar de permitir que la entidad use la diferencia, se menciona que el contratista podrá otorgar lo equivalente en créditos disponibles para la entidad.

Por lo demás, los entregables y requisitos son idénticos en ambos documentos."""),
    ("¿Las certificaciones requeridas para el proveedor de nube pública son las mismas en ambas versiones del documento?", 
     "No exactamente. Ambas versiones requieren certificaciones como ISO 27001, ISO 9001, SOC 1/2/3, FedRAMP y Cloud Security Alliance (CSA). Sin embargo, la versión V4 incluye HIPAA como requisito, mientras que en la versión V6 esta certificación ya no aparece"),
    ("¿En que se diferencia hay en la sección II.?", """En tdr_v4.pdf: "El postor deberá contar con carta y/o certificado de respaldo como partner avanzado oficial de la marca (fabricante) de la nube pública a ofertar."

En tdr_v6.pdf: "El postor deberá contar con carta y/o certificado de respaldo como partner oficial de la marca (fabricante) de la nube pública a ofertar."

La palabra "avanzado" fue eliminada en la versión v6, reduciendo el nivel de partnership requerido."""),
    ("¿En que se diferencia hay en la sección II. Requisitos de Claificacion?", """En tdr_v4.pdf: "El postor deberá contar con carta y/o certificado de respaldo como partner avanzado oficial de la marca (fabricante) de la nube pública a ofertar."

En tdr_v6.pdf: "El postor deberá contar con carta y/o certificado de respaldo como partner oficial de la marca (fabricante) de la nube pública a ofertar."

La palabra "avanzado" fue eliminada en la versión v6, reduciendo el nivel de partnership requerido."""),

    ("Tomando en cuenta que yo soy el desarrollador. dame las diferencias entre las 2 licitaciones", "")
]
# QA = [QA[-1]]

RESET = False
CALL_CLAUDE = True
K = 5       # 5 mejor que 3, 10 mejor que 5 pero no sustancial

if RESET:
    shutil.rmtree('../data/qa', ignore_errors=True)
    os.makedirs('../data/qa', exist_ok=True)


# -----------------------------------------------------------------------------
# PROMPTS
# -----------------------------------------------------------------------------

ALL_DIFERENCES_PROMPT_USER = load_prompt('./prompts/ans_all_diff_user.txt')

ANSWER_QUESTION_PROMPT_SYSTEM = load_prompt('./prompts/ans_q_system.txt')

ANSWER_QUESTION_PROMPT_USER = load_prompt('./prompts/ans_q_user.txt')

ANSWER_QUESTION_DIFFERENCES_PROMPT_SYSTEM = load_prompt('./prompts/ans_qd_system.txt')

ANSWER_QUESTION_DIFFERENCES_PROMPT_USER = load_prompt('./prompts/ans_qd_user.txt')

CLASSIFY_QUESTION_PROMPT_USER = load_prompt('./prompts/classify_question_user.txt')

CHECK_SECTION_PROMPT_USER = load_prompt('./prompts/check_section_user.txt')

TOPK_SECTIONS_PROMPT_SYSTEM = load_prompt('./prompts/topk_sections_system.txt')

TOPK_SECTIONS_PROMPT_USER = load_prompt('./prompts/topk_sections_user.txt')

# -----------------------------------------------------------------------------
# LLM FUNCTIONS
# -----------------------------------------------------------------------------

@retry_with_logging()
def question_all_diferences(query: str) -> dict:
    response_content = claude_call(bedrock=bedrock_runtime, query=ALL_DIFERENCES_PROMPT_USER.format(query=query), system_message="")
    response = response_content['content'][0]['text']
    question_type = retrieve_key_from_xml(response, 'respuesta')

    assert question_type in ['Si', 'No'], f"Invalid question type: {question_type}"

    return {
        "question_type": question_type,
    }

# assert question_all_diferences('¿Que dice la seccion 7.2?')['question_type'] == 'No'
# assert question_all_diferences("Tomando en cuenta que yo soy el desarrollador. dame las diferencias entre las 2 licitaciones")['question_type'] == 'Si'

@retry_with_logging()
def classify_question_type(query: str) -> dict:

    response_content = claude_call(bedrock=bedrock_runtime, query=CLASSIFY_QUESTION_PROMPT_USER.format(query=query), system_message="")
    response = response_content['content'][0]['text']
    question_type = response.split()[-1]

    assert question_type in ['general', 'comparativa'], f"Invalid question type: {question_type}"

    return {
        "response": response,
        "question_type": question_type
    }

# classify_question_type('¿Que dice la seccion 7.2?')
# classify_question_type('Compara la seccion 5 de ambos documentos')


@retry_with_logging()
def check_sections_in_question(query: str) -> dict:

    response_content = claude_call(bedrock=bedrock_runtime, query=CHECK_SECTION_PROMPT_USER.format(query=query), system_message="")
    response = response_content['content'][0]['text']

    sections_in_question = remove_accents(response.split()[-1].lower())
    assert sections_in_question in ['si', 'no'], f"Invalid response: {sections_in_question}"

    return {
        "response": response,
        "sections_in_question": True if sections_in_question == 'si' else False
    }

# print(json.dumps(
#     check_sections_in_question('¿Que dice la seccion 7.2?'), indent=2))
# print(json.dumps(
#     check_sections_in_question('Compara la seccion 5 de ambos documentos'), indent=2))
# print(json.dumps(
#     check_sections_in_question('De acuerdo a las secciones 5 y 6.3.1. ¿Cuales son las diferencias?'), indent=2))
# print(json.dumps(
#     check_sections_in_question('De acuerdo a la seccion 4. ¿Que dice acerca de la seguridad de la informacion?'), indent=2))


class SortSectionTitles(BaseModel):
    key_list: list = Field(..., description="List of keys in the new order")


def get_section_references(text: str) -> list:
    """
    Extrae todos los números de sección en formato 'X', 'X.Y', 'X.Y.Z' del texto.
    Detecta números enteros como '5' y jerarquías completas como '6.3.1'.
    """
    matches = re.findall(r'\b\d+(?:\.\d+)*\b', text)  # Encuentra todas las coincidencias
    return matches if matches else []

# print(get_section_references('¿Que dice la seccion 7.2?'))
# print(get_section_references('Compara la seccion 5 de ambos documentos'))
# print(get_section_references('De acuerdo a las secciones 5 y 6.3.1. ¿Cuales son las diferencias?'))
# print(get_section_references('De acuerdo a la seccion 4. ¿Que dice acerca de la seguridad de la informacion?'))


def sort_by_match(target: str, candidates: list) -> list:
    def match_score(candidate):
        matcher = SequenceMatcher(None, target, candidate)
        return sum(block.size for block in matcher.get_matching_blocks())

    return sorted(candidates, key=match_score, reverse=True)


def get_topk_sections_from_question(question: str, num_to_keys: dict) -> list:
    candidates = num_to_keys.keys()
    section_references = get_section_references(question)
    print(f"{section_references=}")

    fixed_candidates = []
    priority_candidates = []
    other_candidates = []
    for section_number in section_references:
        print(f"{section_number=}")
        if section_number:
            top_level_section = section_number.split('.')[0]
            # p_c = [c for c in candidates if c.startswith(section_number) or section_number.startswith(c)]
            f_c = [
                c for c in candidates if c.startswith(section_number)
            ]
            p_c = [
                c for c in candidates if c.startswith(top_level_section) and c not in f_c
            ]
            o_c = [c for c in candidates if c not in p_c and c not in f_c]
        else:
            p_c = candidates
            o_c = []

        fixed_candidates.extend(f_c)
        priority_candidates.extend(p_c)
        other_candidates.extend(o_c)
        # print(f"{p_c=}")
        # print(f"{o_c=}")

    sorted_fixed = sort_by_match(question, fixed_candidates)
    sorted_priority = sort_by_match(question, priority_candidates)
    sorted_others = sort_by_match(question, other_candidates)

    to_names = lambda x: [num_to_keys[v] for v in x]

    return (
        to_names(sorted_fixed),
        to_names(sorted_priority),
        to_names(sorted_others),
    )

"""
index_tree = json.loads(open(f'../data/tree_tdr_v4.json', "r").read())
num_to_section = {
    k.split()[0] if 'ANEXO' not in k else k: k 
    for k in sorted(list(get_all_keys(index_tree))) 
}

print(get_topk_sections_from_question("Que dice la seccion 5", num_to_section))
print(get_topk_sections_from_question("Que dice la seccion 6.3.1", num_to_section))
print(get_topk_sections_from_question("Que dicen los anexos", num_to_section))
print(get_topk_sections_from_question('¿Que dice la seccion 7.2?', num_to_section))
print(get_topk_sections_from_question('Compara la seccion 5 de ambos documentos', num_to_section))
print(get_topk_sections_from_question('De acuerdo a las secciones 5 y 6.3.1. ¿Cuales son las diferencias?', num_to_section))
print(get_topk_sections_from_question('Explicame lo que dice en la seccion 5.5', num_to_section))
print(get_topk_sections_from_question("¿Qué diferencias hay en la sección 2 'Finalidad publica'?", num_to_section))
"""

# -----------------------------------------------------------------------------
# PROGRAM FUNCTIONS
# -----------------------------------------------------------------------------

def format_pg_section(doc: str, pg_section: str, pg_cntnt: str) -> str:
    tag = os.path.splitext(os.path.basename(doc))[0].upper()
    return f"\n\t|{tag}_CHUNK|\n|TAG|{pg_section}|/TAG|\n{pg_cntnt}\n\t|/{tag}_CHUNK|\n"


def retrieve_sections_from_question_simple(question: str) -> dict:
    query_embedding = embed_call(bedrock_runtime, question)
    retrieved_rag = Session.execute(
        select(
            Embedding.document_name,
            Embedding.section_name,
            Embedding.chunk_content,
            Embedding.embedding.cosine_distance(query_embedding['embedding']).label('cosine_distance')
        )
        .order_by(
            Embedding.embedding.cosine_distance(query_embedding['embedding'])
        ).limit(K)
    )
    
    df_rag = pd.DataFrame(copy.deepcopy(list(retrieved_rag)), columns=['doc', 'section_name', 'section_cntnt', 'cosine_distance'])

    query_df = df_rag[['doc', 'section_name']].drop_duplicates(subset=['section_name'], keep='first')
    
    return {
        record['doc']: record['section_name']
        for record in
        query_df.groupby('doc')['section_name'].apply(list).reset_index().to_dict(orient='records')
    }


def retrieve_sections_from_question_similarity(question: str) -> dict:
    """
    1. question -> top K index names 
    2. RAG of query embeddings on filtered chunk table (by top K index names) to rerank the section names
    3. Retrieve the reranked sections from document table
    """
    retrieve_sections = {}
    for f_path in ['../data/tdr_v4.pdf', '../data/tdr_v6.pdf']:

        # Get Context From Each Document
        print(f"\t -> {f_path=}")
        index_tree = json.loads(open(f'../data/tree_{os.path.basename(f_path).replace(".pdf", "")}.json', "r").read())
        keys_all = sorted(list(get_all_keys(index_tree)))
        num_to_section = {
            k.split()[0] if 'ANEXO' not in k else k: k 
            for k in keys_all
        }
        
        fixed_sections, related_sections, other_sections = get_topk_sections_from_question(question, num_to_section)
        if len(related_sections) == 0:
            related_sections = other_sections
        
        document_sections = [(f_path, v) for v in related_sections]
        print(f"{fixed_sections=} {related_sections=}")

        query_embedding = embed_call(bedrock_runtime, question)
        ranked_chunks = Session.execute(
            select(
                Embedding.document_name,
                Embedding.section_name,
                Embedding.chunk_content,
                Embedding.embedding.cosine_distance(query_embedding['embedding']).label('cosine_distance')
            ).where(
                and_(
                    Embedding.document_name == f_path,
                    or_(*(tuple_(Embedding.document_name, Embedding.section_name) == pair for pair in document_sections))
                )
            ).order_by(text('cosine_distance')).limit(K)
        ).all()
                
        seen = set()
        rag_sections = fixed_sections + [s for _, s, _, _ in ranked_chunks if not (s in seen or seen.add(s))]
        print(f"{rag_sections=}")

        retrieve_sections[f_path] = rag_sections

    return retrieve_sections


def retrieve_docs(file_path: str, sections: list):
    retrieved_docs = Session.execute(
        select(
            DocumentSection.document_name,
            DocumentSection.section_name,
            DocumentSection.section_content
        ).where(
            DocumentSection.document_name == file_path,
            DocumentSection.section_name.in_(sections)
        )   # .distinct()
    ).all()

    return retrieved_docs


def docs_to_context(file_path: str, docs: list) -> str:
    entries = "\n".join([
        format_pg_section(doc, section, cntnt)
        for doc, section, cntnt in docs
    ])
    f_name = os.path.basename(file_path).replace(".pdf", "")
    return f"<CONTEXT_{f_name}>\n{entries}\n</CONTEXT_{f_name}>".strip()


def general_qa_tool(question: str) -> str:
    context_format = """<CONTEXT>\n{entries}\n</CONTEXT>""".strip()
    extracted_docs = []
    all_sections = []

    sections_in_response  = check_sections_in_question(question)
    has_sections = sections_in_response['sections_in_question']
    print(f"{has_sections=}")

    if has_sections:
        extracted_sections = retrieve_sections_from_question_similarity(question)
    else:
        extracted_sections = retrieve_sections_from_question_simple(question)

    print(f"{extracted_sections=}")
    for f_path, sections in extracted_sections.items():
        print(f"{f_path=}")
        extracted_docs.extend(retrieve_docs(f_path, sections))
        all_sections.extend(sections)


    assert len(list(extracted_docs)) > 0, f"No se encontraron documentos similares para la pregunta: {question}"

    print(f"{question=}")
        
    entries = "\n".join([
        format_pg_section(doc, section, cntnt)
        for doc, section, cntnt in extracted_docs
    ])

    context = context_format.format(entries=entries)
    p = ANSWER_QUESTION_PROMPT_USER.format(context=context, query=question)

    return {
        'sections': all_sections,
        'context': context,
        'prompt': p,
        'system_prompt': ANSWER_QUESTION_PROMPT_SYSTEM,
        'sections_in_question': has_sections,
        'sections_in_response': sections_in_response['response'],
    }


def get_doc_chunk_differences(section_name: str, summary: bool = False) -> list:

    if summary:
        chuk_differences = Session.execute(
            select(
                SectionDiff.summary_difference
            ).where(
                and_(
                    SectionDiff.section_v4 == section_name,
                    SectionDiff.section_v6 == section_name
                )
            )
        )
    else:
        chuk_differences = Session.execute(
            select(
                # SectionDiff.section_v4,
                # SectionDiff.section_v6,
                SectionDiff.chunk_v4,
                SectionDiff.chunk_v6
            ).where(
                and_(
                    SectionDiff.section_v4 == section_name,
                    SectionDiff.section_v6 == section_name
                )
            )
        )

    return chuk_differences


def compare_documents_tool(question: str) -> str:
    print(f"{question=}")
    flag = question_all_diferences(question)['question_type'] == 'Si'
    if flag:
        
        sections_v4 = json.loads(open(f'../data/tree_tdr_v4.json', "r").read())
        sections_v6 = json.loads(open(f'../data/tree_tdr_v6.json', "r").read())
        sections_to_compare = list(set.union(set(get_all_keys(sections_v4)),set(get_all_keys(sections_v6))))
    else:
        retrieve_sections = retrieve_sections_from_question_similarity(question)    
        sections_to_compare = list(set(retrieve_sections['../data/tdr_v4.pdf']).intersection(set(retrieve_sections['../data/tdr_v6.pdf'])))

    print(f"{sections_to_compare=}")
    # print(json.dumps(sections_to_compare, indent=2, ensure_ascii=False))

    context_list = []
    for f_path in ['../data/tdr_v4.pdf', '../data/tdr_v6.pdf']:
        retrieved_docs = retrieve_docs(f_path, sections_to_compare)
        context_doc = docs_to_context(f_path, retrieved_docs)
        context_list.append(context_doc)

    diff_context = []
    for sec in sections_to_compare:
        chunk_diff = get_doc_chunk_differences(sec, flag)
        try:
            diff_context.append("\n".join([
                f"<DIFFERENCES_{sec}>\n<CHUNK_V4>{chunk_v4}</CHUNK_V4>\n\n<CHUNK_V6>{chunk_v6}</CHUNK_V6>\n</DIFFERENCES_{sec}>"
                for chunk_v4, chunk_v6 in chunk_diff
            ]))
        except:
            diff_context.append("\n".join([
                f"<DIFFERENCES_{sec}>\n{summary_diff}\n</DIFFERENCES_{sec}>"
                for summary_diff in chunk_diff
            ]))

    diff_context = "\n\n".join(diff_context)
    docs_context = "\n\n".join(context_list)

    if flag:
        docs_context = ""

    p = ANSWER_QUESTION_DIFFERENCES_PROMPT_USER.format(context=docs_context, question=question, diff_context=diff_context)

    return {
        'sections': sections_to_compare,
        'context': docs_context,
        'prompt': p,
        'all': flag,
        'system_prompt': ANSWER_QUESTION_DIFFERENCES_PROMPT_SYSTEM
    }


def handle_query(q_a: tuple) -> dict:
    q, a = q_a
    query_to_fname = q.replace('¿', '').replace('?', '').replace(' ', '_').lower()
    query_to_fname = ''.join(e for e in query_to_fname if e.isalnum() or e == '_')[:50]

    question_type_response = classify_question_type(q)
    question_type = question_type_response['question_type']

    retrieval_dict: dict = {
        'question': q,
        'question_type': question_type, 
        'question_type_response': question_type_response['response']
    }

    print(f"{q=}")
    # print(f"{question_type_response=}")
    print(f"{question_type=}")

    if question_type == 'general':
        retrieval_dict = {**retrieval_dict, **general_qa_tool(q)}
    elif question_type == 'comparativa':
        retrieval_dict = {**retrieval_dict, **compare_documents_tool(q)}
    else:
        raise ValueError(f"Invalid question type: {question_type}")

    
    # print(json.dumps(retrieval_dict, indent=2, ensure_ascii=False))

    if 'prompt' in retrieval_dict and len(retrieval_dict['prompt']):
        with open(f'../data/qa/{question_type}-prompt-{K}-{query_to_fname}.txt', 'w') as f:
            f.write(retrieval_dict['prompt'])

    if 'prompt' in retrieval_dict and CALL_CLAUDE:
        response: dict = claude_call(bedrock_runtime, retrieval_dict['system_prompt'], retrieval_dict['prompt'])
        # print(json.dumps(response['content'], indent=2, ensure_ascii=False))

        llm_answer = retrieve_key_from_xml(response['content'][0]['text'])

        if a:
            b_score = bert_score.compute(predictions=[llm_answer], references=[a], lang='es')
        else:
            b_score = None

        retrieval_dict['answer'] = a
        retrieval_dict['llm_response'] = llm_answer
        retrieval_dict['bert_score_dict'] = b_score
        # 'bleu_dict': bleu.compute(predictions=[llm_answer], references=[answer]),
        # 'rouge_dict': rouge.compute(predictions=[llm_answer], references=[answer]),
        # 'meteor_dict': meteor.compute(predictions=[llm_answer], references=[answer]),

        with open(f'../data/qa/{question_type}-response-{K}-{query_to_fname}.json', 'w') as f:
            f.write(json.dumps(response, indent=2, ensure_ascii=False))

        with open(f'../data/qa/{question_type}-response-{K}-{query_to_fname}.txt', 'w') as f:
            f.write(llm_answer)

    return retrieval_dict


records = []
for q_a in tqdm(QA, total=len(QA)):
    print(f"{q_a[0]=}")
    records.append(handle_query(q_a))

    # try:
    #     records.append(handle_query(q_a))
    # except Exception as e:
    #     print(f"Error: {e}")
print(f"{records=}")


if CALL_CLAUDE:
    df = pd.DataFrame([r for r in records if len(r)])
    # display(df)

    # df['bleu'] = df['bleu_dict'].apply(lambda x: x['bleu'])
    # df['rouge1'] = df['rouge_dict'].apply(lambda x: x['rouge1'])
    # df['rouge2'] = df['rouge_dict'].apply(lambda x: x['rouge2'])
    # df['rougeL'] = df['rouge_dict'].apply(lambda x: x['rougeL'])
    # df['meteor'] = df['meteor_dict'].apply(lambda x: x['meteor'])
    df['bert_score'] = df['bert_score_dict'].apply(lambda x: np.mean(x['f1']) if x else None)

    metric_list = ['bert_score']    # 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor', 

    #  'sections_in_question',
    display(df[['question_type', 'question', 'sections', 'answer', 'llm_response'] + metric_list].sort_values(by=metric_list, ascending=True))
    # display(df)

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

