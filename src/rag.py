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

from db import *
from metrics import *
from utilities import *
import warnings

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
    ("¿Qué diferencias hay en la sección 2 'Finalidad publica'?", ""),
    
    # Seccion 5.1: diferencias importantes, pero contexto muy largo para la LLM
    ("Cuales son las diferencias en la sección 5.1.?", ""),

    # Hay diferencias
    ("¿Qué diferencias hay en la sección 6.3.1.?", ""),
    ("¿Las certificaciones requeridas para el proveedor de nube pública son las mismas en ambas versiones del documento?", 
     "No exactamente. Ambas versiones requieren certificaciones como ISO 27001, ISO 9001, SOC 1/2/3, FedRAMP y Cloud Security Alliance (CSA). Sin embargo, la versión V4 incluye HIPAA como requisito, mientras que en la versión V6 esta certificación ya no aparece​")
]

RESET = False
CALL_CLAUDE = True
K = 10       # 5 mejor que 3, 10 mejor que 5 pero no sustancial

if RESET:
    shutil.rmtree('../data/qa', ignore_errors=True)
    os.makedirs('../data/qa', exist_ok=True)


# -----------------------------------------------------------------------------
# LLM FUNCTIONS
# -----------------------------------------------------------------------------
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


CLASSIFY_QUESTION_PROMPT = """
Eres un sistema sofisticado de clasificación de preguntas. Tu objetivo es categorizar una pregunta en una de dos categorías basándote en si se refiere a un solo documento o compara múltiples documentos.

Objetivo de Clasificación:
1. Si la pregunta trata sobre un solo documento: clasificar como 'general'
2. Si la pregunta compara dos o más documentos: clasificar como 'comparativa'

Proceso de Clasificación:
Debes realizar un análisis detallado considerando:
- Palabras o frases que indiquen una pregunta comparativa
- Mención de múltiples documentos
- Argumentos para clasificación 'general'
- Argumentos para clasificación 'comparativa'

Ejemplos de Clasificación:

Ejemplo 1:
Pregunta: "¿Cuál es el objetivo principal del documento?"
<classification_process>
Palabras clave que indican comparación: Ninguna encontrada
Documentos múltiples mencionados: No
Argumentos para 'general': La pregunta se enfoca en un único documento específico
Argumentos para 'comparativa': Ninguno
Decisión: Clasificación como 'general'
</classification_process>
general

Ejemplo 2:
Pregunta: "Cuales son las diferencias en los anexos?"
<classification_process>
Palabras clave que indican comparación: "diferencias"
Documentos múltiples mencionados: Sí (múltiples anexos)
Argumentos para 'general': Ninguno
Argumentos para 'comparativa': Solicitud explícita de comparación entre anexos
Decisión: Clasificación como 'comparativa'
</classification_process>
comparativa

Ejemplo 3:
Pregunta: "¿Cuáles son los requerimientos principales del proyecto?"
<classification_process>
Palabras clave que indican comparación: Ninguna encontrada
Documentos múltiples mencionados: No
Argumentos para 'general': La pregunta busca información sobre un único proyecto
Argumentos para 'comparativa': Ninguno
Decisión: Clasificación como 'general'
</classification_process>
general

Ejemplo 4:
Pregunta: "En que se diferencia la seccion 7 de los documentos?"
<classification_process>
Palabras clave que indican comparación: "diferencia"
Documentos múltiples mencionados: Sí (múltiples documentos)
Argumentos para 'general': Ninguno
Argumentos para 'comparativa': Comparación explícita de la sección 7 entre varios documentos
Decisión: Clasificación como 'comparativa'
</classification_process>
comparativa

Ejemplo 5:
Pregunta: "¿En que se diferencia el ANEXO A?"
<classification_process>
Palabras clave que indican comparación: "diferencia"
Documentos múltiples mencionados: Implícitamente (referencias a un ANEXO A específico)
Argumentos para 'general': La pregunta parece centrarse en un único anexo
Argumentos para 'comparativa': La palabra "diferencia" sugiere una comparación
Decisión: Clasificación como 'comparativa'
</classification_process>
comparativa

Pasos para Clasificar la Pregunta:
1. Analiza cuidadosamente el contenido y la estructura de la pregunta
2. Identifica si se trata de un solo documento o múltiples documentos
3. Detecta palabras clave comparativas: "diferencia", "comparar", "versus", "ambos"
4. Evalúa los argumentos para cada clasificación
5. Toma una decisión final basada en la solidez de los argumentos

Formato de Salida:
- Proporciona un <classification_process> detallado
- Indica la clasificación final como una sola palabra: 'general' o 'comparativa'

Instrucción Final:
Por favor, realiza el proceso de clasificación para la siguiente pregunta:
<question>
{query}
</question>
"""

CHECK_SECTIONS_PROMPT = """
Eres un sistema especializado en identificar referencias a secciones específicas en una pregunta.

Objetivo:
Determinar si la pregunta hace referencia a una sección específica de un documento.

Criterios de Identificación:
1. Patrones de secciones numeradas: "5.1", "7.2", "Sección 3", etc.
2. Referencias a anexos: "Anexo A", "Anexo 1", "Apéndice B"
3. Referencias a partes específicas: "capítulo", "página", "tabla", "figura"
4. Palabras clave que sugieren referencia específica: "en", "de", "del"

Proceso de Análisis:
- Examinar la pregunta en busca de patrones numéricos o textuales de secciones
- Considerar contexto y estructura de la referencia
- Evaluar la especificidad de la mención

Ejemplos:

Ejemplo 1:
Pregunta: "¿Cuál es el contenido de la sección 5.1?"
<classification_process>
Patron de sección identificado: "5.1"
Tipo de referencia: Sección numerada
Decisión: Si
</classification_process>
Si

Ejemplo 2:
Pregunta: "Describe el Anexo A del documento"
<classification_process>
Patron de sección identificado: "Anexo A"
Tipo de referencia: Anexo específico
Decisión: Si
</classification_process>
Si

Ejemplo 3:
Pregunta: "¿Cuál es el objetivo principal?"
<classification_process>
Patron de sección identificado: Ninguno
Tipo de referencia: Pregunta general
Decisión: No
</classification_process>
No

Pasos para Análisis:
1. Buscar patrones numéricos de secciones
2. Identificar palabras clave de referencias específicas
3. Evaluar el nivel de especificidad de la pregunta
4. Tomar decisión final

Instrucción Final:
Analiza la siguiente pregunta:
<question>
{query}
</question>
"""

SYSTEM_SIMILARITY_PROMPT = """
Eres un asistente de IA avanzado especializado en análisis semántico y clasificación de información textual. Tus capacidades principales incluyen:
Comprender profundamente el contexto semántico de las preguntas
Analizar los títulos de secciones con una evaluación de relevancia matizada
Proporcionar clasificaciones precisas y bien razonadas basadas en la similitud semántica
Generar resultados estructurados que cumplan requisitos específicos

Pautas Clave:
Priorizar la comprensión semántica sobre la coincidencia literal de palabras clave
Ser exhaustivo y sistemático en la evaluación de relevancia
Proporcionar razonamientos claros y explicables para las clasificaciones
Adherirse estrictamente a los requisitos de formato de salida
"""

SIMILARITY_PROMPT = """
Eres un asistente de IA avanzado especializado en análisis semántico y clasificación de información textual. Tu tarea es analizar la relevancia de los títulos de secciones con respecto a una pregunta específica.

## Ejemplos de Análisis
Asume las siguientes SECCIONES DISPONIBLES:

['1. DENOMINACIÓN DE LA CONTRATACIÓN', '2. FINALIDAD PÚBLICA', '3. ANTECEDENTES', '4. OBJETIVOS DE LA CONTRATACIÓN', '4.1. Objetivo General', '4.2. Objetivo Especifico', '5. CARACTERISTICAS Y CONDICIONES DEL SERVICIO A CONTRATAR', '5.1. Descripción y cantidad del servicio a contratar', '5.2. Del procedimiento', '5.3. Seguros', '5.4. Prestaciones accesorias a la prestación principal', '5.4.1. Soporte', '5.4.2. Capacitación', '5.5. Lugar y plazo de prestación del servicio', '5.5.1. Lugar', '5.5.2. Plazo', '6. REQUISITOS Y RECURSOS DEL PROVEEDOR', '6.2. Requisitos de calificación del proveedor', '6.3. Recursos a ser provistos por el proveedor', '6.3.1. Entregables del servicio', '6.3.2. Personal clave', '7. OTRAS CONSIDERACIONES PARA LA EJECUCION DE LA PRESTACION', '7.1. Otras obligaciones', '7.1.1. Medidas de seguridad', '7.2. Confiabilidad', '7.3. Medidas de control durante la ejecución contractual', '7.4. Conformidad de la prestación', '7.4. Forma de pago', '7.5. Penalidades', '7.6. Responsabilidad de vicios ocultos', '8. ANEXOS', 'ANEXO A', 'ANEXO B', 'ANEXO C', 'ANEXO D', 'ANEXO N° E', 'ANEXO N° F', 'I. TERMINOS DE REFERENCIA', 'II. REQUISITOS DE CALIFICACION']

### Ejemplo 1: Periodo de Garantía
Pregunta: "De acuerdo a la sección 5. ¿Cuál es el periodo de garantía que debe tener toda la solución?"
Salida:
['5. CARACTERISTICAS Y CONDICIONES DEL SERVICIO A CONTRATAR', '5.1. Descripción y cantidad del servicio a contratar', '5.2. Del procedimiento', '5.3. Seguros', '5.4. Prestaciones accesorias a la prestación principal', '5.4.1. Soporte', '5.4.2. Capacitación', '5.5. Lugar y plazo de prestación del servicio', '5.5.1. Lugar', '5.5.2. Plazo']

### Ejemplo 2: Controles de Seguridad
Pregunta: "De acuerdo a la sección 4. ¿Qué tipo de controles debe implementar el servicio para garantizar la seguridad de la información?"
Salida:
['4. OBJETIVOS DE LA CONTRATACIÓN', '4.1. Objetivo General', '4.2. Objetivo Especifico']

### Ejemplo 3: Administración de Consola
Pregunta: "De acuerdo a la sección 5. ¿Quién será responsable de administrar la consola de los servicios de Infraestructura?"
Salida:
['5. CARACTERISTICAS Y CONDICIONES DEL SERVICIO A CONTRATAR', '5.1. Descripción y cantidad del servicio a contratar', '5.2. Del procedimiento', '5.3. Seguros', '5.4. Prestaciones accesorias a la prestación principal', '5.4.1. Soporte', '5.4.2. Capacitación', '5.5. Lugar y plazo de prestación del servicio', '5.5.1. Lugar', '5.5.2. Plazo']

### Ejemplo 4: Finalidad Pública
Pregunta: "¿Qué dice la sección 2 'Finalidad Publica'?"
Salida:
['2. FINALIDAD PÚBLICA']

### Ejemplo 5: Contenido de Sección Específica
Pregunta: "¿Qué dice la sección 6.3.1.?"
Salida:
['6.3. Recursos a ser provistos por el proveedor', '6.3.1. Entregables del servicio', '6.3.2. Personal clave']

## Instrucciones para el Análisis Actual

Tu tarea es analizar la pregunta actual siguiendo el mismo enfoque de los ejemplos anteriores:

1. Analizar el significado y contexto central de la pregunta.
2. Evaluar la relevancia de cada título de sección de manera integral.
3. Seleccionar los <k>{k}</k> títulos más relevantes basados en similitud semántica.
4. Producir una lista clasificada de títulos de secciones.

Requisitos:
- Devolver EXACTAMENTE <k>{k}</k> títulos de secciones.
- Basar las clasificaciones en relevancia semántica.
- Mantener el orden original si hay empate en relevancia.

Entrada:
Pregunta: <pregunta>{question}</pregunta>
Secciones Disponibles: <secciones>{section_titles}</secciones>

Proporciona tu salida como un array JSON de los <k>{k}</k> títulos más relevantes.
Si la respuesta no tiene <k>{k}</k> elementos se te considerará un mal analista. 
Asimismo si la respuesta tiene elementos repetidos la respuesta sera considerada como incorrecta.
"""


@retry_with_logging()
def classify_question_type(query: str) -> dict:

    response_content = claude_call(bedrock=bedrock_runtime, query=CLASSIFY_QUESTION_PROMPT.format(query=query), system_message="")
    response = response_content['content'][0]['text']
    question_type = response.split()[-1]

    assert question_type in ['general', 'comparativa'], f"Invalid question type: {question_type}"

    return {
        "response": response,
        "question_type": question_type
    }


@retry_with_logging()
def check_sections_in_question(query: str) -> dict:

    response_content = claude_call(bedrock=bedrock_runtime, query=CHECK_SECTIONS_PROMPT.format(query=query), system_message="")
    response = response_content['content'][0]['text']

    sections_in_question = remove_accents(response.split()[-1].lower())
    assert sections_in_question in ['si', 'no'], f"Invalid response: {sections_in_question}"

    return {
        "response": response,
        "sections_in_question": True if sections_in_question == 'si' else False
    }


class SortSectionTitles(BaseModel):
    key_list: list = Field(..., description="List of keys in the new order")


@retry_with_logging()
def get_topk_sections_from_question(question: str, keys_all: list, k: int = 3) -> list:
    prompt = SIMILARITY_PROMPT.format(
        question=question, 
        section_titles=json.dumps(keys_all, ensure_ascii=False), 
        k=k
    )

    messages = [
        {
            "role": "user",
            "content": [{"text": SYSTEM_SIMILARITY_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    tools = [convert_pydantic_to_bedrock_tool(SortSectionTitles, "Tool to reorder a list of section titles based on relevance to a question")]
    response = bedrock_runtime.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        inferenceConfig={
            'maxTokens': 4096,
            'temperature': 0,
            'topP': 1,
        },
        toolConfig={
            "tools": tools,
            "toolChoice": {
                "tool":{"name": "SortSectionTitles"},
            }
        },
    )
    topk_keys = response['output']['message']['content'][0]['toolUse']['input']['key_list']
    
    print(f"{keys_all=}")
    print(f"{topk_keys=}")

    try:
        if not isinstance(topk_keys, list) or not all(isinstance(key, str) for key in topk_keys):
            raise ValueError("Invalid response format from LLM.")
        if not set(topk_keys).issubset(set(keys_all)):
            raise ValueError("LLM returned unknown section titles.")
        # if len(topk_keys) < k:
        #     raise ValueError("LLM returned fewer section titles than requested.")
    except Exception as e:
        raise ValueError(f"Error parsing or validating LLM response: {e}")

    return list(dict.fromkeys(topk_keys))[:3]


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

        document_sections = [(f_path, v) for v in get_topk_sections_from_question(question, keys_all)]
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
        rag_sections = [s for _, s, _, _ in ranked_chunks if not (s in seen or seen.add(s))]
        print(f"{json.dumps(rag_sections, indent=2, ensure_ascii=False)}")

        retrieve_sections[f_path] = rag_sections

    return retrieve_sections


def retrieve_docs(file_path: str, sections: list) -> list:
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

    if has_sections:
        extracted_sections = retrieve_sections_from_question_similarity(question)
    else:
        extracted_sections = retrieve_sections_from_question_simple(question)

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
    p = PROMPT.format(context=context, query=question)

    return {
        'sections': all_sections,
        'context': context,
        'prompt': p,
        'system_prompt': SYSTEM_PROMPT,
        'sections_in_question': has_sections,
        'sections_in_response': sections_in_response['response'],
    }


def compare_documents_tool(question: str) -> str:
    print(f"{question=}")

    retrieve_sections = retrieve_sections_from_question_similarity(question)    

    sections_to_compare = list(set(retrieve_sections['../data/tdr_v4.pdf']).intersection(set(retrieve_sections['../data/tdr_v6.pdf'])))
    print(f"{sections_to_compare=}")
    # print(json.dumps(sections_to_compare, indent=2, ensure_ascii=False))

    context_list = []
    for f_path in ['../data/tdr_v4.pdf', '../data/tdr_v6.pdf']:
        retrieved_docs = retrieve_docs(f_path, sections_to_compare)
        context_doc = docs_to_context(f_path, retrieved_docs)
        context_list.append(context_doc)

    total_context = "\n\n".join(context_list)
    total_context = f"<TOTAL_CONTEXT>\n{total_context}\n</TOTAL_CONTEXT>"

    p = PROMPT.format(context=total_context, query=question)

    return {
        'sections': sections_to_compare,
        'context': total_context,
        'prompt': p,
        'system_prompt': SYSTEM_PROMPT
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
    try:
        records.append(handle_query(q_a))
    except Exception as e:
        print(f"Error: {e}")
print(f"{records=}")


if CALL_CLAUDE:
    df = pd.DataFrame([r for r in records if len(r)])
    display(df)

    # df['bleu'] = df['bleu_dict'].apply(lambda x: x['bleu'])
    # df['rouge1'] = df['rouge_dict'].apply(lambda x: x['rouge1'])
    # df['rouge2'] = df['rouge_dict'].apply(lambda x: x['rouge2'])
    # df['rougeL'] = df['rouge_dict'].apply(lambda x: x['rougeL'])
    # df['meteor'] = df['meteor_dict'].apply(lambda x: x['meteor'])
    df['bert_score'] = df['bert_score_dict'].apply(lambda x: np.mean(x['f1']) if x else None)

    metric_list = ['bert_score']    # 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor', 

    display(df[['question_type', 'sections_in_question', 'question', 'sections', 'answer', 'llm_response'] + metric_list].sort_values(by=metric_list, ascending=True))

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

