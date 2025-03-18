from sqlalchemy import select
from llm import bedrock_runtime, embed_call, claude_call
import json
from db import *

# https://github.com/amberpe/poc-rag-multidocs/blob/main/RAG.py#L218

def search_similar_text(query, top_k=5) -> list[tuple]:
    query_embedding = embed_call(bedrock_runtime, query)
    return Session.execute(
        select(Embedding.document_name, Embedding.page_index, Embedding.page_content)
        .order_by(
            Embedding.embedding.cosine_distance(query_embedding['embedding'])
        ).limit(top_k)
    )


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

# Prompt Improver: https://console.anthropic.com/dashboard
questions = [
# Chunk question
    'g.\nEl servicio debe permitir asociar una o más direcciones IP elásticas a cualquier\ninstancia\nde\nla\nnube\nprivada\nvirtual,\nde modo que puedan alcanzarse\ndirectamente desde Internet.\nh.\nEl servicio debe permitir conectarse a la nube privada virtual con otras nubes\nprivadas virtuales y obtener acceso a los recursos de otras nubes privadas\nvirtuales a través de direcciones IP privadas mediante la interconexión de nube\nprivada virtual.\ni.\nEl servicio debe permitir conectarse de manera privada a los servicios del\nfabricante de la nube pública sin usar una gateway de Internet, ni una NAT ni\nun proxy de firewall mediante un punto de enlace de la nube privada virtual.\nj.\nEl servicio debe permitir conectar la nube privada virtual y la infraestructura de\nTI local con la VPN del fabricante de la nube pública de sitio a sitio.\nk.\nEl servicio debe permitir asociar grupos de seguridad de la nube privada virtual\ncon instancias en la plataforma.\nl.',
# Cross document question
    '¿Cual es la finalidad publica de los documentos?',
    '¿Cuales son los objetivos generales y especificos de la contratacion en los documentos?',
]

system_prompt = """
Eres un asistente de IA especializado en responder preguntas basadas en contextos proporcionados. Tu tarea es analizar el contexto dado, entender la pregunta y proporcionar una respuesta precisa y relevante en español.
"""

prompt = """
Primero, te presentaré el contexto sobre el cual se basará la pregunta:

{context}

Ahora, te haré una pregunta relacionada con este contexto. Tu objetivo es responder a esta pregunta utilizando únicamente la información proporcionada en el contexto anterior.

<pregunta>
{query}
</pregunta>

Para asegurar una respuesta precisa y bien fundamentada, sigue estos pasos:

1. Analiza cuidadosamente el contexto proporcionado.
2. Extrae y cita la información relevante del contexto para responder a la pregunta.
3. Considera las posibles interpretaciones de la pregunta.
4. Evalúa la fuerza de la evidencia para cada posible respuesta.
5. Escribe tu razonamiento dentro de las etiquetas <analisis> para explicar tu proceso de pensamiento y cómo llegaste a tu respuesta.
6. Proporciona tu respuesta final dentro de las etiquetas <respuesta>.

Recuerda:
- Utiliza solo la información proporcionada en el contexto.
- Si la información necesaria para responder la pregunta no está en el contexto, indica que no puedes responder basándote en la información disponible.
- Mantén tu respuesta clara, concisa y directamente relacionada con la pregunta.

Comienza tu proceso de razonamiento y respuesta ahora.
"""

for query in questions:
    query_normalized = query.replace('¿', '').replace('?', '').replace(' ', '_').lower()
    query_normalized = ''.join(e for e in query_normalized if e.isalnum() or e == '_')[:50]
    print(f"{query_normalized=}")

    retrieved_docs = search_similar_text(query, 5)

    context_format = """<contexto>\n{entries}\n</contexto>""".strip()
    
    entries = "\n".join([
        f"    <{os.path.splitext(os.path.basename(doc))[0]}>\n        {pg_cntnt}\n        <pagina>{pg_indx}</pagina>\n    </{os.path.splitext(os.path.basename(doc))[0]}>"
        for doc, pg_indx, pg_cntnt in retrieved_docs
    ])
    
    context = context_format.format(entries=entries)

    # print(context)
    with open(f'../data/prompt-{query_normalized}.txt', 'w') as f:
        f.write(prompt.format(context=context, query=query))
    # print(prompt)


response = claude_call(bedrock_runtime, system_prompt, prompt)

print(json.dumps(response['content'], indent=2, ensure_ascii=False))

with open('../data/response.json', 'w') as f:
    f.write(json.dumps(response, indent=2, ensure_ascii=False))

# return prompt


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