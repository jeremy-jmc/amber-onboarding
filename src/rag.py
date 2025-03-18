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


"""

questions = [
# Chunk question
    'g.\nEl servicio debe permitir asociar una o más direcciones IP elásticas a cualquier\ninstancia\nde\nla\nnube\nprivada\nvirtual,\nde modo que puedan alcanzarse\ndirectamente desde Internet.\nh.\nEl servicio debe permitir conectarse a la nube privada virtual con otras nubes\nprivadas virtuales y obtener acceso a los recursos de otras nubes privadas\nvirtuales a través de direcciones IP privadas mediante la interconexión de nube\nprivada virtual.\ni.\nEl servicio debe permitir conectarse de manera privada a los servicios del\nfabricante de la nube pública sin usar una gateway de Internet, ni una NAT ni\nun proxy de firewall mediante un punto de enlace de la nube privada virtual.\nj.\nEl servicio debe permitir conectar la nube privada virtual y la infraestructura de\nTI local con la VPN del fabricante de la nube pública de sitio a sitio.\nk.\nEl servicio debe permitir asociar grupos de seguridad de la nube privada virtual\ncon instancias en la plataforma.\nl.',
# Cross document question
    '¿Cual es la finalidad publica de los documentos?',
    '¿Cuales son los objetivos generales y especificos de la contratacion en los documentos?',
]

for query in questions:
    query_normalized = query.replace('¿', '').replace('?', '').replace(' ', '_').lower()
    query_normalized = ''.join(e for e in query_normalized if e.isalnum() or e == '_')[:50]
    print(f"{query_normalized=}")

    retrieved_docs = search_similar_text(query, 5)

    context_format = [f"Document: {doc} - Page ({pg_indx}):\n```\n{pg_cntnt}\n```" for doc, pg_indx, pg_cntnt in retrieved_docs]

    context = '\n'.join(context_format)
    # print(context)


    prompt = f"""
Dado el contexto:
{context}
Responde la siguiente pregunta:
- {query}
    """

    with open(f'../data/prompt-{query_normalized}.txt', 'w') as f:
        f.write(prompt)
    # print(prompt)


response = claude_call(bedrock_runtime, "", prompt)

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