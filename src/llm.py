import botocore
import boto3
import json
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv('../.env')

# TODO: https://python.langchain.com/docs/how_to/structured_output/
# TODO: https://docs.helicone.ai/integrations/bedrock/python
# TODO: try https://python.useinstructor.com/integrations/bedrock/
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock.helicone.ai/v1/us-east-1"
)
# print(help(type(bedrock_runtime)))
bedrock_client = boto3.client(
    service_name='bedrock',
    region_name='us-east-1',
)
# print(f"{type(bedrock_client)=}")

"""
'anthropic.claude-3-5-sonnet-20240620-v1:0'
'anthropic.claude-3-5-haiku-20241022-v1:0'
'anthropic.claude-3-7-sonnet-20250219-v1:0'
"""
# model_id = 'meta.llama3-2-3b-instruct-v1:0'
# print(json.dumps(bedrock_client.get_foundation_model(modelIdentifier=model_id), indent=2))

bedrock_langchain = init_chat_model(
    "us.anthropic.claude-3-5-haiku-20241022-v1:0", 
    model_provider="bedrock_converse", 
    region_name='us-east-1',
)

'''
# https://python.langchain.com/docs/how_to/streaming/
chunks = []
for chunk in bedrock_langchain.stream([
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is"},
], temperature=0, max_tokens=550, performanceConfig={"latency": "optimized"}):
    chunks.append(chunk)
    print(chunk.content, end="\n", flush=True)
'''

event_system = bedrock_runtime.meta.events

def process_custom_arguments(params, context, **kwargs):
    if (custom_headers := params.pop("custom_headers", None)):
        context["custom_headers"] = custom_headers


def add_custom_header_before_call(model, params, request_signer, **kwargs):
    params['headers']['Helicone-Auth'] = f'Bearer {os.getenv("HELICONE_API_KEY")}'
    params['headers']['aws-access-key'] = os.getenv("AWS_ACCESS_KEY_ID")
    params['headers']['aws-secret-key'] = os.getenv("AWS_SECRET_ACCESS_KEY")
    # optionally, you can pass the aws-session-token instead of access and secret key if you are using temporary credentials
    params['headers']['aws-session-token'] = os.getenv("AWS_SESSION_TOKEN")
    if (custom_headers := params.pop("custom_headers", None)):
        params['headers'].update(custom_headers)
    headers = params['headers']
    # print(f'param headers: {headers}')


event_system.register("before-parameter-build.bedrock-runtime.InvokeModel",
                      process_custom_arguments)
event_system.register('before-call.bedrock-runtime.InvokeModel',
                      add_custom_header_before_call)


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------

def embed_body(chunk_message: str):
    return json.dumps({
        'inputText': chunk_message,

    })

# Llamada al modelo de embedding
def embed_call(bedrock: boto3.client, chunk_message: str, model_id: str = "amazon.titan-embed-text-v2:0"):
    body = embed_body(chunk_message)

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        contentType='application/json',
        accept='application/json'
    )

    return json.loads(response['body'].read().decode('utf-8'))


# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------

def claude_body(system_prompt: str, query: str):
    query = [{
        "role": "user",
        "content": query
    }]

    return json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4090,
        "system": system_prompt,
        "messages": query,
        "temperature": 0.0,
        # TODO: research "top_p" and "top_k"
        # "top_k": 0,
    })

# Llamada al LLM


# https://github.com/amberpe/poc-rag-multidocs/blob/main/prompt.py
# https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
def claude_call(bedrock: botocore.client,
                system_message: str,
                query: str,
                model_id='anthropic.claude-3-5-sonnet-20240620-v1:0'):

    body = claude_body(system_message, query=query)

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        contentType='application/json',
        accept='application/json'
    )

    return json.loads(response['body'].read().decode('utf-8'))


'''
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model_with_response_stream.html
# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-invoke.html#inference-examples-stream
def claude_call_stream(bedrock: botocore.client,
                        system_message: str,
                        query: str,
                        model_id='us.anthropic.claude-3-5-sonnet-20240620-v1:0'):
    
    body = json.dumps({
        'prompt': '\n\nHuman: write an essay for living on mars in 1000 words\n\nAssistant:',
        'max_tokens_to_sample': 4000
    })
      

    response = bedrock.invoke_model_with_response_stream(
        body=body,
        modelId=model_id,
    )

    stream = response['body']
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                print(json.loads(chunk.get('bytes').decode()))

claude_call_stream(
    bedrock_runtime, 
    "You are a helpful assistant that translates English to French. Translate the user sentence.", 
    "I love programming."
)
'''


# # -----------------------------------------------------------------------------
# # LangChain
# # -----------------------------------------------------------------------------

# # !python3 -m pip install -qU langchain-aws

# from langchain_aws import ChatBedrock
# from dotenv import load_dotenv
# import os
# load_dotenv('../.env')

# llm = ChatBedrock(
#     model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
#     model_kwargs=dict(temperature=0),
#     region_name=os.getenv("AWS_REGION", "us-east-1"),  # Specify the AWS region
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),  # Load AWS access key
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  # Load AWS secret key
# )

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg
