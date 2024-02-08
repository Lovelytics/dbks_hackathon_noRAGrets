# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" llama-index==0.9.40 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0 protobuf==3.20.0 openai==1.10.0 langchain-openai langchain torch torchvision torchaudio FlagEmbedding
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Ada Vector Search Client
from databricks.vector_search.client import VectorSearchClient
vsc_ada = VectorSearchClient(disable_notice=True)
vs_index_fullname_ada = "demo.hackathon.ada_self_managed_index"
endpoint_name_ada = "ada_vector_search"

# COMMAND ----------

# DBTITLE 1,BGE Vector Search Client
vsc_bge = VectorSearchClient(disable_notice=True)
vs_index_fullname_bge = "demo.hackathon.bge_self_managed_index"
endpoint_name_bge = "bge_vector_search"

# COMMAND ----------

# DBTITLE 1,Filter query for State values
import mlflow.deployments
import ast

def get_state_from_query(query):
    client = mlflow.deployments.get_deploy_client("databricks")
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": f"""
                You determine if there are any US states present in this text: {query}.
                Your response should be JSON like the following:
                {{ 
                    "state": []
                }}

                """
            }
        ],
        "max_tokens": 64,
        "temperature": 0
    }

    response = client.predict(endpoint="databricks-mixtral-8x7b-instruct", inputs=inputs)
    response_content = response["choices"][0]['message']['content']
    cleaned_response = response_content.replace("```json", "")
    cleaned_response = cleaned_response.replace("```", "")
    filters = ast.literal_eval(cleaned_response)

    return filters

# COMMAND ----------

from pprint import pprint

query = f"What does the first section of the Utah Privacy Act say?"
# What is considered biometric data?
# What rights can consumers exercise?

state_filters = get_state_from_query(query)
print(state_filters)

# COMMAND ----------

# DBTITLE 1,Get Ada Embeddings
def open_ai_embeddings(contents):
    embed_model = "nous-ue2-openai-sbx-base-deploy-text-embedding-ada-002"

    response = client.embeddings.create(
        input = contents,
        model = embed_model
    )

    return response.data[0].embedding

# COMMAND ----------

# DBTITLE 1,Search ADA Embeddings
# ADA embedding search
def ada_search(query, filters):
    if filters["state"] != []:
        results_ada = vsc_ada.get_index(endpoint_name_ada, vs_index_fullname_ada).similarity_search(
            query_vector = open_ai_embeddings(query),
            columns=["id","state", "url", "content"],
            filters=filters,
            num_results=10)
        docs_ada = results_ada.get('result', {}).get('data_array', [])
        # print(docs_ada)
        return docs_ada
    else:
        results_ada = vsc_ada.get_index(endpoint_name_ada, vs_index_fullname_ada).similarity_search(
            query_vector = open_ai_embeddings(query),
            columns=["id","state", "url", "content"],
            num_results=10)
        docs_ada = results_ada.get('result', {}).get('data_array', [])
        # print(docs_ada)
        return docs_ada

# COMMAND ----------

# DBTITLE 1,Get BGE Embeddings
# Ad-hoc BGE embedding function
import mlflow.deployments
bge_deploy_client = mlflow.deployments.get_deploy_client("databricks")

def get_bge_embeddings(query):
    #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
    response = bge_deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": query})
    #return [e['embedding'] for e in response.data]
    return response.data[0]['embedding']

# COMMAND ----------

# DBTITLE 1,Search BGE Embeddings
# BGE embedding search
def bge_search(query, filters):
    if filters["state"] != []:
        results_bge = vsc_bge.get_index(endpoint_name_bge, vs_index_fullname_bge).similarity_search(
            query_vector = get_bge_embeddings(query),
            columns=["id","state", "url", "content"],
            filters=filters,
            num_results=10)
        docs_bge = results_bge.get('result', {}).get('data_array', [])
        #pprint(docs_bge)
        return docs_bge
    else:
        results_bge = vsc_bge.get_index(endpoint_name_bge, vs_index_fullname_bge).similarity_search(
            query_vector = get_bge_embeddings(query),
            columns=["id","state", "url", "content"],
            num_results=10)
        docs_bge = results_bge.get('result', {}).get('data_array', [])
        #pprint(docs_bge)
        return docs_bge

# COMMAND ----------

def combine_search_results(docs_bge, docs_ada):
    docs = docs_bge + docs_ada
    dedup_docs = list(set(tuple(i) for i in docs))
    combined_docs = [list(i) for i in dedup_docs]

    #print(combined_docs) # used to be named "final_list"
    return combined_docs

# COMMAND ----------

# DBTITLE 1,Reranking with bge-reranker-large
    # Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import FlagReranker
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

def reranker(docs_to_rerank):
    rerank_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
    query_and_docs = [[query, d[1]] for d in docs_to_rerank]
    scores = rerank_model.compute_score(query_and_docs)
    reranked_docs = sorted(list(zip(docs_to_rerank, scores)), key=lambda x: x[1], reverse=True)
    #print(reranked_docs)
    return reranked_docs
