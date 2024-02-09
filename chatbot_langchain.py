# Databricks notebook source
# %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" llama-index==0.9.40 mlflow==2.9.0 protobuf==3.20.0 openai==1.10.0 langchain-openai langchain torch torchvision torchaudio FlagEmbedding cloudpickle pydantic databricks-sdk databricks-vectorsearch
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 protobuf==3.20.0 openai==1.10.0 cloudpickle pydantic databricks-sdk databricks-vectorsearch mlflow[databricks] protobuf==3.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from openai import AzureOpenAI

# COMMAND ----------

# #Open AI Client
# from openai import AzureOpenAI
# import os

# os.environ["AZURE_OPENAI_API_KEY"] = dbutils.secrets.get(scope='dev_demo', key='azure_openai_api_key')
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://nous-ue2-openai-sbx-openai.openai.azure.com/"

# az_openai_client = AzureOpenAI(
#     api_key = dbutils.secrets.get(scope='dev_demo', key='azure_openai_api_key'),
#     api_version = "2023-05-15",
#     azure_endpoint = "https://nous-ue2-openai-sbx-openai.openai.azure.com/",
#     )

# COMMAND ----------

# DBTITLE 1,Ada Vector Search Client
# from databricks.vector_search.client import VectorSearchClient
# vsc_ada = VectorSearchClient(disable_notice=True)
# vs_index_fullname_ada = "demo.hackathon.ada_self_managed_index"
# endpoint_name_ada = "ada_vector_search"

# COMMAND ----------

import mlflow

# COMMAND ----------

# DBTITLE 1,BGE Vector Search Client
from databricks.vector_search.client import VectorSearchClient
import os

vsc_bge = VectorSearchClient(disable_notice=True)
vs_index_fullname_bge = "demo.hackathon.bge_self_managed_index"
endpoint_name_bge = "bge_vector_search"

# COMMAND ----------

# DBTITLE 1,Filter query for State values
# import mlflow.deployments
# import ast

# def get_state_from_query(query):
#     client = mlflow.deployments.get_deploy_client("databricks")
#     inputs = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": f"""
#                 You determine if there are any US states present in this text: {query}.
#                 Your response should be JSON like the following:
#                 {{ 
#                     "state": []
#                 }}

#                 """
#             }
#         ],
#         "max_tokens": 64,
#         "temperature": 0
#     }

#     response = client.predict(endpoint="databricks-mixtral-8x7b-instruct", inputs=inputs)
#     response_content = response["choices"][0]['message']['content']
#     cleaned_response = response_content.replace("```json", "")
#     cleaned_response = cleaned_response.replace("```", "")
#     filters = ast.literal_eval(cleaned_response)

#     return filters

# COMMAND ----------

# from pprint import pprint

# query = f"What does the first section of the Utah Privacy Act say?"
# # What is considered biometric data?
# # What rights can consumers exercise?

# filters = get_state_from_query(query)
# print(filters)

# COMMAND ----------

# DBTITLE 1,Get Ada Embeddings
# def open_ai_embeddings(contents):
#     embed_model = "nous-ue2-openai-sbx-base-deploy-text-embedding-ada-002"

#     response = az_openai_client.embeddings.create(
#         input = contents,
#         model = embed_model
#     )

#     return response.data[0].embedding

# COMMAND ----------

# DBTITLE 1,Search ADA Embeddings
# # ADA embedding search
# def ada_search(query, filters):
#     if filters["state"] != []:
#         results_ada = vsc_ada.get_index(endpoint_name_ada, vs_index_fullname_ada).similarity_search(
#             query_vector = open_ai_embeddings(query),
#             columns=["id","state", "url", "content"],
#             filters=filters,
#             num_results=10)
#         docs_ada = results_ada.get('result', {}).get('data_array', [])
#         # print(docs_ada)
#         return docs_ada
#     else:
#         results_ada = vsc_ada.get_index(endpoint_name_ada, vs_index_fullname_ada).similarity_search(
#             query_vector = open_ai_embeddings(query),
#             columns=["id","state", "url", "content"],
#             num_results=10)
#         docs_ada = results_ada.get('result', {}).get('data_array', [])
#         # print(docs_ada)
#         return docs_ada

# COMMAND ----------

# DBTITLE 1,Get BGE Embeddings
# # Ad-hoc BGE embedding function
# import mlflow.deployments
# bge_deploy_client = mlflow.deployments.get_deploy_client("databricks")

# def get_bge_embeddings(query):
#     #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
#     response = bge_deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": query})
#     #return [e['embedding'] for e in response.data]
#     return response.data[0]['embedding']

# COMMAND ----------

# DBTITLE 1,Search BGE Embeddings
# # BGE embedding search
# def bge_search(query, filters):
#     if filters["state"] != []:
#         results_bge = vsc_bge.get_index(endpoint_name_bge, vs_index_fullname_bge).similarity_search(
#             query_vector = get_bge_embeddings(query),
#             columns=["id","state", "url", "content"],
#             filters=filters,
#             num_results=10)
#         docs_bge = results_bge.get('result', {}).get('data_array', [])
#         #pprint(docs_bge)
#         return docs_bge
#     else:
#         results_bge = vsc_bge.get_index(endpoint_name_bge, vs_index_fullname_bge).similarity_search(
#             query_vector = get_bge_embeddings(query),
#             columns=["id","state", "url", "content"],
#             num_results=10)
#         docs_bge = results_bge.get('result', {}).get('data_array', [])
#         #pprint(docs_bge)
#         return docs_bge

# COMMAND ----------

# DBTITLE 1,Rework of BGE Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA

# import mlflow.deployments
# bge_deploy_client = mlflow.deployments.get_deploy_client("databricks")
#vsc_bge = VectorSearchClient(disable_notice=True)
vs_index_fullname_bge = "demo.hackathon.bge_self_managed_index"
endpoint_name_bge = "bge_vector_search"

embedding_model_bge = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# def get_bge_embeddings(query):
#     #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
#     response = bge_deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": query})
#     #return [e['embedding'] for e in response.data]
#     return response.data[0]['embedding']

def get_bge_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc_bge = VectorSearchClient(workspace_url=host, disable_notice=True) #, personal_access_token=os.environ["DATABRICKS_TOKEN"]
    vs_index = vsc_bge.get_index(
        endpoint_name=endpoint_name_bge,
        index_name=vs_index_fullname_bge
    )
    
    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, 
        text_column="content", 
        embedding=embedding_model_bge, 
        columns=["id","state", "url", "content"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 5})

# COMMAND ----------

# DBTITLE 1,Use the BGE Retriever
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

retriever = get_bge_retriever()

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)


# COMMAND ----------

print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What rights do Utah customers have in their state's Privacy Act?"}]}))

# COMMAND ----------

# unique_states = spark.sql("SELECT DISTINCT SUBSTRING_INDEX(SUBSTRING_INDEX(path, '/', 6), '/', -1) as states FROM demo.hackathon.pdf_raw").collect()
# state_list = [row.states for row in unique_states]
# state_list

# COMMAND ----------

# DBTITLE 1,Limit model responses to things it knows about
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200) #tried mixtral, it did not these complex follow instructions well. It always gave more verbose answers than were necessary.

is_question_about_databricks_str = """
You are classifying documents to know if this question is related to Privacy Acts (these are legal documents created by States within the United States of America) or something from a very different field. You only know about documents from these States: 'Utah', 'Oregon', 'Texas', 'Connecticut', 'Delaware', 'Montana', 'Virginia', 'New Jersey', 'Iowa', 'Tennessee', 'Indiana', 'California','Colorado'.

Here are some examples:

Question: Knowing this followup history: What does California say about consumer's rights in the California Privacy Act?, classify this question: What sections of the California Privacy Act has that information?
Expected Response: Yes

Question: Knowing this followup history: Does Utah have a Privacy Act?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_databricks_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_databricks_str
)

is_about_databricks_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | chat_model
    | StrOutputParser()
)


# COMMAND ----------

#Returns "Yes" as this is about Privacy Act docs: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}, 
        {"role": "assistant", "content": "Consumers have the right to request, view, delete and modify their private data information."}, 
        {"role": "user", "content": "What steps should a consumer take to request their data?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about Privacy Act docs
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about Privacy Act docs
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "Do you have Privacy Act documents for Alaska?"}
    ]
}))

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant legal documents so that we can better answer the question. The query should be in legal language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query, do not add explanation. Again, it is important NOT to add explanation, only respond with your new query.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

# COMMAND ----------

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}, 
        {"role": "assistant", "content": "Consumers have the right to request, view, delete and modify their private data information."}, 
        {"role": "user", "content": "What is the definition of a consumer in the California Privacy Act?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# # https://github.com/langchain-ai/langchain/issues/13076

# from __future__ import annotations
# from typing import Dict, Optional, Sequence
# from langchain.schema import Document
# from langchain.pydantic_v1 import Extra, root_validator

# from langchain.callbacks.manager import Callbacks
# from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

# from sentence_transformers import CrossEncoder

# from FlagEmbedding import FlagReranker
# # Load model directly
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
# # model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

# # reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # 

# class BgeRerank(BaseDocumentCompressor):
#     tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
#     model_name:str = "BAAI/bge-reranker-large"
#     """Model name to use for reranking."""    
#     top_n: int = 10   
#     """Number of documents to return."""
#     model:CrossEncoder = CrossEncoder(model_name)
#     """CrossEncoder instance to use for reranking."""

#     # def reranker(query, docs):
#     #     tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
#     #     model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

#     #     reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

#     #     query_and_docs = [[query, d[1]] for d in docs]

#     #     scores = reranker_model.compute_score(query_and_docs)

#     #     reranked_docs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

#     #     return reranked_docs

#     def bge_rerank(self,query,docs):
#         model_inputs =  [[query, doc] for doc in docs]
#         scores = self.model.predict(model_inputs)
#         results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
#         return results[:self.top_n]


#     class Config:
#         """Configuration for this pydantic object."""

#         extra = Extra.forbid
#         arbitrary_types_allowed = True

#     def compress_documents(
#         self,
#         documents: Sequence[Document],
#         query: str,
#         callbacks: Optional[Callbacks] = None,
#     ) -> Sequence[Document]:
#         """
#         Compress documents using BAAI/bge-reranker models.

#         Args:
#             documents: A sequence of documents to compress.
#             query: The query to use for compressing the documents.
#             callbacks: Callbacks to run during the compression process.

#         Returns:
#             A sequence of compressed documents.
#         """
#         if len(documents) == 0:  # to avoid empty api call
#             return []
#         doc_list = list(documents)
#         _docs = [d.page_content for d in doc_list]
#         results = self.bge_rerank(query, _docs)
#         final_results = []
#         for r in results:
#             doc = doc_list[r[0]]
#             doc.metadata["relevance_score"] = r[1]
#             final_results.append(doc)
#         return final_results

# COMMAND ----------

# # DOESNT WORK
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import BgeRerank
# from langchain_community.chat_models import ChatDatabricks

# llm = ChatDatabricks(target_uri="databricks",
#                      endpoint="databricks-mixtral-8x7b-instruct",
#                      temperature=0.8)

# compressor = BgeRerank()
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )

# compressed_docs = compression_retriever.get_relevant_documents(
#     "What are consumer's rights in the California Privacy Act?"
# )
# pretty_print_docs(compressed_docs)

# COMMAND ----------

# DBTITLE 1,BGE-reranker
# from FlagEmbedding import FlagReranker
# # Load model directly
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# def reranker(query, docs):
#     tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
#     model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

#     reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

#     query_and_docs = [[query, d[1]] for d in docs]

#     scores = reranker_model.compute_score(query_and_docs)

#     reranked_docs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

#     return reranked_docs

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant legal documents so that we can better answer the question. The query should be in legal language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query, do not add explanation. Again, it is important NOT to add explanation, only respond with your new query. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I can only answer questions about Privacy Act Documents from these states: Utah, Oregon, Texas, Connecticut, Delaware, Montana, Virginia, New Jersey, Iowa, Tennessee, Indiana, California, Colorado.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_databricks_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

# DBTITLE 1,Asking an out-of-scope question
import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}, 
        {"role": "assistant", "content": "Consumers have the right to request, view, delete and modify their private data information."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
print(non_relevant_dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}, 
        {"role": "assistant", "content": "Consumers have the right to request, view, delete and modify their private data information."}, 
        {"role": "user", "content": "How would a California resident request their private data from a company?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
print(dialog["messages"], response)

# COMMAND ----------

# DBTITLE 1,Log LangChain model as MLflow artifact for current run.
import cloudpickle
import langchain
from mlflow.models import infer_signature
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
model_name = f"demo.hackathon.privacy_act_chatbot_model_v0"

with mlflow.start_run(run_name="privacy_chatbot_runs") as run:
    #Get our model signature from input/output
    input_df = pd.DataFrame({"messages": [dialog]})
    output = full_chain.invoke(dialog)
    signature = infer_signature(input_df, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn= get_bge_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
            #"openai=="+ openai.__version__,
            #"databricks-sdk=="+ databricks-sdk.__version__
        ],
        input_example=input_df,
        signature=signature
    )

# COMMAND ----------

# docs_ada = ada_search(query, filters)
# print(docs_ada)

# COMMAND ----------

# docs_bge = bge_search(query, filters)
# print(docs_bge)

# COMMAND ----------

# def combine_search_results(docs_bge, docs_ada):
#     docs = docs_bge + docs_ada
#     dedup_docs = list(set(tuple(i) for i in docs))
#     combined_docs = [list(i) for i in dedup_docs]

#     #print(combined_docs) # used to be named "final_list"
#     return combined_docs

# COMMAND ----------

# combined_docs = combine_search_results(docs_bge, docs_ada)
# print(len(combined_docs))

# COMMAND ----------

# DBTITLE 1,Reranking with bge-reranker-large
# from FlagEmbedding import FlagReranker
# # Load model directly
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# def reranker(query, docs):
#     tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
#     model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

#     reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

#     query_and_docs = [[query, d[1]] for d in docs]

#     scores = reranker_model.compute_score(query_and_docs)

#     reranked_docs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

#     return reranked_docs

# COMMAND ----------

# reranked_docs = reranker(query, combined_docs)
# pprint(reranked_docs)

# COMMAND ----------

# def mixtral_query(userquery, reranked_docs):
#     client = mlflow.deployments.get_deploy_client("databricks")
#     inputs = {
#         "messages": [{"role":"user", "content":f"Summarize this result: {reranked_docs[0][0][3]}"}],
#         "max_tokens": 1500,
#         "temperature": 0.8
#     }

#     response = client.predict(endpoint="databricks-mixtral-8x7b-instruct", inputs=inputs)
#     result = response["choices"][0]['message']['content']
#     result_with_metadata = f"{result}\n\nDocument from State: {reranked_docs[0][0][1]} \nResult id: {reranked_docs[0][0][0]} \nDocument path: {reranked_docs[0][0][2]}"
#     return result_with_metadata.strip()

# COMMAND ----------

# result = mixtral_query(query, reranked_docs)
# print(result)
