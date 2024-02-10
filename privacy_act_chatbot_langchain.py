# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 protobuf==3.20.0 openai==1.10.0 cloudpickle pydantic databricks-sdk databricks-vectorsearch mlflow[databricks] protobuf==3.20.0 langchain==0.1.0 langchain_openai langchain-community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#Open AI Client
from openai import AzureOpenAI
import os
import mlflow

os.environ["AZURE_OPENAI_API_KEY"] = dbutils.secrets.get(scope='dev_demo', key='azure_openai_api_key')
os.environ["AZURE_OPENAI_ENDPOINT"] = dbutils.secrets.get(scope='dev_demo', key='azure_openai_endpoint')

az_openai_client = AzureOpenAI(
    api_key = os.environ["AZURE_OPENAI_API_KEY"],
    api_version = "2023-05-15",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
    )

# COMMAND ----------

# DBTITLE 1,Test Ada Embeddings
def open_ai_embeddings(contents):
    embed_model = "nous-ue2-openai-sbx-base-deploy-text-embedding-ada-002"

    response = az_openai_client.embeddings.create(
        input = contents,
        model = embed_model
    )

    return response.data[0].embedding

print(open_ai_embeddings("Test of embeddings"))

# COMMAND ----------

# DBTITLE 1,Langchain Azure Open AI embeddings
from langchain_openai import AzureOpenAIEmbeddings

lc_az_openai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="nous-ue2-openai-sbx-base-deploy-text-embedding-ada-002",
    openai_api_version="2023-05-15",
)

# COMMAND ----------

# DBTITLE 1,Ada Vector Search Client
from databricks.vector_search.client import VectorSearchClient
vsc_ada = VectorSearchClient(disable_notice=True)
vs_index_fullname_ada = "demo.hackathon.ada_self_managed_index"
endpoint_name_ada = "ada_vector_search"

# COMMAND ----------

# DBTITLE 1,BGE Vector Search Client
from databricks.vector_search.client import VectorSearchClient
import os

vsc_bge = VectorSearchClient(disable_notice=True)
vs_index_fullname_bge = "demo.hackathon.bge_self_managed_index"
endpoint_name_bge = "bge_vector_search"

# COMMAND ----------

# DBTITLE 1,Define Ada Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from operator import itemgetter

lc_az_openai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="nous-ue2-openai-sbx-base-deploy-text-embedding-ada-002",
    openai_api_version="2023-05-15",
)

vs_index_fullname_ada = "demo.hackathon.ada_self_managed_index"
endpoint_name_ada = "ada_vector_search"

embedding_model_ada = lc_az_openai_embeddings
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

def get_ada_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc_ada = VectorSearchClient(workspace_url=host, disable_notice=True) #, personal_access_token=os.environ["DATABRICKS_TOKEN"]
    vs_index = vsc_ada.get_index(
        endpoint_name=endpoint_name_ada,
        index_name=vs_index_fullname_ada
    )
    
    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, 
        text_column="content", 
        embedding=embedding_model_ada, 
        columns=["id","state", "url", "content"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 5})

# COMMAND ----------

# DBTITLE 1,Use Ada Retriever in a function
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

ada_retriever = get_ada_retriever()

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

ada_retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | ada_retriever
)


# COMMAND ----------

# DBTITLE 1,Test Ada Retriever
print(ada_retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What rights do Utah customers have in their state's Privacy Act?"}]}))

# COMMAND ----------

# DBTITLE 1,Define BGE Retriever
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

# DBTITLE 1,Use the BGE Retriever in a function
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

bge_retriever = get_bge_retriever()

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | bge_retriever
)


# COMMAND ----------

# DBTITLE 1,Test BGE retriever
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What rights do Utah customers have in their state's Privacy Act?"}]}))

# COMMAND ----------

# DBTITLE 1,LOTR - Merge Retrievers
# https://python.langchain.com/docs/integrations/retrievers/merger_retriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
     EmbeddingsClusteringFilter,
     EmbeddingsRedundantFilter
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter

filter_embeddings = lc_az_openai_embeddings
lotr = MergerRetriever(retrievers=[ada_retriever, bge_retriever])

# We can remove redundant results from both retrievers using yet another embedding.
# Using multiples embeddings in diff steps could help reduce biases.
# filter = EmbeddingsRedundantFilter(
#     embeddings=filter_embeddings,
#     num_clusters=10,
#     num_closest=1,
#     sorted=True)

splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
relevant_filter = EmbeddingsFilter(embeddings=filter_embeddings,
                                   similarity_threshold=0.62)
pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, redundant_filter, relevant_filter])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=lotr
)

# COMMAND ----------

# DBTITLE 1,Pretty print function - thanks LangChain!
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# COMMAND ----------

compressed_docs = compression_retriever.get_relevant_documents("What rights do Utah customers have in their state's Privacy Act?")
pretty_print_docs(compressed_docs)

# COMMAND ----------

# DBTITLE 1,Test LOTR - Multiple Retrievers
retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | compression_retriever
)
pretty_print_docs(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What rights do Utah customers have in their state's Privacy Act?"}]}))

# COMMAND ----------

# DBTITLE 1,Limit model scope
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 4000) #tried mixtral, it did not these complex follow instructions well. It always gave more verbose answers than were necessary.

parser_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 2500) #tried mixtral, it did not these complex follow instructions well. It always gave more verbose answers than were necessary.

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

is_about_privacy_act_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | parser_model
    | StrOutputParser()
)


# COMMAND ----------

# DBTITLE 1,Ask in-scope question
#Returns "Yes" as this is about Privacy Act docs: 
print(is_about_privacy_act_chain.invoke({
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}, 
        {"role": "assistant", "content": "Consumers have the right to request, view, delete and modify their private data information."}, 
        {"role": "user", "content": "What steps should a consumer take to request their data?"}
    ]
}))

# COMMAND ----------

# DBTITLE 1,Ask out-of-scope question
#Return "no" as this isn't about Privacy Act docs
print(is_about_privacy_act_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

# DBTITLE 1,Ask about a out-of-scope State
#Return "no" as this isn't about Privacy Act docs
print(is_about_privacy_act_chain.invoke({
    "messages": [
        {"role": "user", "content": "Do you have Privacy Act documents for Alaska?"}
    ]
}))

# COMMAND ----------

# DBTITLE 1,Generate query to use with vector store
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
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | parser_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

# COMMAND ----------

# DBTITLE 1,Test - Enhancing the original query
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

# DBTITLE 1,Define a full chain
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
If a user has some chat history, it will be provided here as a Discussion.

Discussion: {chat_history}

Here's some context for that Discussion history: {context}

Based on this history and context, answer this question using the third person point of view. Do NOT use first person, do not use the word "I": {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
  return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
  # return [d.metadata["url"] for d in docs]
  return docs[0].metadata["url"]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | parser_model | StrOutputParser() | compression_retriever,
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
    "question_is_relevant": is_about_privacy_act_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

# DBTITLE 1,Demo - Asking out-of-scope question
import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}, 
        {"role": "assistant", "content": "Consumers have the right to request, view, delete and modify their private data information."}, 
        {"role": "user", "content": "What is your favorite song?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
print(f"{response['result']}")

# COMMAND ----------

# DBTITLE 1,Demo - Requesting Consumer Data
dialog = {
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the California Privacy Act?"}, 
        {"role": "assistant", "content": "Consumers have the right to request, view, delete and modify their private data information."}, 
        {"role": "user", "content": "How would a California resident request their private data from a company?"}
    ]
}
response = full_chain.invoke(dialog)
print(f"{response['result']}\n\nSources:  {response['sources']}")

# COMMAND ----------

# DBTITLE 1,Demo - Consumer Rights
dialog = {
    "messages": [
        {"role": "user", "content": "What are consumer's rights in the Oregon Privacy Act?"}
    ]
}
response = full_chain.invoke(dialog)
print(f"{response['result']}\n\nSources:  {response['sources']}")

# COMMAND ----------

# DBTITLE 1,Demo - Biometric data
dialog = {
    "messages": [
        {"role": "user", "content": "What is considered biometric data in the Colorado Privacy Act?"}
    ]
}
response = full_chain.invoke(dialog)
print(f"{response['result']}\n\nSources:  {response['sources']}")
