# Databricks notebook source
# MAGIC %pip install mlflow[databricks]==2.9.0 lxml==4.9.3 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2 openai==1.10.0 FlagEmbedding==1.2.1 rank_bm25==0.2.2 SQLAlchemy==2.0.25 llama-index==0.9.40 langchain-openai langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser


prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 500)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What is Spark?"}))

# COMMAND ----------

# index_name=f"demo.hackathon.databricks_pdf_documentation_self_managed_vs_index"
index_name = "demo.hackathon.openai_self_managed_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
# from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from llama_index import Document
import os
from langchain_openai import AzureOpenAIEmbeddings

os.environ["AZURE_OPENAI_API_KEY"] = dbutils.secrets.get(scope='dev_demo', key='azure_openai_api_key')
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://nous-ue2-openai-sbx-openai.openai.azure.com/"

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="nous-ue2-openai-sbx-base-deploy-text-embedding-ada-002",
    openai_api_version="2023-05-15",
)

# embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
docs_table = spark.table("demo.hackathon.databricks_pdf_documentation_openai").select("content", "url").collect()

bm25_retriever = BM25Retriever.from_documents(
    documents=[Document(text=str(docs_table[i].__getitem__("content")), metadata={"url": str(docs_table[i].__getitem__("url"))}).to_langchain_format() for i, v in enumerate(docs_table)]
)
bm25_retriever.k = 3

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host)
    vs_index = vsc.get_index(
        endpoint_name="openai_vector_search",
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embeddings, columns=["url", "content"]
    )

    return vectorstore.as_retriever(search_kwargs={'k': 3})

# COMMAND ----------

from llama_index import ServiceContext
from llama_index.postprocessor import LLMRerank
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.schema import QueryBundle
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# score = reranker.compute_score(['query', 'passage'])
# print(score)


os.environ["OPENAI_API_VERSION"] = "2023-05-15"

llm = AzureOpenAI(
    model="gpt-4",
    engine="nous-ue2-openai-sbx-base-deploy-gpt-4-turbo",
)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings, chunk_size=512)

def rerank(
    query_str, vector_top_k=6, reranker_top_n=3, with_reranker=True
):
    query_bundle = QueryBundle(query_str)
    base_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=25)
    # configure retriever
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, get_retriever()], weights=[0.5, 0.5])
    retrieved_nodes = base_splitter.get_nodes_from_documents([Document(text=str(x)) for x in retriever.get_relevant_documents(query_str)])

    if with_reranker:
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
            service_context=service_context,
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes

# COMMAND ----------

from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | RunnableLambda(rerank)
)
results = retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is the effective date of the Colorado privacy act?"}]})

# COMMAND ----------

results[2]

# COMMAND ----------

from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)

# COMMAND ----------

# MAGIC %md
# MAGIC GPT4

# COMMAND ----------

import openai
from llama_index.llms import AzureOpenAI

# openai.api_type: str = "azure"  
# openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  
# openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  
# openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  
# llm_deployment_name = "nous-ue2-openai-sbx-base-deploy-gpt-4-turbo"
# embedding_deployment_name = "nous-ue2-openai-sbx-base-deploy-text-embedding-ada-002"

llm = AzureOpenAI(engine="nous-ue2-openai-sbx-base-deploy-gpt-4-turbo",
                  model="gpt-4",
                  temperature=0.0,
                  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                  api_version="2023-07-01-preview"
                  )
response = llm.complete("The sky is a beautiful blue and")
print(response)                  

# COMMAND ----------

from llama_index.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with colorful personality."
    ),
    ChatMessage(role="user", content="Hello"),
]

response = llm.chat(messages)
print(response)
