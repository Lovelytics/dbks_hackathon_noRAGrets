# Databricks notebook source
# MAGIC %md
# MAGIC https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/2/document-loading

# COMMAND ----------

#transformers==4.30.2 "unstructured[pdf,docx]==0.10.30"

# COMMAND ----------

# %pip install langchain==0.1.0 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.9.0 lxml==4.9.3 pypdf databricks-genai-inference

# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install langchain==0.1.0 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic mlflow==2.9.0 lxml==4.9.3 pypdf databricks-genai-inference
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
#import openai
import sys
sys.path.append('../..')

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file

# openai.api_key  = os.environ['OPENAI_API_KEY']

# COMMAND ----------

# MAGIC %md
# MAGIC # Load PDFs

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader
# loader = PyPDFLoader("/Volumes/demo/hackathon/privacy_act_docs/Iowa/Iowa-privacy-act.pdf")
# pages = loader.load()
# pages_in_pdf = len(pages)
# print(f"total number of pages: {pages_in_pdf}") 
# # Preview page content
# page = pages[0]
# print(page.page_content[0:500])
# page.metadata

# COMMAND ----------

# MAGIC %md
# MAGIC # Split PDFs

# COMMAND ----------

# # not sure which splitter is best for our use case yet, will have to try these to find out
# from langchain.text_splitter import CharacterTextSplitter
# chunk_size = 500
# chunk_overlap = 5

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=1000,
#     chunk_overlap=150,
#     length_function=len
# )
# docs = text_splitter.split_documents(pages)
# docs[10]
# print(f"Number of splits in document: {len(docs)}")

# COMMAND ----------

# Now that we've proved the concept, try reading in multiple files (California has two files in its folder)
folder = "/Volumes/demo/hackathon/privacy_act_docs/California"
pdf_list = []

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        pdf_list.append(os.path.join(folder, file))

print(pdf_list)

# COMMAND ----------

docs = []

loaders = [PyPDFLoader('/Volumes/demo/hackathon/privacy_act_docs/California/ccpa-regulations.pdf'),
           PyPDFLoader('/Volumes/demo/hackathon/privacy_act_docs/California/cppa_act.pdf')]
for loader in loaders:
    docs.extend(loader.load())

# COMMAND ----------

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

# COMMAND ----------

splits = text_splitter.split_documents(docs)
len(splits)

# COMMAND ----------

# MAGIC %md
# MAGIC # Embeddings
# MAGIC Make sure to use an embedding model that is compatible with your LLM.

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as F

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
print(embeddings)

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

df = spark.createDataFrame(splits)
df.show()

# COMMAND ----------

df.select("page_content").show()

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

get_embedding(df["page_content"])

# COMMAND ----------

# from databricks_genai_inference import Embedding

# response = Embedding.create(
#     model="bge-large-en",
#     input=splits.page_content)
# print(f'embeddings: {response.embeddings}')
