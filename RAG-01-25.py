# Databricks notebook source
# MAGIC %pip install langchain==0.1.0 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.9.0 lxml==4.9.3 pypdf
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader
import os
import numpy as np

folder = "/Volumes/demo/hackathon/privacy_act_docs/Colorado"
pdf_list = []

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        pdf_list.append(os.path.join(folder, file))


docs = []
loaders = []
for pdf in pdf_list:
    loaders.append(PyPDFLoader(pdf))
# loaders = [PyPDFLoader('/Volumes/demo/hackathon/privacy_act_docs/California/ccpa-regulations.pdf'),
#            PyPDFLoader('/Volumes/demo/hackathon/privacy_act_docs/California/cppa_act.pdf')]
for loader in loaders:
    docs.extend(loader.load())

for i in range(len(docs)):
    docs[i] = docs[i].page_content

spark_df = {"path": pdf_list, "content": docs}
data = [[*vals] for vals in zip(*spark_df.values())]
df = spark.createDataFrame(data, ["path", "content"])

# COMMAND ----------

from pyspark.sql.types import *

df.write.saveAsTable("demo.hackathon.pdf_raw_v2")

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from pprint import pprint

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
pprint(embeddings)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE demo.hackathon.pdf_raw_v2 SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()

index = client.create_delta_sync_index(
  endpoint_name="vector_search_endpoint",
  source_table_name="demo.hackathon.pdf_raw_v2",
  index_name="demo.hackathon.pdf_index_v2",
  pipeline_type='TRIGGERED',
  primary_key="path",
  embedding_source_column="content",
  embedding_model_endpoint_name="databricks-bge-large-en"
)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()

# question = "What is the effective date for this Coloroda Privacy act?"

# response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
# embeddings = [e['embedding'] for e in response.data]

# results = client.get_index("vector_search_endpoint", "demo.hackathon.pdf_index").similarity_search(
#   query_vector=embeddings[0],
# #   query_text="What is the effective date for this Coloroda Privacy act?",
#   columns=["path", "content"],
#   num_results=1)
# docs = results.get('result', {}).get('data_array', [])
# pprint(docs)

results = client.get_index("vector_search_endpoint", "demo.hackathon.pdf_index_v2").similarity_search(
  query_text="What is the effective date for this Coloroda Privacy act?",
  columns=["path", "content"]
)
pprint(results)

# COMMAND ----------


