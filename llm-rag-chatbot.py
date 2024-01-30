# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 transformers==4.30.2 langchain==0.0.344 databricks-vectorsearch==0.22
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install pyparsing==2.2.1 fonttools==4.22.0 python-dateutil==2.7 charset-normalizer==2 certifi==2017.4.17 joblib==1.0.0 threadpoolctl==2.0.0 greenlet!=0.4.17 smmap==3.0.1 mypy-extensions==0.3.0 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install dbdemos
# MAGIC import dbdemos

# COMMAND ----------

dbdemos.install('llm-rag-chatbot', use_current_cluster=True)
