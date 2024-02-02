# Databricks notebook source
pip install FlagEmbedding

# COMMAND ----------

import mlflow
import numpy as np
mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run():
    components = {
        "model":model,
        "tokenizer":tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="mpt"
        signature=signature
        registered_model_name="ml.llm-catalog.mpt-7b",
        input_example={"prompt": np.array(["Below is an instruction that describes a task. Write a response that appropriately copmletes the request. \n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"]), "max_tokens":np.array([75]), "temperature": np.array([0.0])},
        metadata = {"task": "llm/v1/completions"}
    )

# COMMAND ----------

import mlflow
import transformers

# Define a pre-trained model
model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Log the model
with mlflow.start_run():
    mlflow.transformers.log_model(model, 'model')
