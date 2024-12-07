import openai
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity



api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key


dataset = load_dataset("mteb/stsbenchmark-sts")
sample_data = dataset["validation"][0:10]

client = OpenAI()

def chat_gpt(sentence1, sentence2):
    prompt = f"Provide only a semantic similarity score between the following sentences as a single number. Sentence 1: {sentence1}. Sentence 2: {sentence2}."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens = 50,
        temperature= 0,
        messages=[
                {"role": "system", "content": "You are a helpful assistant providing semantic similarity scores."},
                {"role": "user", "content": prompt}
            ],
    )
    return response.choices[0].message.content.strip()

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(sentence1, sentence2):

    embedding1 = get_embedding(sentence1)
    embedding2 = get_embedding(sentence2)
    similarity = sklearn_cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity


def compute_metrics(scores1, scores2):
    mse = mean_squared_error(scores1, scores2)
    rmse = np.sqrt(mse)
    ae = mean_absolute_error(scores1, scores2)
    return {"MSE": mse, "RMSE": rmse, "AE": ae}



# print(chat_gpt("hello my name is Nikhita", "Hello my name is Guntu"))

print(cosine_similarity("hello my name is Nikhita", "Hello my name is Guntu"))
