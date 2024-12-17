import openai
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from evaluate import load
import pandas as pd

api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key
client = OpenAI()

dataset = load_dataset("mteb/stsbenchmark-sts")

def chat_gpt(sentence1, sentence2, model):
    prompt = f"Provide only a semantic similarity score between the following sentences as a single number. Sentence 1: {sentence1}. Sentence 2: {sentence2}."
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        temperature=0,
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

def compute_rouge(model):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    sample_data = dataset["test"].select(range(10))
    articles = sample_data["article"]
    highlights = sample_data["highlights"]

    rouge = load("rouge")
    model_predictions = []
    for article in articles:
        prompt = f"Summarize the following article: {article}"
        response = client.chat.completions.create(
            model=model,
            max_tokens=150,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing concise summaries of news articles."},
                {"role": "user", "content": prompt}
            ],
        )
        summary = response.choices[0].message.content.strip()
        model_predictions.append(summary)

    rouge_model_input = rouge.compute(predictions=model_predictions, references=articles)
    rouge_input_output = rouge.compute(predictions=articles, references=highlights)

    return rouge_model_input, rouge_input_output

st.title("Semantic Similarity Analysis Dashboard")

st.header("Select GPT Model")
model_options = ["gpt-4o-mini", "gpt-3.5-turbo"]  
selected_model = st.selectbox("Choose a GPT model:", model_options)

st.header("Dataset")
sample_data = dataset["validation"].select(range(10))
st.write("Sample Data:")
st.write(sample_data)

st.header("Semantic Similarity Scores")
chatgpt_scores = []
cosine_scores = []
true_scores = sample_data["score"]

with st.spinner("Calculating scores..."):
    for sent1, sent2 in zip(sample_data["sentence1"], sample_data["sentence2"]):
        gpt_score = float(chat_gpt(sent1, sent2, selected_model))
        chatgpt_scores.append(gpt_score)
        
        cos_score = cosine_similarity(sent1, sent2)
        cosine_scores.append(cos_score)

st.write("**ChatGPT Scores:**", chatgpt_scores)
st.write("**Cosine Similarity Scores:**", cosine_scores)

st.header("Metrics Comparison")
chatgpt_metrics = compute_metrics(true_scores, chatgpt_scores)
cosine_metrics = compute_metrics(true_scores, cosine_scores)

st.subheader("ChatGPT Metrics")
st.write(chatgpt_metrics)

st.subheader("Cosine Similarity Metrics")
st.write(cosine_metrics)

st.header("Text Evaluation Metrics")
rouge_model_input, rouge_input_output = compute_rouge(selected_model)
rouge_comparison = compute_metrics(list(rouge_model_input.values()), list(rouge_input_output.values()))

st.subheader("ROUGE Comparison")
st.write("Model Output vs Dataset Input:", rouge_model_input)
st.write("Dataset Input vs Output:", rouge_input_output)
st.write("Comparison Metrics:", rouge_comparison)

st.header("Conclusions")
st.write("Compare which metric (ChatGPT or Cosine Similarity) is closer to the true scores based on RMSE, MSE, or AE.")
