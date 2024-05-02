import pandas as pd
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf
import numpy as np
import json
import re

# Load the Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Function to clean text
def clean_text(text):
    # Remove non-alphanumeric characters and symbols
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

# Function to compute cosine similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

sen1 = input("Enter Sen 1 : ")
sen2 = input("Enter Sen 2 : ")

clean1 = clean_text(sen1)
clean2 = clean_text(sen2)

print("Clean 1 : ", clean1)
print("Clean 2 : ", clean2)

embedding1 = use_model([clean1])[0]
embedding2 = use_model([clean2])[0]

cosine_sim_title = round(((cosine_similarity(embedding1, embedding2)) * 100) , 2)

print("Matching Percentage : ", cosine_sim_title)