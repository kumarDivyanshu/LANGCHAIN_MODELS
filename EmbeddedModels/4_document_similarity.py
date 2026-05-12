from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Bangalore is the IT hub of India.",
    "Kolkata is the cultural capital of India."
]

query = "tell me about India?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index , score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]
print(query)
print(documents[index])
print("Most similar document:" ,score)