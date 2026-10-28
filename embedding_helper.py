from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import os

# Load configuration
load_dotenv()

# Create AI client
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)


def create_embedding(content):
    embedding1 = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=content
    ).data[0].embedding
    return embedding1


def check_similarity(vector1, vector2):

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) *
        np.linalg.norm(v2)
    )
