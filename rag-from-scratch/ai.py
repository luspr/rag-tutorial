from dataclasses import dataclass
from typing import *

import numpy as np

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())


EMBEDDING_MODEL = "text-embedding-ada-002"
LLM = 'gpt-4-turbo-preview'

@dataclass
class PageEmbedding:
    document_id:    int
    page_number:    int
    embeddings:     list[float] 


@dataclass
class PageEmbeddingStr:
    document_name:  str
    page_number:    int
    embeddings:     list[float] 


def chat_completion(inp: str, model=LLM, temperature=0) -> str:
    client = OpenAI()
    messages = [
        {"role": "system", "content": "You are an expert research assistant helping answering questions about scientific papers."},
        {"role": "user", "content": inp},
        ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message.content


def get_embedding(text, model=EMBEDDING_MODEL):
   client = OpenAI()
   text = text.replace("\n", " ")
   response = client.embeddings.create(input = text, model=model)
   return response.data[0].embedding


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def rank_embeddings(embeddings: list[PageEmbedding], query_string: str):
    """
    Calculates distance to all embeddings for a given query and ranks them.

    Returns a list of indices (related to the page embeddings) and the distances, sorted by distance to query
    """
    query_embedding = get_embedding(query_string)
    indices_and_distances = [(i, vector_similarity(pe.embeddings, query_embedding)) for i, pe in enumerate(embeddings)]
    indices_and_distances.sort(key=lambda x: x[1], reverse=True)
    return indices_and_distances
