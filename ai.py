from dataclasses import dataclass
from typing import *

import numpy as np
import openai

from openai.embeddings_utils import cosine_similarity


with open('api.key', 'r') as fp:
    openai.api_key = fp.read()

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

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



def llm_complete(inp: str, model=COMPLETIONS_MODEL, max_tokens=1000, temperature=0) -> str:
    response = openai.Completion.create(model=model, prompt=inp, temperature=temperature, max_tokens=max_tokens)
    return response['choices'][0]['text']


def chat_completion(inp: str, model='gpt-4', temperature=0) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant and your answers are as concise as possible."},
        {"role": "user", "content": inp},
        ]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
    return response['choices'][0]['message']['content']


def get_embedding(text, model=EMBEDDING_MODEL):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


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


import openai

openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)