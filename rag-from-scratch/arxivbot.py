"""
This is the entry point for this PoC application to find the parameters of kinetic
models automatically from unstructured text.
"""
import argparse
import json
import os
from glob import glob
from dataclasses import dataclass
from typing import *

import fitz
from tqdm import tqdm

from ai import chat_completion, PageEmbeddingStr, get_embedding, rank_embeddings


STRING_MATCHES = ['parameters', 'kinetic', 'model', 'coefficients']

PROMPT = """Answer the question as truthfully as possible, and if you're unsure of the answer, say `I don't know.`

Context:
{context}

Q: {query}
A:
"""


@dataclass
class DocumentPage:
    pagenumber: int # Number in PDF document, starting with 1, 2, ...
    text: str


class EmbeddingStore:
    """
    This is a simplified vector store for embedding PDF documents
    
    Note: It could be improved in various ways.
    1. Compute embeddings in parallel to speed the initial data loading
    2. Update the underlying JSON file
    """
    def __init__(self, docs_path: str):
        docs_path_pattern = os.path.join(docs_path, '*pdf')
        self.document_paths = glob(docs_path_pattern)
        self.pdfs = []
        self.pdf_map = {}
        for dp in self.document_paths:
            try:
                pdf = fitz.open(dp)
                self.pdfs.append(pdf)
                self.pdf_map[dp] = pdf
            except fitz.fitz.FileDataError as e:
                print(f"Broken Document: {dp}")

        self.embeddings_path = os.path.join(docs_path, 'embeddings.json')
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'r') as fp:
                self._embeddings = json.load(fp, object_hook=lambda d: {k: v if k != "page_embeddings" else [PageEmbeddingStr(**e) for e in v] for k, v in d.items()})
        else:
            self._embeddings = {'documents': [], 'page_embeddings': []} # make documents a set later

    def get_embeddings(self):
        for pdf in tqdm(self.pdfs):
            if pdf.name not in self._embeddings['documents']:
                document_embeddings = [PageEmbeddingStr(pdf.name, p, get_embedding(pdf[p].get_text())) for p in range(len(pdf))]
                self._embeddings['documents'].append(pdf.name)
                self._embeddings['page_embeddings'] += document_embeddings
        
        # Serialize for later reuse
        with open(self.embeddings_path, 'w') as fp:
            json.dump(self._embeddings, fp, default=lambda o: o.__dict__)

        return self._embeddings['page_embeddings']


def prompt_per_page(relevant_pages: Dict[str, List[DocumentPage]], query: str):
    """
    Run prompt for all the relevant lages we retrieved
    """
    with open('response-query.md', 'w', encoding='utf-8') as fp:
        for document, pages in relevant_pages.items():
            document_name = os.path.basename(document)
            for page in pages:
                prompt = PROMPT.format(context=page.text, query=query)
                try:
                    response = chat_completion(prompt)
                    # response = llm_complete(prompt)
                except Exception as e:
                    print(e)
                    print("Experienced an error during OpenAI API call.")
                if "I don't know" not in response and "I don't have" not in response:
                    fp.write(f'## {document_name}\npage: {page.pagenumber}\n')
                    print(f'## {document_name}\npage: {page.pagenumber}\n')
                    fp.write(response)
                    print(response)
                    print(25 * '=')
                    fp.write('\n')
                    fp.flush()


def main():
    parser = argparse.ArgumentParser(description='Process some input.')
    parser.add_argument('-i', '--input_path', type=str, help='input file path', default='test_data')
    parser.add_argument('--ask', type=str, help='query string parameter', dest='question')
    parser.add_argument('-k', '--top-k', default=5, type=int, help='number of top hits to search for answer')

    args = parser.parse_args()
    corpus = EmbeddingStore(args.input_path)#'data/val/*pdf')
    if args.question:
        print("Obtaining embeddings")
        page_embeddings = corpus.get_embeddings()

        relevant_pages = []
        print("Ranking documents")
        indices_and_distances = rank_embeddings(page_embeddings, args.question)
        indices = [i for i, _ in indices_and_distances]
        ranked_embeddings = [page_embeddings[i] for i in indices[:args.top_k]]
        relevant_pages = {}
        for e in ranked_embeddings:
            if e.document_name not in relevant_pages:
                relevant_pages[e.document_name] = []
            relevant_pages[e.document_name].append(DocumentPage(e.page_number + 1, text = corpus.pdf_map[e.document_name][e.page_number].get_text()))
        query = args.question
        print("running queries...")
        prompt_per_page(relevant_pages, query)


if __name__ == "__main__":
    main()