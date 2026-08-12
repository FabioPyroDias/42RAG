"""Builds BM25 search indices for document retrieval.

Processes extracted text chunks into a BM25 index and
handles saving and loading it for searches.
"""

from typing import List
from student.models import Chunk
from bm25s import BM25
import bm25s


def generate_bm25_index(chunks: List[Chunk],
                        bm25_k1: float,
                        bm25_b: float) -> BM25:
    """
    Generates a BM25 index from a list of text chunks.

    Args:
        chunks (List[Chunk]): List of chunks to index.
        bm25_k1 (float): Term frequency saturation parameter.
        bm25_b (float): Document length normalization parameter.

    Returns:
        BM25: The trained BM25 index instance.
    """

    texts = []
    for chunk in chunks:
        texts.append(chunk.text)

    tokens = bm25s.tokenize(texts)
    retriever = BM25(k1=bm25_k1, b=bm25_b)
    retriever.index(tokens)

    return retriever
