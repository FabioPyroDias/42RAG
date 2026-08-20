"""Builds BM25 search indices for document retrieval.

Processes extracted text chunks into a BM25 index and
handles saving and loading it for searches.
"""

from typing import List
from src.models import Chunk
import bm25s


def generate_bm25_index(chunks: List[Chunk],
                        bm25_k1: float,
                        bm25_b: float) -> bm25s.BM25:
    """
    Generates a BM25 index from a list of text chunks.

    Args:
        chunks (List[Chunk]): List of chunks to index.
        bm25_k1 (float): Term frequency saturation parameter.
        bm25_b (float): Document length normalization parameter.

    Returns:
        bm25s.BM25: The trained BM25 index instance.

    Raises:
        ValueError
    """

    # This if ensures chunks isn't empty empty before indexing.
    #   If it is, raise ValueError
    if not chunks:
        raise ValueError("Cannot build BM25 index. "
                         "No chunks provided to index")

    # Stores the text content from each Chunk.
    texts = []
    for chunk in chunks:
        texts.append(chunk.text)

    # Tokenize full corpus text into term arrays using bm25s tokenizer.

    # The BM25 pipeline works as follows:
    # 1. Converts raw text strings into token matrix for exact matching.
    #   Takes care of lowercase conversion, word splitting and punctuation
    #   removal, preparing the data to the exact format required for the
    #   internal bm25s structure.
    # 2. Initializes the model with k1, term frequency saturation,
    #   and b, document length penalty.
    #   Initializes the scoring algorithm.
    # 3. Computes corpus statistics and builds the index
    tokens = bm25s.tokenize(texts)
    retriever = bm25s.BM25(k1=bm25_k1, b=bm25_b)
    retriever.index(tokens)

    return retriever
