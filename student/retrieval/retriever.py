"""Executes search queries over indexed document chunks.

Uses pre-built BM25 search indices to rank and retrieve
the most relevant code and documentation contexts.
"""

from typing import List
from pathlib import Path
from student.file_manager import load_json
from student.models import ChunkCollection, MinimalSource
import bm25s


def load_retrieval_index(retrieval_index_path: str,
                         chunks_output_path: str) -> tuple[bm25s.BM25,
                                                           ChunkCollection]:
    """
    Loads a BM25 index and its corresponding chunk collection from disk.

    Args:
        retrieval_index_path: Path to the saved BM25 index directory or file.
        chunks_output_path: Path to the JSON file containing saved chunks.

    Returns:
        tuple[bm25s.BM25, ChunkCollection]: Tuple containing the initialized
            BM25 index and ChunkCollection.

    Raises:
        ValueError
    """

    try:
        index = bm25s.BM25.load(retrieval_index_path)

    except FileNotFoundError:
        raise ValueError(f"Index not found in {retrieval_index_path}. "
                         f"Please run index first")

    json_data = load_json(Path(chunks_output_path))
    collection = ChunkCollection.model_validate(json_data)
    chunks = collection.chunks

    if not chunks:
        raise ValueError(f"Chunks not found in {chunks_output_path}. "
                         f"The file exists but is empty. "
                         f"Please run index first")

    return (index, collection)


def search_chunks(query: str,
                  bm25: bm25s.BM25,
                  chunks: ChunkCollection,
                  k: int) -> List[MinimalSource]:
    """
    Searches document chunks for the top-k most relevant matches using BM25.

    Args:
        query (str): The search query text.
        bm25 (bm25s.BM25): The pre-built BM25 index instance.
        chunks (ChunkCollection): Collection of document chunks corresponding
            to the index entries.
        k (int): Number of top relevant results to retrieve.

    Returns:
        List[MinimalSource]: List of the top-k retrieved source results ranked
            by relevance score.

    Raises:
        ValueError
    """

    if k <= 0:
        raise ValueError("k needs to be positive")

    tokens = bm25s.tokenize([query])

    top_chunks = bm25.retrieve(tokens, corpus=chunks.chunks, k=k)

    sources = []
    for chunk in top_chunks.documents[0]:
        sources.append(
            MinimalSource(file_path=chunk.file_path,
                          first_character_index=chunk.first_character_index,
                          last_character_index=chunk.last_character_index))

    return sources
