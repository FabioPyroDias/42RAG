"""Executes search queries over indexed document chunks.

Uses pre-built BM25 search indices to rank and retrieve
the most relevant code and documentation contexts.
"""

from typing import List
from pathlib import Path
from src.file_manager import load_json
from src.models import Chunk, ChunkCollection, MinimalSource
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
        # Load BM25 index from disk
        index = bm25s.BM25.load(retrieval_index_path)

    except FileNotFoundError:
        raise ValueError(f"Index not found in {retrieval_index_path}. "
                         f"Please run index first")

    # Read saved chunks JSON file from disk
    json_data = load_json(Path(chunks_output_path))

    # Validate JSON into a ChunkCollection Pydantic model
    collection = ChunkCollection.model_validate(json_data)

    # Extract chunk list to verify content existence
    chunks = collection.chunks

    # If the chunks don't exist, raise Error
    if not chunks:
        raise ValueError(f"Chunks not found in {chunks_output_path}. "
                         f"The file exists but is empty. "
                         f"Please run index first")

    return (index, collection)


def search_chunks(query: str,
                  bm25: bm25s.BM25,
                  chunks: ChunkCollection,
                  k: int) -> tuple[List[Chunk], List[MinimalSource]]:
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

    # Ensure top-k count is a positive integer
    if k <= 0:
        raise ValueError("k needs to be positive")

    # Tokenize search query into terms expected by BM25
    tokens = bm25s.tokenize([query])

    # Query BM25 index to retrieve the top-k relevant document chunks
    top_chunks = bm25.retrieve(tokens, corpus=chunks.chunks, k=k)

    # Lists for matching Chunk objects and sources
    chunks_found = []
    sources = []

    # Map retrieved document chunks to Chunk instances
    #   and MinimalSource metadata models.
    for chunk in top_chunks.documents[0]:
        chunks_found.append(chunk)
        sources.append(
            MinimalSource(file_path=chunk.file_path,
                          first_character_index=chunk.first_character_index,
                          last_character_index=chunk.last_character_index))

    return chunks_found, sources


def build_chunk_index(chunks: ChunkCollection) -> dict[
  tuple[str, int, int], Chunk]:
    """
    Creates a dictionary mapping source coordinates to chunk objects.

    Args:
        chunks (ChunkCollection): The collection of chunks to index.

    Returns:
        dict[tuple[str, int, int], Chunk]: Mapping from
            (file_path, first_index, last_index) to Chunk.
    """

    # Index dictionary for chunk retrieval
    chunk_index = {}

    # Iterate through chunks to populate the coordinate index dictionary
    for chunk in chunks.chunks:
        # Create key using file path and character boundaries
        key = (chunk.file_path,
               chunk.first_character_index,
               chunk.last_character_index)

        # Map key to its corresponding Chunk instance
        chunk_index[key] = chunk

    return chunk_index


def match_chunks(sources: List[MinimalSource],
                 chunk_index: dict[tuple[str, int, int],
                                   Chunk]) -> List[Chunk]:
    """
    Maps source references to their corresponding full Chunk objects.

    Args:
        sources (List[MinimalSource]): Source references to match.
        chunk_index (dict): Mapping built by build_chunk_index.

    Returns:
        List[Chunk]: The matched chunks. Sources with no match are skipped.
    """

    # List storing matched Chunk objects
    matched_chunks = []

    # Iterate through source references to match corresponding chunks
    for source in sources:
        # Create key using file path and character boundaries
        key = (source.file_path,
               source.first_character_index,
               source.last_character_index)

        # Append matched chunk if key exists in index dictionary
        if key in chunk_index:
            matched_chunks.append(chunk_index[key])

    return matched_chunks
