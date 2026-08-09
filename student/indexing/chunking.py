"""Splits code and text documents into manageable chunks for indexing.

Processes files line by line using target size thresholds and overlap
to preserve context without breaking code blocks.
"""

from pathlib import Path
from typing import List
from student.file_manager import get_indexable_files, load_text_file
from student.models import Chunk


def generate_file_chunks(path: Path,
                         content: str,
                         max_chunk_size: int) -> List[Chunk]:
    """
    Processes a single indexable files and splits its content into chunks.

    Args:
        path (Path): The Path to the source file.
        content (str): The content file to split into chunks
        max_chunk_size (int): Soft target character limit per chunk.

        Returns:
            List[Chunk]: The list of generated Chunks
    """
    pass


def generate_chunks(path: Path, max_chunk_size: int) -> List[Chunk]:
    """
    Processes all indexable files in a directory and splits them into chunks.

    Args:
        path (Path): The root directory Path containing source files.
        max_chunk_size (int): Soft target character limit per chunk.

    Returns:
        List[Chunk]: The list of generated Chunks of all the files.
    """

    paths = get_indexable_files(path)
    chunks = []

    for index_path in paths:
        content = load_text_file(index_path)
        chunks += generate_file_chunks(index_path, content, max_chunk_size)

    return chunks
