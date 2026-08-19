"""Splits code and text documents into manageable chunks for indexing.

Processes files line by line using target size thresholds and overlap
to preserve context without breaking code blocks.
"""

from pathlib import Path
from typing import List
from tqdm import tqdm
from student.file_manager import get_indexable_files, load_text_file
from student.models import Chunk
from student.indexing.utils import create_chunk


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

    chunks = []

    line = 0
    current_content = ""
    start_position = 0

    data = content.split("\n")
    while line < len(data):
        is_last_line = line == len(data) - 1

        if len(data[line]) + 1 <= max_chunk_size - len(current_content):
            current_content += data[line]
            if not is_last_line:
                current_content += "\n"
            line += 1
        else:
            if len(current_content.strip("\n")) > 0:
                chunks.append(create_chunk(path,
                                           current_content,
                                           start_position))

                start_position += len(current_content)
                current_content = ""

            if len(data[line]) + 1 > max_chunk_size:
                if len(data[line].strip("\n")) > 0:
                    chunks.append(create_chunk(path,
                                               data[line],
                                               start_position))

                start_position += len(data[line])
                if not is_last_line:
                    start_position += 1

                line += 1
            else:
                current_content += data[line]
                if not is_last_line:
                    current_content += "\n"
                line += 1

    if len(current_content.strip("\n")) > 0:
        chunks.append(create_chunk(path,
                                   current_content,
                                   start_position))

    return chunks


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

    for index_path in tqdm(paths, desc="Indexing files"):
        content = load_text_file(index_path)
        chunks += generate_file_chunks(index_path, content, max_chunk_size)

    return chunks
