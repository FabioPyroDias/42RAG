"""Splits code and text documents into manageable chunks for indexing.

Processes files line by line using target size thresholds and overlap
to preserve context without breaking code blocks.
"""

from pathlib import Path
from typing import List
from tqdm import tqdm
from src.file_manager import get_indexable_files, load_text_file
from src.models import Chunk
from src.indexing.utils import create_chunk


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

    # List holding all generated Chunk objects to return
    chunks = []

    # Current line index in the split text array
    line = 0

    # Accumulates lines for the active chunk. Acts as a buffer
    current_content = ""

    # Character offset representing the start of the
    #   current chunk in the document
    start_position = 0

    # Document content split into individual line strings
    data = content.split("\n")

    # Iterates over every line of the document
    while line < len(data):
        # Flag indicating whether the loop is on the final line of the file
        is_last_line = line == len(data) - 1

        # Check if the current line fits in the remaining space of the
        #   current buffer. +1 for newline
        if len(data[line]) + 1 <= max_chunk_size - len(current_content):
            current_content += data[line]
            if not is_last_line:
                current_content += "\n"
            line += 1

        # If current line does not fit in the existing buffer
        else:
            # Add current_content into a new Chunk, removing newline
            if len(current_content.strip("\n")) > 0:
                chunks.append(create_chunk(path,
                                           current_content,
                                           start_position))

                # Update character offset by the length of the
                #   saved chunk text
                start_position += len(current_content)

                # Reset buffer for the next chunk
                current_content = ""

            # Handle an oversized single line that
            #   exceeds max_chunk_size on its own
            if len(data[line]) + 1 > max_chunk_size:
                if len(data[line].strip("\n")) > 0:
                    chunks.append(create_chunk(path,
                                               data[line],
                                               start_position))

                # Update offset by the line length plus
                #   trailing newline character if present
                start_position += len(data[line])
                if not is_last_line:
                    start_position += 1

                line += 1

            # Line fits within max_chunk_size,
            #   so initialize the empty current_content with it
            else:
                current_content += data[line]
                if not is_last_line:
                    current_content += "\n"
                line += 1

    # If there's any remaining text left in current_content
    #   after processing all lines, create a new Chunk and append it
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

    # Retrieve all valid indexable file paths from the target directory
    paths = get_indexable_files(path)

    # Stores generated Chunk objects across all processed files
    chunks = []

    # Iterate through each file path with a progress bar display
    for index_path in tqdm(paths, desc="Indexing files"):
        # Load full text content from the current file
        content = load_text_file(index_path)

        # Split file content into chunks and append them
        chunks += generate_file_chunks(index_path, content, max_chunk_size)

    return chunks
