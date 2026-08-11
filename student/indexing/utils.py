"""Chunking and indexing utilities."""

from pathlib import Path
from student.models import Chunk


def create_chunk(path: Path,
                 content: str,
                 start_position: int) -> Chunk:
    """
    Creates a Chunk instance with calculated character boundaries.

    Args:
        path (Path): Path to the source file.
        content (str): Text content of the chunk.
        start_position (int): Starting character index in the file.

    Returns:
        Chunk: The created chunk object.
    """

    return Chunk(file_path=str(path),
                 first_character_index=start_position,
                 last_character_index=start_position + len(content),
                 text=content)
