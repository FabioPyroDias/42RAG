"""Augments search prompts with retrieved document contexts.

Formats retrieved code and documentation chunks into structured
prompts for generation and context enhancement.
"""

from typing import List
from src.models import Chunk


def augment_context(chunks: List[Chunk], max_context_length: int) -> str:
    """
    Formats and concatenates document chunks into a single context string.

    Args:
        chunks (List[Chunk]): List of retrieved document chunks.
        max_context_length (int): Maximum allowed character length
            for the context.

    Returns:
        str: Formatted context string containing structured
            chunk sources and text.
    """

    current_context = ""
    current_context_size = 0

    context_division = "\n\n"

    for chunk in chunks:
        expected_context = f"## Source: {chunk.file_path}\n{chunk.text}"

        if current_context_size == 0:
            if len(expected_context) <= max_context_length:
                current_context += expected_context
                current_context_size += len(expected_context)

        else:
            if (current_context_size + len(expected_context) +
               len(context_division) <= max_context_length):
                current_context += context_division
                current_context += expected_context

                current_context_size += len(context_division)
                current_context_size += len(expected_context)

    return current_context
