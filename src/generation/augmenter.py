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

    # Accumulator string holding the concatenated, formatted context
    current_context = ""

    # Tracks the length of current_context to enforce the limit
    current_context_size = 0

    # Section separator inserted between consecutive chunks
    context_division = "\n\n"

    # Iterates over chunks to build a single formatted context string
    #   within the character budget
    for chunk in chunks:
        # Format chunk content with a Markdown header specifying its
        #   source file as well as the context
        expected_context = f"## Source: {chunk.file_path}\n{chunk.text}"

        # If the current_context is empty, it doesn't need the separator
        if current_context_size == 0:
            # Append only if the entire chunk fits within max_context_length
            # Also updates the current_context_size to the content added
            if len(expected_context) <= max_context_length:
                current_context += expected_context
                current_context_size += len(expected_context)

        # If the current_context isn't empty, the context_division needs to
        #   be added first, before checking if the current chunk can be added.
        else:
            if (current_context_size + len(expected_context) +
               len(context_division) <= max_context_length):
                current_context += context_division
                current_context += expected_context

                # Update the current_context_size to take into account
                #   the context_division length
                current_context_size += len(context_division)
                current_context_size += len(expected_context)

    return current_context
