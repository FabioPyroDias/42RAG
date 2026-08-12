"""Command-Line Interface for the RAG pipeline processing and evaluation.

This module defines commands to index documents,
search context and generate answers.
"""

from pathlib import Path
from pydantic import ValidationError
from student.config import RAGConfig
from student.indexing.chunking import generate_chunks
from student.indexing.indexer import generate_bm25_index
from student.file_manager import save_json
from student.models import ChunkCollection


class CommandLineInterface():
    """Class exposing CLI commands for the RAG pipeline.

    Each method corresponds to a specific CLI action.
    """

    def __init__(self) -> None:
        """Loads the default RAG pipeline configuration."""

        self.config = RAGConfig()

    def index(self,
              max_chunk_size: int) -> None:
        """
        Indexes documents into chunks and builds the BM25 index.

        Args:
            max_chunk_size (int): Maximum character length for each chunk.

        Returns:
            None
        """

        try:
            self.config = RAGConfig(**{**self.config.model_dump(),
                                       "max_chunk_size": max_chunk_size})

            chunks = generate_chunks(Path(self.config.raw_repository_path),
                                     self.config.max_chunk_size)

            if not chunks:
                raise ValueError(f"No indexable files found in "
                                 f"\"{self.config.raw_repository_path}\". "
                                 f"Verify that the path exists and contains "
                                 f"supported files.")

            bm25 = generate_bm25_index(chunks,
                                       self.config.bm25_k1,
                                       self.config.bm25_b)

            bm25.save(self.config.retrieval_index_path)

            collection = ChunkCollection(chunks=chunks)
            save_json(Path(self.config.chunks_output_path), collection)

            print(f"Ingestion complete! "
                  f"Indices saved under {self.config.processed_path}")

        except ValidationError as error:
            print(f"ERROR: {error.errors()[0]["msg"]}")
        except ValueError as error:
            print(F"ERROR: {error}")

    def search(self,
               query: str,
               k: int = 10) -> None:
        """
        Executes a search query.

        Args:
            query (str): The search query or question text.
            k (int): Number of top context sources to retrieve.

        Returns:
            None
        """

        pass

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = 10) -> None:
        """
        Executes search queries from a dataset.

        Args:
            dataset_path (str): Path to the input dataset JSON file containing
                UnansweredQuestion entries.
            save_directory (str): Directory path where
                StudentSearchResults JSON file will be saved.
            k (int): Number of top context sources to retrieve per question.

        Returns:
            None
        """

        pass

    def answer(self,
               query: str,
               k: int = 10) -> None:
        """
        Retrieves sources and generates an answer.

        Args:
            query (str): The question text to answer.
            k (int): Number of top context sources to retrieve as context.

        Returns:
            None
        """

        pass

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Generates answers from search results.

        Args:
            student_search_results_path (str): Path to the
                StudentSearchResults JSON file.
            save_directory (str): Directory path where
                StudentSearchResultsAndAnswer JSON file will be saved.
        """

        pass
