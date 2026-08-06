"""Command-Line Interface for the RAG pipeline processing and evaluation.

This module defines commands to index documents,
search context and generate answers.
"""


class CommandLineInterface():
    """Class exposing CLI commands for the RAG pipeline.

    Each method corresponds to a specific CLI action.
    """

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
            pass
        except TypeError:
            pass

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
