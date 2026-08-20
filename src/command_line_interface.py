"""Command-Line Interface for the RAG pipeline processing and evaluation.

This module defines commands to index documents,
search context and generate answers.
"""

from pathlib import Path
from tqdm import tqdm
from pydantic import ValidationError
from src.config import RAGConfig
from src.file_manager import save_json, load_dataset, load_json
from src.models import (ChunkCollection,
                        MinimalSearchResults,
                        StudentSearchResults,
                        MinimalAnswer,
                        StudentSearchResultsAndAnswer,
                        RagDataset,
                        AnsweredQuestion)
from src.indexing.chunking import generate_chunks
from src.indexing.indexer import generate_bm25_index
from src.retrieval.retriever import (load_retrieval_index,
                                     search_chunks,
                                     build_chunk_index,
                                     match_chunks)
from src.generation.augmenter import augment_context
from src.generation.generator import load_model, generate_answer
from src.evaluation.evaluator import evaluate_search_results


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
            # Re-validate and update configuration with the newly supplied
            #   max_chunk_size parameter
            # Dumps current config to a dictionary, overrides max_chunk_size,
            #   and instantiates RAGConfig again to trigger Pydantic
            #   validation and create an updated configuration instance
            self.config = RAGConfig(**{**self.config.model_dump(),
                                       "max_chunk_size": max_chunk_size})

            # Load files from source directory and split them into
            #   structured Chunk objects
            chunks = generate_chunks(Path(self.config.raw_repository_path),
                                     self.config.max_chunk_size)

            # Verify that indexable content was found in the repository path
            if not chunks:
                raise ValueError(f"No indexable files found in "
                                 f"\"{self.config.raw_repository_path}\". "
                                 f"Verify that the path exists and contains "
                                 f"supported files.")

            # Build BM25 index over extracted chunks using configured
            #   k1, term frequency saturation, and
            #   b, document length penalty, parameters
            bm25 = generate_bm25_index(chunks,
                                       self.config.bm25_k1,
                                       self.config.bm25_b)

            # Save trained BM25 index files to disk.
            bm25.save(self.config.retrieval_index_path)

            # Wrap chunks in a Pydantic collection container and save them
            #   to JSON for retrieval mapping
            collection = ChunkCollection(chunks=chunks)
            save_json(Path(self.config.chunks_output_path), collection)

            print(f"Ingestion complete! "
                  f"Indices saved under {self.config.processed_path}")

        except ValidationError as error:
            print(f"ERROR: {error.errors()[0]['msg']}")
        except ValueError as error:
            print(f"ERROR: {error}")

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

        try:
            # If query string is empty, raise ValueError
            if not query.strip():
                raise ValueError("Query cannot be empty")

            # Load the BM25 index and corresponding chunk dataset from disk
            bm25, chunks = (
                load_retrieval_index(self.config.retrieval_index_path,
                                     self.config.chunks_output_path))

            # Perform lexical search and extract retrieved source objects.
            # Since search_chunks returns the list of Chunks and the list
            #   with MinimalSource, we can simply use the second field
            #   since StudentSearchResults expects a list of MinimalSource
            #   for the retrieved_sources field in MinimalSearchResults
            sources = search_chunks(query, bm25, chunks, k)[1]

            search_result = StudentSearchResults(
                search_results=[MinimalSearchResults(
                    question_id="question_query",
                    question=query,
                    retrieved_sources=sources)], k=k)

            print(search_result.model_dump_json(indent=4))

        except ValueError as error:
            print(f"ERROR: {error}")

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

        try:
            # Load the BM25 index and corresponding chunk dataset from disk
            bm25, chunks = (
                load_retrieval_index(self.config.retrieval_index_path,
                                     self.config.chunks_output_path))

            # Parse input evaluation dataset from the provided JSON path
            dataset = load_dataset(Path(dataset_path))

            # Stores MinimalSearchResults instances
            search_results = []

            # Process each dataset question sequentially
            #   with a progress bar display
            for question in tqdm(dataset.rag_questions, desc="Searching"):
                # Perform lexical search and extract retrieved source objects.
                # Since search_chunks returns the list of Chunks and the list
                #   with MinimalSource, we can simply use the second field
                #   since StudentSearchResults expects a list of MinimalSource
                #   for the retrieved_sources field in MinimalSearchResults
                sources = search_chunks(question.question, bm25, chunks, k)[1]

                search_results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources))

            # Encapsulate all search results in
            #   StudentSearchResults container model
            student_search_results = StudentSearchResults(
                search_results=search_results, k=k)

            # Build output path maintaining dataset filename
            #   and save results to JSON.
            output_path = Path(save_directory) / Path(dataset_path).name
            save_json(output_path, student_search_results)

            print(f"Saved student_search_results to {output_path}")

        except ValueError as error:
            print(f"ERROR: {error}")

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

        try:
            # If query string is empty, raise ValueError
            if not query.strip():
                raise ValueError("Cannot generate answer for an empty query")

            # Load the BM25 index and corresponding chunk dataset from disk
            bm25, chunks = (
                load_retrieval_index(self.config.retrieval_index_path,
                                     self.config.chunks_output_path))

            # Perform lexical search and extract retrieved source objects.
            # For both answer and answer_dataset, both fields in the tuple
            #   are needed
            chunks_found, sources = search_chunks(query, bm25, chunks, k)

            # Load the text generation model pipeline specified
            model = load_model(self.config.model_name)

            # Combine retrieved chunks into a bounded prompt context string
            context = augment_context(chunks_found,
                                      self.config.max_context_length)

            # Generate answer from query and context using parameters
            answer_text = (
                generate_answer(query, context, model,
                                self.config.generation_max_new_tokens,
                                self.config.generation_temperature,
                                self.config.generation_top_p))

            # Encapsulate query, sources, and generated answer in
            #   StudentSearchResultsAndAnswer class model
            result = StudentSearchResultsAndAnswer(
                search_results=[MinimalAnswer(
                    question_id="question_query",
                    question=query,
                    retrieved_sources=sources,
                    answer=answer_text)], k=k)

            print(result.model_dump_json(indent=4))

        except ValueError as error:
            print(f"ERROR: {error}")

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

        try:

            # Load the text generation model pipeline specified
            model = load_model(self.config.model_name)

            # Load search results JSON and validate them
            #   with a StudentSearchResults Pydantic model
            data = load_json(Path(student_search_results_path))
            results = StudentSearchResults.model_validate(data)

            # Load the BM25 index and corresponding chunk dataset from disk
            chunks = load_retrieval_index(self.config.retrieval_index_path,
                                          self.config.chunks_output_path)[1]

            # Build a index dictionary mapping the source to the chunk objects
            chunk_index = build_chunk_index(chunks)

            # Stores MinimalAnswer model instances
            answers = []

            # Iterates through search results sequentially
            #   with a progress bar display
            for result in tqdm(results.search_results,
                               desc="Generating answers"):

                # Handle empty query text.
                # To not stop the entire dataset from getting
                #   answers generated, if queries are empty,
                #   an error placeholder will be added instead
                if not result.question.strip():
                    answer_text = "ERROR: empty query"

                    print(f"WARNING: skipping empty query for "
                          f"question_id={result.question_id}")

                else:
                    # Retrieve chunks matching source from matching index
                    matched_chunks = match_chunks(result.retrieved_sources,
                                                  chunk_index)

                    # Combine matched chunks into a prompt context string
                    context = augment_context(matched_chunks,
                                              self.config.max_context_length)

                    # Generate answer using model pipeline and parameters
                    answer_text = (
                        generate_answer(
                            result.question, context, model,
                            self.config.generation_max_new_tokens,
                            self.config.generation_temperature,
                            self.config.generation_top_p))

                # Store query, retrieved sources, and generated answer
                #   in MinimalAnswer model
                answers.append(MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text))

            # Wrap generated answers in StudentSearchResultsAndAnswer
            #   container class model
            results_answers = StudentSearchResultsAndAnswer(
                search_results=answers, k=results.k)

            # Determine output path maintaining original
            #   dataset filename and save to JSON
            output_path = (
                Path(save_directory) / Path(student_search_results_path).name)

            save_json(output_path, results_answers)

            print(f"Saved student_search_results_and_answer to {output_path}")

        except ValidationError as error:
            print(f"ERROR: {error.errors()[0]}")
        except ValueError as error:
            print(f"ERROR: {error}")

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str,
                 k: int = 10,
                 max_context_length: int = 2000) -> None:
        """
        Evaluates retrieved sources against ground truth annotations.

        Args:
            student_search_results_path (str): Path to a
                StudentSearchResults JSON file.
            dataset_path (str): Path to the AnsweredQuestions dataset.
            k (int): Maximum k for recall reporting.
            max_context_length (int): Kept for parity with the
                moulinette CLI signature; unused in the recall
                computation itself.

        Returns:
            None
        """

        try:
            # Ensure top-k count is a positive integer
            if k <= 0:
                raise ValueError("k must be positive integer")

            # Load search results JSON and validate into
            #   StudentSearchResults Pydantic model
            data = load_json(Path(student_search_results_path))
            student_results = StudentSearchResults.model_validate(data)

            # Load reference dataset JSON and validate into
            #   RagDataset Pydantic model
            reference_data = load_json(Path(dataset_path))
            reference_dataset = RagDataset.model_validate(reference_data)

            # Filter dataset to extract valid questions entries
            reference_questions = []
            for question in reference_dataset.rag_questions:
                if isinstance(question, AnsweredQuestion):
                    reference_questions.append(question)

            # Index Dictionary for evaluation matching
            student_sources_by_id = {}

            # Map each question ID to its retrieved sources
            for res in student_results.search_results:
                student_sources_by_id[res.question_id] = res.retrieved_sources

            # Determine evaluation cutoff thresholds, k-values,
            #   up to specified k
            k_values = []

            for k_value in (1, 3, 5, 10):
                if k_value <= k:
                    k_values.append(k_value)

            # If specified k isn't in k_values, added it
            if k not in k_values:
                k_values.append(k)

            # Determine Recall@K metrics against truth sources
            results = evaluate_search_results(student_sources_by_id,
                                              reference_questions,
                                              k_values)

            # Print structured evaluation results summary
            print("Evaluation Results")
            print("=" * 40)
            print(f"Questions evaluated: "
                  f"{results['questions_evaluated']}")
            print()

            for k_value in k_values:
                print(f"Recall@{k_value}: "
                      f"{results[f'recall@{k_value}']:.3f}")

        except ValidationError as error:
            print(f"ERROR: {error.errors()[0]['msg']}")
        except ValueError as error:
            print(f"ERROR: {error}")
