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
            bm25, chunks = (
                load_retrieval_index(self.config.retrieval_index_path,
                                     self.config.chunks_output_path))

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
            bm25, chunks = (
                load_retrieval_index(self.config.retrieval_index_path,
                                     self.config.chunks_output_path))

            dataset = load_dataset(Path(dataset_path))

            search_results = []

            for question in tqdm(dataset.rag_questions, desc="Searching"):
                sources = search_chunks(question.question, bm25, chunks, k)[1]

                search_results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources))

            student_search_results = StudentSearchResults(
                search_results=search_results, k=k)

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
            if not query.strip():
                raise ValueError("Cannot generate answer for an empty query")

            bm25, chunks = (
                load_retrieval_index(self.config.retrieval_index_path,
                                     self.config.chunks_output_path))

            chunks_found, sources = search_chunks(query, bm25, chunks, k)

            model = load_model(self.config.model_name)

            context = augment_context(chunks_found,
                                      self.config.max_context_length)

            answer_text = (
                generate_answer(query, context, model,
                                self.config.generation_max_new_tokens,
                                self.config.generation_temperature,
                                self.config.generation_top_p))

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
            model = load_model(self.config.model_name)

            data = load_json(Path(student_search_results_path))
            results = StudentSearchResults.model_validate(data)

            chunks = load_retrieval_index(self.config.retrieval_index_path,
                                          self.config.chunks_output_path)[1]

            chunk_index = build_chunk_index(chunks)

            answers = []

            for result in tqdm(results.search_results,
                               desc="Generating answers"):
                if not result.question.strip():
                    answer_text = "ERROR: empty query"

                    print(f"WARNING: skipping empty query for "
                          f"question_id={result.question_id}")

                else:
                    matched_chunks = match_chunks(result.retrieved_sources,
                                                  chunk_index)

                    context = augment_context(matched_chunks,
                                              self.config.max_context_length)

                    answer_text = (
                        generate_answer(
                            result.question, context, model,
                            self.config.generation_max_new_tokens,
                            self.config.generation_temperature,
                            self.config.generation_top_p))

                answers.append(MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text))

            results_answers = StudentSearchResultsAndAnswer(
                search_results=answers, k=results.k)

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
            data = load_json(Path(student_search_results_path))
            student_results = StudentSearchResults.model_validate(data)

            reference_data = load_json(Path(dataset_path))
            reference_dataset = RagDataset.model_validate(reference_data)

            reference_questions = [
                question for question in reference_dataset.rag_questions
                if isinstance(question, AnsweredQuestion)
            ]

            student_sources_by_id = {
                result.question_id: result.retrieved_sources
                for result in student_results.search_results
            }

            k_values = [k_value for k_value in (1, 3, 5, 10)
                        if k_value <= k]
            if k not in k_values:
                k_values.append(k)

            results = evaluate_search_results(student_sources_by_id,
                                              reference_questions,
                                              k_values)

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
