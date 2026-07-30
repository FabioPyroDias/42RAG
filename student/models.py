"""Data models for type-safe RAG pipeline data handling.

This module defines the Pydantic models used to validate, serialize, and
deserialize documents, questions, search results, and generated answers
across the RAG pipeline.
"""

from pydantic import BaseModel, Field
from typing import List, Sequence
import uuid


class MinimalSource(BaseModel):
    """Represents a minimal source of information.

    This model points to a specific text span using its start and
    end character indices, rather than storing the actual text.

    Attributes:
        file_path (str): Path to the source file where the text span resides.
        first_character_index (int): Starting character index of the
            retrieved text span.
        last_character_index (int): Ending character index of the
            retrieved text span.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents an unanswered question.

    Holds the original question text before
    search retrieval or answer generation.

    Attributes:
        question_id (str): Unique identifier for the question.
        question (str): The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents an answered question.

    Extends UnansweredQuestion, adding the sources and the answer.

    Attributes:
        sources (List[MinimalSource]): List of sources used to
            answer the question.
        answer (str): The answer text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG questions.

    Attributes:
        rag_questions (List[AnsweredQuestion | UnansweredQuestion]):
            List of all questions in the dataset.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent the search results.

    Stores search results for a single question.

    Attributes:
        question_id (str): The unique identifier of the question.
            The same identifier of the original question (UnansweredQuestion).
        question (str): The text of the question.
            The same text of the original question (UnansweredQuestion).
        retrieved_sources (List[MinimalSource]): List of sources found
            by the search.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent an answer to a question.

    Extends MinimalSearchResults, adding the answer text for the question.

    Attributes:
        answer (str): The text of the generated answer.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Represent search results.

    Stores the collection of search results for all questions.

    Attributes:
        search_results (Sequence[MinimalSearchResults]): Sequence of search
            results for each question.
        k (int): The number of sources to retrieve per question.
    """

    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Represent search results with answers.

    Extends StudentSearchResults, adding answers to the search results.

    Attributes:
        search_results (Sequence[MinimalAnswer]): Sequence of search results
            with answers for each question.
    """

    search_results: Sequence[MinimalAnswer]
