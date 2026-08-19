"""Configuration settings for the RAG pipeline.

Defines default parameters for indexing, retrieval, and generation.
"""

from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Configuration parameters for the RAG pipeline.

    Stores default paths and settings for indexing, retrieval, and generation.

    Attributes:
            raw_repository_path (str): Directory of the repository
                to be processed by the RAG
            datasets_root_path (str): Directory containing test datasets.
            chunks_output_path (str): Directory to save processed chunks.
            retrieval_index_path (str): Directory to save the BM25 index.
            output_search_path (str): Directory to save search results.
            output_answer_path (str): Directory to save generated answers.

            max_chunk_size (int): Maximum character length per chunk.
            bm25_k1 (float): BM25 term frequency saturation parameter.
                Determines how much repeating a search term increases
                    the document's score.
                Higher values reward documents that repeat the word
                    many times, while lower values cap the score quickly.
            bm25_b (float): BM25 document length normalization parameter.
                Controls whether long documents are penalized for being long.
                A higher value strongly favors shorter, concise documents,
                    while a lower value completely ignores document length
                    in the score.

            max_context_length (int): Maximum character length of
                context passed to the LLM.

            model_name (str): LLM identifier for answer generation.
            generation_temperature (float): Sampling temperature for
                generation.
                Controls the randomness of the LLM's output.
                Lower values make the answers strictly deterministic
                    and focused, while higher values make the output more
                    creative and varied.
            generation_top_p (float): Nucleus sampling parameter.
                Filters out unlikely words during text generation.
                A lower value restricts the model to choosing only from the
                    most probable next words, while a higher value considers
                    a wider vocabulary
            generation_max_new_tokens (int): Maximum new tokens to generate.
    """

    raw_repository_path: str = Field(default="data/raw/vllm-0.10.1")
    datasets_root_path: str = Field(default="data/datasets")
    processed_path: str = Field(default="data/processed")
    chunks_output_path: str = Field(default="data/processed/chunks")
    retrieval_index_path: str = Field(default="data/processed/bm25_index")
    output_search_path: str = Field(default="data/output/search_results")
    output_answer_path: str = Field(default="data/output/"
                                            "search_results_and_answer/")

    max_chunk_size: int = Field(default=2000, ge=100, le=2000)
    bm25_k1: float = Field(default=1.5, ge=0.1, le=3.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    max_context_length: int = Field(default=10000, ge=2000, le=12000)

    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    generation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    generation_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    generation_max_new_tokens: int = Field(default=200, ge=100, le=500)
