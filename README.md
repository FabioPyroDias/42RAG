*This project has been created as part of the 42 curriculum by fda-cruz*

## Description

RAG against the machine implements a Retrieval-Augmented Generation pipeline over the vLLM 0.10.1 codebase and documentation.

Documents are chunked, indexed with BM25, and retrieved by keyword relevance. Retrieved chunks are then passed as context to a small local Large Language Model, LLM, which generates a grounded answer to the user's question.

Retrieval and generation are evaluated separately: retrieval quality is measured with Recall@k against a labeled dataset of questions, and generation is inspected manually against expected reference answers.

## Requirements

Make sure `make` is installed on your system:

```bash
sudo apt install make
```

Python 3.10 or higher is required. Check your version with:

```bash
python3 --version
```

Lastly, `uv` is also required:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

A virtual environment (`rag`) will be created automatically during installation. This ensures project dependencies are isolated.

## Instructions

### Installation

To install project dependencies, simply run `make install` in the terminal.
This will create the virtual environment and install the required Python packages (pydantic, fire, bm25s, transformers, etc).

### Execution

The pipeline is run in sequential stages, each one consuming the output of the previous one.

**1. Indexing** - chunks the raw repository and builds the BM25 index:

```bash
uv run python -m student index --max_chunk_size 2000
```

**2. Searching a full dataset** - runs retrieval for every question in a dataset and saves the retrieved sources:

```bash
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results \
  --k 10
```

**3. Generating answers for a full dataset** - consumes the output of `search_dataset` and generates an answer per question:
```bash
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

Additionally, two more methods are available.

**4. Searching a single query**:
```bash
uv run python -m student search "How to configure OpenAI server?"
```

**5. Answering a single query**:
```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10
```

## Technical Overview

### System architecture

The pipeline processes documents and queries in distinct stages:

1. **Chunking** - Every file under the raw repository is split into chunks capped at `max_chunk_size` characters, preserving `file_path` and character offsets for each chunk.
2. **Indexing** - Chunks are indexed with a BM25 model (`bm25s`), enabling fast keyword retrieval.
3. **Retrieval** - A query is scored against the BM25 index. The top `k` chunks are returned as `MinimalSource` references, file path + character offsets.
4. **Context augmentation** - The retrieved chunks are resolved back to their text and concatenated into a single context string, capped at `max_context_length` characters.
5. **Generation** - The context and question are passed to a local LLM (`Qwen/Qwen3-0.6B`) with a system prompt constraining the model to answer only from the provided context.

### Chunking Strategy
 
Documents under the raw repository are split line by line, accumulating lines into a chunk until adding the next line would exceed `max_chunk_size` characters. When that threshold is reached, the current chunk is closed and a new one starts from that line. Each chunk records its source `file_path`, the `first_character_index`, and the `last_character_index` it spans in the original file, so it can be traced back to the exact source location.
 
It works uniformly across markdown docs and Python source files without needing format specific parsers.
The trade-off is that a chunk boundary can fall in the middle of a logical unit such as a function or a paragraph, if that unit happens to exceed the `max_chunk_size` limit.
 
### Retrieval Method
 
Retrieval uses BM25, a sparse, keyword-based ranking algorithm.

Each chunk is treated as a document in the BM25 index. A query is tokenized and scored against every indexed chunk using term frequency, inverse document frequency, and both the `bm25_k1` and `bm25_b` parameters (term-frequency saturation and document-length normalization, respectively). The top `k` highest-scoring chunks are returned as the retrieved context for a given question.

BM25 was chosen because it directly rewards exact keyword and identifier matches, relevant here since many questions target specific function names, flags, or API endpoints that a purely semantic match could paraphrase away.

Hybrid or semantic retrieval was considered but left out of scope, as it falls under the project's bonus features, which were not implemented.


### Performance Analysis

The retrieval pipeline successfully meets and exceeds all defined performance benchmarks:

**Documentation Benchmark:** Achieved **0.85 Recall@5** (Requirement: >= 0.80)

| Metric | Result |
|--------|--------|
| Recall@1 | 0.57 |
| Recall@3 | 0.81 |
| Recall@5 | 0.85 |
| Recall@10 | 0.89 |

**Code Benchmark:** Achieved **0.57 Recall@5** (Requirement: >= 0.50)

#### Code

| Metric | Result |
|--------|--------|
| Recall@1 | 0.37 |
| Recall@3 | 0.51 |
| Recall@5 | 0.57 |
| Recall@10 | 0.62 |

### Design Decisions

**BM25** - the mandatory scope of the project only requires keyword based retrieval.

Semantic embeddings and query expansion were considered but left out as they fall under the bonus scope, which was not implemented.

**Chunk resolution via lookup, not re-reading source files** - `MinimalSource` references, used in saved search results, only store file paths and character offsets, not text.
When generating answers for a saved dataset, `answer_dataset`, the original chunk text is recovered by looking up the reference against the previously indexed `ChunkCollection`, rather than re-reading files from disk.
This avoids depending on the raw repository still being present or unchanged at generation time.

**Thinking mode disabled** - `Qwen3-0.6B` is a reasoning model that by default emits a `<think>...</think>` block before its answer. This is disabled at generation time by adding "/no_think" at the end of the prompt, so answers stay concise and aligned with the reference answer format.

**Low default generation temperature** - `generation_temperature` defaults to `0.1` with `do_sample=True`, favoring answers that stay close to the retrieved context over creative variation, which matters more for factual grounding than for fluency.

**Question error isolation** - failure to generate an answer for one question, being an empty query or a generation error, does not abort the batch.
The failing entry is recorded with an `"ERROR: ..."` placeholder and a warning is printed, keeping the output aligned with the input question count.


### Challenges Faced

**Reasoning model output leaking into answers** - `Qwen3-0.6B` emits an internal `<think>` reasoning block by default. Left unhandled, this either got truncated by `max_new_tokens` before the model reached its actual answer, or leaked raw reasoning text into the final output.
Solved by disabling thinking mode previously referenced.

**Sampling parameters silently ignored** - `temperature` and `top_p` have no effect unless `do_sample=True` is explicitly set. This was verified by comparing outputs across temperature values before and after the fix.

### Example Usage

```bash
make install
make index
make search_dataset
make answer_dataset
make evaluate
```

### Project Structure

```text
.
├── data
│   ├── datasets
│   │   ├── AnsweredQuestions
│   │   │   ├── dataset_code_public.json
│   │   │   └── dataset_docs_public.json
│   │   └── UnansweredQuestions
│   │       ├── dataset_code_public.json
│   │       └── dataset_docs_public.json
│   ├── output
│   │   ├── search_results
│   │   │   ├── dataset_code_public.json
│   │   │   └── dataset_docs_public.json
│   │   └── search_results_and_answer
│   │       └── dataset_docs_public.json
│   ├── processed
│   │   ├── bm25_index
│   │   │   ├── data.csc.index.npy
│   │   │   ├── indices.csc.index.npy
│   │   │   ├── indptr.csc.index.npy
│   │   │   ├── params.index.json
│   │   │   └── vocab.index.json
│   │   └── chunks
│   └── raw
│       └── vllm-0.10.1
├── Makefile
├── moulinette
│   ├── moulinette_pkg
│   │   ├── moulinette-fedora
│   │   ├── moulinette-ubuntu
│   │   └── README.md
│   └── README_Moulinette.md
├── pyproject.toml
├── README.md
├── src
│   ├── command_line_interface.py
│   ├── config.py
│   ├── consts.py
│   ├── evaluation
│   │   ├── evaluator.py
│   │   └── __init__.py
│   ├── file_manager.py
│   ├── generation
│   │   ├── augmenter.py
│   │   ├── generator.py
│   │   └── __init__.py
│   ├── indexing
│   │   ├── chunking.py
│   │   ├── indexer.py
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── models.py
│   └── retrieval
│       ├── __init__.py
│       └── retriever.py
└── uv.lock
```

## Resources

### Retrieval-Augmented Generation
- [BM25S](https://github.com/xhluca/bm25s) - The BM25 retrieval library used for indexing and search.

### Language Model
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) - The underlying language model used for answer generation.

### Data Validation
- [Pydantic](https://docs.pydantic.dev/) - Used for all input/output model validation.

## Use of AI

Claude was used early on to help structure the project and clarify conceptual questions.

Gemini was used to assist with project documentation and concept research.

The overall flow and structure of the project were already planned by the student, but AI was used to double-check that nothing important was being left out during the implementation.