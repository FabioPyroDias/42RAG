export UV_PROJECT_ENVIRONMENT = rag

UV_CACHE_DIR_SG = /sgoinfre/$(USER)/uv-cache
UV_ENV_DIR_SG = /sgoinfre/$(USER)/venvs/$(UV_PROJECT_ENVIRONMENT)
HF_HOME_SG = /sgoinfre/$(USER)/huggingface

PYTHON = UV_PROJECT_ENVIRONMENT=$(UV_PROJECT_ENVIRONMENT) uv run python -m

MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs --explicit-package-bases \
		--namespace-packages

RM = rm -rf

install:
	@if [ -d /sgoinfre ]; then \
		echo "Using /sgoinfre for storage..."; \
		mkdir -p $(UV_CACHE_DIR_SG) /sgoinfre/$(USER)/venvs $(HF_HOME_SG); \
		UV_CACHE_DIR=$(UV_CACHE_DIR_SG) \
		UV_PROJECT_ENVIRONMENT=$(UV_ENV_DIR_SG) \
		UV_LINK_MODE=copy \
		uv sync; \
		ln -sfn $(UV_ENV_DIR_SG) $(UV_PROJECT_ENVIRONMENT); \
	else \
		UV_PROJECT_ENVIRONMENT=$(UV_PROJECT_ENVIRONMENT) uv sync; \
	fi

run: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) src; \
	else \
		$(PYTHON) src; \
	fi

debug: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) pdb -m src; \
	else \
		$(PYTHON) pdb -m src; \
	fi

clean:
	$(RM) .mypy_cache/
	$(RM) src/__pycache__
	$(RM) src/indexing/__pycache__
	$(RM) src/retrieval/__pycache__
	$(RM) src/generation/__pycache__
	$(RM) src/evaluation/__pycache__

lint: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) flake8; \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) mypy $(MYPY_FLAGS) src; \
	else \
		$(PYTHON) flake8; \
		$(PYTHON) mypy $(MYPY_FLAGS) src; \
	fi

lint-strict: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) flake8; \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) mypy --strict src; \
	else \
		$(PYTHON) flake8; \
		$(PYTHON) mypy --strict src; \
	fi

destroy: clean
	$(RM) rag
	$(RM) .venv

index: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) src index 2000; \
	else \
		$(PYTHON) src index 2000; \
	fi

search: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) src search "How to configure OpenAI server?"; \
	else \
		$(PYTHON) src search "How to configure OpenAI server?"; \
	fi

search_dataset: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--save_directory data/output/search_results --k 10; \
	else \
		$(PYTHON) src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--save_directory data/output/search_results --k 10; \
	fi

answer: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) src answer "How to configure OpenAI server?" --k 10; \
	else \
		$(PYTHON) src answer "How to configure OpenAI server?" --k 10; \
	fi

answer_dataset: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer; \
	else \
		$(PYTHON) src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer; \
	fi

evaluate: install
	@if [ -d /sgoinfre ]; then \
		HF_HOME=$(HF_HOME_SG) UV_CACHE_DIR=$(UV_CACHE_DIR_SG) $(PYTHON) src evaluate \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10; \
	else \
		$(PYTHON) src evaluate \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10; \
	fi