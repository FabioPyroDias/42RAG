export UV_PROJECT_ENVIRONMENT = rag

PYTHON = UV_PROJECT_ENVIRONMENT=$(UV_PROJECT_ENVIRONMENT) uv run python -m

MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs --explicit-package-bases \
		--namespace-packages

RM = rm -rf

install:
	UV_PROJECT_ENVIRONMENT=$(UV_PROJECT_ENVIRONMENT) uv sync

run: install
	$(PYTHON) student

debug:

clean:
	$(RM) .mypy_cache/
	$(RM) student/__pycache__
	$(RM) student/indexing/__pycache__

lint:
	$(PYTHON) flake8
	$(PYTHON) mypy $(MYPY_FLAGS) student

lint-strict:
	$(PYTHON) flake8
	$(PYTHON) mypy --strict student

destroy: clean
	$(RM) rag
	$(RM) .venv