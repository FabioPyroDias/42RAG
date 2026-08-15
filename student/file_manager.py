"""File manager operations for the RAG pipeline.

Handles reading source code, recursively searching directories and
JSON input and output.
"""

from typing import Any, List
from pathlib import Path
from student.models import RagDataset
from pydantic import BaseModel, ValidationError
from student.consts import TEXT_EXTENSIONS, CODE_EXTENSIONS
import json


def load_text_file(path: Path) -> str:
    """
    Reads the entire content of a text file into a string.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The file content

    Raises:
        ValueError
    """

    try:
        with open(path, "r", encoding="utf-8") as fd:
            return fd.read()
    except FileNotFoundError:
        raise ValueError(f"File {path} not found")
    except PermissionError:
        raise ValueError(f"Permission denied to file {path}")
    except IsADirectoryError:
        raise ValueError(f"{path} is a directory, not a file")
    except TypeError:
        raise ValueError(f"{path} expected to be a Path")
    except UnicodeDecodeError:
        raise ValueError(f"File {path} is not valid UTF-8 text")
    except OSError as error:
        raise ValueError(f"Unexpected OS error {error}")


def get_indexable_files(path: Path) -> List[Path]:
    """
    Recursively retrieves all indexable code and text files from a directory.

    Args:
        path (Path): The root directory path to search.

    Returns:
        List[Path]: A list of Path objects for all matching files.

    Raises:
        ValueError
    """

    if not path.exists():
        raise ValueError(f"Directory {path} not found")

    if not path.is_dir():
        raise ValueError(f"{path} is a file, not a directory")

    # Recursively walks through path and all of its subdirectories,
    #   matching every entry, both files and directories, against the "*"
    #   pattern, which matches any name.
    # Returns an iterator,
    #   so nothing is loaded into memory until we consume it below.
    files = path.rglob("*")

    indexable_files = []

    for file in files:
        if file.is_file():
            if file.suffix in (TEXT_EXTENSIONS | CODE_EXTENSIONS):
                indexable_files.append(file)

    return indexable_files


def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path (Path): The path to the file.

    Returns:
        Any: dict formatted JSON file.

    Raises:
        ValueError
    """

    try:
        with open(path, "r", encoding="utf-8") as fd:
            return json.load(fd)
    except FileNotFoundError:
        raise ValueError(f"File {path} not found")
    except PermissionError:
        raise ValueError(f"Permission denied to file {path}")
    except IsADirectoryError:
        raise ValueError(f"{path} is a directory, not a file")
    except TypeError:
        raise ValueError(f"{path} expected to be a Path")
    except UnicodeDecodeError:
        raise ValueError(f"File {path} is not valid UTF-8 text")
    except OSError as error:
        raise ValueError(f"Unexpected OS error {error}")
    except json.JSONDecodeError:
        raise ValueError(f"JSON file {path} not properly formatted")


def load_dataset(path: Path) -> RagDataset:
    """
    Loads a JSON dataset and validates using the RagDataset model.

    Args:
        path (Path): The Path to the file.

    Returns:
        RagDataset: The validated Pydantic model containing the dataset.

    Raises:
        ValueError
    """

    try:
        return RagDataset.model_validate(load_json(path))
    except ValidationError as error:
        raise ValueError(f"Dataset in {path} does not match the "
                         f"expected format: {error}")


def save_json(path: Path, data: BaseModel) -> None:
    """
    Serialize and saves a Pydantic BaseModel to a JSON file.

    Args:
        path (Path): Path where the output JSON file will be saved.
        data (BaseModel): Validated model to be written.

    Returns:
        None

    Raises:
        ValueError
    """

    try:
        # path.parent returns the directory containing the file,
        #   as a Path object.
        directory = path.parent

        # -> if the directory doesn't exist yet, create it and any missing
        #   parent directories
        if not directory.exists():
            directory.mkdir(parents=True)

        with open(path, "w") as fd:
            # Returns a standard Python dict.
            # Using mode='json' serializes the data, meaning it
            #   translates complex memory structures into
            #   JSON-safe formats so that json.dumps() can safely
            #   convert it to text later.
            formatted_data = data.model_dump(mode='json', by_alias=True)

            # Converts the Python dict into a
            #   formatted JSON string, adding 4 spaces of
            #   indentation for readability matching the
            #   desired output by the subject standards.
            converted_json = json.dumps(formatted_data, indent=4)

            fd.write(converted_json)

    except PermissionError:
        raise ValueError(f"No permission to open the file at {path}")
    except FileNotFoundError:
        raise ValueError(f"\"{path}\" file does not exist")
    except IsADirectoryError:
        raise ValueError(f"\"{path}\" expected file, received directory")
    except TypeError:
        raise ValueError(f"{path} expected to be a Path")
    except OSError as error:
        raise ValueError(f"Unexpected OS error {error}")
