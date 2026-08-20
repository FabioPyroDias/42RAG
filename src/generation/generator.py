"""Generates answers using augmented context prompts.

Interfaces with language models to produce responses
based on retrieved code and documentation.
"""

from typing import Any
from transformers import pipeline
from transformers.utils import logging as hf_logging


# Silence warnings from HuggingFace
hf_logging.set_verbosity_error()


def load_model(model_name: str) -> Any:
    """
    Loads the Hugging Face text-generation pipeline.

    Args:
        model_name (str): Hugging Face model identifier or local path.

    Returns:
        Any: The initialized transformers text-generation pipeline.

    Raises:
        ValueError
    """

    try:
        # Initialize the HuggingFace text-generation pipeline.
        generator = pipeline("text-generation", model=model_name)

        # Preserve exact whitespace/formatting,
        #   crucial for code generation & context prompts
        generator.tokenizer.clean_up_tokenization_spaces = False

        # Remove default max_length cap to prevent conflicts between
        #   long prompt contexts and max_new_tokens parameters
        #   during generation
        generator.model.generation_config.max_length = None
        return generator
    except OSError as error:
        raise ValueError(f"Unexpected OS error {error}")
    except ImportError as error:
        raise ValueError(f"Missing libraries {error}")
    except ValueError:
        raise ValueError(f"Failed to load model '{model_name}'. "
                         f"Check model path or network connection.")


def generate_answer(query: str, context: str,
                    generator: Any, max_tokens: int,
                    temperature: float, top_p: float) -> str:
    """
    Generates an answer to a query using a language model and context.

    Args:
        query (str): The question text to answer.
        context (str): The retrieved context.
        generator (Any): The initialized transformers text-generation
            pipeline.
        max_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature for generation.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The generated answer text.

    Raises:
        ValueError
    """

    # Define system instructions enforcing strict
    #   RAG grounding and fallback behavior
    system_prompt = (
        "You are a documentation assistant. Answer the question using "
        "ONLY the information in the provided context. Be concise and "
        "factual. If the context does not contain enough information "
        "to answer, say so explicitly instead of guessing."
        )

    # Format query with retrieved context and disable
    #   internal thinking tags "/no_think"
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    message = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": f"{prompt} /no_think"}]

    try:
        answer_text = ""

        # Invoke HuggingFace text generation pipeline with sampling parameters
        outputs = generator(message, max_new_tokens=max_tokens,
                            max_length=None,
                            temperature=temperature, top_p=top_p,
                            return_full_text=False, do_sample=True)

        # generator returns the following structure:
        # [
        #   {'generated_text': '<think>\n\n</think>\n\nThe HTTP endpoint
        #       used to dynamically load a LoRA adapter in vLLM is
        #       `http://localhost:8000/v1/load_lora_adapter`.'
        #   }
        # ]
        #
        # So we get the first index, with outputs[0] and gets the dict.
        #   After, we try to get the "generated_text" key with get(),
        #   since the key might not exist. This leaves generated_data with:
        #
        #   '<think>\n\n</think>\n\nThe HTTP endpoint used to dynamically
        #   load a LoRA adapter in vLLM is
        #   `http://localhost:8000/v1/load_lora_adapter`.'
        generated_data = outputs[0].get("generated_text", "")

        # We verifiy if it's a list and if it's not empty.
        if isinstance(generated_data, list) and generated_data:
            answer_text = generated_data[-1].get("content", "").strip()
        # In case it returns a str
        elif isinstance(generated_data, str):
            answer_text = generated_data.strip()
        else:
            raise ValueError("Unexpected output structure from "
                             "model pipeline.")

        # As of now, generated_data is a string without leading or trailing
        #   whitespaces, but "<think>\n\n</think>" still exists. Therefore,
        #   we need to remove that entire section in order to have
        #   the valid generated answer
        if "</think>" in answer_text:
            answer_text = answer_text.split("</think>", 1)[1].strip()

        if not answer_text:
            raise ValueError("Model generated an empty answer.")

        return answer_text

    except (ValueError, RuntimeError) as error:
        raise ValueError(f"Generation failed: {error}")
