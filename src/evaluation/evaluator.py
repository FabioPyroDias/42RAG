"""Evaluates retrieval quality against ground truth annotations.

Implements the recall@k metric where a retrieved source counts
as "found" for a correct source if they share the same file and their
character ranges overlap by at least 5% of the correct source's length.
A question's score is the number of correct sources found divided by
the total number of correct sources for that question.
"""

from typing import List
from src.models import MinimalSource, AnsweredQuestion


def determine_overlap_ratio(retrieved: MinimalSource,
                            correct: MinimalSource) -> float:
    """
    Calculates the character-range overlap ratio between two sources.

    The ratio is the overlapping character span divided by the length
    of the correct (ground truth) source, since that is the source
    the retrieval is being scored against finding.

    Args:
        retrieved (MinimalSource): A source returned by retrieval.
        correct (MinimalSource): A ground truth source.

    Returns:
        float: Overlap ratio in [0, 1]. 0 if the sources are in
            different files or do not overlap at all.
    """

    # Verifies if retrieved file is the same as the correct one.
    # If not, no need to test overlap
    if retrieved.file_path != correct.file_path:
        return 0.0

    # Finds where the shared region begins by selecting the
    #   furthest starting point in the text
    overlap_start = max(retrieved.first_character_index,
                        correct.first_character_index)

    # Finds where the shared region ends by selecting the
    #   earliest ending point
    overlap_end = min(retrieved.last_character_index,
                      correct.last_character_index)

    # Subtracts the start from the end to obtain the number of
    #   shared characters.
    # If the chunks do not overlap, this subtraction yields a
    #   negative number, which max forces to 0
    overlap_length = max(0, overlap_end - overlap_start)

    # Calculates the total character length of the expected answer chunk
    correct_length = (correct.last_character_index -
                      correct.first_character_index)

    # If the reference chunk is invalid or empty,
    #   it prevents a Python error and returns 0.0
    if correct_length <= 0:
        return 0.0

    # Divides the overlap span by the size of the correct chunk.
    # The result represents the fraction of relevant text
    #   successfully returned by the retriever
    return overlap_length / correct_length


def determine_question_recall(retrieved_sources: List[MinimalSource],
                              correct_sources: List[MinimalSource],
                              k: int) -> float:
    """
    Determines the recall score for a single question.

    Args:
        retrieved_sources (List[MinimalSource]): Sources returned by
            retrieval, ranked by relevance.
        correct_sources (List[MinimalSource]): Ground truth sources
            for this question.
        k (int): Only the top-k retrieved sources are considered.

    Returns:
        float: number of correct sources found divided by the total
            number of correct sources. 0.0 if there are no correct
            sources for this question.
    """

    # Checks if there are any ground-truth sources for the question.
    #   If none exist, it immediately returns 0.0 to prevent errors
    if not correct_sources:
        return 0.0

    # Float between 0.0 and 1.0 defining the minimum ratio of a ground-truth
    # source that must be covered by a retrieved chunk to count as a hit
    overlap_threshold = 0.05

    # Slices the ranked retrieval results to isolate only the top k sources
    top_k_sources = retrieved_sources[:k]

    # Tracks the total number of unique ground-truth sources successfully
    #   recovered by the retriever
    found_count = 0

    # Iterate over every single correct source
    # After, each top retrieved source is cross compared and checked with
    #   determine_overlap_ratio. If it exceeds the threshold required,
    #   this source is conseidered as a valid source
    for correct in correct_sources:
        found = False

        for retrieved in top_k_sources:
            if (determine_overlap_ratio(retrieved, correct)
               >= overlap_threshold):
                found = True
                break

        if found:
            found_count += 1

    # Computes the recall score for this question as the ratio of
    #   successfully matched ground-truth sources to total expected sources
    return found_count / len(correct_sources)


def evaluate_search_results(retrieved_sources_dict: dict[str,
                                                         List[MinimalSource]],
                            reference_questions: List[AnsweredQuestion],
                            k_values: List[int]) -> dict[str, float]:
    """
    Evaluates recall@k for a set of k values across all questions.

    Args:
        retrieved_sources_dict (dict): Mapping of question_id to the
            list of MinimalSource retrieved.
        reference_questions (List[AnsweredQuestion]): Ground truth
            questions, each with its correct sources.
        k_values (List[int]): The k values to report recall for.

    Returns:
        dict: Mapping of "recall@{k}" to the average recall score
            across all evaluated questions, plus "questions_evaluated".
    """

    # Stores the final evaluation recall metrics
    # These will be less or equal to k and the final key will hold the
    #   total number of questions processed
    results = {}
    questions_evaluated = 0

    # Iterate over each k value
    for k in k_values:
        # Accumulator list storing recall scores for all
        #   individual questions at current depth k
        scores = []

        # Process each ground truth question in the reference dataset
        for question in reference_questions:
            # List of sources retrieved for the current question_id
            retrieved_sources = retrieved_sources_dict.get(
                question.question_id)

            # Skip questions that have no corresponding submission
            if retrieved_sources is None:
                continue

            # Individual recall score for this single question at depth k
            score = determine_question_recall(retrieved_sources,
                                              question.sources, k)

            scores.append(score)

        # Track total evaluated questions on the first k iteration
        questions_evaluated = len(scores)

        # Determine average recall across all evaluated questions for depth k
        # If no question was evaluated, then the result is 0
        if scores:
            results[f"recall@{k}"] = sum(scores) / len(scores)
        else:
            results[f"recall@{k}"] = 0.0

    # Add total count of processed questions to the final output
    results["questions_evaluated"] = questions_evaluated

    return results
