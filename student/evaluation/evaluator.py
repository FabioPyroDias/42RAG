"""Evaluates retrieval quality against ground truth annotations.

Implements the recall@k metric described in the subject (VI.1.1):
a retrieved source counts as "found" for a correct source if they
share the same file and their character ranges overlap by at least
5% of the correct source's length. A question's score is the number
of correct sources found divided by the total number of correct
sources for that question.
"""

from typing import List
from student.models import MinimalSource, AnsweredQuestion


OVERLAP_THRESHOLD = 0.05


def compute_overlap_ratio(retrieved: MinimalSource,
                          correct: MinimalSource) -> float:
    """
    Computes the character-range overlap ratio between two sources.

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

    if retrieved.file_path != correct.file_path:
        return 0.0

    overlap_start = max(retrieved.first_character_index,
                        correct.first_character_index)
    overlap_end = min(retrieved.last_character_index,
                      correct.last_character_index)

    overlap_length = max(0, overlap_end - overlap_start)
    correct_length = (correct.last_character_index -
                      correct.first_character_index)

    if correct_length <= 0:
        return 0.0

    return overlap_length / correct_length


def compute_question_recall(retrieved_sources: List[MinimalSource],
                            correct_sources: List[MinimalSource],
                            k: int) -> float:
    """
    Computes the recall score for a single question.

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

    if not correct_sources:
        return 0.0

    top_k_sources = retrieved_sources[:k]

    found_count = 0

    for correct in correct_sources:
        found = False

        for retrieved in top_k_sources:
            if compute_overlap_ratio(retrieved, correct) >= OVERLAP_THRESHOLD:
                found = True
                break

        if found:
            found_count += 1

    return found_count / len(correct_sources)


def evaluate_search_results(student_sources_by_id: dict[str,
                                                        List[MinimalSource]],
                            reference_questions: List[AnsweredQuestion],
                            k_values: List[int]) -> dict[str, float]:
    """
    Computes recall@k for a set of k values across all questions.

    Args:
        student_sources_by_id (dict): Mapping of question_id to the
            list of MinimalSource retrieved by the student's system.
        reference_questions (List[AnsweredQuestion]): Ground truth
            questions, each with its correct sources.
        k_values (List[int]): The k values to report recall for.

    Returns:
        dict: Mapping of "recall@{k}" to the average recall score
            across all evaluated questions, plus "questions_evaluated".
    """

    results = {}
    questions_evaluated = 0

    for k in k_values:
        scores = []

        for question in reference_questions:
            retrieved_sources = student_sources_by_id.get(
                question.question_id)

            if retrieved_sources is None:
                continue

            score = compute_question_recall(retrieved_sources,
                                            question.sources, k)
            scores.append(score)

        if k == k_values[0]:
            questions_evaluated = len(scores)

        results[f"recall@{k}"] = (sum(scores) / len(scores)
                                  if scores else 0.0)

    results["questions_evaluated"] = questions_evaluated

    return results
