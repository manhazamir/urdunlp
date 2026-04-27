"""
urdunlp.evaluate
~~~~~~~~~~~~~~~~

Accuracy metrics for sentence segmentation.

The matching strategy compares the last three words of each labeled sentence
against the segmented output — this is a boundary-precision approach that
tolerates minor tokenisation differences in the interior of sentences.
"""

from __future__ import annotations


def boundary_accuracy(
    predicted: list[str],
    reference: list[str],
    window: int = 3,
) -> float:
    """
    Compute boundary-precision accuracy between predicted and reference sentences.

    A predicted sentence is counted as correct if its final ``window`` words
    match the final ``window`` words of any reference sentence.  Sentences
    with fewer than ``window`` words are skipped (counted as errors).

    Parameters
    ----------
    predicted:
        List of sentences produced by the segmenter.
    reference:
        List of ground-truth sentences.
    window:
        Number of trailing words to compare (default: 3).

    Returns
    -------
    float
        Accuracy as a percentage (0–100).  Returns 0.0 if no valid reference
        sentences exist.

    Raises
    ------
    ValueError
        If either list is empty.
    """
    if not predicted or not reference:
        raise ValueError("Both predicted and reference sentence lists must be non-empty.")

    correct = 0
    valid = 0

    for ref_sentence in reference:
        ref_words = ref_sentence.split()
        if len(ref_words) < window:
            continue
        valid += 1
        ref_tail = ref_words[-window:]

        for pred_sentence in predicted:
            pred_words = pred_sentence.split()
            if len(pred_words) < window:
                continue
            if pred_words[-window:] == ref_tail:
                correct += 1
                break

    if valid == 0:
        return 0.0

    return (correct / valid) * 100


def corpus_accuracy(
    predicted_corpus: list[list[str]],
    reference_corpus: list[list[str]],
    window: int = 3,
) -> float:
    """
    Mean boundary accuracy across a corpus of (predicted, reference) pairs.

    Parameters
    ----------
    predicted_corpus:
        List of segmented sentence lists (one per document).
    reference_corpus:
        List of reference sentence lists (one per document).
    window:
        Trailing-word window passed to :func:`boundary_accuracy`.

    Returns
    -------
    float
        Mean accuracy across all valid documents.
    """
    if len(predicted_corpus) != len(reference_corpus):
        raise ValueError("predicted_corpus and reference_corpus must have the same length.")

    total = 0.0
    count = 0

    for pred, ref in zip(predicted_corpus, reference_corpus):
        try:
            total += boundary_accuracy(pred, ref, window=window)
            count += 1
        except ValueError:
            continue

    return total / count if count > 0 else 0.0
