"""
urdunlp
~~~~~~~

Urdu NLP toolkit — sentence segmentation, boundary evaluation, and domain
adaptation for Urdu text.

Quick start::

    from urdunlp import segment

    sentences = segment("میں گھر گیا۔ وہ باہر تھا۔")
    # ['میں گھر گیا', 'وہ باہر تھا']

To improve accuracy on your own domain, learn endwords and conjunctions from
a hand-labeled reference text first::

    from urdunlp import segment, learn_from_text

    endwords, conjunctions = learn_from_text(my_labeled_text)
    sentences = segment(my_raw_text, endwords=endwords, conjunctions=conjunctions)
"""

from urdunlp.segmenter import segment, learn_from_text, SENTENCE_DELIMITER
from urdunlp.evaluate import boundary_accuracy, corpus_accuracy

__version__ = "0.1.0"
__author__ = "Manha Zamir"
__email__ = "manhazamir@gmail.com"
__license__ = "MIT"

__all__ = [
    "segment",
    "learn_from_text",
    "boundary_accuracy",
    "corpus_accuracy",
    "SENTENCE_DELIMITER",
]
