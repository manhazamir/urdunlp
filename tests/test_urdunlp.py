"""
Tests for urdunlp — core segmentation and evaluation.
"""

import pytest
from urdunlp import segment, learn_from_text, boundary_accuracy, corpus_accuracy
from urdunlp.segmenter import _split_on_delimiter, SENTENCE_DELIMITER


# ---------------------------------------------------------------------------
# _split_on_delimiter
# ---------------------------------------------------------------------------

class TestSplitOnDelimiter:
    def test_empty_string(self):
        assert _split_on_delimiter("") == []

    def test_single_sentence(self):
        result = _split_on_delimiter("میں گھر گیا۔")
        non_empty = [r for r in result if r.strip()]
        assert len(non_empty) == 1
        assert "میں گھر گیا" in non_empty[0]

    def test_two_sentences(self):
        result = _split_on_delimiter("میں گھر گیا۔ وہ باہر تھا۔")
        non_empty = [r for r in result if r.strip()]
        assert len(non_empty) == 2

    def test_no_delimiter(self):
        result = _split_on_delimiter("میں گھر گیا")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------

class TestSegment:
    def test_returns_list(self):
        result = segment("میں گھر گیا۔")
        assert isinstance(result, list)

    def test_empty_input(self):
        assert segment("") == []

    def test_two_clear_sentences(self):
        text = "میں گھر گیا۔ وہ باہر تھا۔"
        result = segment(text)
        assert len(result) == 2

    def test_segments_do_not_contain_delimiter(self):
        text = "میں گھر گیا۔ وہ باہر تھا۔"
        for s in segment(text):
            assert SENTENCE_DELIMITER not in s

    def test_no_empty_strings_in_output(self):
        text = "میں گھر گیا۔ وہ باہر تھا۔"
        for s in segment(text):
            assert s.strip() != ""

    def test_custom_endwords(self):
        custom_endwords = {"تھا", "تھی", "ہے", "گیا"}
        result = segment("وہ گھر میں تھا۔ وہ باہر گیا۔", endwords=custom_endwords)
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# learn_from_text
# ---------------------------------------------------------------------------

class TestLearnFromText:
    LABELED = "میں گھر گیا۔ وہ باہر تھا اور کھیل رہا تھا۔"

    def test_returns_two_sets(self):
        endwords, conjunctions = learn_from_text(self.LABELED)
        assert isinstance(endwords, set)
        assert isinstance(conjunctions, set)

    def test_endwords_not_empty(self):
        endwords, _ = learn_from_text(self.LABELED)
        assert len(endwords) > 0

    def test_learns_from_delimiter_context(self):
        # "گیا" appears before ۔ so it should be learned as an endword
        endwords, _ = learn_from_text(self.LABELED)
        assert "گیا" in endwords

    def test_empty_text(self):
        endwords, conjunctions = learn_from_text("")
        # Should fall back to defaults
        assert len(endwords) > 0

    def test_custom_base_lists(self):
        endwords, conjunctions = learn_from_text(
            self.LABELED,
            base_endwords=["تھا"],
            base_conjunctions=["اور"],
        )
        assert "تھا" in endwords
        assert "اور" in conjunctions


# ---------------------------------------------------------------------------
# boundary_accuracy
# ---------------------------------------------------------------------------

class TestBoundaryAccuracy:
    def test_perfect_match(self):
        pred = ["میں گھر گیا", "وہ باہر تھا"]
        ref  = ["میں گھر گیا", "وہ باہر تھا"]
        assert boundary_accuracy(pred, ref) == 100.0

    def test_zero_match(self):
        pred = ["کچھ اور الفاظ یہاں"]
        ref  = ["میں گھر گیا تھا"]
        assert boundary_accuracy(pred, ref) == 0.0

    def test_empty_lists_raise(self):
        with pytest.raises(ValueError):
            boundary_accuracy([], ["میں گھر گیا تھا"])
        with pytest.raises(ValueError):
            boundary_accuracy(["میں گھر گیا تھا"], [])

    def test_returns_float(self):
        result = boundary_accuracy(["میں گھر گیا تھا"], ["میں گھر گیا تھا"])
        assert isinstance(result, float)

    def test_partial_match(self):
        pred = ["میں گھر گیا تھا", "وہ مختلف جگہ ہے"]
        ref  = ["میں گھر گیا تھا", "وہ گھر میں تھا"]
        acc = boundary_accuracy(pred, ref)
        assert 0.0 <= acc <= 100.0


# ---------------------------------------------------------------------------
# corpus_accuracy
# ---------------------------------------------------------------------------

class TestCorpusAccuracy:
    def test_perfect_corpus(self):
        pred_corpus = [["میں گھر گیا تھا"], ["وہ باہر کھیل رہا تھا"]]
        ref_corpus  = [["میں گھر گیا تھا"], ["وہ باہر کھیل رہا تھا"]]
        assert corpus_accuracy(pred_corpus, ref_corpus) == 100.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            corpus_accuracy([["ایک"]], [["ایک"], ["دو"]])

    def test_returns_float(self):
        result = corpus_accuracy(
            [["میں گھر گیا تھا"]],
            [["میں گھر گیا تھا"]],
        )
        assert isinstance(result, float)
