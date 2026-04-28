# urdunlp

Urdu sentence segmentation that outperforms urduhack.

```bash
pip install urdu-segmentation-nlp
```

---

## The problem

Urdu has 230 million speakers and almost no production-ready NLP tooling. The dominant open-source library, [urduhack](https://github.com/urduhack/urduhack), implements sentence tokenisation — but its accuracy on real Urdu text leaves room to improve.

`urdunlp` is a lightweight, zero-dependency library built around a rule-based segmentation algorithm that was benchmarked against urduhack and outperformed it by ~7 percentage points on the same dataset.

| Model | Boundary accuracy |
|---|---|
| urduhack `sentence_tokenizer` | 74.1% |
| **urdunlp `segment`** | **81.5%** |

---

## Quickstart

```python
from urdunlp import segment

text = "وزیراعظم نے کہا کہ ملک ترقی کر رہا ہے۔ عوام خوشحال ہیں اور مستقبل روشن ہے۔"
sentences = segment(text)
# ['وزیراعظم نے کہا کہ ملک ترقی کر رہا ہے', 'عوام خوشحال ہیں اور مستقبل روشن ہے']
```

### Domain adaptation

The default endword and conjunction lists were learned from a news corpus. For better accuracy on your domain (legal, literary, conversational), pass a hand-labeled reference text to `learn_from_text` first:

```python
from urdunlp import segment, learn_from_text

with open("my_labeled_text.txt", encoding="utf-8") as f:
    labeled = f.read()

endwords, conjunctions = learn_from_text(labeled)
sentences = segment(my_raw_text, endwords=endwords, conjunctions=conjunctions)
```

### Evaluating accuracy

```python
from urdunlp import segment, boundary_accuracy

predicted = segment(raw_text)
reference = labeled_text.split("۔")

score = boundary_accuracy(predicted, reference)
print(f"Accuracy: {score:.1f}%")
```

For a corpus of documents:

```python
from urdunlp import corpus_accuracy

score = corpus_accuracy(predicted_corpus, reference_corpus)
```

---

## API reference

### `segment(text, endwords=None, conjunctions=None) → list[str]`

Segment Urdu text into sentences.

- `text` — raw Urdu string
- `endwords` — optional `set[str]` of sentence-ending words (uses built-in defaults if `None`)
- `conjunctions` — optional `set[str]` of continuation words (uses built-in defaults if `None`)

Returns a list of sentence strings, without the trailing `۔`.

---

### `learn_from_text(labeled_text, base_endwords=None, base_conjunctions=None) → tuple[set, set]`

Learn endwords and conjunctions from a hand-labeled text. The labeled text should use `۔` as the sentence boundary marker.

Returns `(endwords, conjunctions)` — pass these directly to `segment`.

---

### `boundary_accuracy(predicted, reference, window=3) → float`

Compute segmentation accuracy using a trailing-word boundary match strategy. Returns a percentage (0–100).

---

### `corpus_accuracy(predicted_corpus, reference_corpus, window=3) → float`

Mean `boundary_accuracy` across a list of (predicted, reference) document pairs.

---

## How it works

The segmentation algorithm works in two passes:

1. **Primary split** — divide on `۔` (the Urdu full stop), which handles the majority of sentence boundaries cleanly.

2. **Secondary rule** — after a primary split, check whether the final word of each chunk is a known *endword* (a verb form that typically closes a clause). If the word immediately following an endword is a *conjunction* or another *endword*, treat the boundary as a continuation rather than a full stop — this handles compound sentences and subordinate clauses that urduhack mis-segments.

The endword and conjunction lists are seeded with defaults learned from a news corpus and can be extended for any domain using `learn_from_text`.

---

## Development

```bash
git clone https://github.com/manhazamir/urdu-segmentation-nlp
cd urdunlp
pip install pytest
python -m pytest tests/ -v
```

---

## Research

This library packages the algorithm from original research comparing Urdu sentence segmentation approaches. The benchmark used a hand-labeled dataset of Urdu news paragraphs; accuracy was computed using boundary-precision matching on the final three words of each sentence.