## Why this phase exists

- After tokenization, you have tokens (strings). Models need **numbers**.
- Text representation = converting tokens/documents into numeric vectors.
- "Sparse vector" = a vector that's mostly zeros, with a huge dimensionality equal to vocabulary size. This whole note is the classical (pre-deep-learning) approach — feeds directly into classical ML models (Naive Bayes, SVM, Logistic Regression).

---

## 1. One-Hot Encoding (theory only)

- Each word gets a vector of size = vocabulary size, with a single `1` at that word's index, rest `0`.
- Example: vocab = `[cat, dog, sat]` → "dog" = `[0, 1, 0]`
- **Limitation**: every word is equidistant from every other word — no notion of similarity ("cat" and "dog" are just as unrelated as "cat" and "sat"). No frequency info either. Rarely used standalone in practice; it's more a stepping stone concept before BoW.

---

## 2. Bag of Words (BoW)

**Working**:

1. Build a vocabulary of all unique words across the entire document collection (corpus).
2. Represent each document as a vector of length = vocab size.
3. Each position = **count** of how many times that word appears in the document.
4. Word order is completely discarded (hence "bag" — like dumping words in a bag, order lost).

**Example**:

Corpus:

- Doc1: "the cat sat"
- Doc2: "the dog sat"

Vocabulary: `[cat, dog, sat, the]`

|Doc|cat|dog|sat|the|
|---|---|---|---|---|
|Doc1|1|0|1|1|
|Doc2|0|1|1|1|

**Code:**

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["the cat sat", "the dog sat"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
# ['cat' 'dog' 'sat' 'the']
print(X.toarray())
# [[1 0 1 1]
#  [0 1 1 1]]
```

**Limitations**:

- Every word treated as equally important — "the" (no signal) counted same weight as "cat" (real signal).
- Vocabulary grows huge with corpus size → very high-dimensional, very sparse vectors.
- No word order/context captured at all ("dog bites man" and "man bites dog" get identical-ish vectors if same words).
- No notion of semantic similarity between words.

---

## 3. TF-IDF (Term Frequency – Inverse Document Frequency)

**Working**: fixes BoW's biggest flaw — treats all words as equally important — by **downweighting words that appear in many documents** (common, low-signal) and **upweighting words that are frequent in one document but rare across the corpus** (specific, high-signal).

**Formula**:

```
TF(t, d)  = (count of term t in document d) / (total terms in d)

IDF(t)    = log( N / df(t) )
            N = total number of documents
            df(t) = number of documents containing term t

TF-IDF(t, d) = TF(t, d) × IDF(t)
```

(scikit-learn uses a smoothed variant: `IDF = log((1+N)/(1+df(t))) + 1` to avoid divide-by-zero and always give unseen-but-present terms a small positive weight)

**Example** (same corpus):

- Doc1: "the cat sat", Doc2: "the dog sat"
- "the" appears in both docs → **low IDF** → low weight despite appearing in both.
- "cat" appears in only Doc1 → **high IDF** for Doc1 → higher weight, since it's _distinctive_ to that document.

**Code:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["the cat sat", "the dog sat"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
# ['cat' 'dog' 'sat' 'the']
print(X.toarray().round(2))
# [[0.63 0.   0.45 0.45]
#  [0.   0.63 0.45 0.45]]
```

Notice: "cat"/"dog" (distinctive) get higher weight than "sat"/"the" (common to both docs).

**Limitations**:

- Still a **sparse, bag-of-words-based** representation — word order is still lost.
- Still no semantic understanding — "good" and "great" are treated as completely unrelated dimensions.
- IDF is corpus-dependent — the same word can get very different weights depending on what corpus you compute it on; doesn't generalize across domains without recomputing.
- Still suffers vocabulary explosion on large corpora.

---

## 4. Other sparse methods (theory only)

**N-grams as features**

- Instead of single words (unigrams), use contiguous sequences of N words as vocabulary units — bigrams ("the cat", "cat sat"), trigrams, etc.
- Partially recovers some word-order/context info BoW/TF-IDF lose.
- Trade-off: vocabulary size explodes even further (combinatorially) as N increases.

**Count Vectorizer variants (binary BoW)**

- Same as BoW but just presence/absence (1/0) instead of raw counts — used when frequency doesn't matter, just whether a term appears.

**Co-occurrence Matrix**

- Matrix capturing how often word pairs appear near each other within a context window across the corpus.
- Foundational concept — this is actually the bridge idea toward embeddings (GloVe is literally built by factorizing a co-occurrence matrix).

---

## Comparison Table

|Method|Captures frequency|Captures importance/weighting|Captures word order|Vocabulary size|Semantic similarity|
|---|---|---|---|---|---|
|One-Hot|No|No|No|Huge|None|
|BoW|Yes|No|No|Huge|None|
|TF-IDF|Yes|Yes|No|Huge|None|
|N-grams|Yes|Depends|Partial (local)|Even huger|None|

---

## The core limitation of ALL sparse methods

- **High dimensionality + sparsity**: vectors are the size of the entire vocabulary (can be 50k+), with almost all values zero → wasteful memory and compute.
- **No semantic meaning**: "king" and "queen" are just as mathematically unrelated as "king" and "banana" — cosine similarity between any two different words' one-hot-style dimensions is always 0.
- **No generalization**: a word never seen during vocabulary-building is completely unrepresentable (OOV problem again, same issue tokenization solved for subwords — sparse vectorization reintroduces it at the word level).
- **Curse of dimensionality**: with huge sparse vectors, classical ML models need proportionally more data to learn meaningful patterns, and distance/similarity metrics become less meaningful in very high dimensions.

**This is exactly the problem dense vectors/embeddings solve** — instead of a 50,000-dimension mostly-zero vector per word, you get a compact (e.g. 300-dimension, or 4096-dimension for LLMs) dense vector where every value is meaningful, similar words end up close together in the vector space, and the vector is learned (not hand-counted) from actual usage patterns in text.

Covering embeddings in depth next.