## Are these dense vectors?

Yes. Word embeddings = dense vectors. Direct contrast with the last note:

||Sparse (BoW/TF-IDF)|Dense (Embeddings)|
|---|---|---|
|Dimension|Vocab size (huge, e.g. 50k)|Fixed, small (e.g. 100–300D)|
|Most values|Zero|Non-zero, meaningful|
|How obtained|Hand-counted|**Learned** by a neural network from usage patterns|
|Similar words|No relationship captured|Close together in vector space|

---

## The core idea (Distributional Hypothesis)

- **"You shall know a word by the company it keeps"** — a word's meaning is defined by the words that typically appear around it.
- "king" and "queen" show up in similar contexts ("the ___ ruled the land") → embeddings push them to similar vector positions.
- Every embedding algorithm below is just a different mathematical trick for exploiting this idea at scale.

---

## The Embedding Matrix — the actual object you're learning

- An embedding matrix is just a **lookup table**: shape `[vocab_size, embedding_dim]`.
- Example: vocab = 50,000 words, embedding dim = 300 → matrix is `50000 x 300`.
- Row `i` of this matrix IS the vector for word `i`. Looking up a word's embedding = indexing into this matrix by its vocab ID (the same ID tokenization assigned it).
- **This matrix is literally a neural network's weight matrix.** Training the embedding = training this weight matrix via backprop, same as any other NN weights.
- Once trained, you throw away the rest of the training network and just keep this matrix — that's your reusable embedding lookup table for any downstream task.

```
word "cat" → vocab ID 452 → embedding_matrix[452] → [0.12, -0.44, 0.09, ..., 0.31]  (300 numbers)
```

---

## 1. Word2Vec (Google, 2013)

Two architectures, both shallow (1 hidden layer) neural networks trained on a **self-supervised fake task** — the real goal isn't the task itself, it's the by-product weight matrix.

### Architecture A: CBOW (Continuous Bag of Words)

**Working**:

1. Take a context window around a target word, e.g. window size 2: `["The", "cat", ___, "on", "mat"]`.
2. Input = surrounding context words ("The", "cat", "on", "mat"), each one-hot encoded.
3. Task: **predict the missing middle word** ("sat").
4. Network: Input (one-hot) → Hidden layer (this IS the embedding matrix, no activation, just a linear projection) → Output layer (softmax over entire vocabulary) → predicted word.
5. Backprop adjusts the hidden layer weights so the prediction gets better.
6. After training on millions of such windows across the corpus, the hidden layer weight matrix = your word embeddings.

**Why it works**: words that tend to appear in similar contexts end up needing similar weight patterns to predict correctly → their vectors converge to similar positions.

**Good for**: smaller datasets, faster training, works well for frequent words.

### Architecture B: Skip-gram

**Working**: exact mirror of CBOW.

1. Input = the **target word** (one-hot).
2. Task: **predict the surrounding context words**.
3. Same shallow network shape, just input/output swapped.

**Example**: input "sat" → predict "The", "cat", "on", "mat" (as separate training pairs: (sat, The), (sat, cat), (sat, on), (sat, mat)).

**Good for**: larger datasets, better at representing rare words (each rare word gets many training signals — one per context word it predicts).

**CBOW vs Skip-gram, one line**: CBOW predicts the word from context (faster, smooths over frequent words). Skip-gram predicts context from the word (slower, better for rare words).

**Training trick — Negative Sampling**: computing full softmax over a 50k+ vocabulary for every training step is too slow. Instead, the network only updates weights for the correct word + a handful (e.g. 5–20) of randomly sampled "negative" (wrong) words, turning it into a much cheaper binary classification problem ("is this word/context pair real or fake?"). This is why Word2Vec became practically trainable at scale.

**Code:**

```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat", "on", "the", "mat"],
             ["the", "dog", "sat", "on", "the", "rug"]]

model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=1)  # sg=1 -> skip-gram, sg=0 -> CBOW
model.train(sentences, total_examples=len(sentences), epochs=10)

vector = model.wv["cat"]
print(vector.shape)   # (100,)
print(model.wv.most_similar("cat"))
```

**Limitations**:

- **Static/context-independent**: "bank" gets exactly ONE vector regardless of "river bank" vs "money bank" — no context sensitivity (this is the big limitation that transformer-based contextual embeddings later fixed).
- OOV problem: a word never seen during training has no vector at all — Word2Vec has zero fallback (FastText below fixes this).
- Needs a large corpus to produce good quality vectors for rare words.

---

## 2. GloVe (Global Vectors, Stanford, 2014)

**Core difference from Word2Vec**: Word2Vec is _predictive_ (learns via a prediction task using local context windows). GloVe is _count-based_ — it directly uses **global co-occurrence statistics** across the whole corpus.

**Working**:

1. Build a global word-word co-occurrence matrix `X`, where `X[i][j]` = how often word `j` appears in the context of word `i`, across the ENTIRE corpus (not just local windows like Word2Vec's training pairs).
2. Factorize this matrix into two smaller dense matrices (word vectors and context vectors) such that the dot product of two word vectors approximates the **log of their co-occurrence probability ratio**.
3. Objective: `w_i · w_j + b_i + b_j ≈ log(X_ij)` — minimized via weighted least squares, not classification.

**Intuition**: if "ice" co-occurs a lot with "solid" but rarely with "gas", and "steam" is the reverse, the ratio of co-occurrence probabilities encodes the meaningful relationship — GloVe directly optimizes vectors to capture that ratio.

**Example**: the famous result — `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`. This works because GloVe (and Word2Vec) end up encoding relationships as consistent directions in vector space (gender direction, royalty direction, etc.), purely as an emergent property of the training objective — not explicitly programmed.

**Code:**

```python
import gensim.downloader as api

glove_vectors = api.load("glove-wiki-gigaword-100")  # pretrained, 100D
print(glove_vectors["king"].shape)   # (100,)
print(glove_vectors.most_similar("king"))
result = glove_vectors.most_similar(positive=["king", "woman"], negative=["man"])
print(result[0])   # ('queen', 0.76...)
```

**Limitations**: same static/context-independent limitation as Word2Vec, same OOV problem (no subword fallback), and building the full co-occurrence matrix is memory-heavy for huge corpora.

---

## 3. FastText (Facebook/Meta, 2016)

**Core difference**: extends Word2Vec's skip-gram idea, but represents each word as a **bag of character n-grams**, not an atomic unit.

**Working**:

1. Word "apple" (with boundary markers) → `<ap, app, ppl, ple, le>` (character n-grams, e.g. n=3) `+` the whole word `<apple>` itself.
2. Each n-gram gets its own embedding vector.
3. The final word vector = **sum (or average) of all its n-gram vectors**.
4. Trained the same way as Word2Vec skip-gram (predict context), but now predicting using this composed n-gram-sum vector instead of one atomic word vector.

**Why this matters — solves Word2Vec/GloVe's OOV problem**:

- A brand new word never seen during training (e.g. a typo, a rare brand name, "unbelievability") can still be embedded — just break it into character n-grams (which were likely seen as substrings in other words during training) and sum those.
- Also captures **morphology** naturally: "run", "running", "runner" share n-grams like "run" → their vectors end up related even without explicit lemmatization.

**Especially good for**: morphologically rich languages (Finnish, Turkish, Hindi), typo-prone text (social media), rare/technical vocabulary.

**Code:**

```python
from gensim.models import FastText

sentences = [["the", "cat", "sat", "on", "the", "mat"],
             ["the", "dog", "sat", "on", "the", "rug"]]

model = FastText(sentences, vector_size=100, window=2, min_count=1, sg=1)
model.train(sentences, total_examples=len(sentences), epochs=10)

print(model.wv["cat"].shape)          # (100,)
print(model.wv["caterpillar"].shape)  # works even if "caterpillar" was never in training data!
```

**Limitations**: larger model size (has to store n-gram vectors, not just word vectors), slightly slower training than plain Word2Vec, and it's still ultimately a **static** embedding once training is done (same word = same vector, regardless of sentence context).

---

## How similarity is actually checked

- Word vectors live in a continuous vector space — "similar meaning" = "close together" in that space.
- **Cosine similarity** is the standard metric (not Euclidean distance) — measures the angle between two vectors, ignoring magnitude:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- Range: -1 (opposite) to 1 (identical direction). Magnitude is ignored deliberately — word frequency can inflate vector length, but direction is what encodes meaning.
- `most_similar()` in gensim (used above) computes cosine similarity between your query vector and every vector in the vocabulary, returns top-K closest.
- **Vector arithmetic** works because relationships are encoded as consistent directions: `king - man + woman ≈ queen` is literally vector addition/subtraction, then a nearest-neighbor lookup for the closest real word vector to the result.
- At production scale (millions of vectors), brute-force cosine similarity against every vector is too slow — real systems (search engines, RAG pipelines) use **Approximate Nearest Neighbor (ANN)** algorithms (FAISS, HNSW, Annoy) to search efficiently. Worth knowing this term exists even if not deep-diving now — it's the same underlying search operation your future RAG-based projects will lean on.

---

## Comparison Table

||Word2Vec|GloVe|FastText|
|---|---|---|---|
|Approach|Predictive (NN, local context window)|Count-based (global co-occurrence matrix factorization)|Predictive (skip-gram) + subword n-grams|
|Unit of representation|Whole word|Whole word|Character n-grams summed into word|
|Handles OOV words|No|No|**Yes**|
|Captures morphology|No|No|**Yes**|
|Training data need|Large corpus|Large corpus (needs full co-occurrence matrix)|Large corpus|
|Context-sensitive|No (static)|No (static)|No (static)|
|Good for|General-purpose, speed|Capturing global corpus statistics well|Rare words, typos, morphologically rich languages|

---

## The remaining limitation (shared by all three)

- All three are **static embeddings** — one fixed vector per word, no matter the sentence.
- "The bank raised interest rates" vs "I sat by the river bank" → same "bank" vector in Word2Vec/GloVe/FastText, even though the meanings are completely different.
- This exact gap is what **contextual embeddings** (ELMo, then BERT/GPT-style transformer embeddings) were built to solve — the vector for a word changes depending on the sentence it's in, generated fresh by running the whole sentence through the model each time instead of a static lookup table.

Covering contextual embeddings / how transformers generate these dynamically, separately later.

---

## Interview-style facts to remember

- Word embeddings are a **by-product** of a fake self-supervised training task (predict context / predict word) — the actual useful output is the learned weight matrix, not the prediction itself.
- The embedding matrix is literally a neural network weight matrix — `[vocab_size x embedding_dim]` — retrieving a word's vector is just a row lookup (equivalent to multiplying a one-hot vector by the matrix, but implemented as direct indexing for speed).
- CBOW predicts word-from-context (faster, better for frequent words). Skip-gram predicts context-from-word (slower, better for rare words).
- GloVe uses global co-occurrence counts; Word2Vec uses local sliding-window prediction — different mechanism, but they converge to embeddings with very similar properties in practice.
- FastText is the only one of the three that handles OOV words and captures subword/morphological structure, because it operates on character n-grams, not atomic words.
- `king - man + woman ≈ queen` is real and reproducible — it's the standard example used to demonstrate that embedding spaces encode semantic relationships as consistent vector directions, not because it was explicitly programmed.
- Cosine similarity, not Euclidean distance, is the standard metric for word vector similarity — direction matters, magnitude doesn't.
- Typical embedding dimensions: 100–300D for classical Word2Vec/GloVe/FastText. Modern LLMs use much higher dimensional embeddings (e.g. 1536D, 4096D+) as part of much larger models.
- All three (Word2Vec, GloVe, FastText) produce **static** embeddings — this is the key limitation that motivated the shift to transformer-based contextual embeddings.