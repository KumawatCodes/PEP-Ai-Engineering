## What is NLP (skip the textbook def)

NLP = making a machine **read, understand, and generate human language** (text or speech) well enough to _do something useful with it_.

Think of it as the bridge between:

- Human language → messy, ambiguous, context-heavy, unstructured
- Machine → wants numbers, structure, fixed-size vectors

NLP is the entire engineering pipeline that converts one into the other and back.

## Use-case first (why it exists)

|Use case|What's actually happening under the hood|
|---|---|
|Spam filter|Text classification|
|Google Translate|Sequence-to-sequence generation|
|Siri/Alexa|Speech-to-text → NLU → intent classification → response generation|
|Grammarly|Sequence labeling + language modeling|
|ChatGPT / Claude|Next-token prediction at massive scale (autoregressive language modeling)|
|Resume screening bots|NER (extract skills, companies, dates) + classification|
|Search engines|Semantic similarity between query and documents|

**Pattern**: almost every NLP use case boils down to one of these core tasks — classify, extract, generate, or match/rank.

## How it "works" (30,000 ft view)

1. Text comes in (raw, dirty, human-written).
2. It gets cleaned and broken into processable units.
3. Those units get converted into numbers (vectors) — because models only do math.
4. A model (rule-based → statistical → neural → transformer) processes those vectors.
5. Output is either a number/label (classification), a structure (tags, entities), or more text (generation).

That's it. Steps 2–3 are "traditional NLP." Step 4, since ~2017, is dominated by **transformers**. Step 4 pre-2017 used RNNs/LSTMs, and before that, statistical models (n-grams, HMMs) and rule-based systems.

## NLP's role in the current AI boom

The AI boom you're seeing (ChatGPT, Claude, Gemini, etc.) is fundamentally an **NLP boom**, powered by 3 things converging:

1. **Transformer architecture (2017)** — solved the long-range dependency problem RNNs had, and is fully parallelizable (trainable on GPUs at scale).
2. **Scale** — massive web-scale text corpora + massive compute (GPT-3 onward proved "bigger model + more data = emergent capability").
3. **Self-supervised pretraining** — no need for labeled data anymore; the model learns from raw internet text by predicting the next token. This removed the biggest bottleneck traditional NLP had (labeled data scarcity).

So LLMs are NOT a separate field from NLP — they ARE modern NLP. Everything you're learning here (cleaning, tokenization, embeddings) is still literally what happens before text enters a model like GPT or Claude. LLMs just replaced the "hand-crafted feature engineering" era of NLP with "learned representations at scale."

```
Rule-based NLP (1950s-1990s)
        ↓
Statistical NLP (1990s-2010s)  → n-grams, HMM, CRF, Naive Bayes
        ↓
Neural NLP (2013-2017)          → Word2Vec, RNN, LSTM, GRU
        ↓
Transformer era (2017-now)      → BERT, GPT, T5
        ↓
LLM era (2020-now)               → GPT-3/4, Claude, Gemini (scale + pretraining + RLHF)
```

## The NLP Pipeline — full picture

Every NLP system, whether it's a spam filter or GPT-4's training, roughly follows this pipeline:

```
Text Acquisition
      ↓
Text Cleaning
      ↓
Sentence Segmentation
      ↓
Tokenization        ← (covering separately, deep dive later)
      ↓
Text Normalization (stemming/lemmatization, casing)
      ↓
Feature Engineering / Vectorization (BoW, TF-IDF, embeddings)
      ↓
Modeling (classification / generation / etc.)
      ↓
Evaluation
      ↓
Deployment
```

Below: everything **up to tokenization**, in depth.

---

## Phase 1: Text Acquisition

Getting raw text data into the system. Sources:

- Scraping (web pages, PDFs, HTML)
- APIs (Twitter/X, Reddit, news APIs)
- Databases / internal logs
- Public datasets (Common Crawl, Wikipedia dumps — this is what pretrains LLMs)
- User-generated input (chat, forms)

**Engineering concern here**: format heterogeneity. You'll get HTML, PDFs, JSON, plain text — all mixed with noise (ads, nav bars, encoding artifacts). This stage is basically "ETL for text," which is directly relevant to your data engineering background — same extraction mindset as WDI ETL, just unstructured instead of structured.

## Phase 2: Text Cleaning

Raw text is dirty. Cleaning removes stuff that adds noise but no linguistic signal.

Typical operations:

- Remove HTML/XML tags, markdown artifacts
- Remove/normalize special characters, emojis (depends on task — sentiment analysis might _want_ emojis)
- Fix encoding issues (mojibake — e.g. `â€™` instead of `'`)
- Remove extra whitespace/newlines
- Lowercasing (careful — this destroys info like "US" vs "us", NER cares about casing)
- Remove URLs, mentions, hashtags (task-dependent, not universal)
- Spelling correction (optional, expensive)

**Key principle**: cleaning is **task-dependent**, not universal. Sentiment analysis wants emojis and punctuation (`!!!` matters). Legal document classification wants exact casing preserved. There's no single "correct" cleaning pipeline — you design it around the downstream task.

## Phase 3: Sentence Segmentation

Splitting a text blob into individual sentences before you tokenize into words.

Sounds trivial — split on `.`? — but isn't:

- "Dr. Smith went to the U.S. yesterday." → naive `.` split breaks this into garbage
- Abbreviations, decimal numbers (3.14), ellipses (...), quoted sentences within sentences

This is why real systems use rule-based + ML sentence boundary detection (e.g. Punkt tokenizer in NLTK) rather than a plain string split.

**Why this phase matters**: a lot of downstream tasks (translation, summarization, NER) operate at sentence-level, not document-level, so getting boundaries wrong propagates errors downstream.

---

## Phase 4: Stopword Removal

- Stopwords = high-frequency, low-information words: "the", "is", "in", "and", "to".
- Removing them shrinks vocabulary size and cuts noise before vectorization.
- Standard practice for: document classification, clustering, search/information retrieval.
- **Not universal** — removing stopwords blindly can hurt:
    - Sentiment analysis / QA / machine translation — words like "not", "no", "very" are stopwords in generic lists but flip meaning ("not good" ≠ "good").
    - So: filter your stopword list per task, don't use the default list unquestioned.
- Stopword lists are corpus/language-specific — no single universal list.

## Phase 5: POS Tagging (Part-of-Speech Tagging)

- Labels every token with its grammatical role: noun, verb, adjective, adverb, etc.
- Why it matters here: it's a **prerequisite for accurate lemmatization** (see below) — without knowing if "worst" is being used as an adjective, a lemmatizer can guess wrong.
- Done using pretrained taggers (statistical or neural) — language-specific, since grammar rules differ per language.
- Also directly useful downstream for: NER, parsing, grammar checking.

## Phase 6: Stemming vs Lemmatization

Both reduce a word to a base/root form to shrink vocabulary and group word variants together — but they work very differently.

**Stemming**

- Rule-based, chops off prefixes/suffixes using heuristics (no dictionary, no grammar knowledge).
- Fast, but crude — output isn't always a real word.
- Example: `studies → studi`, `learning → learn`, `better → better` (misses irregular forms).
- Classic algorithm: Porter Stemmer.
- Use when: speed > accuracy (large-scale search indexing, quick prototyping).

**Lemmatization**

- Dictionary + grammar based (uses POS tag) → always returns a valid word (the "lemma").
- Slower, but linguistically accurate.
- Example: `worst → bad` (only works correctly if POS-tagged as adjective first).
- Use when: accuracy matters more than speed (most model-training pipelines).

**Rule of thumb from practice**: run POS tagging → then lemmatize using that tag → prefer lemmatization over stemming for anything feeding into a model.

## Phase 7: Other normalization steps (grouped — same spirit as cleaning)

- Punctuation removal
- Case normalization (lowercasing) — same caveat as Phase 2, can lose information
- Removing digits/numbers (task-dependent — finance/legal text needs numbers kept)
- Symbol/special-character removal

These often get merged into "Cleaning" (Phase 2) or "Normalization" depending on which book/pipeline you read — no strict universal boundary, just do it before vectorization.

---

## Important reality check: does modern NLP (LLMs) even do all this?

Short answer: **mostly no, and that's a deliberate design shift, not an oversight.**

- Classical ML pipelines (Naive Bayes, SVM, TF-IDF-based models) — need the **full pipeline**: cleaning → stopword removal → stemming/lemmatization → POS tagging → vectorization. These models can't learn structure themselves, so you hand-engineer it.
- Deep learning / transformer models (BERT, GPT, LLaMA, Claude) — use **minimal preprocessing**: basically just tokenization + normalization. No stopword removal, no stemming, no lemmatization.
    - Why: these models learn language structure directly from raw text at scale (self-supervised pretraining). Removing "not" or reducing "running"→"run" by hand would actually **destroy information** the model could've learned to use itself.
    - This is the same "hand-crafted features → learned representations" shift mentioned earlier in the AI boom section.

**Takeaway**: learn the full classical pipeline (it's foundational and still used in traditional ML/industrial systems), but know that LLM-era models intentionally skip most of it beyond tokenization.

---

## Next in the pipeline (not covered yet — separate note)

**Tokenization** → breaking sentences into tokens (words/subwords) → this is where modern LLMs get interesting (BPE, WordPiece, SentencePiece). Doing this as its own deep-dive note next since it's the bridge into how transformers actually consume text.

---

## Quick recap (for fast revision)

- NLP = converting messy human language ↔ structured numeric form a model can use.
- Every use case reduces to: classify / extract / generate / match.
- Modern AI boom = NLP + transformers + scale + self-supervised pretraining.
- Full classical pipeline: **Acquire → Clean → Segment → Tokenize → Stopword removal → POS tag → Stem/Lemmatize → Vectorize.**
- Cleaning and stopword removal are task-dependent — no universal script.
- Lemmatization needs POS tagging first; stemming doesn't (and is less accurate because of that).
- LLMs (BERT/GPT/Claude-era) skip stopword removal/stemming/lemmatization — they learn from raw tokenized text directly.

# NLP Libraries — What to Use, When

Mapped to the pipeline phases from the previous note. Goal: know which library owns which phase, not memorize every API.

## Quick map (phase → library)

|Pipeline phase|Go-to library|
|---|---|
|Scraping/acquiring text|`requests`, `BeautifulSoup`, `Scrapy`|
|Regex-based cleaning|`re` (built-in)|
|Sentence segmentation, tokenization, POS, stemming, lemmatization, stopwords|`NLTK`, `spaCy`|
|Quick prototyping / sentiment|`TextBlob`|
|Topic modeling, word embeddings (Word2Vec)|`Gensim`|
|Vectorization (BoW, TF-IDF) + classical ML models|`scikit-learn`|
|Social-media-style rule-based sentiment|`VADER`|
|Deep learning / transformers / LLMs|`Hugging Face Transformers` (+ `PyTorch`)|

---

## 1. `re` (built-in Python)

- Regex module — used in the **cleaning phase** (Phase 2 from last note).
- Strip HTML tags, URLs, special characters, fix whitespace.
- Not NLP-specific, but every pipeline starts here before real NLP libraries touch the text.

## 2. `BeautifulSoup` / `Scrapy`

- Used in **text acquisition** phase — parsing HTML/XML when scraping web pages.
- `BeautifulSoup` — simple, good for parsing a page you already fetched.
- `Scrapy` — full scraping framework, use when crawling many pages at scale (your ETL background makes this one intuitive — same role as an extractor in a data pipeline).

## 3. NLTK (Natural Language Toolkit)

- The **oldest and most academic** NLP library — built for teaching/learning, not production speed.
- Covers almost the entire classical pipeline: tokenization, stopwords, POS tagging, stemming (Porter Stemmer), lemmatization (WordNet Lemmatizer), sentence segmentation (Punkt).
- Comes with 50+ corpora and lexical resources (WordNet is the big one).
- **Use when**: learning concepts, prototyping, academic work. Pure Python → slow for production-scale text.
- **Skip when**: you need speed or you're deploying — reach for spaCy instead.

## 4. spaCy

- **Industrial-strength**, Cython-optimized (fast — actual C-speed under the hood, not pure Python like NLTK).
- Same core tasks as NLTK (tokenization, POS tagging, lemmatization, sentence segmentation) but production-grade + adds **NER** and **dependency parsing** out of the box.
- Opinionated — gives you one good way to do things instead of ten options like NLTK.
- Ships pretrained pipelines per language (`en_core_web_sm`, etc.) — one line to load a full pipeline.
- **Use when**: building anything real-world/production — chatbots, document processing, real-time APIs.
- **Skip when**: you want to hand-tune every low-level algorithm choice — spaCy hides that by design.

**NLTK vs spaCy, one line**: NLTK = classroom, spaCy = production.

## 5. TextBlob

- Built on top of NLTK + Pattern, wraps everything in a dead-simple API.
- Good for: sentiment analysis, POS tagging, quick prototypes, translation — in 2-3 lines of code.
- **Use when**: you want a fast throwaway script or are just starting out.
- **Skip when**: production system — it's slow and not deep-learning-based, mostly gets replaced by spaCy or Hugging Face as projects mature.

## 6. Gensim

- Specialist library — not a general NLP toolkit, focused on:
    - Topic modeling (LDA — Latent Dirichlet Allocation)
    - Word embeddings (Word2Vec, Doc2Vec)
    - Document similarity / semantic search
- Memory-efficient — built to handle large corpora that don't fit in RAM.
- **Use when**: you need "what topics exist in this corpus" or "how similar are these two documents" — common in RAG pipelines for building/searching embeddings at scale.
- **Skip when**: basic preprocessing (tokenize/clean) — that's not its job, pair it with NLTK/spaCy for that part.

## 7. scikit-learn

- Not NLP-specific, but essential for the **vectorization + modeling** phase.
- Provides `CountVectorizer` (Bag of Words) and `TfidfVectorizer` (TF-IDF) — turns cleaned text into numeric feature vectors.
- Then plug those vectors into any classical ML model (Naive Bayes, SVM, Logistic Regression) it also provides.
- **Use when**: you're doing classical ML on text (not deep learning) — e.g. spam classifier, sentiment classifier on small-medium data.

## 8. VADER (Valence Aware Dictionary and sEntiment Reasoner)

- Rule-based (lexicon + heuristics), not ML-based — no training needed, works out of the box.
- Tuned specifically for **social media text** — handles emojis, slang, capitalization ("GREAT!!!" scores higher than "great"), punctuation emphasis.
- **Use when**: quick sentiment scoring on tweets/reviews/informal text, no time/data to train a model.
- **Skip when**: formal/domain-specific text (legal, medical) — it wasn't tuned for that vocabulary.

## 9. Hugging Face Transformers (+ PyTorch/TensorFlow underneath)

- The library for **modern deep learning NLP / LLMs** — BERT, GPT-family, T5, LLaMA, etc.
- Gives pretrained models + a simple `pipeline()` API to run text classification, generation, summarization, translation, Q&A in a few lines.
- Also the standard toolkit for **fine-tuning** these models on your own data.
- This is the library that matters most once you move past classical NLP into how ChatGPT/Claude/BERT actually get used and fine-tuned — directly relevant to where the field is heading.
- **Use when**: state-of-the-art accuracy needed, working with embeddings for RAG, or fine-tuning an LLM.
- **Skip when**: task is simple and classical ML would do the job just as well with far less compute (don't reach for BERT to classify 500 emails, that's what scikit-learn is for — don't over-engineer).

## 10. Honorable mentions (know they exist, don't need mastery yet)

- **Stanza** (Stanford NLP) — pretrained pipelines for 70+ languages, academic-grade accuracy, similar role to spaCy but multilingual research use.
- **Textacy** — sits on top of spaCy, adds convenience preprocessing/cleaning utilities.
- **AllenNLP** — PyTorch-based, research-oriented, used less in production now.

---

## Practical decision rule

- Learning a concept for the first time → **NLTK**
- Building something real that needs to run fast → **spaCy**
- Need embeddings/topic modeling/semantic similarity → **Gensim**
- Turning cleaned text into vectors for a classical ML model → **scikit-learn**
- Need an LLM / pretrained deep model / fine-tuning → **Hugging Face Transformers**
- Quick sentiment on informal social text, zero setup → **VADER**

---

## Quick recap

- `re` + `BeautifulSoup` → acquisition & cleaning (pre-NLP-library stage).
- `NLTK` → learning/prototyping, full classical pipeline, slow.
- `spaCy` → production version of the same pipeline, fast, adds NER/parsing.
- `TextBlob` → simplest API, throwaway scripts only.
- `Gensim` → topic modeling + embeddings, not general preprocessing.
- `scikit-learn` → vectorization (BoW/TF-IDF) + classical ML models.
- `VADER` → rule-based sentiment for informal/social text.
- `Hugging Face Transformers` → deep learning/LLM era — where the field is now.