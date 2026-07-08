## What is Tokenization

- Process of breaking raw text into smaller units called **tokens** — could be words, subwords, or characters.
- Sits right after sentence segmentation in the pipeline, and is the **direct input format for every downstream NLP model** — classical or LLM.
- Why it's needed: models don't understand strings. Tokenization is step 1 of converting text → numbers (each token gets mapped to an integer ID via a vocabulary, then that ID gets embedded into a vector).

```
Raw text → Tokens → Token IDs (vocab lookup) → Embeddings → Model
```

- This is the **single most consequential preprocessing decision** in an LLM pipeline — vocabulary is frozen after training and can't be changed without retraining the whole model. Bad tokenizer choice = permanent bottleneck (cost, speed, multilingual quality all depend on it).

---

## Types of Tokenization

### 1. Word Tokenization

**Working**: split text into words, typically using whitespace + punctuation rules.

**Example**: `"NLP isn't easy."` → `["NLP", "is", "n't", "easy", "."]`

**Code (NLTK):**

```python
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

text = "NLP isn't easy."
print(word_tokenize(text))
# ['NLP', 'is', "n't", 'easy', '.']
```

**Code (spaCy):**

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("NLP isn't easy.")
print([token.text for token in doc])
# ['NLP', 'is', "n't", 'easy', '.']
```

**Limitations**:

- Vocabulary explosion — every inflection ("run", "running", "runs") is a separate vocab entry.
- Out-of-vocabulary (OOV) problem — any word not seen during training/vocab-building becomes an `<UNK>` token, losing all information.
- Fails on languages without whitespace word boundaries (Chinese, Japanese, Thai).
- Struggles with contractions, hyphenated words, compound words (German).

---

### 2. Character Tokenization

**Working**: every single character (including spaces/punctuation) is its own token.

**Example**: `"NLP"` → `["N", "L", "P"]`

**Code:**

```python
text = "NLP"
tokens = list(text)
print(tokens)
# ['N', 'L', 'P']
```

**Limitations**:

- Vocabulary is tiny (just the character set) → no OOV problem at all.
- But sequences become extremely long — a 10-word sentence becomes 50+ tokens.
- Model has to learn word structure completely from scratch (no built-in notion of "word") — needs far more compute/data to reach the same understanding.
- Rarely used as the primary tokenizer in production LLMs (used more as a fallback mechanism, e.g. byte-level fallback inside BPE).

---

### 3. Subword Tokenization (the industry standard)

Core idea: break rare/complex words into meaningful sub-units, but keep common words whole. Best of both worlds between word-level (small sequences, big vocab, OOV problem) and character-level (huge sequences, tiny vocab, no OOV).

There are 3 major algorithms. All three are "learned" — trained on a corpus first to build the vocabulary, not rule-based.

#### 3a. BPE — Byte Pair Encoding

**Architecture / working**:

1. Start with vocabulary = every individual character (or byte, in byte-level BPE).
2. Count all adjacent symbol pairs in the training corpus.
3. Merge the most frequent pair into a new single token, add it to vocabulary.
4. Repeat merging until you hit the target vocabulary size (e.g. 50,000 tokens).
5. At inference: apply the same learned merge rules to new text.

**Example**: training corpus has "hug", "pug", "hugs" repeatedly.

- "u"+"g" is most frequent pair → merge into "ug"
- "u"+"n" next → merge into "un"
- Vocabulary grows: `[b, g, h, n, p, s, u, ug, un, ...]`

**Code (Hugging Face tokenizers — using GPT-2's BPE tokenizer):**

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "unbelievably tokenization"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ization']
# Ġ = marks a preceding space (GPT-2's byte-level BPE convention)
```

**Code (OpenAI's tiktoken — what GPT-4/GPT-4o actually use in production):**

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("unbelievably tokenization")
print(tokens)
print([enc.decode([t]) for t in tokens])
```

**Limitations**:

- Purely frequency-based merges — no linguistic/semantic awareness, can produce odd splits.
- Greedy — once a merge happens it's permanent, no going back to reconsider.
- Byte-level BPE (used by GPT/LLaMA) needs no `<UNK>` token — any byte sequence is encodable — but non-Latin scripts can still get split into inefficiently many tokens.

**Used by**: GPT-2/3/4, LLaMA 2/3, Mistral, RoBERTa, Qwen2, most modern decoder-only (generative) LLMs.

---

#### 3b. WordPiece

**Architecture / working**:

- Very similar bottom-up merging process to BPE, but the merge criterion is different:
    - BPE merges the **most frequent** pair.
    - WordPiece merges the pair that **maximizes the likelihood of the training data** (a probabilistic/likelihood score, not raw frequency).
- Subword continuation pieces are prefixed with `##` to mark "this continues the previous token" (not a new word).

**Example**: `"unbelievable"` → `["un", "##believ", "##able"]`

**Code:**

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "unbelievable tokenization"
print(tokenizer.tokenize(text))
# ['unbelievable', 'token', '##ization']  (varies by vocab)
```

**Limitations**:

- Same greedy, irreversible merge issue as BPE.
- Longest-match-first decoding at inference can occasionally miss a more optimal split.
- Largely specific to BERT-family encoder models — has been mostly supplanted by BPE for generative LLMs.

**Used by**: BERT, DistilBERT, ELECTRA.

---

#### 3c. Unigram / SentencePiece

**Architecture / working** (opposite direction from BPE/WordPiece):

- Starts with a **large** candidate vocabulary (all substrings/frequent subwords).
- Iteratively **prunes** the least useful tokens (those that reduce corpus likelihood the least) until it reaches target vocab size.
- Top-down instead of bottom-up.
- SentencePiece is the _implementation/framework_ — it treats text as a raw stream of Unicode characters (including spaces as a normal character, usually shown as `▁`), so it needs no pre-tokenization step and is fully language-agnostic (no whitespace assumption — critical for Chinese/Japanese/Thai).
- SentencePiece can run either the BPE algorithm or the Unigram algorithm internally — "SentencePiece" and "Unigram" are often used interchangeably but technically SentencePiece is the tool, Unigram is one of two algorithms it can run.

**Example**: `"tokenization"` → `["▁token", "ization"]` (▁ marks start of a new word)

**Code:**

```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
text = "unbelievable tokenization"
print(tokenizer.tokenize(text))
# ['▁un', 'believ', 'able', '▁token', 'ization']
```

**Limitations**:

- Training is more computationally expensive than BPE (likelihood-based pruning vs simple frequency counting).
- Less common for decoder-only generative LLMs (GPT/LLaMA family prefers BPE) — more popular for encoder-decoder and multilingual models.

**Used by**: T5, mT5, ALBERT, XLNet, mBART, Gemma (uses the Unigram variant specifically), most multilingual models.

---

### 4. Whitespace / Regex Tokenization

**Working**: simplest possible approach — split purely on whitespace or a regex pattern, no learned vocabulary at all.

**Code:**

```python
import re
text = "NLP isn't easy!"
tokens = re.findall(r'\w+|[^\w\s]', text)
print(tokens)
# ['NLP', 'isn', "'", 't', 'easy', '!']
```

**Limitations**: crude — breaks on punctuation attached to words, doesn't handle contractions/hyphens intelligently. Mainly used for quick scripts or as a pre-tokenization step before BPE (BPE still needs an initial rough split before it starts merging).

---

## Comparison Table

|Type|Vocab size|Sequence length|OOV handling|Used by|
|---|---|---|---|---|
|Word|Large (unbounded)|Short|Poor (`<UNK>` for unseen words)|Legacy/classical ML pipelines|
|Character|Tiny|Very long|None needed|Rare, fallback mechanism only|
|BPE|Medium (~30k-100k+)|Medium|None (byte-level = always encodable)|GPT-2/3/4, LLaMA, Mistral, RoBERTa|
|WordPiece|Medium|Medium|Rare, well-handled|BERT, DistilBERT, ELECTRA|
|Unigram/SentencePiece|Medium|Medium|None, language-agnostic|T5, Gemma, ALBERT, multilingual models|
|Whitespace/Regex|Large (unbounded)|Short|Poor|Quick scripts, pre-tokenization step|

## When to use which (practical)

- Quick prototyping / classical ML (Naive Bayes, TF-IDF pipeline) → **word tokenization** (NLTK/spaCy) is enough.
- Building/fine-tuning a decoder-only generative model (GPT-style) → **BPE** (tiktoken or Hugging Face tokenizers).
- Working with BERT-family encoder models for classification/embeddings → **WordPiece** (comes bundled with the pretrained model, you don't choose it).
- Multilingual or encoder-decoder work (translation, T5-style) → **SentencePiece/Unigram**.
- In practice: **you rarely choose the algorithm yourself** — you use whatever tokenizer ships with the pretrained model you're using. You only train a tokenizer from scratch if you're pretraining a model from scratch.

## Industry standard, and why

**BPE (byte-level variant) has become the dominant choice for modern LLMs** — GPT-4, LLaMA 2/3, Mistral, Qwen2, and most current generative models use it. Reasons:

- Byte-level base means zero OOV — any Unicode text, any language, any emoji is always encodable, no `<UNK>` ever needed.
- Simpler and cheaper to train than Unigram's likelihood-based pruning.
- Good empirical compression efficiency (fewer tokens per sentence = cheaper inference, longer effective context).
- SentencePiece/Unigram still remains important for multilingual and encoder-decoder models, and WordPiece persists mainly because BERT-family models are still widely used, but new frontier generative models overwhelmingly choose BPE.

Real production detail worth knowing: OpenAI's vocabulary jump from ~50k tokens (GPT-2/3, `cl100k_base`-era) to ~100k+ tokens (`o200k_base`, GPT-4o generation) cut average token counts for non-English text substantially — vocabulary size directly affects your API cost and effective context window for multilingual traffic.

---

## Interview-style facts to remember

- Tokenization is done **once, at training time**, to build the vocabulary — and that vocabulary is **frozen forever** after. You cannot add new tokens to a deployed model without retraining/re-embedding.
- BPE = bottom-up (merge small → big). Unigram = top-down (prune big → small). WordPiece = bottom-up like BPE but with a likelihood-based merge criterion instead of raw frequency.
- The `##` prefix (WordPiece) and `▁` prefix (SentencePiece) both solve the same problem: marking whether a token starts a new word or continues the previous one — necessary because you need to reconstruct the original text (detokenize) losslessly.
- "Token" ≠ "word" — this trips people up. A single word can be 1 token or split across many tokens (e.g. "tokenization" → "token" + "ization"). This is exactly why LLM context windows are measured in tokens, not words.
- Byte-level BPE guarantees **no OOV problem** — this is a big deal, because pre-2017 word-level models used to hard-fail or lose meaning on unseen words.
- Rare/made-up words getting split into unusual subword chunks is why LLMs sometimes struggle with tasks like counting letters in a word ("how many r's in strawberry") — the model never sees the raw characters, it sees whatever subword chunks the tokenizer produced.
- Sentence vs word vs subword tokenization are all "tokenization" — don't just default to thinking it means word-splitting in an interview.
- Multilingual tokenization inefficiency is a real, named problem: languages that weren't well represented in the tokenizer's training corpus get split into far more tokens per word than English, directly increasing their API cost and reducing effective context length for those languages.