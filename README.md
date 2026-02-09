# llmClassificR <img src="man/figures/logo_llmClassificR.png" align="right" height="260" />

## Text Classification with Large Language Models in R

A user-friendly toolkit for text classification using Large Language 
Model APIs and local HuggingFace models. Provides batch zero-shot classification
(direct, few-shot, chain-of-thought, structured JSON, multi-model ensemble,
scoring, and LLM-as-judge), embedding extraction (API-based via Voyage AI and
Jina AI, and local via HuggingFace transformers), retrieval-augmented generation
(RAG), automatic topic discovery, and integration with Exploratory Graph Analysis
(EGA) for unsupervised text classification. All LLM classification functions
send texts in a single batch API call for speed. 

Developed and maintained by Dr. Hudson Golino (University of Virginia).

## Installation

```r
# Install from GitHub
devtools::install_github("hfgolino/llmClassificR")
```

## Quick Start

```r
library(llmClassificR)

# One-time setup (creates Python environment)
setup_llm_env()

# Every session
activate_llm_env()
set_groq_key("your-groq-api-key")     # Free at https://console.groq.com
set_voyage_key("your-voyage-api-key")  # Free at https://dash.voyageai.com
set_jina_key("your-jina-api-key")     # Free at https://jina.ai/embeddings/

# Load example data
data(extraversion)

# Classify!
results <- classify_direct(extraversion$text, unique(extraversion$subtopic))
head(results)
```

## Features

### Classification (7 methods)

All methods send texts in a **single batch API call** for speed.

| Function | Description |
|----------|-------------|
| `classify_direct()` | Simple zero-shot classification |
| `classify_fewshot()` | Few-shot with labeled examples |
| `classify_cot()` | Chain-of-thought with reasoning |
| `classify_json()` | Structured output with confidence scores |
| `classify_ensemble()` | Multi-model majority vote |
| `classify_scores()` | Per-class probability scores (0-100) |
| `classify_judge()` | LLM-as-judge with evidence extraction |

### Embeddings

| Function | Description |
|----------|-------------|
| `embed_voyage()` | Voyage AI API embeddings (1024-dim) |
| `embed_jina()` | Jina AI API embeddings (pure R via httr2, no Python needed) |
| `embed_local()` | Local HuggingFace models (all hidden layers) |
| `evaluate_layers()` | Layer quality evaluation (silhouette, PCA, separation) |

### RAG (Retrieval-Augmented Generation)

| Function | Description |
|----------|-------------|
| `retrieve_similar()` | Find similar documents by cosine similarity |
| `rag_generate()` | Generate answers from retrieved context |
| `discover_topics()` | Automatic topic discovery (no labels needed) |
| `classify_with_rag()` | Full unsupervised RAG pipeline |

### EGA Integration

| Function | Description |
|----------|-------------|
| `ega_classify()` | EGA on embeddings for text classification |
| `prime_weight_encode()` | Prime-weight encoding across transformer layers |

### Utilities

| Function | Description |
|----------|-------------|
| `llm_generate()` | Generate text (item creation, summarization) |
| `compare_methods()` | Compare classification methods (NMI table) |
| `cosine_similarity()` | Cosine similarity between vectors |
| `cosine_similarity_matrix()` | Pairwise similarity matrix |
| `list_models()` | List available Groq models (live API query) |
| `list_jina_models()` | List available Jina AI embedding models |
| `verify_setup()` | Check environment and API connectivity |

## Example: Compare Classification Methods

```r
library(llmClassificR)
data(extraversion)

texts <- extraversion$text
classes <- unique(extraversion$subtopic)

# Run multiple approaches
res_direct  <- classify_direct(texts, classes)
res_cot     <- classify_cot(texts, classes)
res_json    <- classify_json(texts, classes)
res_judge   <- classify_judge(texts, classes)

# Compare all at once
compare_methods(
  true_labels = extraversion$subtopic,
  "Direct"           = res_direct,
  "Chain-of-Thought" = res_cot,
  "Structured JSON"  = res_json,
  "LLM-as-Judge"     = res_judge
)
```

## Example: Full RAG Pipeline (No Labels Needed)

```r
# Discover topics and classify — fully unsupervised
pipeline <- classify_with_rag(extraversion$text, n_topics = 4)
print(pipeline$topics)          # What topics were discovered?
table(pipeline$classifications$prediction)  # How were texts classified?
```

## Example: Embeddings + EGA

```r
# Voyage AI embeddings (requires Python environment)
embeddings <- embed_voyage(extraversion$text)
result <- ega_classify(embeddings)
aricode::NMI(extraversion$subtopic, result$communities)

# Jina AI embeddings (pure R — no Python needed!)
jina_emb <- embed_jina(extraversion$text)
result2 <- ega_classify(jina_emb)
aricode::NMI(extraversion$subtopic, result2$communities)

# Jina with Matryoshka dimension reduction (256 instead of 1024)
jina_small <- embed_jina(extraversion$text, dimensions = 256)
```

## Datasets

| Dataset | Description |
|---------|-------------|
| `extraversion` | 120 first-person extraversion descriptions (4 subtopics) |
| `extraversion_3p` | 120 third-person extraversion descriptions (4 subtopics) |
| `personality_descriptions` | 110 Big Five personality descriptions (5 traits) |

## Dependencies

**R packages** (required): reticulate, jsonlite, httr2, stats, cluster

**R packages** (suggested): EGAnet, aricode, ggpubr, ggplot2, tidyr, dplyr, corrplot

**Python** (managed automatically via UV): groq, voyageai, torch, transformers

> **Note:** `embed_jina()` uses pure R (httr2) and does **not** require the Python
> environment. You can use Jina AI embeddings even without running `setup_llm_env()`.

## Citation

If you use llmClassificR in your research, please cite:

```
Golino, H. (2025). llmClassificR: Text Classification with Large Language Models in R.
R package version 0.1.0. https://github.com/hfgolino/llmClassificR
```

## License

GPL-3
