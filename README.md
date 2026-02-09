# textLLM <img src="man/figures/logo.png" align="right" height="139" />

## Text Classification with Large Language Models in R

A teaching-friendly R package for text classification using LLM APIs. Designed for the **"Text Classification with LLMs in R"** workshop by Dr. Hudson Golino (University of Virginia).

## Installation

```r
# Install from GitHub
devtools::install_github("hfgolino/textLLM")
```

## Quick Start

```r
library(textLLM)

# One-time setup (creates Python environment)
setup_llm_env()

# Every session
activate_llm_env()
set_groq_key("your-groq-api-key")     # Free at https://console.groq.com
set_voyage_key("your-voyage-api-key")  # Free at https://dash.voyageai.com

# Load example data
data(extraversion)

# Classify!
results <- classify_direct(extraversion$text, unique(extraversion$subtopic))
head(results)
```

## Features

### 🏷️ Classification (7 methods)

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

### 📐 Embeddings

| Function | Description |
|----------|-------------|
| `embed_voyage()` | Voyage AI API embeddings (1024-dim) |
| `embed_local()` | Local HuggingFace models (all layers) |
| `evaluate_layers()` | Layer quality evaluation |

### 🔍 RAG (Retrieval-Augmented Generation)

| Function | Description |
|----------|-------------|
| `retrieve_similar()` | Find similar documents by embedding |
| `rag_generate()` | Generate answers from retrieved context |
| `discover_topics()` | Automatic topic discovery |
| `classify_with_rag()` | Full unsupervised RAG pipeline |

### 🕸️ EGA Integration

| Function | Description |
|----------|-------------|
| `ega_classify()` | EGA on embeddings for text classification |
| `prime_weight_encode()` | Prime-weight encoding across layers |

### 🛠️ Utilities

| Function | Description |
|----------|-------------|
| `llm_generate()` | Generate text (item creation, summarization) |
| `compare_methods()` | Compare classification methods (NMI table) |
| `cosine_similarity()` | Cosine similarity between vectors |
| `cosine_similarity_matrix()` | Pairwise similarity matrix |
| `list_models()` | List available Groq models |
| `verify_setup()` | Check environment and API connectivity |

## Example: Compare Classification Methods

```r
library(textLLM)
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
  "Direct"          = res_direct,
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

## Example: EGA on Embeddings

```r
embeddings <- embed_voyage(extraversion$text)
result <- ega_classify(embeddings)
aricode::NMI(extraversion$subtopic, result$communities)
```

## Dependencies

**R packages**: reticulate, jsonlite (required); EGAnet, aricode, ggpubr (suggested)

**Python** (managed automatically via UV): groq, voyageai, torch, transformers

## Citation

If you use textLLM in your research, please cite:

```
Golino, H. (2025). textLLM: Text Classification with Large Language Models in R.
R package version 0.1.0. https://github.com/hfgolino/textLLM
```

## License

GPL-3
