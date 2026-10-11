# llmClassificR <img src="man/figures/logo_llmClassificR.png" align="right" height="260" />

## Text Classification with Large Language Models in R

A user-friendly toolkit for text classification with Large Language Models
in R, from bag-of-words to Jev. It provides robust batch zero-shot
classification with Groq LLMs (direct, few-shot, chain-of-thought, JSON,
strict JSON Schema, multi-model ensemble, self-consistency, scoring and
LLM-as-judge), **Jev** "System One" classification (TypeSafe AI) directly
from R, embeddings (Voyage AI, Jina AI and local HuggingFace models), a
do-it-yourself Jev-like classifier from label descriptions, **calibration**
tools, **bag-of-words baselines**, learning curves and linear probes,
retrieval-augmented generation (RAG), synthetic data generation and quality
checks, and integration with Exploratory Graph Analysis (EGA) for
unsupervised text classification.

Developed and maintained by Dr. Hudson Golino (University of Virginia).

## What's new in 0.2.0

Inspired by Raschka (2026), *Language Models for Text Classification: From
Bag-of-Words to Jev*:

- **Jev from pure R** (no Python): `jev_classify()` (Choice),
  `jev_multilabel()` (Noul), `jev_rate()` (Score), and the building blocks
  `jev_choice()`, `jev_noul()`, `jev_score()`, `jev_ask()` and
  `jev_ask_many()` (parallel requests, retries, cost report).
- **Evaluation and calibration**: `evaluate_classification()` (accuracy,
  macro-F1, kappa, NMI, invalid answers), `compute_nmi()` (works with
  aricode >= 1.1.0 and EGA's unassigned texts), `calibration_metrics()`
  (ECE, Brier, NLL), reliability diagrams, `fit_temperature()`,
  `fit_platt()`, `roc_auc()` and `rlcr_reward()`.
- **More robust LLM classification**: every `classify_*()` function now
  shuffles and chunks the texts, validates the labels, and survives missing
  answers and failed chunks. New `classify_schema()` (strict JSON Schema)
  and `classify_self_consistency()` (vote probabilities).
- **Baselines and probes**: `bow_baseline()` (TF-IDF + logistic
  regression), `learning_curve()`, `cv_logistic()`, `linear_probe()` and
  `probe_layers()` (EGA vs. a linear head, layer by layer).
- **DIY Jev-like classifier**: `embed_sentences()`, `classify_by_description()`
  and `train_shared_head()`.
- **Synthetic data**: `generate_labeled_texts()` and `diversity_report()`.
- **New data**: `extraversion_llama`, `neuroticism`, `label_descriptions` and
  `minilm_embeddings` (pre-computed embeddings, so many examples run offline).
- Updated models (voyage-4, current Groq models), API keys read from
  `~/.Renviron`, and an optional ModernBERT environment.

See [NEWS.md](NEWS.md) for all changes, including behavior changes.

> **0.2.3:** Groq deprecated `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`
> for free and developer-tier keys on August 16, 2026. Every Groq function now
> defaults to models on the free tier: `openai/gpt-oss-120b` for single-model
> functions, and `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and
> `qwen/qwen3.8-27b` for `classify_ensemble()`. Run `list_models()` to see the
> models your key can use today.

> **0.2.1:** `embed_local()` runs in a separate background R process by default,
> so a PyTorch crash can no longer abort your RStudio session. Start every
> session with `library(llmClassificR)` and `activate_llm_env()`, before other
> packages.

## Installation

```r
# Install from GitHub
devtools::install_github("hfgolino/llmClassificR")
```

## Quick Start

Store your API keys in `~/.Renviron` (run `usethis::edit_r_environ()`, add
one line per key, save and restart R). Keys then never appear in your
scripts:

```
GROQ_API_KEY=gsk_...
VOYAGE_API_KEY=pa-...
JINA_API_KEY=jina_...
TYPESAFE_API_KEY=...
```

```r
library(llmClassificR)

# One-time setup (creates the Python environment for Groq, Voyage, HuggingFace)
setup_llm_env()

# Every session
activate_llm_env()
set_groq_key()       # Free at https://console.groq.com
set_voyage_key()     # Free at https://dash.voyageai.com
set_jina_key()       # Free at https://jina.ai/embeddings/
set_typesafe_key()   # Jev: https://console.typesafe.ai

# Load example data
data(extraversion)

# Classify!
results <- classify_direct(extraversion$text, unique(extraversion$subtopic))
evaluate_classification(extraversion$subtopic, results$prediction, "GPT-OSS 120B")
```

## Features

### LLM classification with Groq (9 methods)

Texts are **shuffled and sent in chunks** (40 per call by default), answers
are **validated** against the classes (invalid answers become `NA` and count
as errors) and results come back in the original order.

| Function | Description |
|----------|-------------|
| `classify_direct()` | Simple zero-shot classification |
| `classify_fewshot()` | Few-shot with labeled examples |
| `classify_cot()` | Chain-of-thought with reasoning |
| `classify_json()` | JSON output with verbalized confidence |
| `classify_schema()` | **New.** Strict JSON Schema: labels guaranteed to be valid |
| `classify_ensemble()` | Multi-model majority vote |
| `classify_self_consistency()` | **New.** Vote probabilities from repeated runs |
| `classify_scores()` | Per-class scores (0-100) |
| `classify_judge()` | LLM-as-judge with evidence extraction |
| `run_in_chunks()`, `normalize_labels()` | **New.** Building blocks for your own batch classifiers |

### Jev (TypeSafe AI), pure R

| Function | Description |
|----------|-------------|
| `jev_classify()` | Choice: one class per text, with probabilities |
| `jev_multilabel()` | Noul: one yes/no probability per label |
| `jev_rate()` | Score: ordinal rubric with 2 to 10 levels |
| `jev_choice()`, `jev_noul()`, `jev_score()` | Question builders |
| `jev_ask()`, `jev_ask_many()` | Several questions in one call; many texts in parallel |
| `jev_parse_choice()`, `jev_parse_noul()`, `jev_parse_score()`, `jev_usage()` | Parse answers; tokens and cost |
| `jev_list_models()` | Available Jev versions |

### Evaluation and calibration (offline)

| Function | Description |
|----------|-------------|
| `evaluate_classification()` | Accuracy, macro-F1, Cohen's kappa, NMI, invalid answers |
| `compare_methods()` | Compare methods (and EGA clusters) in one table |
| `compute_nmi()` | NMI without aricode; character labels and `NA` handled |
| `calibration_metrics()` | Accuracy, mean confidence, ECE, Brier, NLL |
| `reliability_table()`, `plot_reliability()` | Reliability diagrams |
| `softmax()`, `fit_temperature()`, `fit_platt()` | Repair over- or under-confidence |
| `expected_calibration_error()`, `brier_score()`, `roc_auc()` | Single metrics |
| `rlcr_reward()` | Why honest confidence pays |

### Baselines, probes and a DIY Jev-like classifier

| Function | Description |
|----------|-------------|
| `bow_baseline()` | TF-IDF + elastic-net logistic regression in one call |
| `bow_features()`, `top_terms()` | Sparse TF-IDF matrix; most predictive words per class |
| `cv_logistic()` | Cross-validated logistic regression (with probabilities) |
| `learning_curve()` | More labeled data or more tuning? |
| `linear_probe()`, `probe_layers()` | Supervised heads on embeddings, layer by layer (with EGA) |
| `embed_sentences()` | Local sentence embeddings (MiniLM; free) |
| `classify_by_description()` | Zero-shot classification from label descriptions |
| `train_shared_head()` | Train a shared head; test it on new label sets |

### Embeddings

| Function | Description |
|----------|-------------|
| `embed_voyage()` | Voyage AI API embeddings (voyage-4; 256 to 2048 dimensions) |
| `embed_jina()` | Jina AI API embeddings: v3 (default), v4 and v5 text models (pure R via httr2, no Python needed) |
| `embed_local()` | Local HuggingFace models, all hidden layers (BERT, RoBERTa, ModernBERT, ...) |
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
| `plot_prime_weight()` | Plot the encoding, next to the true labels (matched colors) |
| `match_communities()` | Rename communities after the labels they overlap most |

### Synthetic data and utilities

| Function | Description |
|----------|-------------|
| `generate_labeled_texts()` | **New.** Labeled synthetic texts with a strict JSON Schema |
| `diversity_report()` | **New.** Duplicates, lexical diversity, distinct-2, within-label overlap |
| `llm_generate()` | Generate text (item creation, summarization) |
| `cosine_similarity()`, `cosine_similarity_matrix()` | Cosine similarities |
| `list_models()` | List available Groq models (live API query) |
| `list_jina_models()` | List available Jina AI embedding models |
| `verify_setup()` | Check environment and API connectivity (Groq, Voyage, Jina, Jev) |

## Example: Start with a Baseline (Runs Offline)

```r
library(llmClassificR)
data(extraversion)

bow <- bow_baseline(extraversion$text, extraversion$subtopic)
#>                        method accuracy macro_F1 kappa   NMI n_invalid
#>  TF-IDF + logistic regression    0.975    0.975 0.967 0.921         0

head(bow$top_terms)        # the most predictive (stemmed) words per facet

# More labeled texts or more tuning?
learning_curve(bow$X, extraversion$subtopic)
```

A cheap, local, interpretable baseline at 97.5% accuracy is the bar every LLM
classifier has to clear.

## Example: Jev (Choice, Noul and Score)

```r
set_typesafe_key()   # reads TYPESAFE_API_KEY
classes <- unique(extraversion$subtopic)

# Choice: one facet per text, with the probability of every facet
res_jev <- jev_classify(
  extraversion$text,
  criteria = label_descriptions$extraversion,
  instructions = "Which facet of extraversion does this text express most clearly?"
)
attr(res_jev, "usage")   # tokens and cost (input tokens only)

evaluate_classification(extraversion$subtopic, res_jev$prediction, "Jev")
calibration_metrics(res_jev[, classes], extraversion$subtopic, "Jev")

# Triage: send the least confident texts to a human coder
res_jev[order(res_jev$confidence)[1:10], c("text", "prediction", "confidence")]

# Noul: multi-label (a text can express more than one facet)
ml <- jev_multilabel(extraversion$text, c(
  Friendly  = "The writer is warm, welcoming and kind toward other people.",
  Assertive = "The writer takes charge and speaks up directly."
))

# Score: an ordinal rubric
data(neuroticism)
intensity <- jev_rate(neuroticism$text,
                      levels = c("None", "Mild", "Clear", "Intense"),
                      instructions = "How intense is the NEGATIVE emotion in this text?")
tapply(intensity$score, neuroticism$subtopic, mean)
```

Jev is proprietary: texts are sent to TypeSafe's servers (check your IRB
protocol), and results can change between versions. Pin the model version
(`model = "jev-1.13.0"`) and validate against human coding.

## Example: Compare LLM Methods

```r
texts <- extraversion$text
classes <- unique(extraversion$subtopic)

res_direct <- classify_direct(texts, classes)
res_schema <- classify_schema(texts, classes)                 # openai/gpt-oss-20b, strict
res_sc     <- classify_self_consistency(texts, classes, n_runs = 5)

compare_methods(
  true_labels = extraversion$subtopic,
  "Bag-of-words"     = bow$predictions,
  "Direct"           = res_direct,
  "Strict schema"    = res_schema,
  "Self-consistency" = res_sc,
  "Jev"              = res_jev
)

# Are the probabilities honest?
rbind(
  calibration_metrics(res_sc[, classes], extraversion$subtopic, "Self-consistency"),
  calibration_metrics(res_jev[, classes], extraversion$subtopic, "Jev")
)
```

## Example: Calibration and a DIY Jev-like Classifier (Runs Offline)

```r
data(minilm_embeddings)
data(neuroticism)
emb <- minilm_embeddings

# Zero-shot from label DESCRIPTIONS only: cosine similarity + softmax
S <- cosine_scores(emb$texts$neuroticism, emb$labels$emotions)
y <- neuroticism$subtopic

# Fit a temperature on half of the texts, evaluate on the other half
set.seed(2026)
calib <- sample(seq_along(y), 60)
test  <- setdiff(seq_along(y), calib)
T_hat <- fit_temperature(S[calib, ], y[calib])

rbind(
  calibration_metrics(softmax(S[test, ]), y[test], "T = 1"),
  calibration_metrics(softmax(S[test, ], T_hat), y[test], "Temperature-scaled")
)
#>               method  n n_missing accuracy mean_confidence   ECE Brier   NLL
#>                T = 1 60         0    0.867           0.300 0.567 0.659 1.216
#>   Temperature-scaled 60         0    0.867           0.941 0.074 0.177 0.368

rel <- rbind(reliability_table(softmax(S[test, ]), y[test], "T = 1"),
             reliability_table(softmax(S[test, ], T_hat), y[test], "Temperature-scaled"))
plot_reliability(rel)
```

Same accuracy, honest confidence. With your own texts,
`classify_by_description(texts, descriptions)` embeds them locally with
`embed_sentences()` (or with `embed_voyage` / `embed_jina`).

## Example: Full RAG Pipeline (No Labels Needed)

```r
# Discover topics and classify: fully unsupervised
pipeline <- classify_with_rag(extraversion$text, n_topics = 4)
print(pipeline$topics)                          # What topics were discovered?
compute_nmi(extraversion$subtopic, pipeline$classifications$prediction)
```

## Example: Embeddings + EGA

```r
# Voyage AI embeddings (requires Python environment)
embeddings <- embed_voyage(extraversion$text)
result <- ega_classify(embeddings)
compute_nmi(extraversion$subtopic, result$communities)   # unsupervised
linear_probe(embeddings, extraversion$subtopic)         # supervised head

# Jina AI embeddings (pure R: no Python needed!)
jina_emb <- embed_jina(extraversion$text)                  # jina-embeddings-v3
result2 <- ega_classify(jina_emb)
compute_nmi(extraversion$subtopic, result2$communities)

# Newer Jina v5 text models: v5-text-small (1024-dim) or v5-text-nano (768-dim)
jina_v5 <- embed_jina(extraversion$text, model = "jina-embeddings-v5-text-nano")

# Layer by layer: EGA vs. a linear head (BERT or ModernBERT)
emb <- embed_local(extraversion$text, model_name = "bert-base-uncased")
probe_layers(emb$layers, extraversion$subtopic, ega = TRUE)
```

For ModernBERT, create a second environment once with
`setup_llm_env("modernbert_env", modernbert = TRUE)` and start a fresh R
session with `activate_llm_env("modernbert_env")`.

## Datasets

| Dataset | Description |
|---------|-------------|
| `extraversion` | 120 first-person extraversion descriptions (4 facets; gpt-oss-20b) |
| `extraversion_3p` | 120 third-person extraversion descriptions (4 facets; gpt-oss-120b) |
| `extraversion_llama` | 120 extraversion descriptions (4 facets; llama-3.1-8b) |
| `neuroticism` | 120 emotion-laden texts (Sad, Excited, Angry, Friendly) |
| `personality_descriptions` | 110 Big Five personality descriptions (5 traits) |
| `label_descriptions` | Label descriptions (Jev-style criteria) for three tasks |
| `minilm_embeddings` | Pre-computed MiniLM embeddings of all texts and label descriptions |

## Dependencies

**R packages** (required): reticulate, jsonlite, httr2 (>= 1.1.1), stats, cluster, utils, graphics, grDevices, callr, clue

**R packages** (suggested): EGAnet, glmnet, Matrix, tm, SnowballC, ggplot2, aricode, ggpubr, tidyr, dplyr, corrplot

**Python** (managed automatically via UV): groq, voyageai, torch, transformers

> **Note:** Jev (`jev_*()`), `embed_jina()`, and all evaluation, calibration
> and baseline functions use pure R and do **not** require the Python
> environment.

## Citation

If you use llmClassificR in your research, please cite:

```
Golino, H. (2026). llmClassificR: Text Classification with Large Language Models in R.
R package version 0.2.3. https://github.com/hfgolino/llmClassificR
```

## License

GPL-3
