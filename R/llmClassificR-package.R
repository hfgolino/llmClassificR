#' @keywords internal
"_PACKAGE"
#' @title llmClassificR: Text Classification with Large Language Models in R
#'
#' @description A user-friendly toolkit for text classification with Large
#' Language Models in R. It covers the whole path from bag-of-words to Jev:
#' robust batch zero-shot classification with Groq LLMs (direct, few-shot,
#' chain-of-thought, JSON, strict JSON Schema, ensembles, self-consistency,
#' scoring and LLM-as-judge), Jev "System One" classification (TypeSafe AI)
#' from pure R, embeddings (Voyage AI, Jina AI and local HuggingFace
#' models), a do-it-yourself Jev-like classifier from label descriptions,
#' calibration tools (ECE, Brier score, reliability diagrams, temperature and
#' Platt scaling), bag-of-words baselines, learning curves and linear probes,
#' retrieval-augmented generation (RAG), synthetic data generation and
#' quality checks, and Exploratory Graph Analysis (EGA) for unsupervised
#' text classification.
#'
#' @section Getting Started:
#' Store your API keys in \code{~/.Renviron} (one per line; e.g., with
#' \code{usethis::edit_r_environ()}) and restart R:
#' \preformatted{
#' GROQ_API_KEY=gsk_...
#' VOYAGE_API_KEY=pa-...
#' JINA_API_KEY=jina_...
#' TYPESAFE_API_KEY=...
#' }
#' The Groq and Voyage AI functions and the local HuggingFace models need a
#' Python environment managed by UV. Run \code{\link{setup_llm_env}} once,
#' then \code{\link{activate_llm_env}} at the start of each R session. Jev
#' and Jina AI are called from pure R (httr2), and the calibration,
#' evaluation and baseline functions run fully offline.
#'
#' \preformatted{
#' # One-time setup
#' setup_llm_env()
#'
#' # Every session
#' activate_llm_env()
#' set_groq_key()       # reads GROQ_API_KEY
#' set_voyage_key()     # reads VOYAGE_API_KEY
#' set_typesafe_key()   # reads TYPESAFE_API_KEY (Jev)
#' }
#'
#' @section Setup Functions:
#' \itemize{
#'   \item \code{\link{setup_llm_env}} - Create the Python environment (run once; \code{modernbert = TRUE} for ModernBERT)
#'   \item \code{\link{activate_llm_env}} - Activate the environment (every session)
#'   \item \code{\link{set_groq_key}} - Set Groq API key (free at \url{https://console.groq.com})
#'   \item \code{\link{set_voyage_key}} - Set Voyage AI API key (free at \url{https://dash.voyageai.com})
#'   \item \code{\link{set_jina_key}} - Set Jina AI API key (free at \url{https://jina.ai/embeddings/})
#'   \item \code{\link{set_typesafe_key}} - Set TypeSafe AI (Jev) API key (\url{https://console.typesafe.ai})
#'   \item \code{\link{verify_setup}} - Check environment and API connectivity
#'   \item \code{\link{list_models}} - List available Groq models
#'   \item \code{\link{list_jina_models}} - List available Jina AI embedding models
#'   \item \code{\link{jev_list_models}} - List available Jev models
#' }
#'
#' @section LLM Classification Functions (Groq):
#' All functions accept a character vector of texts and a character vector
#' of class labels, and return a \code{data.frame} with one row per text.
#' Texts are shuffled and sent in chunks (40 per call by default); answers
#' are validated against the classes (invalid answers are \code{NA}).
#'
#' \itemize{
#'   \item \code{\link{classify_direct}} - Simple zero-shot classification
#'   \item \code{\link{classify_fewshot}} - Few-shot with user-provided labeled examples
#'   \item \code{\link{classify_cot}} - Chain-of-thought reasoning with explanations
#'   \item \code{\link{classify_json}} - JSON output with verbalized confidence
#'   \item \code{\link{classify_schema}} - Strict JSON Schema (labels guaranteed to be valid)
#'   \item \code{\link{classify_ensemble}} - Multi-model majority vote
#'   \item \code{\link{classify_self_consistency}} - Vote probabilities from repeated runs
#'   \item \code{\link{classify_scores}} - Per-class scores (0-100)
#'   \item \code{\link{classify_judge}} - LLM-as-judge with evidence extraction
#'   \item \code{\link{run_in_chunks}}, \code{\link{normalize_labels}},
#'     \code{\link{classification_schema}} - Building blocks for your own batch classifiers
#' }
#'
#' @section Jev (TypeSafe AI) Functions:
#' Jev answers typed questions about a text with probabilities (pure R, no
#' Python).
#' \itemize{
#'   \item \code{\link{jev_classify}} - Choice: one class per text, with probabilities
#'   \item \code{\link{jev_multilabel}} - Noul: one yes/no probability per label
#'   \item \code{\link{jev_rate}} - Score: ordinal rubric (2 to 10 levels)
#'   \item \code{\link{jev_choice}}, \code{\link{jev_noul}}, \code{\link{jev_score}} - Question builders
#'   \item \code{\link{jev_ask}}, \code{\link{jev_ask_many}} - Several questions in one call; many texts in parallel
#'   \item \code{\link{jev_parse_choice}}, \code{\link{jev_parse_noul}}, \code{\link{jev_parse_score}}, \code{\link{jev_usage}} - Parse answers; tokens and cost
#' }
#'
#' @section Evaluation and Calibration:
#' \itemize{
#'   \item \code{\link{evaluate_classification}} - Accuracy, macro-F1, kappa, NMI and invalid answers
#'   \item \code{\link{compare_methods}} - Compare several methods (and EGA clusters) in one table
#'   \item \code{\link{compute_nmi}} - NMI without aricode (handles character labels and NA)
#'   \item \code{\link{roc_auc}} - Area under the ROC curve for yes/no probabilities
#'   \item \code{\link{calibration_metrics}} - Accuracy, confidence, ECE, Brier and NLL
#'   \item \code{\link{expected_calibration_error}}, \code{\link{brier_score}} - Single calibration metrics
#'   \item \code{\link{reliability_table}}, \code{\link{plot_reliability}} - Reliability diagrams
#'   \item \code{\link{softmax}}, \code{\link{fit_temperature}}, \code{\link{fit_platt}} - Repair confidence
#'   \item \code{\link{rlcr_reward}} - Why honest confidence pays (RLCR)
#' }
#'
#' @section Do-It-Yourself Jev-like Classifier:
#' \itemize{
#'   \item \code{\link{embed_sentences}} - Local sentence embeddings (MiniLM; free)
#'   \item \code{\link{cosine_scores}} - Cosine similarity between texts and label descriptions
#'   \item \code{\link{classify_by_description}} - Zero-shot classification from label descriptions
#'   \item \code{\link{train_shared_head}} - Train a shared head; test it on new label sets
#' }
#'
#' @section Baselines and Probes:
#' \itemize{
#'   \item \code{\link{bow_baseline}} - TF-IDF + logistic regression in one call
#'   \item \code{\link{bow_features}}, \code{\link{top_terms}} - Features and interpretable top words
#'   \item \code{\link{cv_logistic}} - Cross-validated elastic-net logistic regression (with probabilities)
#'   \item \code{\link{learning_curve}} - More data or more tuning?
#'   \item \code{\link{linear_probe}}, \code{\link{probe_layers}} - Supervised heads on embeddings, layer by layer
#' }
#'
#' @section Synthetic Data:
#' \itemize{
#'   \item \code{\link{generate_labeled_texts}} - Labeled texts with a strict JSON Schema
#'   \item \code{\link{diversity_report}} - Duplicates, lexical diversity, distinct-2 and overlap
#'   \item \code{\link{llm_generate}} - General-purpose text generation (item creation, summarization)
#' }
#'
#' @section Embedding Functions:
#' \itemize{
#'   \item \code{\link{embed_voyage}} - Voyage AI API embeddings (voyage-4; 256 to 2048 dimensions)
#'   \item \code{\link{embed_jina}} - Jina AI API embeddings (pure R, no Python needed)
#'   \item \code{\link{embed_local}} - Local HuggingFace encoder models (all hidden layers; BERT, ModernBERT, ...)
#'   \item \code{\link{evaluate_layers}} - Evaluate layer quality using silhouette, PCA, and separation ratio
#' }
#'
#' @section RAG (Retrieval-Augmented Generation):
#' \itemize{
#'   \item \code{\link{retrieve_similar}} - Retrieve most similar documents by cosine similarity
#'   \item \code{\link{rag_generate}} - Generate answers grounded in retrieved document context
#'   \item \code{\link{discover_topics}} - Automatic topic discovery (no prior labels needed)
#'   \item \code{\link{classify_with_rag}} - Full unsupervised RAG classification pipeline
#' }
#'
#' @section EGA Integration:
#' \itemize{
#'   \item \code{\link{ega_classify}} - Exploratory Graph Analysis on embedding matrices
#'   \item \code{\link{prime_weight_encode}} - Prime-weight encoding across transformer layers
#' }
#'
#' @section Utility Functions:
#' \itemize{
#'   \item \code{\link{cosine_similarity}} - Cosine similarity between two vectors
#'   \item \code{\link{cosine_similarity_matrix}} - Pairwise cosine similarity matrix
#' }
#'
#' @section Datasets:
#' \itemize{
#'   \item \code{\link{extraversion}} - 120 first-person extraversion descriptions (4 facets; gpt-oss-20b)
#'   \item \code{\link{extraversion_3p}} - 120 third-person extraversion descriptions (4 facets; gpt-oss-120b)
#'   \item \code{\link{extraversion_llama}} - 120 extraversion descriptions (4 facets; llama-3.1-8b)
#'   \item \code{\link{neuroticism}} - 120 emotion-laden texts (Sad, Excited, Angry, Friendly)
#'   \item \code{\link{personality_descriptions}} - 110 Big Five personality descriptions (5 traits)
#'   \item \code{\link{label_descriptions}} - Label descriptions (Jev-style criteria) for three tasks
#'   \item \code{\link{minilm_embeddings}} - Pre-computed MiniLM embeddings (run examples offline)
#' }
#'
#' @section Quick Example:
#' \preformatted{
#' library(llmClassificR)
#' data(extraversion)
#' classes <- unique(extraversion$subtopic)
#'
#' # A cheap local baseline first (no API)
#' bow <- bow_baseline(extraversion$text, extraversion$subtopic)
#'
#' # LLM and Jev
#' res_llm <- classify_direct(extraversion$text, classes)
#' res_jev <- jev_classify(extraversion$text, label_descriptions$extraversion)
#'
#' # Compare
#' compare_methods(
#'   true_labels = extraversion$subtopic,
#'   "Bag-of-words" = bow$predictions,
#'   "Llama 3.3 70B" = res_llm,
#'   "Jev" = res_jev
#' )
#'
#' # Are Jev's probabilities honest?
#' calibration_metrics(res_jev[, classes], extraversion$subtopic, "Jev")
#' }
#'
#' @references
#' Raschka, S. (2026, September 29). Language models for text
#' classification: From bag-of-words to Jev. \emph{Ahead of AI}.
#' \url{https://magazine.sebastianraschka.com/p/classifier-history-and-jev}
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @name llmClassificR-package
#'
#' @aliases llmClassificR
#'
#' @keywords package
#'
NULL

# Internal package environment for storing state
.pkg_env <- new.env(parent = emptyenv())

# Column names used inside ggplot2::aes() (tidy evaluation)
utils::globalVariables(".data")
