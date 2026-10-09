# ==============================================================================
# Batch zero-shot classification with LLMs on Groq
# ==============================================================================
#
# All classify_*() functions share the same robust batch engine (since 0.2.0):
#   - texts are SHUFFLED and sent in CHUNKS (default 40 texts per call), so the
#     prompt is never ordered by label and never too long;
#   - answers are mapped back to the allowed classes (normalize_labels());
#     anything else becomes NA instead of a new class;
#   - a missing answer or a failed chunk gives NA for those texts instead of
#     an error for the whole batch;
#   - results always come back in the ORIGINAL order of `texts`.
# Set chunk_size = NULL and shuffle = FALSE to reproduce the 0.1.0 behavior
# (one call with all texts in their original order).

#' Batch settings shared by the classify functions
#'
#' @section Batch settings:
#' \describe{
#'   \item{\code{chunk_size}}{Texts per API call. Default \code{40}. Use
#'     \code{NULL} to send all texts in one call.}
#'   \item{\code{shuffle}}{Shuffle texts before chunking, so texts sorted by
#'     label do not bias the model. Results are returned in the original
#'     order. Default \code{TRUE}.}
#'   \item{\code{seed}}{Seed for the shuffle (the global random state is
#'     restored). Default \code{2026}.}
#' }
#' Answers are mapped to \code{classes} ignoring case and punctuation (see
#' \code{\link{normalize_labels}}); invalid or missing answers are \code{NA}.
#' If a whole chunk fails (for example, invalid JSON), its texts get \code{NA}
#' and a warning is shown. The pause between chunks is
#' \code{getOption("llmClassificR.pause", 2)} seconds; set
#' \code{options(llmClassificR.verbose = FALSE)} to hide the progress messages.
#' @name batch_settings
#' @keywords internal
NULL


#' Direct zero-shot text classification
#'
#' Classifies texts into categories with a Groq LLM. This is the simplest and
#' fastest classification approach: the model receives a numbered list of
#' texts and returns one category for each, as JSON.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'   See \code{\link{list_models}} for options.
#' @param chunk_size Number of texts per API call. Default \code{40};
#'   \code{NULL} sends all texts in one call.
#' @param shuffle Logical. Shuffle the texts before chunking (results are
#'   returned in the original order). Default \code{TRUE}.
#' @param seed Seed for the shuffle. Default \code{2026}.
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index of each text}
#'   \item{text}{The original text (truncated to 200 characters)}
#'   \item{prediction}{The predicted category (one of \code{classes}, or
#'     \code{NA} if the model gave no valid answer)}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' data(extraversion)
#' activate_llm_env()
#' set_groq_key()   # reads GROQ_API_KEY from ~/.Renviron
#'
#' results <- classify_direct(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic)
#' )
#' head(results)
#'
#' # Evaluate
#' evaluate_classification(extraversion$subtopic, results$prediction, "Direct")
#' table(True = extraversion$subtopic, Predicted = results$prediction, useNA = "ifany")
#'
#' # The 0.1.0 behavior: one call, texts in their original order
#' results_one_call <- classify_direct(extraversion$text, unique(extraversion$subtopic),
#'                                     chunk_size = NULL, shuffle = FALSE)
#' }
#'
#' @seealso \code{\link{classify_fewshot}}, \code{\link{classify_cot}},
#'   \code{\link{classify_schema}}, \code{\link{evaluate_classification}},
#'   \code{\link{compare_methods}}
#'
#' @export
classify_direct <- function(texts, classes,
                            model = "llama-3.3-70b-versatile",
                            chunk_size = 40, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  chunk_fun <- function(x) {
    prompt <- sprintf(
      'Classify each numbered text into exactly ONE of these categories: %s

%s

Respond with ONLY a JSON object where keys are the text numbers (as strings)
and values are the category names.
Example format: {"1": "category_a", "2": "category_b"}',
      paste(classes, collapse = ", "), .format_numbered_texts(x)
    )
    content <- .groq_chat(prompt, model = model,
                          max_tokens = length(x) * 30L + 200L,
                          response_format = list(type = "json_object"))
    results <- .parse_json(content)
    raw <- vapply(seq_along(x), function(i) .get_string(results, i), character(1))
    data.frame(prediction = normalize_labels(raw, classes), stringsAsFactors = FALSE)
  }

  res <- .batch_classify(texts, chunk_fun, chunk_size, shuffle, seed,
                         template = data.frame(prediction = NA_character_))
  .report_invalid(res$prediction)
  .with_text_cols(texts, res)
}


#' Few-shot text classification
#'
#' Classifies texts by providing labeled examples to guide the model. The
#' model sees example text-label pairs before classifying the target texts.
#' This often improves accuracy over direct zero-shot, especially for
#' ambiguous categories. Never evaluate on the texts used as examples.
#'
#' @inheritParams classify_direct
#' @param examples A data.frame with columns \code{text} and \code{label},
#'   containing labeled examples (2-3 per class is typical).
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with columns \code{text_id}, \code{text} and
#'   \code{prediction}.
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' # Examples: first 2 texts from each class
#' example_idx <- unlist(lapply(unique(extraversion$subtopic), function(cls) {
#'   head(which(extraversion$subtopic == cls), 2)
#' }))
#' examples <- data.frame(text = extraversion$text[example_idx],
#'                        label = extraversion$subtopic[example_idx])
#'
#' # Classify (and evaluate) only the remaining texts
#' test_idx <- setdiff(seq_len(nrow(extraversion)), example_idx)
#' results <- classify_fewshot(
#'   texts = extraversion$text[test_idx],
#'   classes = unique(extraversion$subtopic),
#'   examples = examples
#' )
#' evaluate_classification(extraversion$subtopic[test_idx], results$prediction, "Few-shot")
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_cot}}
#'
#' @export
classify_fewshot <- function(texts, classes, examples,
                             model = "llama-3.3-70b-versatile",
                             chunk_size = 40, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  if (!all(c("text", "label") %in% names(examples))) {
    stop("'examples' must be a data.frame with columns 'text' and 'label'.", call. = FALSE)
  }
  if (any(examples$text %in% texts)) {
    warning("Some few-shot examples are also in 'texts'; evaluate only on the other texts.",
            call. = FALSE)
  }

  # Shuffle the examples too, so they are not ordered by label
  ex <- examples[.with_seed(seed, sample(nrow(examples))), , drop = FALSE]
  example_str <- paste(vapply(seq_len(nrow(ex)), function(i) {
    sprintf('Text: "%s"\nCategory: %s', substr(ex$text[i], 1, 300), ex$label[i])
  }, character(1)), collapse = "\n\n")

  chunk_fun <- function(x) {
    prompt <- sprintf(
      'Classify each numbered text into one of these categories: %s

Here are some labeled examples:
%s

Now classify ALL of these texts:
%s

Respond with ONLY a JSON object where keys are the text numbers (as strings)
and values are the category names.',
      paste(classes, collapse = ", "), example_str, .format_numbered_texts(x)
    )
    content <- .groq_chat(prompt, model = model,
                          max_tokens = length(x) * 30L + 200L,
                          response_format = list(type = "json_object"))
    results <- .parse_json(content)
    raw <- vapply(seq_along(x), function(i) .get_string(results, i), character(1))
    data.frame(prediction = normalize_labels(raw, classes), stringsAsFactors = FALSE)
  }

  res <- .batch_classify(texts, chunk_fun, chunk_size, shuffle, seed,
                         template = data.frame(prediction = NA_character_))
  .report_invalid(res$prediction)
  .with_text_cols(texts, res)
}


#' Chain-of-thought text classification
#'
#' Classifies texts by asking the model to reason briefly before assigning a
#' category. Returns both the prediction and the reasoning for each text.
#'
#' @inheritParams classify_direct
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{text}{Original text (truncated)}
#'   \item{prediction}{Predicted category (or \code{NA})}
#'   \item{reasoning}{Model's reasoning for the classification}
#' }
#'
#' @details Reasoning models (\code{openai/gpt-oss-20b},
#'   \code{openai/gpt-oss-120b}, \code{qwen/qwen3.8-27b}) already "think"
#'   internally; with them, explicit chain-of-thought prompting usually adds
#'   little.
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' results <- classify_cot(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic)
#' )
#'
#' # Inspect reasoning for first 3 texts
#' results[1:3, c("prediction", "reasoning")]
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_json}}
#'
#' @export
classify_cot <- function(texts, classes,
                         model = "llama-3.3-70b-versatile",
                         chunk_size = 40, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  system_prompt <- "You are an expert text classifier. For each text, briefly
analyze the key features before classifying."

  chunk_fun <- function(x) {
    prompt <- sprintf(
      'For each numbered text below, classify it into exactly ONE of these categories: %s

%s

Think step by step for each text. Then return a JSON object with this structure:
{
  "1": {"reasoning": "brief analysis", "category": "chosen_category"},
  "2": {"reasoning": "brief analysis", "category": "chosen_category"}
}
Return ONLY valid JSON.',
      paste(classes, collapse = ", "), .format_numbered_texts(x)
    )
    content <- .groq_chat(prompt, model = model, system = system_prompt,
                          max_tokens = length(x) * 80L + 200L,
                          response_format = list(type = "json_object"))
    results <- .parse_json(content)
    data.frame(
      prediction = normalize_labels(
        vapply(seq_along(x), function(i) .get_string(results, i, "category"),
               character(1)), classes),
      reasoning = vapply(seq_along(x), function(i) .get_string(results, i, "reasoning"),
                         character(1)),
      stringsAsFactors = FALSE
    )
  }

  res <- .batch_classify(texts, chunk_fun, chunk_size, shuffle, seed,
                         template = data.frame(prediction = NA_character_,
                                               reasoning = NA_character_))
  .report_invalid(res$prediction)
  .with_text_cols(texts, res)
}


#' Structured JSON text classification with confidence
#'
#' Classifies texts and returns the predicted category, a verbalized
#' confidence level (high/medium/low) and a brief reasoning for each text.
#' Uses JSON mode. For output that is \emph{guaranteed} to follow the format
#' (categories restricted to \code{classes}), see \code{\link{classify_schema}}.
#'
#' @inheritParams classify_direct
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with columns \code{text_id}, \code{text},
#'   \code{prediction}, \code{confidence} ("high", "medium" or "low") and
#'   \code{reasoning}.
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' results <- classify_json(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic)
#' )
#'
#' # Is stated confidence related to accuracy?
#' tapply(results$prediction == extraversion$subtopic, results$confidence, mean)
#' }
#'
#' @seealso \code{\link{classify_schema}}, \code{\link{classify_judge}}
#'
#' @export
classify_json <- function(texts, classes,
                          model = "llama-3.3-70b-versatile",
                          chunk_size = 40, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  chunk_fun <- function(x) {
    prompt <- sprintf(
      'Classify each numbered text into one of these categories: %s

%s

For each text, provide:
- "category": the assigned category (must be one of the listed categories)
- "confidence": your confidence level ("high", "medium", or "low")
- "reasoning": a brief one-sentence explanation

Return a JSON object with text numbers as keys:
{
  "1": {"category": "cat_name", "confidence": "high", "reasoning": "explanation"},
  "2": {"category": "cat_name", "confidence": "medium", "reasoning": "explanation"}
}
Return ONLY valid JSON.',
      paste(classes, collapse = ", "), .format_numbered_texts(x)
    )
    content <- .groq_chat(prompt, model = model,
                          max_tokens = length(x) * 80L + 200L,
                          response_format = list(type = "json_object"))
    results <- .parse_json(content)
    get <- function(field) vapply(seq_along(x), function(i) {
      .get_string(results, i, field)
    }, character(1))
    data.frame(prediction = normalize_labels(get("category"), classes),
               confidence = normalize_labels(get("confidence"), c("high", "medium", "low")),
               reasoning = get("reasoning"),
               stringsAsFactors = FALSE)
  }

  res <- .batch_classify(texts, chunk_fun, chunk_size, shuffle, seed,
                         template = data.frame(prediction = NA_character_,
                                               confidence = NA_character_,
                                               reasoning = NA_character_))
  .report_invalid(res$prediction)
  .with_text_cols(texts, res)
}


#' Classification with a strict JSON Schema (constrained decoding)
#'
#' With a JSON Schema in strict mode, the model can only produce output that
#' matches the schema: every text gets an \code{id}, and the category MUST be
#' one of \code{classes} (an \code{enum}). Parsing cannot fail and invalid
#' labels cannot occur.
#'
#' @inheritParams classify_direct
#' @param model A Groq model that supports strict structured outputs. Default
#'   is \code{"openai/gpt-oss-20b"}; \code{"openai/gpt-oss-120b"} and
#'   \code{"qwen/qwen3.8-27b"} also work (September 2026).
#' @param with_confidence Logical. Also return a verbalized confidence
#'   (high/medium/low) and a one-sentence reasoning. Default \code{TRUE}.
#' @param chunk_size Number of texts per API call. Default \code{30}.
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with \code{text_id}, \code{text}, \code{prediction}
#'   and (if \code{with_confidence = TRUE}) \code{confidence} and
#'   \code{reasoning}.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' # The schema sent to the model (runs without an API key)
#' str(classification_schema(c("Friendly", "Positive")), max.level = 3)
#'
#' \dontrun{
#' data(extraversion)
#' res <- classify_schema(extraversion$text, unique(extraversion$subtopic))
#' table(res$prediction, useNA = "ifany")   # only the four facets (or NA)
#' evaluate_classification(extraversion$subtopic, res$prediction, "Strict schema")
#' }
#'
#' @seealso \code{\link{classify_json}}, \code{\link{generate_labeled_texts}}
#' @export
classify_schema <- function(texts, classes,
                            model = "openai/gpt-oss-20b",
                            with_confidence = TRUE,
                            chunk_size = 30, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  schema_format <- list(
    type = "json_schema",
    json_schema = list(name = "classifications", strict = TRUE,
                       schema = classification_schema(classes, with_confidence))
  )

  chunk_fun <- function(x) {
    extra <- if (with_confidence) {
      ', your "confidence" ("high", "medium", or "low") and a one-sentence "reasoning"'
    } else {
      ""
    }
    prompt <- sprintf(
      'Classify each numbered text into one of these categories: %s

%s

For each text return its number as "id", the "category"%s.',
      paste(classes, collapse = ", "), .format_numbered_texts(x), extra
    )
    content <- .groq_chat(prompt, model = model,
                          max_tokens = length(x) * (if (with_confidence) 80L else 25L) + 300L,
                          response_format = schema_format)
    results <- .parse_json(content)$classifications
    m <- match(seq_along(x), results$id)   # align by id, whatever the order
    out <- data.frame(prediction = normalize_labels(results$category[m], classes),
                      stringsAsFactors = FALSE)
    if (with_confidence) {
      out$confidence <- as.character(results$confidence[m])
      out$reasoning <- as.character(results$reasoning[m])
    }
    out
  }

  template <- data.frame(prediction = NA_character_)
  if (with_confidence) {
    template$confidence <- NA_character_
    template$reasoning <- NA_character_
  }
  res <- .batch_classify(texts, chunk_fun, chunk_size, shuffle, seed, template)
  .report_invalid(res$prediction)
  .with_text_cols(texts, res)
}


#' JSON Schema for batch classification
#'
#' Builds the JSON Schema used by \code{\link{classify_schema}}: an object
#' with an array \code{classifications}; each item has an integer \code{id}
#' and a \code{category} restricted to \code{classes}.
#'
#' @param classes Character vector of allowed categories.
#' @param with_confidence Logical. Add \code{confidence} (high/medium/low)
#'   and \code{reasoning} fields. Default \code{TRUE}.
#'
#' @return A nested list (converted to JSON when sent to the API).
#'
#' @examples
#' schema <- classification_schema(c("Friendly", "Positive", "Assertive", "Energetic"))
#' cat(jsonlite::toJSON(schema, auto_unbox = TRUE, pretty = TRUE))
#'
#' @export
classification_schema <- function(classes, with_confidence = TRUE) {
  item_properties <- list(
    id = list(type = "integer"),
    category = list(type = "string", enum = as.list(classes))
  )
  required <- list("id", "category")
  if (with_confidence) {
    item_properties$confidence <- list(type = "string", enum = list("high", "medium", "low"))
    item_properties$reasoning <- list(type = "string")
    required <- c(required, list("confidence", "reasoning"))
  }
  list(
    type = "object",
    properties = list(
      classifications = list(
        type = "array",
        items = list(type = "object", properties = item_properties,
                     required = required, additionalProperties = FALSE)
      )
    ),
    required = list("classifications"),
    additionalProperties = FALSE
  )
}


#' Multi-model ensemble classification
#'
#' Classifies texts with several LLMs (each with \code{\link{classify_direct}})
#' and takes a majority vote. Ties are broken by the order of \code{models}.
#'
#' @inheritParams classify_direct
#' @param models Character vector of Groq model names. Default uses three
#'   production models: \code{llama-3.3-70b-versatile},
#'   \code{llama-3.1-8b-instant} and \code{openai/gpt-oss-20b}.
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{text}{Original text (truncated)}
#'   \item{prediction}{Majority vote prediction}
#'   \item{agreement}{Logical: did all models with a valid answer agree?}
#'   \item{votes}{Comma-separated individual model predictions}
#' }
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' results <- classify_ensemble(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic)
#' )
#'
#' # How often do all models agree? Are unanimous votes more accurate?
#' mean(results$agreement)
#' tapply(results$prediction == extraversion$subtopic, results$agreement, mean)
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_self_consistency}},
#'   \code{\link{list_models}}
#'
#' @export
classify_ensemble <- function(texts, classes,
                              models = c("llama-3.3-70b-versatile",
                                         "llama-3.1-8b-instant",
                                         "openai/gpt-oss-20b"),
                              chunk_size = 40, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  cat("Running ensemble across", length(models), "models...\n")

  model_preds <- list()
  for (m in seq_along(models)) {
    cat("  Model", m, ":", models[m], "\n")
    model_preds[[m]] <- tryCatch({
      res <- classify_direct(texts, classes, model = models[m],
                             chunk_size = chunk_size, shuffle = shuffle, seed = seed)
      res$prediction
    }, error = function(e) {
      cat("  FAILED:", conditionMessage(e), "\n")
      rep(NA_character_, length(texts))
    })
  }

  predictions <- character(length(texts))
  agreement <- logical(length(texts))
  votes_str <- character(length(texts))

  for (i in seq_along(texts)) {
    votes <- vapply(model_preds, function(p) p[i], character(1))
    votes_str[i] <- paste(votes, collapse = ", ")
    valid <- votes[!is.na(votes)]
    if (length(valid) == 0) {
      predictions[i] <- NA_character_
      agreement[i] <- FALSE
    } else {
      counts <- table(factor(valid, levels = unique(valid)))   # ties -> first model
      predictions[i] <- names(counts)[which.max(counts)]
      agreement[i] <- length(unique(valid)) == 1
    }
  }

  cat("Model agreement rate:", round(mean(agreement) * 100, 1), "%\n")

  data.frame(
    text_id = seq_along(texts),
    text = substr(texts, 1, 200),
    prediction = predictions,
    agreement = agreement,
    votes = votes_str,
    stringsAsFactors = FALSE
  )
}


#' Self-consistency classification (vote probabilities)
#'
#' Asks the same model the same question several times, with some randomness
#' (\code{temperature > 0}) and a different text order every run. The share
#' of runs that voted for each class is a probability estimate: 5 of 5 votes
#' gives 1.0, 3 of 5 gives 0.6. This gives probabilities without access to
#' token log-probabilities, so they can be evaluated with
#' \code{\link{calibration_metrics}}.
#'
#' @inheritParams classify_direct
#' @param n_runs Number of runs. Default \code{5}. Cost grows linearly.
#' @param temperature Sampling temperature. Default \code{0.8}.
#'
#' @return A data.frame with \code{text_id}, \code{text}, \code{prediction}
#'   (majority vote), \code{confidence} (vote share of the prediction) and one
#'   probability column per class. The attribute \code{"votes"} holds the
#'   texts x runs matrix of individual answers.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' classes <- unique(extraversion$subtopic)
#'
#' sc <- classify_self_consistency(extraversion$text, classes, n_runs = 5)
#' head(sc)
#' calibration_metrics(as.matrix(sc[, classes]), extraversion$subtopic, "Self-consistency")
#' }
#'
#' @seealso \code{\link{classify_ensemble}}, \code{\link{calibration_metrics}}
#' @export
classify_self_consistency <- function(texts, classes, n_runs = 5,
                                      temperature = 0.8,
                                      model = "llama-3.3-70b-versatile",
                                      chunk_size = 40, seed = 2026) {
  classes <- as.character(classes)
  votes <- vapply(seq_len(n_runs), function(r) {
    cat("Self-consistency run", r, "of", n_runs, "\n")
    chunk_fun <- function(x) {
      prompt <- sprintf(
        'Classify each numbered text into exactly ONE of these categories: %s

%s

Respond with ONLY a JSON object where keys are the text numbers (as strings)
and values are the category names.',
        paste(classes, collapse = ", "), .format_numbered_texts(x)
      )
      content <- .groq_chat(prompt, model = model, temperature = temperature,
                            max_tokens = length(x) * 30L + 200L,
                            response_format = list(type = "json_object"))
      results <- .parse_json(content)
      raw <- vapply(seq_along(x), function(i) .get_string(results, i), character(1))
      data.frame(prediction = normalize_labels(raw, classes), stringsAsFactors = FALSE)
    }
    # A different shuffle in every run
    .batch_classify(texts, chunk_fun, chunk_size, shuffle = TRUE,
                    seed = if (is.null(seed)) NULL else seed + r,
                    template = data.frame(prediction = NA_character_))$prediction
  }, character(length(texts)))
  votes <- matrix(votes, nrow = length(texts))

  probs <- matrix(vapply(seq_len(nrow(votes)), function(i) {
    v <- votes[i, ]
    v <- v[!is.na(v)]
    if (length(v) == 0) return(rep(NA_real_, length(classes)))
    vapply(classes, function(cl) mean(v == cl), numeric(1))
  }, numeric(length(classes))), ncol = length(classes), byrow = TRUE,
  dimnames = list(NULL, classes))
  answered <- stats::complete.cases(probs)
  prediction <- rep(NA_character_, length(texts))
  prediction[answered] <- classes[max.col(probs[answered, , drop = FALSE], ties.method = "first")]
  confidence <- rep(NA_real_, length(texts))
  confidence[answered] <- apply(probs[answered, , drop = FALSE], 1, max)

  out <- data.frame(text_id = seq_along(texts), text = substr(texts, 1, 200),
                    prediction = prediction, confidence = confidence,
                    probs, check.names = FALSE, stringsAsFactors = FALSE)
  attr(out, "votes") <- votes
  .report_invalid(prediction)
  out
}


#' Classification with per-class scores
#'
#' Instead of assigning a single label, this function asks the model to
#' rate how well each text matches each category on a 0-100 scale
#' (verbalized scores). Rescaled to sum to 1, they can be treated as
#' probabilities and checked with \code{\link{calibration_metrics}}.
#'
#' @inheritParams classify_direct
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{prediction}{Predicted category (highest score), \code{NA} if the
#'     text received no scores}
#'   \item{...}{One numeric column per class (named as in \code{classes})
#'     with 0-100 scores}
#' }
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' classes <- unique(extraversion$subtopic)
#'
#' results <- classify_scores(extraversion$text, classes)
#' head(results)
#'
#' # Scores -> probabilities -> calibration
#' calibration_metrics(as.matrix(results[, classes]), extraversion$subtopic,
#'                     "Verbalized scores")
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_self_consistency}}
#'
#' @export
classify_scores <- function(texts, classes,
                            model = "llama-3.3-70b-versatile",
                            chunk_size = 40, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  classes_str <- paste(sprintf('"%s"', classes), collapse = ", ")

  chunk_fun <- function(x) {
    prompt <- sprintf(
      'Rate how well each numbered text matches each of these categories on a scale from 0 to 100.

Categories: %s

%s

Return a JSON object where keys are text numbers (as strings) and values are objects
with each category as a key and its score (0-100) as the value.
Scores for each text should sum to approximately 100.
Example: {"1": {"cat_a": 70, "cat_b": 20, "cat_c": 10}}
Return ONLY valid JSON.',
      classes_str, .format_numbered_texts(x)
    )
    content <- .groq_chat(prompt, model = model,
                          max_tokens = length(x) * (12L * length(classes) + 10L) + 200L,
                          response_format = list(type = "json_object"))
    results <- .parse_json(content)
    scores <- t(vapply(seq_along(x), function(i) {
      entry <- results[[as.character(i)]]
      if (is.null(entry) || !is.list(entry)) return(rep(NA_real_, length(classes)))
      keys <- normalize_labels(names(entry), classes)
      vapply(classes, function(cl) {
        k <- which(keys == cl)[1]
        if (is.na(k)) 0 else suppressWarnings(as.numeric(entry[[k]]))
      }, numeric(1))
    }, numeric(length(classes))))
    scores <- matrix(scores, nrow = length(x))
    colnames(scores) <- classes
    as.data.frame(scores, check.names = FALSE)
  }

  template <- as.data.frame(matrix(NA_real_, 1, length(classes),
                                   dimnames = list(NULL, classes)), check.names = FALSE)
  res <- .batch_classify(texts, chunk_fun, chunk_size, shuffle, seed, template)
  S <- as.matrix(res[, classes, drop = FALSE])
  answered <- stats::complete.cases(S) & rowSums(S) > 0
  prediction <- rep(NA_character_, length(texts))
  prediction[answered] <- classes[max.col(S[answered, , drop = FALSE], ties.method = "first")]
  .report_invalid(prediction)

  cbind(data.frame(text_id = seq_along(texts), prediction = prediction,
                   stringsAsFactors = FALSE),
        res[, classes, drop = FALSE])
}


#' LLM-as-Judge classification with evidence
#'
#' Uses the LLM as an expert judge that provides evidence extraction and
#' reasoning for each classification decision. This creates an audit trail
#' suitable for qualitative research, where you need to justify why each
#' text was assigned to a category.
#'
#' @inheritParams classify_direct
#' @param persona Character string describing the expert persona for the
#'   system prompt. Default is a personality psychologist.
#'
#' @inheritSection batch_settings Batch settings
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{text}{Original text (truncated)}
#'   \item{prediction}{Predicted category (or \code{NA})}
#'   \item{confidence}{Confidence level: "high", "medium", or "low"}
#'   \item{reasoning}{One-sentence explanation}
#'   \item{evidence}{Key phrases from the text supporting the decision}
#' }
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' results <- classify_judge(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic)
#' )
#'
#' # Inspect evidence for first 5 texts
#' results[1:5, c("prediction", "reasoning", "evidence")]
#'
#' # Custom persona
#' results2 <- classify_judge(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic),
#'   persona = "You are a social psychologist specializing in personality traits."
#' )
#' }
#'
#' @seealso \code{\link{classify_json}}, \code{\link{classify_cot}}
#'
#' @export
classify_judge <- function(texts, classes,
                           persona = "You are an expert personality psychologist acting as a text classifier.
Your job is to carefully read personality descriptions and classify them into categories.
You must provide reasoning for your classification, citing specific evidence from the text.",
                           model = "llama-3.3-70b-versatile",
                           chunk_size = 40, shuffle = TRUE, seed = 2026) {
  classes <- as.character(classes)

  chunk_fun <- function(x) {
    prompt <- sprintf(
      'Classify each numbered personality description into ONE of these categories: %s

%s

For each text, provide:
- "evidence": list of 2-3 key phrases from the text that inform your decision
- "reasoning": one sentence explaining why this evidence points to your chosen category
- "category": the final classification
- "confidence": "high", "medium", or "low"

Return a JSON object with text numbers as keys:
{
  "1": {"evidence": ["phrase1", "phrase2"], "reasoning": "...", "category": "...", "confidence": "high"}
}
Return ONLY valid JSON.',
      paste(classes, collapse = ", "), .format_numbered_texts(x)
    )
    content <- .groq_chat(prompt, model = model, system = persona,
                          max_tokens = length(x) * 120L + 200L,
                          response_format = list(type = "json_object"))
    results <- .parse_json(content)
    get <- function(field) vapply(seq_along(x), function(i) {
      .get_string(results, i, field)
    }, character(1))
    evidence <- vapply(seq_along(x), function(i) .get_string(results, i, "evidence"),
                       character(1))
    data.frame(prediction = normalize_labels(get("category"), classes),
               confidence = normalize_labels(get("confidence"), c("high", "medium", "low")),
               reasoning = get("reasoning"),
               evidence = evidence,
               stringsAsFactors = FALSE)
  }

  res <- .batch_classify(texts, chunk_fun, chunk_size, shuffle, seed,
                         template = data.frame(prediction = NA_character_,
                                               confidence = NA_character_,
                                               reasoning = NA_character_,
                                               evidence = NA_character_))
  .report_invalid(res$prediction)
  .with_text_cols(texts, res)
}


#' Generate text with an LLM
#'
#' Sends a prompt to a Groq LLM and returns the generated text. Useful for
#' text generation tasks like item creation, summarization, and exploration.
#'
#' @param prompt The user prompt.
#' @param system_prompt Optional system prompt to set a persona or context.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#' @param temperature Creativity parameter (0 = deterministic, 1 = creative).
#'   Default is \code{0.3}.
#' @param max_tokens Maximum response length (sent as
#'   \code{max_completion_tokens}). Default is \code{500}. Reasoning models
#'   get 2,000 extra tokens to think.
#' @param ... Additional arguments for the Groq chat completions API, for
#'   example \code{top_p}, \code{stop}, or the built-in browser tool of
#'   \code{openai/gpt-oss-120b}: \code{tools = list(list(type = "browser_search"))}
#'   with \code{tool_choice = "required"}. They override the package defaults
#'   (e.g., \code{reasoning_effort = "medium"}).
#'
#' @return A character string containing the model's response.
#'
#' @examples
#' \dontrun{
#' # Simple generation
#' llm_generate("Generate 3 survey items measuring academic self-efficacy.")
#'
#' # With system prompt
#' llm_generate(
#'   prompt = "Generate 5 Likert-scale items for conscientiousness.",
#'   system_prompt = "You are an expert psychometrician.",
#'   temperature = 0.4
#' )
#'
#' # With web browsing (gpt-oss models on Groq)
#' llm_generate(
#'   prompt = paste("Summarize in 2-3 sentences one news story about AI published",
#'                  "in the last 5 days (today is", format(Sys.Date(), "%B %d, %Y"), ")."),
#'   model = "openai/gpt-oss-120b", temperature = 1, max_tokens = 2048L,
#'   tools = list(list(type = "browser_search")), tool_choice = "required"
#' )
#' }
#'
#' @seealso \code{\link{generate_labeled_texts}}
#' @export
llm_generate <- function(prompt,
                         system_prompt = NULL,
                         model = "llama-3.3-70b-versatile",
                         temperature = 0.3,
                         max_tokens = 500L,
                         ...) {
  out <- .groq_chat(prompt, model = model, system = system_prompt,
                    temperature = temperature, max_tokens = as.integer(max_tokens),
                    extra = list(...))
  attr(out, "usage") <- NULL
  out
}
