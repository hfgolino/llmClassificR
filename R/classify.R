#' Direct zero-shot text classification
#'
#' Classifies texts into categories using a single LLM API call. This is the
#' simplest and fastest classification approach: the model receives all texts
#' and returns a category for each.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'   See \code{\link{list_models}} for options.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index of each text}
#'   \item{text}{The original text (truncated to 200 characters)}
#'   \item{prediction}{The predicted category}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' data(extraversion)
#' activate_llm_env()
#' set_groq_key("your-key")
#'
#' results <- classify_direct(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic)
#' )
#' head(results)
#'
#' # Compare with true labels
#' table(True = extraversion$subtopic, Predicted = results$prediction)
#' }
#'
#' @seealso \code{\link{classify_fewshot}}, \code{\link{classify_cot}},
#'   \code{\link{classify_json}}, \code{\link{compare_methods}}
#'
#' @export
classify_direct <- function(texts, classes,
                            model = "llama-3.3-70b-versatile") {

  client <- .get_groq_client()
  all_texts <- .format_numbered_texts(texts)

  prompt <- sprintf(
    'Classify each numbered text into exactly ONE of these categories: %s

%s

Respond with ONLY a JSON object where keys are the text numbers (as strings)
and values are the category names.
Example format: {"1": "category_a", "2": "category_b"}',
    paste(classes, collapse = ", "), all_texts
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(list("role" = "user", "content" = prompt)),
    temperature = 0.0,
    max_tokens = as.integer(length(texts) * 30),
    response_format = list("type" = "json_object")
  )

  results <- jsonlite::fromJSON(response$choices[[1]]$message$content)

  predictions <- character(length(texts))
  for (i in seq_along(texts)) {
    predictions[i] <- tolower(results[[as.character(i)]])
  }

  data.frame(
    text_id = seq_along(texts),
    text = substr(texts, 1, 200),
    prediction = predictions,
    stringsAsFactors = FALSE
  )
}


#' Few-shot text classification
#'
#' Classifies texts by providing labeled examples to guide the model. The
#' model sees example text-label pairs before classifying the target texts.
#' This often improves accuracy over direct zero-shot, especially for
#' ambiguous categories.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param examples A data.frame with columns \code{text} and \code{label},
#'   containing labeled examples (2-3 per class is typical).
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index of each text}
#'   \item{text}{The original text (truncated to 200 characters)}
#'   \item{prediction}{The predicted category}
#' }
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' # Create examples: first 2 texts from each class
#' examples <- do.call(rbind, lapply(unique(extraversion$subtopic), function(cls) {
#'   idx <- which(extraversion$subtopic == cls)
#'   data.frame(text = extraversion$text[idx[1:2]],
#'              label = cls, stringsAsFactors = FALSE)
#' }))
#'
#' # Classify the remaining texts
#' test_idx <- setdiff(1:nrow(extraversion), which(extraversion$text %in% examples$text))
#' results <- classify_fewshot(
#'   texts = extraversion$text[test_idx],
#'   classes = unique(extraversion$subtopic),
#'   examples = examples
#' )
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_cot}}
#'
#' @export
classify_fewshot <- function(texts, classes, examples,
                             model = "llama-3.3-70b-versatile") {

  client <- .get_groq_client()

  # Format examples
  example_str <- paste(sapply(seq_len(nrow(examples)), function(i) {
    sprintf('Text: "%s"\nCategory: %s',
            substr(examples$text[i], 1, 200), examples$label[i])
  }), collapse = "\n\n")

  all_texts <- .format_numbered_texts(texts)

  prompt <- sprintf(
    'Classify each numbered text into one of these categories: %s

Here are some labeled examples:
%s

Now classify ALL of these texts:
%s

Respond with ONLY a JSON object where keys are the text numbers (as strings)
and values are the category names.',
    paste(classes, collapse = ", "), example_str, all_texts
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(list("role" = "user", "content" = prompt)),
    temperature = 0.0,
    max_tokens = as.integer(length(texts) * 30),
    response_format = list("type" = "json_object")
  )

  results <- jsonlite::fromJSON(response$choices[[1]]$message$content)

  predictions <- character(length(texts))
  for (i in seq_along(texts)) {
    predictions[i] <- tolower(results[[as.character(i)]])
  }

  data.frame(
    text_id = seq_along(texts),
    text = substr(texts, 1, 200),
    prediction = predictions,
    stringsAsFactors = FALSE
  )
}


#' Chain-of-thought text classification
#'
#' Classifies texts by asking the model to reason step-by-step before
#' assigning a category. Returns both the prediction and the reasoning
#' for each text. This approach often improves accuracy on harder
#' classification tasks by encouraging the model to analyze key features.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{text}{Original text (truncated)}
#'   \item{prediction}{Predicted category}
#'   \item{reasoning}{Model's reasoning for the classification}
#' }
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
                         model = "llama-3.3-70b-versatile") {

  client <- .get_groq_client()
  all_texts <- .format_numbered_texts(texts)

  system_prompt <- "You are an expert text classifier. For each text, briefly
analyze the key features before classifying."

  prompt <- sprintf(
    'For each numbered text below, classify it into exactly ONE of these categories: %s

%s

Think step by step for each text. Then return a JSON object with this structure:
{
  "1": {"reasoning": "brief analysis", "category": "chosen_category"},
  "2": {"reasoning": "brief analysis", "category": "chosen_category"}
}
Return ONLY valid JSON.',
    paste(classes, collapse = ", "), all_texts
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(
      list("role" = "system", "content" = system_prompt),
      list("role" = "user", "content" = prompt)
    ),
    temperature = 0.0,
    max_tokens = as.integer(length(texts) * 80),
    response_format = list("type" = "json_object")
  )

  results <- jsonlite::fromJSON(response$choices[[1]]$message$content)

  predictions <- character(length(texts))
  reasoning <- character(length(texts))
  for (i in seq_along(texts)) {
    entry <- results[[as.character(i)]]
    predictions[i] <- tolower(entry$category)
    reasoning[i] <- entry$reasoning
  }

  data.frame(
    text_id = seq_along(texts),
    text = substr(texts, 1, 200),
    prediction = predictions,
    reasoning = reasoning,
    stringsAsFactors = FALSE
  )
}


#' Structured JSON text classification with confidence
#'
#' Classifies texts and returns structured output including the predicted
#' category, a confidence level (high/medium/low), and a brief reasoning
#' for each text. Uses JSON mode for reliable parsing.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{text}{Original text (truncated)}
#'   \item{prediction}{Predicted category}
#'   \item{confidence}{Confidence level: "high", "medium", or "low"}
#'   \item{reasoning}{Brief explanation of the classification}
#' }
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
#' # Confidence distribution
#' table(results$confidence)
#'
#' # Look at low-confidence predictions
#' results[results$confidence == "low", ]
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_judge}}
#'
#' @export
classify_json <- function(texts, classes,
                          model = "llama-3.3-70b-versatile") {

  client <- .get_groq_client()
  all_texts <- .format_numbered_texts(texts)

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
    paste(classes, collapse = ", "), all_texts
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(list("role" = "user", "content" = prompt)),
    temperature = 0.0,
    max_tokens = as.integer(length(texts) * 80),
    response_format = list("type" = "json_object")
  )

  results <- jsonlite::fromJSON(response$choices[[1]]$message$content)

  predictions <- character(length(texts))
  confidence <- character(length(texts))
  reasoning <- character(length(texts))
  for (i in seq_along(texts)) {
    entry <- results[[as.character(i)]]
    predictions[i] <- tolower(entry$category)
    confidence[i] <- tolower(entry$confidence)
    reasoning[i] <- entry$reasoning
  }

  data.frame(
    text_id = seq_along(texts),
    text = substr(texts, 1, 200),
    prediction = predictions,
    confidence = confidence,
    reasoning = reasoning,
    stringsAsFactors = FALSE
  )
}


#' Multi-model ensemble classification
#'
#' Classifies texts using multiple LLM models and takes a majority vote.
#' Each model classifies all texts in a single batch call, then predictions
#' are combined via majority vote. This approach is more robust than any
#' single model.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param models Character vector of Groq model names. Default uses three
#'   models: \code{llama-3.3-70b-versatile}, \code{llama-3.1-8b-instant},
#'   and \code{gemma2-9b-it}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{text}{Original text (truncated)}
#'   \item{prediction}{Majority vote prediction}
#'   \item{agreement}{Logical: did all models agree?}
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
#' # How often do all models agree?
#' mean(results$agreement)
#'
#' # Look at disagreements
#' results[!results$agreement, c("text", "prediction", "votes")]
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{list_models}}
#'
#' @export
classify_ensemble <- function(texts, classes,
                              models = c("llama-3.3-70b-versatile",
                                         "llama-3.1-8b-instant",
                                         "openai/gpt-oss-20b")) {

  cat("Running ensemble across", length(models), "models...\n")

  # Get batch predictions from each model
  model_preds <- list()
  for (m in seq_along(models)) {
    cat("  Model", m, ":", models[m], "...")
    model_preds[[m]] <- tryCatch({
      res <- classify_direct(texts, classes, model = models[m])
      cat(" done\n")
      res$prediction
    }, error = function(e) {
      cat(" FAILED\n")
      rep(NA_character_, length(texts))
    })
  }

  # Majority vote
  predictions <- character(length(texts))
  agreement <- logical(length(texts))
  votes_str <- character(length(texts))

  for (i in seq_along(texts)) {
    votes <- sapply(model_preds, function(p) p[i])
    votes_str[i] <- paste(votes, collapse = ", ")
    valid <- votes[!is.na(votes)]
    if (length(valid) == 0) {
      predictions[i] <- NA_character_
      agreement[i] <- FALSE
    } else {
      vote_table <- sort(table(valid), decreasing = TRUE)
      predictions[i] <- names(vote_table)[1]
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


#' Classification with per-class scores
#'
#' Instead of assigning a single label, this function asks the model to
#' rate how well each text matches each category on a 0-100 scale. This
#' is analogous to the entailment scores produced by NLI-based zero-shot
#' models and is useful when you need soft assignments or confidence
#' calibration.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{prediction}{Predicted category (highest score)}
#'   \item{...}{One numeric column per class with 0-100 scores}
#' }
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' results <- classify_scores(
#'   texts = extraversion$text,
#'   classes = unique(extraversion$subtopic)
#' )
#'
#' # View score distribution
#' head(results)
#'
#' # Mean scores by true category (requires dplyr/tidyr)
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_json}}
#'
#' @export
classify_scores <- function(texts, classes,
                            model = "llama-3.3-70b-versatile") {

  client <- .get_groq_client()
  all_texts <- .format_numbered_texts(texts)
  classes_str <- paste(sprintf('"%s"', classes), collapse = ", ")

  prompt <- sprintf(
    'Rate how well each numbered text matches each of these categories on a scale from 0 to 100.

Categories: %s

%s

Return a JSON object where keys are text numbers (as strings) and values are objects
with each category as a key and its score (0-100) as the value.
Scores for each text should sum to approximately 100.
Example: {"1": {"cat_a": 70, "cat_b": 20, "cat_c": 10}}
Return ONLY valid JSON.',
    classes_str, all_texts
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(list("role" = "user", "content" = prompt)),
    temperature = 0.0,
    max_tokens = as.integer(length(texts) * 50),
    response_format = list("type" = "json_object")
  )

  results <- jsonlite::fromJSON(response$choices[[1]]$message$content)

  # Build scores matrix
  scores_list <- list()
  for (i in seq_along(texts)) {
    entry <- results[[as.character(i)]]
    scores_list[[i]] <- as.data.frame(t(unlist(entry)))
  }
  scores_df <- do.call(rbind, scores_list)
  colnames(scores_df) <- tolower(colnames(scores_df))

  # Ensure numeric
  for (col in colnames(scores_df)) {
    scores_df[[col]] <- as.numeric(scores_df[[col]])
  }

  # Predicted class = highest score
  predictions <- apply(scores_df, 1, function(x) names(x)[which.max(x)])

  cbind(
    data.frame(text_id = seq_along(texts), prediction = predictions,
               stringsAsFactors = FALSE),
    scores_df
  )
}


#' LLM-as-Judge classification with evidence
#'
#' Uses the LLM as an expert judge that provides evidence extraction and
#' reasoning for each classification decision. This creates an audit trail
#' suitable for qualitative research, where you need to justify why each
#' text was assigned to a category.
#'
#' @param texts Character vector of texts to classify.
#' @param classes Character vector of possible category labels.
#' @param persona Character string describing the expert persona for the
#'   system prompt. Default is a personality psychologist.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{text_id}{Integer index}
#'   \item{text}{Original text (truncated)}
#'   \item{prediction}{Predicted category}
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
                           model = "llama-3.3-70b-versatile") {

  client <- .get_groq_client()
  all_texts <- .format_numbered_texts(texts)

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
    paste(classes, collapse = ", "), all_texts
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(
      list("role" = "system", "content" = persona),
      list("role" = "user", "content" = prompt)
    ),
    temperature = 0.0,
    max_tokens = as.integer(length(texts) * 120),
    response_format = list("type" = "json_object")
  )

  results <- jsonlite::fromJSON(response$choices[[1]]$message$content)

  predictions <- character(length(texts))
  confidence <- character(length(texts))
  reasoning <- character(length(texts))
  evidence <- character(length(texts))

  for (i in seq_along(texts)) {
    entry <- results[[as.character(i)]]
    predictions[i] <- tolower(entry$category)
    confidence[i] <- tolower(entry$confidence)
    reasoning[i] <- entry$reasoning
    ev <- entry$evidence
    if (!is.null(ev) && length(ev) > 0) {
      evidence[i] <- paste(ev, collapse = "; ")
    } else {
      evidence[i] <- NA_character_
    }
  }

  data.frame(
    text_id = seq_along(texts),
    text = substr(texts, 1, 200),
    prediction = predictions,
    confidence = confidence,
    reasoning = reasoning,
    evidence = evidence,
    stringsAsFactors = FALSE
  )
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
#' @param max_tokens Maximum response length. Default is \code{500}.
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
#' }
#'
#' @export
llm_generate <- function(prompt,
                         system_prompt = NULL,
                         model = "llama-3.3-70b-versatile",
                         temperature = 0.3,
                         max_tokens = 500L) {

  client <- .get_groq_client()

  messages <- list()
  if (!is.null(system_prompt)) {
    messages <- c(messages, list(list("role" = "system", "content" = system_prompt)))
  }
  messages <- c(messages, list(list("role" = "user", "content" = prompt)))

  response <- client$chat$completions$create(
    model = model,
    messages = messages,
    temperature = temperature,
    max_tokens = as.integer(max_tokens)
  )

  response$choices[[1]]$message$content
}
