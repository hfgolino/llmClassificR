#' Map model answers to the allowed class labels
#'
#' LLMs often return labels with different case, spaces or punctuation
#' (\code{" friendly."}, \code{"ASSERTIVE"}). This function maps each answer
#' to one of the allowed \code{classes}: first by exact match, then ignoring
#' case and any character that is not a letter or a digit (in any script).
#' Answers that do not match any class (for example \code{"Friendliness"} or
#' \code{"sociable"}) become \code{NA}, so they can be counted as invalid
#' instead of silently creating a new class. Classes that become identical
#' after this cleaning (e.g., \code{"C"} and \code{"C++"}) are matched
#' exactly only.
#'
#' @param pred Character vector of model answers.
#' @param classes Character vector of allowed labels.
#'
#' @return A character vector with the same length as \code{pred}, holding
#'   the matching class label (in its original spelling) or \code{NA}.
#'
#' @examples
#' classes <- c("Friendly", "Positive", "Assertive", "Energetic")
#' normalize_labels(c(" friendly.", "ASSERTIVE", "Friendliness", NA), classes)
#'
#' @seealso \code{\link{evaluate_classification}}
#' @export
normalize_labels <- function(pred, classes) {
  clean <- function(x) gsub("[^\\p{L}\\p{N}]", "", tolower(as.character(x)), perl = TRUE)
  classes <- as.character(classes)
  pred <- as.character(pred)
  out <- classes[match(pred, classes)]                 # 1. exact match
  cc <- clean(classes)
  usable <- nzchar(cc) & !(cc %in% cc[duplicated(cc)])  # unambiguous after cleaning
  todo <- is.na(out) & !is.na(pred)
  if (any(todo) && any(usable)) {
    out[todo] <- classes[usable][match(clean(pred[todo]), cc[usable])]   # 2. cleaned match
  }
  out
}


#' Send texts in shuffled chunks and reassemble the results
#'
#' Batch prompting (many texts in one prompt) is fast, but two things can go
#' wrong: long prompts hit token limits, and texts sorted by label (all
#' "Friendly" texts first, then all "Positive" ones) let the model copy the
#' previous answer. \code{run_in_chunks()} shuffles the texts, sends them in
#' chunks, and returns the results in the \strong{original} order.
#'
#' @param texts Character vector of texts.
#' @param classify_fun A function that takes a character vector of texts and
#'   returns a data.frame with one row per text, in the same order.
#' @param chunk_size Number of texts per call. Default is \code{40}.
#' @param shuffle Logical. Shuffle the texts before chunking? Default
#'   \code{TRUE}.
#' @param seed Random seed for the shuffle (the global random state is
#'   restored afterwards). Default is \code{2026}.
#' @param pause Seconds to wait between chunks (free-tier rate limits).
#'   Default is \code{getOption("llmClassificR.pause", 2)}.
#' @param verbose Logical. Print progress. Default
#'   \code{getOption("llmClassificR.verbose", TRUE)}.
#'
#' @return A data.frame with one row per text in the original order, plus a
#'   column \code{row} with the original position. If a chunk fails, its
#'   rows are kept with \code{NA} values and a warning is given. If every
#'   chunk fails (e.g., a wrong API key or model name), an error is raised
#'   with the first error message.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @examples
#' # A toy "classifier": label each text by its first word
#' first_word <- function(x) data.frame(prediction = sub(" .*", "", x))
#' texts <- c("Friendly hello", "Positive day", "Assertive boss", "Energetic run")
#' run_in_chunks(texts, first_word, chunk_size = 2, pause = 0)
#'
#' \dontrun{
#' # With an LLM: any batch function that returns one row per text
#' res <- run_in_chunks(extraversion$text, function(x) {
#'   classify_direct(x, unique(extraversion$subtopic), chunk_size = NULL)
#' }, chunk_size = 30)
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{normalize_labels}}
#' @export
run_in_chunks <- function(texts, classify_fun, chunk_size = 40, shuffle = TRUE,
                          seed = 2026, pause = getOption("llmClassificR.pause", 2),
                          verbose = getOption("llmClassificR.verbose", TRUE)) {
  n <- length(texts)
  if (n == 0) stop("'texts' is empty.", call. = FALSE)
  if (is.null(chunk_size) || !is.finite(chunk_size)) chunk_size <- n
  order_sent <- if (shuffle) .with_seed(seed, sample(n)) else seq_len(n)
  chunks <- split(order_sent, ceiling(seq_along(order_sent) / chunk_size))

  out <- vector("list", length(chunks))
  failed <- logical(length(chunks))
  errors <- character(length(chunks))
  for (k in seq_along(chunks)) {
    idx <- chunks[[k]]
    if (verbose && length(chunks) > 1) {
      cat("  Chunk", k, "of", length(chunks), "(", length(idx), "texts )\n")
    }
    res <- tryCatch(classify_fun(texts[idx]), error = function(e) {
      errors[k] <<- conditionMessage(e)
      NULL
    })
    if (!is.null(res) && (!is.data.frame(res) || nrow(res) != length(idx))) {
      errors[k] <- paste0("returned ", NROW(res), " rows for ", length(idx), " texts")
      res <- NULL
    }
    if (is.null(res)) {
      failed[k] <- TRUE
      res <- data.frame(row.names = seq_along(idx))
    }
    res$row <- idx
    out[[k]] <- res
    if (k < length(chunks) && pause > 0) Sys.sleep(pause)
  }
  if (all(failed)) {
    stop("Every chunk failed. First error: ", errors[1], call. = FALSE)
  }
  for (k in which(failed)) {
    warning("Chunk ", k, " failed: ", errors[k], ". Its texts are returned as NA.",
            call. = FALSE)
  }
  res <- .rbind_fill(out)
  res <- res[order(res$row), , drop = FALSE]
  rownames(res) <- NULL
  res
}


# ------------------------------------------------------------------------------
# Internal helpers shared by the classify_*() functions
# ------------------------------------------------------------------------------

#' Evaluate an expression with a temporary seed, restoring the RNG state
#' @noRd
.with_seed <- function(seed, expr) {
  if (is.null(seed)) return(expr)
  old <- if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv(), inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (is.null(old)) {
      if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
        rm(".Random.seed", envir = globalenv())
      }
    } else {
      assign(".Random.seed", old, envir = globalenv())
    }
  })
  set.seed(seed)
  expr
}

#' rbind data.frames that may have different columns (missing -> NA)
#' @noRd
.rbind_fill <- function(dfs) {
  dfs <- Filter(Negate(is.null), dfs)
  cols <- unique(unlist(lapply(dfs, names)))
  template <- Find(function(d) all(cols %in% names(d)), dfs)
  dfs <- lapply(dfs, function(d) {
    for (cl in setdiff(cols, names(d))) {
      proto <- if (!is.null(template)) template[[cl]][NA_integer_] else NA
      d[[cl]] <- rep(proto, nrow(d))
      if (length(d[[cl]]) == 0) d[[cl]] <- proto[0]
    }
    d[, cols, drop = FALSE]
  })
  do.call(rbind, dfs)
}

#' Is this a reasoning model that accepts reasoning_effort = "low"?
#' (gpt-oss and qwen3.8 on Groq, September 2026)
#' @noRd
.is_reasoning_model <- function(model) grepl("gpt-oss|qwen3\\.8", model)

#' One call to the Groq chat API
#'
#' Uses max_completion_tokens, low reasoning effort for reasoning models
#' (they think before answering), and the client's built-in retries.
#' Returns the text content, with token usage as an attribute.
#' @noRd
.groq_chat <- function(prompt, model = "openai/gpt-oss-120b", system = NULL,
                       temperature = 0, max_tokens = 4000L, response_format = NULL,
                       extra = list()) {
  client <- .get_groq_client()
  messages <- list()
  if (!is.null(system)) messages <- c(messages, list(list(role = "system", content = system)))
  messages <- c(messages, list(list(role = "user", content = prompt)))

  if (.is_reasoning_model(model)) max_tokens <- max_tokens + 2000L   # room to "think"
  args <- list(model = model, messages = messages, temperature = temperature,
               max_completion_tokens = as.integer(max_tokens))
  if (!is.null(response_format)) args$response_format <- response_format
  if (.is_reasoning_model(model)) args$reasoning_effort <- "low"
  if (length(extra) > 0) args <- utils::modifyList(args, extra)   # user arguments win

  response <- do.call(client$chat$completions$create, args)
  content <- response$choices[[1]]$message$content
  if (is.null(content)) content <- ""
  # Some reasoning models put their thinking in <think> blocks: drop them
  content <- gsub("(?s)<think>.*?</think>\\s*", "", content, perl = TRUE)
  attr(content, "usage") <- tryCatch(
    c(prompt_tokens = response$usage$prompt_tokens,
      completion_tokens = response$usage$completion_tokens),
    error = function(e) NULL)
  content
}

#' Parse JSON from a model answer (tolerates text around the JSON object)
#' @noRd
.parse_json <- function(content, simplify = TRUE) {
  out <- tryCatch(jsonlite::fromJSON(content, simplifyVector = simplify),
                  error = function(e) NULL)
  if (is.null(out)) {
    # Try the outermost {...} block (e.g., JSON wrapped in ```json fences)
    m <- regmatches(content, regexpr("\\{.*\\}", content))
    if (length(m) == 1) {
      out <- tryCatch(jsonlite::fromJSON(m, simplifyVector = simplify),
                      error = function(e) NULL)
    }
  }
  if (is.null(out)) stop("The model did not return valid JSON.", call. = FALSE)
  out
}

#' Safely pull entry i (keys "1", "2", ...) from a batch JSON object
#' @noRd
.get_entry <- function(results, i, field = NULL) {
  entry <- if (is.list(results) && !is.null(names(results))) results[[as.character(i)]] else NULL
  if (is.null(entry)) return(NA)
  if (is.null(field)) {
    if (is.list(entry) || length(entry) != 1) return(NA)
    return(as.character(entry))
  }
  if (!is.list(entry)) return(NA)   # e.g., a bare string where an object was expected
  value <- entry[[field]]
  if (is.null(value) || length(value) == 0) NA else value
}

#' Entry i (and field) as ONE string: arrays are collapsed with "; "
#' @noRd
.get_string <- function(results, i, field = NULL) {
  value <- .get_entry(results, i, field)
  value <- unlist(value)
  if (length(value) == 0 || all(is.na(value))) return(NA_character_)
  paste(as.character(value), collapse = "; ")
}

#' Shared engine: shuffle + chunk + reassemble + tidy output
#'
#' `template` is a one-row data.frame of NA values with the expected columns
#' and types; columns missing because every chunk failed are filled from it.
#' @noRd
.batch_classify <- function(texts, chunk_fun, chunk_size, shuffle, seed, template) {
  texts <- as.character(texts)
  res <- run_in_chunks(texts, chunk_fun, chunk_size = chunk_size, shuffle = shuffle,
                       seed = seed)
  res$row <- NULL
  for (cl in setdiff(names(template), names(res))) {
    res[[cl]] <- rep(template[[cl]], length(texts))
  }
  res[, union(names(template), names(res)), drop = FALSE]
}

#' Standard output columns
#' @noRd
.with_text_cols <- function(texts, res) {
  cbind(data.frame(text_id = seq_along(texts), text = substr(texts, 1, 200),
                   stringsAsFactors = FALSE),
        res, stringsAsFactors = FALSE)
}

#' Warn about invalid answers
#' @noRd
.report_invalid <- function(prediction) {
  n_bad <- sum(is.na(prediction))
  if (n_bad > 0) {
    message(n_bad, " of ", length(prediction), " texts received no valid label (NA). ",
            "They count as errors in evaluate_classification().")
  }
  invisible(n_bad)
}
