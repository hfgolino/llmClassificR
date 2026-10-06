# ==============================================================================
# Jev (TypeSafe AI): a "System One" classification model, called from pure R
# ==============================================================================
#
# Jev answers typed questions about a text and returns probabilities instead of
# generated text. Three question types:
#   - Choice: pick ONE option from a set (2 to 255 options)
#   - Noul:   probability that a yes/no statement is true
#   - Score:  rate the text on an ordered rubric of 2 to 10 levels
#
# API: POST https://api.typesafe.ai/v1/systemone (one text per request).
# All functions below use httr2 only; no Python environment is needed.

.jev_defaults <- list(
  base_url = "https://api.typesafe.ai/v1",
  model = "jev-1.13.0",
  price_per_million = 0.042   # USD per million input tokens; output tokens are free (Sept 2026)
)


#' Set the TypeSafe AI (Jev) API key
#'
#' Stores the TypeSafe AI API key for the current R session. Jev is called
#' through pure R (httr2), so no Python environment is needed. Get a key at
#' \url{https://console.typesafe.ai/settings/keys}.
#'
#' @param key Your TypeSafe API key. Defaults to the \code{TYPESAFE_API_KEY}
#'   environment variable, so the recommended workflow is to add a line
#'   \code{TYPESAFE_API_KEY=your-key} to your \code{~/.Renviron} file
#'   (for example with \code{usethis::edit_r_environ()}), restart R and call
#'   \code{set_typesafe_key()} with no arguments. Avoid typing keys in scripts
#'   you share.
#'
#' @return Invisible \code{NULL}.
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' # Recommended: key stored in ~/.Renviron as TYPESAFE_API_KEY=...
#' set_typesafe_key()
#' }
#'
#' @seealso \code{\link{jev_classify}}, \code{\link{jev_ask}}
#' @export
set_typesafe_key <- function(key = Sys.getenv("TYPESAFE_API_KEY")) {
  if (is.null(key) || !nzchar(key)) {
    stop("Please provide a valid TypeSafe API key (or set TYPESAFE_API_KEY in ~/.Renviron).\n",
         "Get one at https://console.typesafe.ai/settings/keys", call. = FALSE)
  }
  .pkg_env$typesafe_key <- key
  cat("TypeSafe (Jev) API key set successfully.\n")
  invisible(NULL)
}

#' Get the TypeSafe key, with a helpful error if not set
#' @noRd
.get_typesafe_key <- function() {
  key <- .pkg_env$typesafe_key
  if (is.null(key)) {
    env_key <- Sys.getenv("TYPESAFE_API_KEY")
    if (nzchar(env_key)) return(env_key)
    stop("TypeSafe API key not set. Run set_typesafe_key() first ",
         "(or set TYPESAFE_API_KEY in ~/.Renviron).", call. = FALSE)
  }
  key
}


# ------------------------------------------------------------------------------
# Question builders
# ------------------------------------------------------------------------------

#' Build Jev questions (Choice, Noul, Score)
#'
#' Helpers that build the three question types understood by Jev. Each
#' question has \code{instructions} (what to decide) and \code{criteria}
#' (how to decide). The criteria are the heart of the method: describe
#' \emph{when} each option applies, as you would in a coding manual for
#' human raters.
#'
#' \describe{
#'   \item{\code{jev_choice()}}{Pick ONE option (multi-class; 2 to 255
#'     options). The answer contains the chosen option, its probability
#'     (\code{confidence}) and the probability of every option.}
#'   \item{\code{jev_noul()}}{A yes/no statement. The answer is the
#'     probability that the statement is true (\code{noul}). Use several
#'     Noul questions for multi-label problems.}
#'   \item{\code{jev_score()}}{An ordered rubric of 2 to 10 levels. The
#'     answer contains the expected level (probability-weighted average of
#'     the levels 0, 1, 2, ...) and the probability of each level.}
#' }
#'
#' @param instructions Character string: the question asked about the text.
#' @param criteria For \code{jev_choice()}: a named character vector where
#'   names are the options and values describe when each option applies. An
#'   unnamed vector uses the values as both option names and descriptions.
#' @param true,false For \code{jev_noul()}: optional descriptions of what
#'   counts as "true" and as "false". Give both or neither.
#' @param levels For \code{jev_score()}: character vector of 2 to 10 level
#'   descriptions in increasing order. The first level is scored 0.
#'
#' @return A list that can be used as one element of the \code{questions}
#'   argument of \code{\link{jev_ask}} and \code{\link{jev_ask_many}}.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' # These helpers only build lists, so they run without an API key
#' facets <- jev_choice(
#'   "Which facet of extraversion does this text express most clearly?",
#'   c(Friendly  = "Warm, welcoming and kind toward other people.",
#'     Positive  = "Optimistic and upbeat; looks on the bright side.",
#'     Assertive = "Takes charge, speaks up and leads decisively.",
#'     Energetic = "Full of energy; fast-paced, lively and active.")
#' )
#' str(facets)
#'
#' others <- jev_noul(
#'   "The writer describes an interaction with other people.",
#'   true  = "Other people are part of the situation described.",
#'   false = "The text is only about the writer."
#' )
#'
#' level <- jev_score(
#'   "How extraverted does the writer sound?",
#'   c("Reserved", "Balanced", "Outgoing", "Highly outgoing")
#' )
#'
#' # Several questions are answered in ONE call (names identify the answers)
#' questions <- list(facet = facets, mentions_others = others, level = level)
#' names(questions)
#'
#' @seealso \code{\link{jev_ask}}, \code{\link{jev_classify}},
#'   \code{\link{jev_multilabel}}, \code{\link{jev_rate}}
#' @name jev_questions
NULL

#' @rdname jev_questions
#' @export
jev_choice <- function(instructions, criteria) {
  .check_string(instructions, "instructions")
  criteria <- .as_named_criteria(criteria)
  if (length(criteria) < 2 || length(criteria) > 255) {
    stop("A Choice question needs between 2 and 255 options.", call. = FALSE)
  }
  list(type = "choice", instructions = instructions, criteria = as.list(criteria))
}

#' @rdname jev_questions
#' @export
jev_noul <- function(instructions, true = NULL, false = NULL) {
  .check_string(instructions, "instructions")
  q <- list(type = "noul", instructions = instructions)
  if (xor(is.null(true), is.null(false))) {
    stop("Give both 'true' and 'false' criteria, or neither.", call. = FALSE)
  }
  if (!is.null(true)) q$criteria <- list(true = true, false = false)
  q
}

#' @rdname jev_questions
#' @export
jev_score <- function(instructions, levels) {
  .check_string(instructions, "instructions")
  levels <- as.character(levels)
  if (length(levels) < 2 || length(levels) > 10) {
    stop("A Score question needs between 2 and 10 levels.", call. = FALSE)
  }
  list(type = "score", instructions = instructions, criteria = as.list(unname(levels)))
}

#' @noRd
.as_named_criteria <- function(criteria) {
  criteria <- unlist(criteria)
  if (is.null(names(criteria))) names(criteria) <- criteria
  empty <- is.na(names(criteria)) | names(criteria) == ""
  names(criteria)[empty] <- criteria[empty]
  if (anyDuplicated(names(criteria))) stop("Option names must be unique.", call. = FALSE)
  criteria
}

#' @noRd
.check_string <- function(x, name) {
  if (!is.character(x) || length(x) != 1 || is.na(x) || !nzchar(x)) {
    stop("'", name, "' must be a single non-empty character string.", call. = FALSE)
  }
}


# ------------------------------------------------------------------------------
# Calling the API
# ------------------------------------------------------------------------------

#' Build one Jev request (not sent). Retries, throttling and timeout built in.
#' @noRd
.jev_request <- function(state, questions, model) {
  if (!is.list(questions) || is.null(names(questions)) || any(names(questions) == "")) {
    stop("'questions' must be a NAMED list of questions, e.g. list(facet = jev_choice(...)).",
         call. = FALSE)
  }
  httr2::request(paste0(.jev_defaults$base_url, "/systemone")) |>
    httr2::req_auth_bearer_token(.get_typesafe_key()) |>
    httr2::req_body_json(list(model = model, state = state, questions = questions)) |>
    httr2::req_retry(
      max_tries = 4,
      is_transient = function(resp) {
        httr2::resp_status(resp) %in% c(408, 429, 500, 502, 503, 504, 529)
      }
    ) |>
    httr2::req_throttle(capacity = 10, fill_time_s = 1, realm = "typesafe") |>   # <= 10 per second
    httr2::req_timeout(60) |>
    httr2::req_error(body = .jev_error_body)
}

#' Turn an API error into a readable message
#' @noRd
.jev_error_body <- function(resp) {
  status <- httr2::resp_status(resp)
  detail <- tryCatch({
    b <- httr2::resp_body_json(resp)
    err <- if (is.list(b$error)) b$error$message else b$error
    msg <- b$detail %||% err %||% b$message
    if (is.list(msg)) jsonlite::toJSON(msg, auto_unbox = TRUE) else msg
  }, error = function(e) NULL)
  hint <- switch(as.character(status),
                 "401" = "Check your TypeSafe API key (set_typesafe_key()).",
                 "422" = "The request is invalid: check the question types and criteria.",
                 "429" = "Rate limit reached: wait a moment or lower max_active.",
                 "529" = "The service is overloaded: try again in a few minutes.",
                 NULL)
  c(detail, hint)
}


#' Ask Jev questions about one text
#'
#' Sends one text (the \code{state}) and one or more named questions to Jev
#' and returns the parsed response. Several questions (of any type) are
#' answered in a single call.
#'
#' @param text The text to classify (a single string). It can also be a
#'   named list, which is sent as a JSON object, for example
#'   \code{list(text = "...", context = "Self-description from a survey")}.
#' @param questions A \emph{named} list of questions built with
#'   \code{\link{jev_choice}}, \code{\link{jev_noul}} and
#'   \code{\link{jev_score}}. The names identify the answers.
#' @param model Jev model version. Default is \code{"jev-1.13.0"}. Pin a
#'   version for research; \code{"jev-latest"} changes with new releases.
#'   See \code{\link{jev_list_models}}.
#'
#' @return The parsed JSON response (a list). \code{$answers} holds one entry
#'   per question and \code{$usage$input_tokens} the tokens billed.
#'
#' @details
#' Requests are retried automatically on temporary errors (408, 429, 5xx and
#' 529) and throttled to at most 10 requests per second (the API limit is
#' 1,200 requests per minute). Jev charges only for input tokens
#' ($0.042 per million in September 2026).
#'
#' \strong{Research note.} Jev is proprietary: texts are sent to TypeSafe's
#' servers (check your IRB protocol before sending participant data), the
#' training data are unknown, and results can change between versions.
#' Pin the model version and validate against human coding.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' set_typesafe_key()
#'
#' hello <- jev_ask(
#'   "I always greet newcomers with a warm smile and ask about their interests.",
#'   questions = list(
#'     facet = jev_choice("Which facet of extraversion is expressed?",
#'                        label_descriptions$extraversion),
#'     mentions_others = jev_noul("The writer describes an interaction with other people."),
#'     level = jev_score("How extraverted does the writer sound?",
#'                       c("Reserved", "Balanced", "Outgoing", "Highly outgoing"))
#'   )
#' )
#' hello$answers$facet$choice
#' unlist(hello$answers$facet$probabilities)
#' hello$answers$mentions_others$noul
#' hello$answers$level$score
#' }
#'
#' @seealso \code{\link{jev_ask_many}}, \code{\link{jev_classify}}
#' @export
jev_ask <- function(text, questions, model = "jev-1.13.0") {
  .jev_request(text, questions, model) |>
    httr2::req_perform() |>
    httr2::resp_body_json()
}


#' Ask Jev the same questions about many texts (in parallel)
#'
#' Sends one request per text, several at a time, and returns the parsed
#' responses in the original order. Failed requests are returned as
#' \code{NULL} (and appear as \code{NA} after parsing), so one error never
#' stops a whole batch.
#'
#' @param texts Character vector of texts.
#' @param questions A named list of questions (see \code{\link{jev_ask}}).
#' @param model Jev model version. Default is \code{"jev-1.13.0"}.
#' @param max_active Maximum number of simultaneous requests. Default is
#'   \code{10}.
#' @param verbose Logical. Print a summary (time, failures, tokens and cost).
#'   Default is \code{TRUE}.
#'
#' @return A list with one parsed response (or \code{NULL}) per text. The
#'   attribute \code{"usage"} holds a one-row data.frame with the number of
#'   texts, failures, input tokens, cost in USD and seconds.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' set_typesafe_key()
#'
#' questions <- list(
#'   facet = jev_choice("Which facet of extraversion does this text express?",
#'                      label_descriptions$extraversion),
#'   level = jev_score("How extraverted does the writer sound?",
#'                     c("Reserved", "Balanced", "Outgoing", "Highly outgoing"))
#' )
#' responses <- jev_ask_many(extraversion$text, questions)
#' attr(responses, "usage")
#'
#' # One data.frame per question
#' facets <- jev_parse_choice(responses, "facet")
#' levels <- jev_parse_score(responses, "level")
#' }
#'
#' @seealso \code{\link{jev_parse_choice}}, \code{\link{jev_classify}}
#' @export
jev_ask_many <- function(texts, questions, model = "jev-1.13.0",
                         max_active = 10, verbose = TRUE) {
  texts <- as.list(texts)
  reqs <- lapply(texts, function(tx) .jev_request(tx, questions, model))
  t0 <- Sys.time()
  resps <- httr2::req_perform_parallel(reqs, on_error = "continue",
                                       max_active = max_active, progress = verbose)
  out <- lapply(resps, function(r) {
    if (inherits(r, "httr2_response")) {
      tryCatch(httr2::resp_body_json(r), error = function(e) NULL)
    } else {
      NULL
    }
  })
  errors <- vapply(resps, function(r) {
    if (inherits(r, "error")) conditionMessage(r) else NA_character_
  }, character(1))

  usage <- jev_usage(out)
  usage$seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  if (verbose) {
    cat(sprintf("Jev: %d texts in %.1f s | failed: %d | input tokens: %s | cost: $%.5f\n",
                usage$n_texts, usage$seconds, usage$n_failed,
                format(usage$input_tokens, big.mark = ","), usage$cost_usd))
    if (usage$n_failed > 0) {
      cat("  First error:", errors[!is.na(errors)][1], "\n")
    }
  }
  attr(out, "usage") <- usage
  attr(out, "errors") <- errors
  out
}


#' Token usage and cost of Jev responses
#'
#' @param responses A list of Jev responses, as returned by
#'   \code{\link{jev_ask_many}} (or a single response from
#'   \code{\link{jev_ask}}).
#' @param price_per_million Price in USD per million input tokens. Default is
#'   \code{0.042} (September 2026). Output tokens are free.
#'
#' @return A one-row data.frame with \code{n_texts}, \code{n_failed},
#'   \code{input_tokens} and \code{cost_usd}.
#'
#' @examples
#' # Two fake responses (the second one failed)
#' fake <- list(list(usage = list(input_tokens = 250)), NULL)
#' jev_usage(fake)
#'
#' @export
jev_usage <- function(responses, price_per_million = 0.042) {
  if (!is.null(responses$usage)) responses <- list(responses)   # a single response
  tokens <- vapply(responses, function(r) {
    if (is.null(r) || is.null(r$usage$input_tokens)) 0 else as.numeric(r$usage$input_tokens)
  }, numeric(1))
  data.frame(
    n_texts = length(responses),
    n_failed = sum(vapply(responses, is.null, logical(1))),
    input_tokens = sum(tokens),
    cost_usd = round(sum(tokens) / 1e6 * price_per_million, 6)
  )
}


#' List the available Jev models
#'
#' @return A data.frame with one row per model (name, description and
#'   release date, when available).
#'
#' @examples
#' \dontrun{
#' set_typesafe_key()
#' jev_list_models()
#' }
#' @export
jev_list_models <- function() {
  res <- httr2::request(paste0(.jev_defaults$base_url, "/models")) |>
    httr2::req_auth_bearer_token(.get_typesafe_key()) |>
    httr2::req_retry(max_tries = 3) |>
    httr2::req_error(body = .jev_error_body) |>
    httr2::req_perform() |>
    httr2::resp_body_json()
  models <- res$models %||% res$data %||% res
  df <- do.call(rbind, lapply(models, function(m) {
    data.frame(model = as.character(m$name %||% m$id %||% NA),
               description = as.character(m$description %||% NA),
               release_date = as.character(m$release_date %||% NA),
               stringsAsFactors = FALSE)
  }))
  df
}


# ------------------------------------------------------------------------------
# Parsing responses
# ------------------------------------------------------------------------------

#' Convert Jev answers to data frames
#'
#' Turn a list of Jev responses (from \code{\link{jev_ask_many}}) into one
#' data.frame per question type, with one row per text. Failed requests or
#' missing answers become \code{NA}.
#'
#' @param responses A list of Jev responses.
#' @param question Name of the question to extract.
#' @param questions For \code{jev_parse_noul()}: names of one or more Noul
#'   questions (one column each).
#' @param options For \code{jev_parse_choice()}: the option names. If
#'   \code{NULL} (default), they are taken from the first successful answer.
#' @param n_levels For \code{jev_parse_score()}: number of rubric levels. If
#'   \code{NULL} (default), taken from the first successful answer.
#'
#' @return
#' \describe{
#'   \item{\code{jev_parse_choice()}}{\code{choice}, \code{confidence} and
#'     one probability column per option.}
#'   \item{\code{jev_parse_noul()}}{One column per question with the
#'     probability that the statement is true.}
#'   \item{\code{jev_parse_score()}}{\code{score} (expected level),
#'     \code{level} (most probable level), \code{confidence} and one
#'     probability column per level (\code{"0"}, \code{"1"}, ...).}
#' }
#'
#' @examples
#' # A fake response with the same structure as the API output
#' fake <- list(list(answers = list(
#'   facet = list(type = "choice", choice = "Friendly", confidence = 0.8,
#'                probabilities = list(Friendly = 0.8, Positive = 0.2)),
#'   others = list(type = "noul", noul = 0.93),
#'   level = list(type = "score", score = 1.7, confidence = 0.6,
#'                probabilities = list("0" = 0.1, "1" = 0.1, "2" = 0.8))
#' )), NULL)   # the second request failed
#'
#' jev_parse_choice(fake, "facet")
#' jev_parse_noul(fake, "others")
#' jev_parse_score(fake, "level")
#'
#' @name jev_parse
NULL

#' @rdname jev_parse
#' @export
jev_parse_choice <- function(responses, question, options = NULL) {
  answers <- lapply(responses, function(r) if (is.null(r)) NULL else r$answers[[question]])
  if (is.null(options)) {
    first <- Find(Negate(is.null), answers)
    if (is.null(first)) stop("No successful answers for question '", question, "'.", call. = FALSE)
    options <- names(first$probabilities)
  }
  rows <- lapply(answers, function(a) {
    probs <- vapply(options, function(o) {
      p <- a$probabilities[[o]]
      if (is.null(p)) NA_real_ else as.numeric(p)
    }, numeric(1))
    data.frame(choice = if (is.null(a$choice)) NA_character_ else as.character(a$choice),
               confidence = if (is.null(a$confidence)) NA_real_ else as.numeric(a$confidence),
               t(probs), check.names = FALSE, stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' @rdname jev_parse
#' @export
jev_parse_noul <- function(responses, questions) {
  out <- do.call(rbind, lapply(responses, function(r) {
    vals <- vapply(questions, function(q) {
      v <- if (is.null(r)) NULL else r$answers[[q]]$noul
      if (is.null(v)) NA_real_ else as.numeric(v)
    }, numeric(1))
    as.data.frame(t(vals), check.names = FALSE)
  }))
  rownames(out) <- NULL
  out
}

#' @rdname jev_parse
#' @export
jev_parse_score <- function(responses, question, n_levels = NULL) {
  answers <- lapply(responses, function(r) if (is.null(r)) NULL else r$answers[[question]])
  if (is.null(n_levels)) {
    first <- Find(Negate(is.null), answers)
    if (is.null(first)) stop("No successful answers for question '", question, "'.", call. = FALSE)
    n_levels <- length(first$probabilities)
  }
  lev <- as.character(seq(0, n_levels - 1))
  rows <- lapply(answers, function(a) {
    probs <- vapply(lev, function(k) {
      p <- a$probabilities[[k]]
      if (is.null(p)) NA_real_ else as.numeric(p)
    }, numeric(1))
    data.frame(score = if (is.null(a$score)) NA_real_ else as.numeric(a$score),
               level = if (all(is.na(probs))) NA_integer_ else as.integer(which.max(probs) - 1L),
               confidence = if (is.null(a$confidence)) NA_real_ else as.numeric(a$confidence),
               t(probs), check.names = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}


# ------------------------------------------------------------------------------
# High-level wrappers (same output style as the classify_*() functions)
# ------------------------------------------------------------------------------

#' Classify texts with Jev (Choice question)
#'
#' Zero-shot multi-class classification with Jev. Each text receives one
#' option, its probability (\code{confidence}) and the probability of every
#' option, so the output can go straight into \code{\link{calibration_metrics}}
#' and \code{\link{evaluate_classification}}.
#'
#' @param texts Character vector of texts to classify.
#' @param criteria Named character vector: names are the classes and values
#'   describe when each class applies (see \code{\link{label_descriptions}}
#'   for examples). An unnamed vector of class labels also works, but good
#'   descriptions usually help.
#' @param instructions The question asked about every text.
#' @param model Jev model version. Default is \code{"jev-1.13.0"}.
#' @param max_active Maximum number of simultaneous requests. Default \code{10}.
#' @param verbose Print a progress bar and a cost summary. Default \code{TRUE}.
#'
#' @return A data.frame with \code{text_id}, \code{text} (truncated to 200
#'   characters), \code{prediction}, \code{confidence} and one probability
#'   column per class. Failed requests give \code{NA}. The attribute
#'   \code{"usage"} holds tokens and cost.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' set_typesafe_key()
#'
#' res <- jev_classify(
#'   extraversion$text, label_descriptions$extraversion,
#'   instructions = "Which facet of extraversion does this text express most clearly?"
#' )
#' head(res)
#' attr(res, "usage")
#'
#' # Evaluation and calibration
#' evaluate_classification(extraversion$subtopic, res$prediction, "Jev")
#' probs <- as.matrix(res[, names(label_descriptions$extraversion)])
#' calibration_metrics(probs, extraversion$subtopic, "Jev")
#'
#' # Triage: send the least confident texts to a human coder
#' res[order(res$confidence)[1:10], c("text", "prediction", "confidence")]
#' }
#'
#' @seealso \code{\link{jev_multilabel}}, \code{\link{jev_rate}},
#'   \code{\link{classify_direct}}, \code{\link{classify_by_description}}
#' @export
jev_classify <- function(texts, criteria,
                         instructions = "Which option best describes this text?",
                         model = "jev-1.13.0",
                         max_active = 10, verbose = TRUE) {
  criteria <- .as_named_criteria(criteria)
  questions <- list(label = jev_choice(instructions, criteria))
  responses <- jev_ask_many(texts, questions, model = model,
                            max_active = max_active, verbose = verbose)
  parsed <- jev_parse_choice(responses, "label", options = names(criteria))
  out <- data.frame(text_id = seq_along(texts), text = substr(texts, 1, 200),
                    prediction = normalize_labels(parsed$choice, names(criteria)),
                    parsed[, -1, drop = FALSE],
                    check.names = FALSE, stringsAsFactors = FALSE)
  attr(out, "usage") <- attr(responses, "usage")
  out
}


#' Multi-label classification with Jev (one Noul question per label)
#'
#' Asks one yes/no (Noul) question per label about every text, all in one
#' call per text. Unlike \code{\link{jev_classify}}, labels are not mutually
#' exclusive: a text can express several facets at once, and the
#' probabilities do not need to sum to 1.
#'
#' @param texts Character vector of texts.
#' @param statements Named character vector: names are the labels and values
#'   are statements that are "true" when the label applies, e.g.
#'   \code{c(Friendly = "The writer is warm and welcoming toward others.")}.
#' @param true,false Optional general descriptions of what counts as true and
#'   false, shared by all statements.
#' @inheritParams jev_classify
#'
#' @return A data.frame with \code{text_id}, \code{text} and one column per
#'   label with the probability that the statement is true.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' set_typesafe_key()
#'
#' statements <- c(
#'   Friendly  = "The writer is warm, welcoming and kind toward other people.",
#'   Positive  = "The writer is optimistic and upbeat.",
#'   Assertive = "The writer takes charge and speaks up directly.",
#'   Energetic = "The writer is full of energy and enthusiasm."
#' )
#' ml <- jev_multilabel(extraversion$text, statements)
#'
#' # How many texts express more than one facet (P > 0.5)?
#' table(rowSums(ml[, names(statements)] > 0.5))
#'
#' # AUC of each statement for its own facet
#' sapply(names(statements), function(f) {
#'   roc_auc(ml[[f]], extraversion$subtopic == f)
#' })
#' }
#'
#' @seealso \code{\link{jev_classify}}, \code{\link{roc_auc}}
#' @export
jev_multilabel <- function(texts, statements, true = NULL, false = NULL,
                           model = "jev-1.13.0",
                           max_active = 10, verbose = TRUE) {
  statements <- .as_named_criteria(statements)
  questions <- lapply(unname(statements), function(s) jev_noul(s, true = true, false = false))
  keys <- paste0("q", seq_along(statements))
  names(questions) <- keys
  responses <- jev_ask_many(texts, questions, model = model,
                            max_active = max_active, verbose = verbose)
  probs <- jev_parse_noul(responses, keys)
  names(probs) <- names(statements)
  out <- data.frame(text_id = seq_along(texts), text = substr(texts, 1, 200),
                    probs, check.names = FALSE, stringsAsFactors = FALSE)
  attr(out, "usage") <- attr(responses, "usage")
  out
}


#' Rate texts on an ordinal rubric with Jev (Score question)
#'
#' @param texts Character vector of texts.
#' @param levels Character vector of 2 to 10 level descriptions, in
#'   increasing order (the first level is 0).
#' @param instructions The rating question.
#' @inheritParams jev_classify
#'
#' @return A data.frame with \code{text_id}, \code{text}, \code{score} (the
#'   expected level, a probability-weighted average), \code{level} (the most
#'   probable level), \code{confidence} and one probability column per level.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(neuroticism)
#' set_typesafe_key()
#'
#' intensity <- jev_rate(
#'   neuroticism$text,
#'   levels = c("No negative emotion",
#'              "Mild negative emotion",
#'              "Clear negative emotion",
#'              "Intense negative emotion"),
#'   instructions = "How intense is the NEGATIVE emotion expressed in this text?"
#' )
#' tapply(intensity$score, neuroticism$subtopic, mean)
#' }
#'
#' @seealso \code{\link{jev_score}}, \code{\link{jev_classify}}
#' @export
jev_rate <- function(texts, levels,
                     instructions = "Rate this text on the following scale.",
                     model = "jev-1.13.0",
                     max_active = 10, verbose = TRUE) {
  questions <- list(rating = jev_score(instructions, levels))
  responses <- jev_ask_many(texts, questions, model = model,
                            max_active = max_active, verbose = verbose)
  parsed <- jev_parse_score(responses, "rating", n_levels = length(levels))
  out <- data.frame(text_id = seq_along(texts), text = substr(texts, 1, 200),
                    parsed, check.names = FALSE, stringsAsFactors = FALSE)
  attr(out, "usage") <- attr(responses, "usage")
  out
}
