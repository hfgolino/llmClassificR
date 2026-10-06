# ------------------------------------------------------------------------------
# Offline mocks for the Groq client (reticulate object) and the TypeSafe API.
# The Groq mock "classifies" a text by the class name it contains, and reacts
# to special words in the texts:
#   MISSING -> the answer for that text is left out of the JSON
#   WEIRD   -> the label comes back as " friendly." (case and punctuation)
#   INVALID -> the label is "Sociable" (not one of the classes)
#   BADJSON -> the whole chunk returns invalid JSON
# ------------------------------------------------------------------------------

options(llmClassificR.pause = 0, llmClassificR.verbose = FALSE)

mock_log <- new.env()
mock_log$calls <- list()

test_classes <- c("Friendly", "Positive", "Assertive", "Energetic")

make_texts <- function(n_per_class = 5, classes = test_classes) {
  lab <- rep(classes, each = n_per_class)
  data.frame(text = paste("Text", seq_along(lab), "shows a", lab, "person."),
             label = lab, stringsAsFactors = FALSE)
}

mock_label <- function(tx, classes) {
  if (grepl("INVALID", tx)) return("Sociable")
  hit <- classes[vapply(classes, function(cl) grepl(cl, tx, fixed = TRUE), logical(1))]
  lab <- if (length(hit)) hit[1] else classes[1]
  if (grepl("WEIRD", tx)) lab <- paste0(" ", tolower(lab), ".")
  lab
}

mock_create <- function(model, messages, temperature = 0, max_completion_tokens = NULL,
                        response_format = NULL, reasoning_effort = NULL, ...) {
  prompt <- messages[[length(messages)]]$content
  mock_log$calls[[length(mock_log$calls) + 1]] <- list(
    model = model, max_completion_tokens = max_completion_tokens,
    response_format = response_format, reasoning_effort = reasoning_effort,
    temperature = temperature, prompt = prompt, n_messages = length(messages))

  lines <- strsplit(prompt, "\n")[[1]]
  num <- regmatches(lines, regexec("^([0-9]+)\\. (.*)$", lines))
  num <- num[lengths(num) == 3]
  ids <- as.integer(vapply(num, `[`, "", 2))
  txs <- vapply(num, `[`, "", 3)
  m <- regmatches(prompt, regexec("[Cc]ategories: ([^\n]+)", prompt))[[1]]
  classes <- if (length(m)) trimws(strsplit(gsub('"', "", m[2]), ",")[[1]]) else test_classes

  keep <- !grepl("MISSING", txs)
  labs <- vapply(txs, mock_label, "", classes = classes, USE.NAMES = FALSE)
  to_json <- function(x) as.character(jsonlite::toJSON(x, auto_unbox = TRUE))

  content <- if (any(grepl("BADJSON", txs))) {
    "Sorry, I cannot answer in JSON."
  } else if (!is.null(response_format) && response_format$type == "json_schema") {
    if (response_format$json_schema$name == "labeled_documents") {
      topics <- unlist(response_format$json_schema$schema$properties$documents$items$properties$subtopic$enum)
      to_json(list(documents = data.frame(doc_id = seq_along(topics), subtopic = topics,
                                          text = paste("A new text about", topics))))
    } else {
      d <- data.frame(id = ids, category = labs, confidence = "high", reasoning = "because")[keep, ]
      to_json(list(classifications = d[rev(seq_len(nrow(d))), ]))   # reversed order
    }
  } else if (grepl("Rate how well", prompt)) {
    out <- lapply(labs, function(l) {
      v <- stats::setNames(as.list(rep(10, length(classes))), tolower(classes))
      k <- normalize_labels(l, classes)
      if (!is.na(k)) v[[tolower(k)]] <- 70
      v
    })
    names(out) <- ids
    to_json(out[keep])
  } else if (grepl('"evidence"', prompt)) {
    out <- lapply(labs, function(l) list(evidence = c("phrase one", "phrase two"),
                                         reasoning = "r", category = l, confidence = "High"))
    names(out) <- ids
    to_json(out[keep])
  } else if (grepl('"reasoning": "brief analysis"', prompt)) {
    out <- lapply(labs, function(l) list(reasoning = "r", category = l))
    names(out) <- ids
    to_json(out[keep])
  } else if (grepl("your confidence level", prompt)) {
    out <- lapply(labs, function(l) list(category = l, confidence = "medium", reasoning = "r"))
    names(out) <- ids
    to_json(out[keep])
  } else if (!is.null(response_format) && response_format$type == "json_object") {
    out <- as.list(labs)
    names(out) <- ids
    to_json(out[keep])
  } else {
    "Some generated prose."
  }
  list(choices = list(list(message = list(content = content))),
       usage = list(prompt_tokens = 100L, completion_tokens = 50L))
}

local_mock_groq <- function(env = parent.frame()) {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- pkg$groq_client
  pkg$groq_client <- list(chat = list(completions = list(create = mock_create)))
  mock_log$calls <- list()
  withr::defer(pkg$groq_client <- old, envir = env)
  invisible(NULL)
}

# ---- TypeSafe (Jev) mock for httr2 -------------------------------------------
jev_mock <- function(req) {
  if (grepl("/v1/models", req$url)) {
    return(httr2::response_json(body = list(models = list(
      list(name = "jev-1.13.0", description = "General-purpose system one model.",
           release_date = "2026-09-15")))))
  }
  b <- req$body$data
  tx <- if (is.list(b$state)) b$state$text else b$state
  if (grepl("FAIL", tx)) {
    return(httr2::response_json(status_code = 401L, body = list(detail = "Invalid API key")))
  }
  ans <- lapply(b$questions, function(q) {
    if (q$type == "choice") {
      opts <- names(q$criteria)
      lab <- mock_label(tx, opts)
      p <- stats::setNames(rep(0.1 / (length(opts) - 1), length(opts)), opts)
      p[lab] <- 0.9
      list(type = "choice", choice = lab, confidence = 0.9, probabilities = as.list(p))
    } else if (q$type == "noul") {
      hit <- any(vapply(test_classes, function(cl) grepl(cl, tx) && grepl(cl, q$instructions),
                        logical(1)))
      list(type = "noul", noul = if (hit) 0.95 else 0.05)
    } else {
      k <- length(q$criteria)
      p <- stats::setNames(rep(1 / k, k), 0:(k - 1))
      list(type = "score", score = sum((0:(k - 1)) * p), confidence = max(p),
           legend = stats::setNames(q$criteria, 0:(k - 1)), probabilities = as.list(p))
    }
  })
  httr2::response_json(body = list(model = b$model, answers = ans,
                                   usage = list(input_tokens = 300L, output_tokens = 20L)))
}

local_mock_jev <- function(env = parent.frame()) {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- pkg$typesafe_key
  pkg$typesafe_key <- "test-key"
  httr2::local_mocked_responses(jev_mock, env = env)
  withr::defer(pkg$typesafe_key <- old, envir = env)
  invisible(NULL)
}
