# ==============================================================================
# Synthetic labeled texts: generation (strict JSON Schema) and quality checks
# ==============================================================================

#' Generate labeled synthetic texts with a strict JSON Schema
#'
#' Asks a Groq reasoning model to write short texts for each topic
#' (for example, the four facets of extraversion). A JSON Schema in strict
#' mode guarantees the format: every document has an id, a topic that is one
#' of \code{topics} (an \code{enum}) and a text, so parsing cannot fail.
#'
#' @param topics Character vector of topic labels, or a named vector where
#'   names are the labels and values describe each topic (descriptions give
#'   the model a clearer target; see \code{\link{label_descriptions}}).
#' @param n_per_topic Number of texts per topic in each call. Default
#'   \code{5}. Keep it small (5-10) and use \code{n_batches} for more texts.
#' @param n_batches Number of calls. Default \code{1}.
#' @param domain Optional short description of the domain, e.g.
#'   \code{"extraversion facets in first-person self-descriptions"}.
#' @param instructions Optional extra instructions (style, length, setting).
#' @param model A Groq model that supports strict structured outputs. Default
#'   \code{"openai/gpt-oss-20b"}.
#' @param temperature Sampling temperature. Default \code{1} (diversity).
#' @param system_prompt System prompt (persona).
#'
#' @return A data.frame with \code{doc_id}, \code{batch}, \code{subtopic} and
#'   \code{text}. Check its quality with \code{\link{diversity_report}}
#'   before using it.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' set_groq_key()
#' synthetic <- generate_labeled_texts(
#'   topics = label_descriptions$extraversion,
#'   n_per_topic = 5, n_batches = 2,
#'   domain = "facets of extraversion",
#'   instructions = paste("2-3 sentences; vary first and third person and",
#'                        "the setting (work, family, leisure, school).")
#' )
#' table(synthetic$subtopic)
#' diversity_report(synthetic$text, synthetic$subtopic, "gpt-oss-20b")
#'
#' # Do the new texts look like the old ones? A baseline trained on one set
#' # and tested on the other checks for distribution shift.
#' }
#'
#' @seealso \code{\link{diversity_report}}, \code{\link{classify_schema}}
#' @export
generate_labeled_texts <- function(topics, n_per_topic = 5, n_batches = 1,
                                   domain = NULL, instructions = NULL,
                                   model = "openai/gpt-oss-20b", temperature = 1,
                                   system_prompt = paste(
                                     "You are a behavioral scientist and expert psychometrician",
                                     "skilled at generating diverse, realistic text data for",
                                     "computational analysis.")) {
  topics <- .as_named_criteria(topics)
  labels <- names(topics)
  described <- any(topics != labels)

  schema <- list(
    type = "object",
    properties = list(
      documents = list(
        type = "array",
        items = list(
          type = "object",
          properties = list(
            doc_id = list(type = "integer"),
            subtopic = list(type = "string", enum = as.list(labels)),
            text = list(type = "string")
          ),
          required = list("doc_id", "subtopic", "text"),
          additionalProperties = FALSE
        )
      )
    ),
    required = list("documents"),
    additionalProperties = FALSE
  )

  topic_lines <- if (described) {
    paste0("- ", labels, ": ", topics, collapse = "\n")
  } else {
    paste0("- ", labels, collapse = "\n")
  }
  prompt <- paste0(
    "Generate ", n_per_topic * length(labels), " diverse short documents",
    if (!is.null(domain)) paste0(" about ", domain) else "",
    ", exactly ", n_per_topic, " for each of these topics:\n", topic_lines, "\n\n",
    "Each document must clearly express its topic without naming it. ",
    "Vary vocabulary, sentence structure, perspective and setting; avoid repeating openings.",
    if (!is.null(instructions)) paste0("\n", instructions) else ""
  )
  response_format <- list(type = "json_schema",
                          json_schema = list(name = "labeled_documents", strict = TRUE,
                                             schema = schema))
  out <- list()
  for (b in seq_len(n_batches)) {
    cat("Batch", b, "of", n_batches, "\n")
    docs <- tryCatch({
      content <- .groq_chat(prompt, model = model, system = system_prompt,
                            temperature = temperature,
                            max_tokens = n_per_topic * length(labels) * 120L + 300L,
                            response_format = response_format)
      d <- .parse_json(content)$documents
      data.frame(batch = b, subtopic = as.character(d$subtopic),
                 text = trimws(as.character(d$text)), stringsAsFactors = FALSE)
    }, error = function(e) {
      warning("Batch ", b, " failed: ", conditionMessage(e), call. = FALSE)
      NULL
    })
    out[[b]] <- docs
  }
  res <- do.call(rbind, out)
  if (is.null(res)) stop("All batches failed.", call. = FALSE)
  res <- cbind(doc_id = seq_len(nrow(res)), res)
  rownames(res) <- NULL
  res
}


#' Quality checks for (synthetic) text data
#'
#' Synthetic data are only useful if they are diverse and on target
#' (Raschka, 2026, stresses that curating training data mattered more than
#' model tweaks for Jev-like classifiers). Quick checks:
#' \itemize{
#'   \item \strong{exact_duplicates}: repeated texts (ignoring case);
#'   \item \strong{lexical_diversity}: distinct words / all words;
#'   \item \strong{distinct_2}: distinct word pairs / all word pairs;
#'   \item \strong{within_label_overlap}: mean word overlap (Jaccard) between
#'     texts of the same label (higher = more repetitive);
#'   \item \strong{top_opening_share}: share of texts starting with the most
#'     common three-word opening.
#' }
#'
#' @param texts Character vector of texts.
#' @param labels Optional vector of labels (for the within-label overlap).
#' @param name Name of the data set or generator (first column).
#'
#' @return A one-row data.frame. Use \code{rbind()} to compare generators.
#'
#' @examples
#' data(extraversion); data(extraversion_3p); data(extraversion_llama)
#' rbind(
#'   diversity_report(extraversion$text, extraversion$subtopic, "gpt-oss-20b"),
#'   diversity_report(extraversion_3p$text, extraversion_3p$subtopic, "gpt-oss-120b"),
#'   diversity_report(extraversion_llama$text, extraversion_llama$subtopic, "llama-3.1-8b")
#' )
#'
#' @seealso \code{\link{generate_labeled_texts}}
#' @export
diversity_report <- function(texts, labels = NULL, name = "") {
  words_of <- function(x) {
    w <- strsplit(tolower(gsub("[^[:alpha:][:space:]']", " ", x)), "\\s+")
    lapply(w, function(v) v[v != ""])
  }
  jaccard <- function(a, b) {
    u <- length(union(a, b))
    if (u == 0) 0 else length(intersect(a, b)) / u
  }
  texts <- as.character(texts)
  w <- words_of(texts)
  all_words <- unlist(w)
  bigrams <- unlist(lapply(w, function(v) {
    if (length(v) > 1) paste(utils::head(v, -1), utils::tail(v, -1)) else NULL
  }))
  within <- NA_real_
  if (!is.null(labels)) {
    per_label <- vapply(split(seq_along(texts), labels), function(idx) {
      if (length(idx) < 2) return(NA_real_)
      pairs <- utils::combn(idx, 2)
      mean(apply(pairs, 2, function(p) jaccard(unique(w[[p[1]]]), unique(w[[p[2]]]))))
    }, numeric(1))
    within <- mean(per_label, na.rm = TRUE)
  }
  openings <- vapply(w, function(v) paste(utils::head(v, 3), collapse = " "), character(1))
  data.frame(
    name = name,
    n_texts = length(texts),
    exact_duplicates = sum(duplicated(tolower(trimws(texts)))),
    mean_words = round(mean(lengths(w)), 1),
    lexical_diversity = round(length(unique(all_words)) / max(1, length(all_words)), 3),
    distinct_2 = round(length(unique(bigrams)) / max(1, length(bigrams)), 3),
    within_label_overlap = round(within, 3),
    top_opening_share = round(max(table(openings)) / length(texts), 3),
    stringsAsFactors = FALSE
  )
}
