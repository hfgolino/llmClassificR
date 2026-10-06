#' Cosine similarity between two vectors
#'
#' Computes the cosine similarity between two numeric vectors. Values range
#' from -1 (opposite) to 1 (identical direction), with 0 meaning orthogonal.
#'
#' @param a Numeric vector.
#' @param b Numeric vector of the same length.
#'
#' @return A single numeric value (the cosine similarity).
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' a <- c(1, 2, 3)
#' b <- c(1, 2, 3)
#' cosine_similarity(a, b)  # 1.0 (identical)
#'
#' c <- c(-1, -2, -3)
#' cosine_similarity(a, c)  # -1.0 (opposite)
#'
#' @export
cosine_similarity <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}


#' Cosine similarity matrix
#'
#' Computes pairwise cosine similarities between all rows of a matrix.
#'
#' @param x A numeric matrix (rows = observations, cols = dimensions).
#'
#' @return A symmetric numeric matrix of cosine similarities.
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' embeddings <- embed_voyage(texts)
#' sim_mat <- cosine_similarity_matrix(embeddings)
#' image(sim_mat, main = "Cosine Similarity Matrix")
#' }
#'
#' @export
cosine_similarity_matrix <- function(x) {
  # Normalize rows to unit length
  norms <- sqrt(rowSums(x^2))
  x_norm <- x / norms
  # Dot product of normalized rows = cosine similarity
  sim <- tcrossprod(x_norm)
  rownames(sim) <- rownames(x)
  colnames(sim) <- rownames(x)
  sim
}


#' Compare multiple classification methods
#'
#' Evaluates several sets of predictions against the same true labels and
#' returns one summary table, sorted by Normalized Mutual Information (NMI).
#' Works with class predictions (LLMs, Jev, embeddings) and with cluster
#' memberships (EGA communities), because NMI does not need the label names
#' to match.
#'
#' @param true_labels Character or factor vector of true labels.
#' @param ... Named arguments where each name is a method label and each
#'   value is either a vector of predictions (or cluster memberships) or a
#'   data.frame with a \code{prediction} column.
#'
#' @return A data.frame (returned invisibly and printed) with columns:
#' \describe{
#'   \item{method}{Name of the classification method}
#'   \item{NMI}{Normalized Mutual Information (see \code{\link{compute_nmi}})}
#'   \item{accuracy}{Proportion of exact matches (ignoring case and
#'     punctuation). Only meaningful for class predictions, not clusters}
#'   \item{macro_F1}{Average per-class F1 score}
#'   \item{kappa}{Cohen's kappa}
#'   \item{n_invalid}{Predictions that are missing or not one of the true
#'     classes (for cluster numbers this equals the number of texts)}
#'   \item{n_classes_predicted}{Number of unique predicted classes}
#' }
#'
#' @details
#' Since version 0.2.0, NMI is computed with \code{\link{compute_nmi}}
#' (no aricode needed). Missing predictions are treated as one "(invalid)"
#' category for class predictions, so they count as errors.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' truth <- rep(c("Friendly", "Positive", "Assertive"), each = 4)
#' good  <- truth
#' good[c(2, 7)] <- c("Positive", "Friendly")
#' noisy <- sample(truth)
#' clusters <- rep(c(2, 3, 1), each = 4)   # e.g., EGA communities
#'
#' compare_methods(truth, "Good model" = good, "Random" = noisy, "EGA" = clusters)
#'
#' \dontrun{
#' data(extraversion)
#' classes <- unique(extraversion$subtopic)
#'
#' res1 <- classify_direct(extraversion$text, classes)
#' res2 <- classify_cot(extraversion$text, classes)
#' res3 <- classify_schema(extraversion$text, classes)
#'
#' compare_methods(
#'   true_labels = extraversion$subtopic,
#'   "Direct"           = res1,
#'   "Chain-of-Thought" = res2,
#'   "Strict schema"    = res3
#' )
#' }
#'
#' @seealso \code{\link{evaluate_classification}}, \code{\link{compute_nmi}}
#'
#' @export
compare_methods <- function(true_labels, ...) {

  methods <- list(...)
  method_names <- names(methods)

  if (is.null(method_names) || any(method_names == "")) {
    stop("All arguments must be named (method labels).", call. = FALSE)
  }

  true_labels <- as.character(true_labels)
  classes <- sort(unique(true_labels[!is.na(true_labels)]))

  rows <- lapply(seq_along(methods), function(i) {
    preds <- methods[[i]]
    if (is.data.frame(preds) && "prediction" %in% colnames(preds)) {
      preds <- preds$prediction
    }
    preds <- .match_to_classes(preds, classes)
    ev <- evaluate_classification(true_labels, preds, method_names[i], digits = 4)
    # NMI on the raw memberships: missing EGA assignments are singletons,
    # missing LLM answers one "(invalid)" category
    is_cluster <- any(!is.na(preds)) && all(is.na(preds) | !(preds %in% classes))
    ev$NMI <- round(compute_nmi(true_labels, preds,
                                na = if (is_cluster) "singleton" else "category"), 4)
    ev$n_classes_predicted <- length(unique(preds[!is.na(preds)]))
    ev
  })
  results <- do.call(rbind, rows)
  results <- results[, c("method", "NMI", "accuracy", "macro_F1", "kappa",
                         "n_invalid", "n_classes_predicted")]

  # Sort by NMI
  results <- results[order(-results$NMI), ]
  rownames(results) <- NULL

  cat("\n==========================================\n")
  cat("  CLASSIFICATION METHOD COMPARISON\n")
  cat("==========================================\n")
  print(results, row.names = FALSE)
  cat("==========================================\n")

  invisible(results)
}


# Null-coalescing operator (base R has it only from R 4.4.0)
`%||%` <- function(x, y) if (is.null(x)) y else x
