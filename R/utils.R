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
#' Computes Normalized Mutual Information (NMI) between true labels and
#' predictions from multiple classification methods. Returns a summary
#' table for easy comparison.
#'
#' @param true_labels Character or factor vector of true labels.
#' @param ... Named arguments where each name is a method label and each
#'   value is either a character vector of predictions or a data.frame
#'   with a \code{prediction} column.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{method}{Name of the classification method}
#'   \item{NMI}{Normalized Mutual Information score}
#'   \item{accuracy}{Proportion of exact matches (if labels align)}
#'   \item{n_classes_predicted}{Number of unique predicted classes}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' classes <- unique(extraversion$subtopic)
#'
#' res1 <- classify_direct(extraversion$text, classes)
#' res2 <- classify_cot(extraversion$text, classes)
#' res3 <- classify_json(extraversion$text, classes)
#'
#' comparison <- compare_methods(
#'   true_labels = extraversion$subtopic,
#'   "Direct"          = res1,
#'   "Chain-of-Thought" = res2,
#'   "Structured JSON"  = res3
#' )
#' print(comparison)
#' }
#'
#' @seealso \code{\link{classify_direct}}, \code{\link{classify_cot}}
#'
#' @export
compare_methods <- function(true_labels, ...) {

  if (!requireNamespace("aricode", quietly = TRUE)) {
    stop("Package 'aricode' is required. Install with: install.packages('aricode')",
         call. = FALSE)
  }

  methods <- list(...)
  method_names <- names(methods)

  if (is.null(method_names) || any(method_names == "")) {
    stop("All arguments must be named (method labels).", call. = FALSE)
  }

  results <- data.frame(
    method = character(),
    NMI = numeric(),
    accuracy = numeric(),
    n_classes_predicted = integer(),
    stringsAsFactors = FALSE
  )

  true_labels <- as.character(true_labels)

  for (i in seq_along(methods)) {
    preds <- methods[[i]]

    # Accept data.frame or vector
    if (is.data.frame(preds) && "prediction" %in% colnames(preds)) {
      preds <- preds$prediction
    }
    preds <- as.character(preds)

    nmi_val <- aricode::NMI(true_labels, preds)
    acc <- mean(true_labels == preds, na.rm = TRUE)
    n_cls <- length(unique(preds[!is.na(preds)]))

    results <- rbind(results, data.frame(
      method = method_names[i],
      NMI = round(nmi_val, 4),
      accuracy = round(acc, 4),
      n_classes_predicted = n_cls,
      stringsAsFactors = FALSE
    ))
  }

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
