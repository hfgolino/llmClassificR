#' Normalized Mutual Information between two labelings
#'
#' Computes the Normalized Mutual Information (NMI) between true labels and
#' predicted labels or cluster memberships (for example, EGA communities).
#' NMI is 1 when the two labelings carry the same information and 0 when
#' they are independent. Label names do not need to match, so NMI can
#' compare classes with cluster numbers.
#'
#' This function does not depend on \pkg{aricode}. It gives the same values
#' as \code{aricode::NMI(factor(x), factor(y))} (the default "max" variant),
#' but also accepts character vectors (aricode >= 1.1.0 returns
#' "NA are not supported" for character labels) and handles missing values
#' explicitly.
#'
#' @param truth Vector of true labels (character, factor or integer).
#' @param pred Vector of predicted labels or cluster memberships.
#' @param na How to treat \code{NA} in \code{pred}:
#'   \code{"singleton"} (default) makes each \code{NA} its own one-member
#'   cluster, which is the right choice for EGA texts that were not assigned
#'   to any community; \code{"category"} puts all \code{NA}s in one extra
#'   "(invalid)" class, the usual choice for invalid LLM answers;
#'   \code{"drop"} removes those texts. Texts with \code{NA} in \code{truth}
#'   are always removed.
#' @param variant Normalization: \code{"max"} (default, as in aricode),
#'   \code{"sqrt"}, \code{"sum"} or \code{"min"}.
#'
#' @return A single number between 0 and 1.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' truth <- c("Friendly", "Friendly", "Positive", "Positive", "Assertive", "Assertive")
#'
#' # Same partition, different names: NMI = 1
#' compute_nmi(truth, c(3, 3, 1, 1, 2, 2))
#'
#' # One text left unassigned by EGA (NA)
#' compute_nmi(truth, c(3, 3, 1, 1, 2, NA))
#'
#' # Invalid LLM answers as one extra category
#' compute_nmi(truth, c("Friendly", NA, "Positive", "Positive", "Assertive", NA),
#'             na = "category")
#'
#' @seealso \code{\link{evaluate_classification}}, \code{\link{compare_methods}}
#' @export
compute_nmi <- function(truth, pred, na = c("singleton", "category", "drop"),
                        variant = c("max", "sqrt", "sum", "min")) {
  na <- match.arg(na)
  variant <- match.arg(variant)
  if (length(truth) != length(pred)) {
    stop("'truth' and 'pred' must have the same length.", call. = FALSE)
  }
  keep <- !is.na(truth)
  truth <- as.character(truth)[keep]
  pred <- as.character(pred)[keep]

  miss <- is.na(pred)
  if (any(miss)) {
    if (na == "singleton") {
      pred[miss] <- paste0(".unassigned_", seq_len(sum(miss)))
    } else if (na == "category") {
      pred[miss] <- "(invalid)"
    } else {
      truth <- truth[!miss]
      pred <- pred[!miss]
    }
  }
  if (length(truth) == 0) return(NA_real_)

  pxy <- table(truth, pred) / length(truth)
  px <- rowSums(pxy)
  py <- colSums(pxy)
  expected <- outer(px, py)
  nz <- pxy > 0
  mi <- sum(pxy[nz] * log(pxy[nz] / expected[nz]))
  hx <- -sum(px[px > 0] * log(px[px > 0]))
  hy <- -sum(py[py > 0] * log(py[py > 0]))
  denom <- switch(variant,
                  max = max(hx, hy),
                  sqrt = sqrt(hx * hy),
                  sum = (hx + hy) / 2,
                  min = min(hx, hy))
  if (denom == 0) return(if (hx == 0 && hy == 0) NaN else 0)
  max(0, min(1, mi / denom))
}


#' Evaluate a classification: accuracy, macro-F1, kappa and NMI
#'
#' Computes the metrics used throughout the course for any vector of
#' predictions (LLM, Jev, embeddings, bag-of-words):
#' \itemize{
#'   \item \strong{accuracy}: proportion of correct predictions;
#'   \item \strong{macro-F1}: average of the per-class F1 scores (every
#'     class counts the same, useful when classes are unbalanced);
#'   \item \strong{kappa}: Cohen's kappa, agreement corrected for chance (as
#'     in human coding);
#'   \item \strong{NMI}: Normalized Mutual Information (see
#'     \code{\link{compute_nmi}});
#'   \item \strong{n_invalid}: predictions that are missing or not one of
#'     the true classes. They are counted as errors, never dropped.
#' }
#'
#' Predictions are matched to the true class names ignoring case, spaces and
#' punctuation (\code{" friendly."} counts as \code{"Friendly"}; see
#' \code{\link{normalize_labels}}).
#'
#' @param truth Vector of true labels.
#' @param pred Vector of predicted labels (\code{NA} = no valid answer), or a
#'   data.frame with a \code{prediction} column.
#' @param method Optional name of the method (first column of the output).
#' @param digits Number of decimals. Default is \code{3}.
#'
#' @return A one-row data.frame with columns \code{method}, \code{accuracy},
#'   \code{macro_F1}, \code{kappa}, \code{NMI} and \code{n_invalid}. Use
#'   \code{rbind()} to combine several methods.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' truth <- c("Friendly", "Positive", "Assertive", "Energetic",
#'            "Friendly", "Positive", "Assertive", "Energetic")
#' pred  <- c("friendly", "Positive", "Assertive", "Positive",
#'            "Friendly", NA, "assertive.", "Energetic")
#' evaluate_classification(truth, pred, "Example")
#'
#' # Combine several methods in one table
#' rbind(
#'   evaluate_classification(truth, pred, "Model A"),
#'   evaluate_classification(truth, rev(truth), "Model B")
#' )
#'
#' @seealso \code{\link{compare_methods}}, \code{\link{calibration_metrics}}
#' @export
evaluate_classification <- function(truth, pred, method = "", digits = 3) {
  if (is.data.frame(pred) && "prediction" %in% names(pred)) pred <- pred$prediction
  if (length(truth) != length(pred)) {
    stop("'truth' and 'pred' must have the same length.", call. = FALSE)
  }
  keep <- !is.na(truth)
  truth <- as.character(truth)[keep]
  pred <- .match_to_classes(pred[keep], sort(unique(truth)))
  classes <- sort(unique(truth))

  invalid <- is.na(pred) | !(pred %in% classes)
  pred[is.na(pred)] <- "(invalid)"

  f1 <- vapply(classes, function(k) {
    tp <- sum(pred == k & truth == k)
    precision <- if (sum(pred == k) == 0) 0 else tp / sum(pred == k)
    recall <- tp / sum(truth == k)
    if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
  }, numeric(1))

  levs <- union(classes, unique(pred))
  tab <- table(factor(truth, levs), factor(pred, levs))
  po <- sum(diag(tab)) / sum(tab)
  pe <- sum(rowSums(tab) * colSums(tab)) / sum(tab)^2
  kappa <- if (pe == 1) NA_real_ else (po - pe) / (1 - pe)

  data.frame(method = method,
             accuracy = round(mean(pred == truth), digits),
             macro_F1 = round(mean(f1), digits),
             kappa = round(kappa, digits),
             NMI = round(compute_nmi(truth, pred, na = "category"), digits),
             n_invalid = sum(invalid),
             stringsAsFactors = FALSE)
}


#' Map predictions onto class names where possible (keep others as they are)
#' @noRd
.match_to_classes <- function(pred, classes) {
  pred <- as.character(pred)
  mapped <- normalize_labels(pred, classes)
  ifelse(is.na(mapped), pred, mapped)
}


#' Area under the ROC curve
#'
#' The probability that a randomly chosen positive text receives a higher
#' score than a randomly chosen negative text (ties count one half).
#' 0.5 = chance, 1 = perfect ranking. Useful for yes/no probabilities such as
#' Jev's Noul answers or any per-class score.
#'
#' @param score Numeric vector of scores or probabilities (\code{NA}s are
#'   ignored).
#' @param is_positive Logical vector (or 0/1): is the text a positive case?
#'
#' @return A single number between 0 and 1.
#'
#' @examples
#' roc_auc(c(0.9, 0.8, 0.3, 0.2), c(TRUE, TRUE, FALSE, FALSE))  # 1
#' roc_auc(c(0.9, 0.2, 0.8, 0.3), c(TRUE, TRUE, FALSE, FALSE))  # 0.5
#'
#' @seealso \code{\link{jev_multilabel}}
#' @export
roc_auc <- function(score, is_positive) {
  is_positive <- as.logical(is_positive)
  ok <- !is.na(score) & !is.na(is_positive)
  score <- score[ok]
  is_positive <- is_positive[ok]
  n1 <- sum(is_positive)
  n0 <- sum(!is_positive)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(score)
  (sum(r[is_positive]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
