#' Softmax: turn scores into probabilities
#'
#' Converts scores (logits, cosine similarities, log-probabilities) into
#' probabilities that sum to 1. With a matrix, the softmax is applied to each
#' row (texts in rows, classes in columns).
#'
#' The \code{temperature} controls how sharp the probabilities are:
#' \code{temperature < 1} makes them more confident, \code{temperature > 1}
#' softer. Dividing all scores by the same temperature never changes which
#' class is largest, so accuracy is unchanged (see
#' \code{\link{fit_temperature}}).
#'
#' @param scores Numeric vector or matrix (rows = texts, columns = classes).
#' @param temperature Positive number. Default is \code{1}.
#'
#' @return Probabilities with the same shape (and names) as \code{scores}.
#'
#' @examples
#' softmax(c(Friendly = 2, Positive = 1, Assertive = 0))
#'
#' # Cosine similarities are small numbers: T = 1 gives nearly uniform
#' # probabilities, a small temperature sharpens them
#' cosines <- c(Friendly = 0.45, Positive = 0.30, Assertive = 0.20)
#' softmax(cosines)
#' softmax(cosines, temperature = 0.05)
#'
#' @seealso \code{\link{fit_temperature}}, \code{\link{calibration_metrics}}
#' @export
softmax <- function(scores, temperature = 1) {
  if (!is.numeric(temperature) || length(temperature) != 1 || temperature <= 0) {
    stop("'temperature' must be a single positive number.", call. = FALSE)
  }
  if (is.data.frame(scores)) scores <- .as_prob_matrix(scores)
  if (is.matrix(scores)) {
    s <- scores / temperature
    s <- s - apply(s, 1, max)   # numerical stability
    e <- exp(s)
    return(e / rowSums(e))
  }
  s <- scores / temperature
  e <- exp(s - max(s))
  e / sum(e)
}


#' Expected Calibration Error and Brier score
#'
#' A classifier is \emph{calibrated} when, among all texts it labels with
#' probability 0.80, about 80\% are correct.
#' \itemize{
#'   \item \strong{ECE} (Expected Calibration Error): texts are grouped into
#'     confidence bins, and ECE is the weighted average of
#'     |accuracy - confidence| across bins. 0 = perfect.
#'   \item \strong{Brier score}: mean squared distance between the
#'     probability vector and the one-hot truth. 0 = perfect. It rewards both
#'     accuracy and honest confidence (it is a proper scoring rule; see
#'     \code{\link{rlcr_reward}}).
#' }
#'
#' @param confidence Numeric vector of confidences (probability of the chosen
#'   class) between 0 and 1.
#' @param correct Logical vector: was the prediction correct?
#' @param n_bins Number of equal-width bins. Default is \code{10}.
#' @param probs For \code{brier_score()}: a probability matrix (rows = texts,
#'   columns = classes, with class names as column names), or a numeric vector
#'   of probabilities for a yes/no question.
#' @param truth For \code{brier_score()}: true labels (matrix case) or a
#'   logical / 0-1 vector (yes/no case).
#'
#' @return A single number. Texts with missing values are ignored.
#'
#' @examples
#' # Over-confident classifier: says 0.95 but is right 60% of the time
#' conf <- rep(0.95, 10)
#' correct <- c(rep(TRUE, 6), rep(FALSE, 4))
#' expected_calibration_error(conf, correct)   # 0.35
#'
#' probs <- rbind(c(A = 0.7, B = 0.2, C = 0.1),
#'                c(A = 0.1, B = 0.8, C = 0.1))
#' brier_score(probs, c("A", "C"))
#'
#' # Yes/no probabilities
#' brier_score(c(0.9, 0.2, 0.6), c(TRUE, FALSE, FALSE))
#'
#' @seealso \code{\link{calibration_metrics}}, \code{\link{reliability_table}}
#' @name calibration_errors
NULL

#' @rdname calibration_errors
#' @export
expected_calibration_error <- function(confidence, correct, n_bins = 10) {
  .check_unit_interval(confidence)
  ok <- !is.na(confidence) & !is.na(correct)
  confidence <- confidence[ok]
  correct <- as.numeric(correct[ok])
  if (length(confidence) == 0) return(NA_real_)
  bins <- cut(confidence, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  gaps <- tapply(seq_along(confidence), bins, function(i) {
    length(i) * abs(mean(correct[i]) - mean(confidence[i]))
  })
  sum(gaps, na.rm = TRUE) / length(confidence)
}

#' @rdname calibration_errors
#' @export
brier_score <- function(probs, truth) {
  if (is.null(dim(probs))) {
    y <- as.numeric(as.logical(truth))
    ok <- !is.na(probs) & !is.na(y)
    return(mean((probs[ok] - y[ok])^2))
  }
  probs <- as.matrix(probs)
  truth <- normalize_labels(truth, colnames(probs))
  ok <- stats::complete.cases(probs) & !is.na(truth)
  onehot <- outer(truth[ok], colnames(probs), "==") * 1
  mean(rowSums((probs[ok, , drop = FALSE] - onehot)^2))
}

#' Confidences must be probabilities (not percentages)
#' @noRd
.check_unit_interval <- function(x) {
  if (any(x < 0 | x > 1 + 1e-8, na.rm = TRUE)) {
    stop("Confidences must be between 0 and 1 (divide percentages by 100).", call. = FALSE)
  }
}

#' Negative log-likelihood of the true class (internal)
#' @noRd
.nll <- function(probs, truth) {
  idx <- match(truth, colnames(probs))
  p_true <- probs[cbind(seq_along(truth), idx)]
  -mean(log(pmax(p_true, 1e-12)))
}


#' Calibration summary for a matrix of class probabilities
#'
#' One-line summary of how accurate and how honest a set of probabilities
#' is: accuracy, mean confidence, ECE, Brier score and negative
#' log-likelihood (NLL). Works for any source of probabilities: Jev
#' (\code{\link{jev_classify}}), self-consistency votes
#' (\code{\link{classify_self_consistency}}), verbalized scores
#' (\code{\link{classify_scores}}), embeddings
#' (\code{\link{classify_by_description}}) or bag-of-words
#' (\code{\link{cv_logistic}}).
#'
#' @param probs Probability matrix or data.frame (rows = texts, columns =
#'   classes, with class names as column names). Rows are rescaled to sum
#'   to 1. Rows with missing values (failed requests) are dropped and
#'   counted in \code{n_missing}. For a data.frame, non-numeric columns and
#'   the bookkeeping columns \code{text_id}, \code{fold},
#'   \code{confidence}, \code{score}, \code{level} and \code{row} are
#'   ignored, so whole outputs of \code{\link{jev_classify}} or
#'   \code{\link{cv_logistic}} can be passed directly.
#' @param truth Vector of true labels (matched to the column names ignoring
#'   case and punctuation).
#' @param method Optional name of the method.
#' @param n_bins Number of bins for ECE. Default is \code{10}.
#' @param digits Number of decimals. Default is \code{3}.
#'
#' @return A one-row data.frame with \code{method}, \code{n},
#'   \code{n_missing}, \code{accuracy}, \code{mean_confidence}, \code{ECE},
#'   \code{Brier} and \code{NLL}.
#'
#' @details
#' If \code{mean_confidence} is much lower than \code{accuracy}, the model is
#' under-confident; if it is much higher, over-confident. Both can be
#' repaired with \code{\link{fit_temperature}} without changing the
#' predictions.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' data(minilm_embeddings)
#' data(extraversion)
#' emb <- minilm_embeddings
#'
#' # Zero-shot probabilities from label descriptions (no API, no training)
#' S <- cosine_scores(emb$texts$extraversion, emb$labels$extraversion)
#' calibration_metrics(softmax(S), extraversion$subtopic, "T = 1")
#' calibration_metrics(softmax(S, 0.05), extraversion$subtopic, "T = 0.05")
#'
#' @seealso \code{\link{reliability_table}}, \code{\link{fit_temperature}}
#' @export
calibration_metrics <- function(probs, truth, method = "", n_bins = 10, digits = 3) {
  probs <- .as_prob_matrix(probs)
  truth <- normalize_labels(truth, colnames(probs))
  ok <- stats::complete.cases(probs) & rowSums(probs) > 0 & !is.na(truth)
  P <- probs[ok, , drop = FALSE]
  P <- P / rowSums(P)
  y <- truth[ok]
  pred <- colnames(P)[max.col(P, ties.method = "first")]
  conf <- apply(P, 1, max)
  data.frame(method = method,
             n = sum(ok),
             n_missing = sum(!ok),
             accuracy = round(mean(pred == y), digits),
             mean_confidence = round(mean(conf), digits),
             ECE = round(expected_calibration_error(conf, pred == y, n_bins), digits),
             Brier = round(brier_score(P, y), digits),
             NLL = round(.nll(P, y), digits),
             stringsAsFactors = FALSE)
}

#' Probability matrix from a matrix or a data.frame. For data frames, only
#' numeric columns are kept, and bookkeeping columns returned by the package
#' (text_id, fold, confidence, score, level, row) are dropped.
#' @noRd
.as_prob_matrix <- function(probs) {
  if (is.data.frame(probs)) {
    meta <- c("text_id", "fold", "confidence", "score", "level", "row")
    num <- vapply(probs, is.numeric, logical(1)) & !(names(probs) %in% meta)
    probs <- as.matrix(probs[, num, drop = FALSE])
  }
  probs <- as.matrix(probs)
  if (is.null(colnames(probs))) stop("'probs' needs class names as column names.", call. = FALSE)
  probs
}


#' Reliability diagram data and plot
#'
#' A reliability diagram plots the stated confidence (x) against the
#' observed accuracy (y) within confidence bins. Points on the diagonal mean
#' perfect calibration; points below it mean over-confidence.
#'
#' @param probs Either a probability matrix (rows = texts, columns = classes)
#'   or a numeric vector of confidences.
#' @param truth True labels (matrix case) or a logical vector indicating
#'   whether each prediction was correct (vector case).
#' @param method Optional name of the method (used for colors in the plot).
#' @param n_bins Number of equal-width bins. Default is \code{10}.
#'
#' @return \code{reliability_table()} returns a data.frame with one row per
#'   non-empty bin: \code{method}, \code{bin}, \code{confidence} (mean
#'   confidence), \code{accuracy} and \code{n}. \code{plot_reliability()}
#'   returns a ggplot object (if \pkg{ggplot2} is installed) or draws a base
#'   R plot.
#'
#' @examples
#' data(minilm_embeddings)
#' data(extraversion)
#' S <- cosine_scores(minilm_embeddings$texts$extraversion,
#'                    minilm_embeddings$labels$extraversion)
#'
#' rel <- rbind(
#'   reliability_table(softmax(S, 1), extraversion$subtopic, "T = 1"),
#'   reliability_table(softmax(S, 0.02), extraversion$subtopic, "T = 0.02")
#' )
#' rel
#' plot_reliability(rel)
#'
#' # Vector input: stated confidence and correctness
#' reliability_table(c(0.9, 0.8, 0.6, 0.95), c(TRUE, TRUE, FALSE, TRUE), "LLM")
#'
#' @seealso \code{\link{calibration_metrics}}
#' @export
reliability_table <- function(probs, truth, method = "", n_bins = 10) {
  if (is.null(dim(probs)) && !is.data.frame(probs)) {
    conf <- as.numeric(probs)
    .check_unit_interval(conf)
    correct <- as.logical(truth)
  } else {
    P <- .as_prob_matrix(probs)
    y <- normalize_labels(truth, colnames(P))
    ok <- stats::complete.cases(P) & rowSums(P) > 0 & !is.na(y)
    P <- P[ok, , drop = FALSE] / rowSums(P[ok, , drop = FALSE])
    conf <- apply(P, 1, max)
    correct <- colnames(P)[max.col(P, ties.method = "first")] == y[ok]
  }
  ok <- !is.na(conf) & !is.na(correct)
  conf <- conf[ok]
  correct <- correct[ok]
  bin <- cut(conf, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  idx <- split(seq_along(conf), bin, drop = TRUE)
  out <- data.frame(
    method = method,
    bin = names(idx),
    confidence = vapply(idx, function(i) mean(conf[i]), numeric(1)),
    accuracy = vapply(idx, function(i) mean(correct[i]), numeric(1)),
    n = vapply(idx, length, integer(1)),
    stringsAsFactors = FALSE
  )
  rownames(out) <- NULL
  out
}

#' @rdname reliability_table
#' @param rel A data.frame from \code{reliability_table()} (several methods
#'   can be combined with \code{rbind()}).
#' @param title Plot title.
#' @export
plot_reliability <- function(rel, title = "Reliability diagram") {
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    return(
      ggplot2::ggplot(rel, ggplot2::aes(x = .data$confidence, y = .data$accuracy,
                                        color = .data$method)) +
        ggplot2::geom_abline(linetype = "dashed", color = "grey50") +
        ggplot2::geom_line() +
        ggplot2::geom_point(ggplot2::aes(size = .data$n)) +
        ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
        ggplot2::labs(title = title, subtitle = "Dashed line = perfect calibration",
                      x = "Confidence (probability of the chosen class)",
                      y = "Observed accuracy", size = "Texts", color = NULL) +
        ggplot2::theme_minimal(base_size = 12)
    )
  }
  methods <- unique(rel$method)
  cols <- seq_along(methods) + 1
  graphics::plot(c(0, 1), c(0, 1), type = "n", xlab = "Confidence", ylab = "Observed accuracy",
                 main = title, asp = 1)
  graphics::abline(0, 1, lty = 2, col = "grey50")
  for (m in seq_along(methods)) {
    r <- rel[rel$method == methods[m], ]
    graphics::lines(r$confidence, r$accuracy, col = cols[m], type = "b", pch = 19)
  }
  graphics::legend("topleft", legend = methods, col = cols, lty = 1, bty = "n")
  invisible(NULL)
}


#' Temperature scaling
#'
#' Finds the single number \eqn{T} that makes \code{softmax(scores / T)}
#' as honest as possible (minimum negative log-likelihood) on a set of
#' labeled texts (Guo et al., 2017). \eqn{T > 1} softens over-confident
#' probabilities; \eqn{0 < T < 1} sharpens under-confident ones.
#' Predictions (and accuracy) never change: only the confidence is repaired.
#'
#' Fit \eqn{T} on a calibration split and evaluate it on a different test
#' split.
#'
#' @param scores Score matrix (rows = texts, columns = classes). Use cosine
#'   similarities, logits, or \code{log(probs)} to recalibrate probabilities
#'   from another model (e.g., Jev or self-consistency votes, adding a small
#'   constant before the log).
#' @param truth True labels for the rows of \code{scores}.
#' @param interval Search interval for \eqn{T}. Default \code{c(0.001, 10)}.
#'
#' @return The fitted temperature (a single number).
#'
#' @references Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
#'   On calibration of modern neural networks. \emph{ICML}.
#'
#' @examples
#' data(minilm_embeddings)
#' data(neuroticism)
#' S <- cosine_scores(minilm_embeddings$texts$neuroticism,
#'                    minilm_embeddings$labels$emotions)
#' y <- neuroticism$subtopic
#'
#' set.seed(2026)
#' calib <- sample(seq_along(y), 60)   # fit T here
#' test  <- setdiff(seq_along(y), calib)   # evaluate here
#'
#' T_hat <- fit_temperature(S[calib, ], y[calib])
#' T_hat
#'
#' rbind(
#'   calibration_metrics(softmax(S[test, ]), y[test], "T = 1"),
#'   calibration_metrics(softmax(S[test, ], T_hat), y[test], "Temperature-scaled")
#' )
#'
#' @seealso \code{\link{softmax}}, \code{\link{fit_platt}},
#'   \code{\link{calibration_metrics}}
#' @export
fit_temperature <- function(scores, truth, interval = c(0.001, 10)) {
  scores <- .as_prob_matrix(scores)
  truth <- normalize_labels(truth, colnames(scores))
  ok <- stats::complete.cases(scores) & !is.na(truth)
  S <- scores[ok, , drop = FALSE]
  y <- truth[ok]
  stats::optimize(function(t) .nll(softmax(S, t), y), interval = interval)$minimum
}


#' Platt scaling for yes/no scores
#'
#' Turns any score for a yes/no question (a cosine similarity, an LLM rating,
#' a Jev Noul probability) into a calibrated probability with a logistic
#' regression: \eqn{p(yes) = sigmoid(a \cdot score + b)}. Unlike a softmax
#' over several classes, here the bias \eqn{b} matters.
#'
#' @param score Numeric vector of scores.
#' @param is_positive Logical (or 0/1) vector: is the answer "yes"?
#' @param object A \code{platt_fit} object.
#' @param newdata Numeric vector of new scores. If missing, the fitted
#'   probabilities are returned.
#' @param x A \code{platt_fit} object (for printing).
#' @param ... Not used.
#'
#' @return \code{fit_platt()} returns an object of class \code{platt_fit}
#'   with the slope \code{a}, the intercept \code{b} and the \code{glm} fit.
#'   \code{predict()} returns probabilities.
#'
#' @examples
#' data(minilm_embeddings)
#' data(neuroticism)
#' S <- cosine_scores(minilm_embeddings$texts$neuroticism,
#'                    minilm_embeddings$labels$emotions)
#' is_angry <- neuroticism$subtopic == "Angry"
#'
#' set.seed(2026)
#' calib <- sample(seq_along(is_angry), 60)
#' test  <- setdiff(seq_along(is_angry), calib)
#'
#' platt <- fit_platt(S[calib, "Angry"], is_angry[calib])
#' platt
#' p_angry <- predict(platt, S[test, "Angry"])
#' brier_score(p_angry, is_angry[test])
#' roc_auc(p_angry, is_angry[test])
#'
#' @seealso \code{\link{fit_temperature}}
#' @export
fit_platt <- function(score, is_positive) {
  y <- as.numeric(as.logical(is_positive))
  fit <- stats::glm(y ~ score, family = stats::binomial())
  structure(list(a = unname(stats::coef(fit)[2]), b = unname(stats::coef(fit)[1]),
                 model = fit), class = "platt_fit")
}

#' @rdname fit_platt
#' @export
predict.platt_fit <- function(object, newdata, ...) {
  if (missing(newdata)) return(unname(stats::fitted(object$model)))
  unname(stats::plogis(object$b + object$a * newdata))
}

#' @rdname fit_platt
#' @export
print.platt_fit <- function(x, ...) {
  cat(sprintf("Platt scaling: p(yes) = sigmoid(%.3f * score + %.3f)\n", x$a, x$b))
  invisible(x)
}


#' The RLCR reward: why honest confidence pays
#'
#' Reinforcement Learning with Calibration Rewards (RLCR; Damani et al.,
#' 2025) trains reasoning models with the reward
#' \deqn{R = c - (q - c)^2}
#' where \eqn{c = 1} if the answer is correct (0 otherwise) and \eqn{q} is
#' the stated confidence. The second part is the Brier score, a proper
#' scoring rule: if the true chance of being right is \eqn{p}, the expected
#' reward is maximized by stating \eqn{q = p}.
#'
#' @param q Stated confidence(s) between 0 and 1.
#' @param correct Logical: was the answer correct?
#'
#' @return Numeric reward(s).
#'
#' @references Damani, M., et al. (2025). Beyond binary rewards: Training
#'   LMs to reason about their uncertainty. arXiv:2507.16806.
#'
#' @examples
#' rlcr_reward(0.9, FALSE)   # confident and wrong: -0.81
#' rlcr_reward(0.2, FALSE)   # humble and wrong:    -0.04
#' rlcr_reward(0.9, TRUE)    # confident and right:  0.99
#'
#' # Expected reward when the true chance of being right is 0.7
#' q <- seq(0, 1, 0.01)
#' expected <- 0.7 * rlcr_reward(q, TRUE) + 0.3 * rlcr_reward(q, FALSE)
#' q[which.max(expected)]    # 0.7: stating the truth pays best
#' plot(q, expected, type = "l", xlab = "Stated confidence", ylab = "Expected reward")
#' abline(v = 0.7, lty = 2)
#'
#' @export
rlcr_reward <- function(q, correct) {
  as.numeric(correct) - (q - as.numeric(correct))^2
}
