# ==============================================================================
# Baselines and linear probes (penalized logistic regression with glmnet)
# ==============================================================================
#
# Raschka (2026): always compare LLM classifiers with a simple, cheap baseline
# (bag-of-words + logistic regression), check learning curves before deciding
# between "more data" and "more tuning", and test robustness to distribution
# shift. The same logistic "head" on top of embeddings is a linear probe.

#' @noRd
.need <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("This function needs the package(s): ", paste(missing, collapse = ", "),
         ".\nInstall with: install.packages(c(",
         paste0('"', missing, '"', collapse = ", "), "))", call. = FALSE)
  }
}


#' Bag-of-words (TF-IDF) features
#'
#' Turns texts into a sparse document-term matrix: lowercase, remove
#' punctuation, numbers and English stop words, stem ("greets", "greeting"
#' -> "greet") and weight by TF-IDF (words frequent in a few documents get
#' high weights; a word in every document gets zero).
#'
#' @param texts Character vector of texts.
#' @param weighting \code{"tfidf"} (default), \code{"tf"} (counts) or
#'   \code{"binary"}.
#' @param stem Logical. Apply Porter stemming (needs \pkg{SnowballC}).
#'   Default \code{TRUE}.
#' @param remove_stopwords Logical. Remove English stop words. Default
#'   \code{TRUE}.
#' @param min_docs Keep only terms that appear in at least this many texts.
#'   Default \code{1}.
#'
#' @return A sparse matrix (class \code{dgCMatrix}): texts x terms.
#'
#' @details
#' The vocabulary and IDF weights are computed on all \code{texts}. Inside
#' cross-validation this lets held-out texts share the vocabulary weights
#' (a small, common simplification). For a transfer experiment (train on one
#' source, test on another), build the features for all texts together and
#' subset the rows.
#'
#' @examples
#' data(extraversion)
#' if (requireNamespace("tm", quietly = TRUE) && requireNamespace("SnowballC", quietly = TRUE)) {
#'   X <- bow_features(extraversion$text)
#'   dim(X)
#'   X[1:3, 1:6]
#' }
#'
#' @seealso \code{\link{bow_baseline}}, \code{\link{cv_logistic}}
#' @export
bow_features <- function(texts, weighting = c("tfidf", "tf", "binary"),
                         stem = TRUE, remove_stopwords = TRUE, min_docs = 1) {
  weighting <- match.arg(weighting)
  .need(c("tm", "Matrix", if (stem) "SnowballC"))
  corp <- tm::VCorpus(tm::VectorSource(as.character(texts)))
  corp <- tm::tm_map(corp, tm::content_transformer(tolower))
  corp <- tm::tm_map(corp, tm::removePunctuation)
  corp <- tm::tm_map(corp, tm::removeNumbers)
  if (remove_stopwords) corp <- tm::tm_map(corp, tm::removeWords, tm::stopwords("english"))
  if (stem) corp <- tm::tm_map(corp, tm::stemDocument)
  corp <- tm::tm_map(corp, tm::stripWhitespace)

  wfun <- switch(weighting, tfidf = tm::weightTfIdf, tf = tm::weightTf, binary = tm::weightBin)
  dtm <- tm::DocumentTermMatrix(corp, control = list(weighting = wfun))
  X <- Matrix::sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v, dims = dim(dtm),
                            dimnames = list(NULL, colnames(dtm)))
  if (min_docs > 1) X <- X[, Matrix::colSums(X > 0) >= min_docs, drop = FALSE]
  X
}


#' Stratified inner folds for cv.glmnet (5, or fewer for very small classes),
#' so that every inner training set contains every class
#' @noRd
.inner_folds <- function(y) {
  k <- max(3, min(5, min(table(y))))   # cv.glmnet needs at least 3 folds
  .make_folds(as.character(y), k)
}

#' Clear errors for missing labels and tiny classes
#' @noRd
.check_labels <- function(y, min_per_class, what) {
  if (anyNA(y)) {
    stop("The labels contain missing values; remove those texts first.", call. = FALSE)
  }
  if (min(table(y)) < min_per_class) {
    stop(what, " needs at least ", min_per_class, " texts in every class ",
         "(smallest class: ", min(table(y)), ").", call. = FALSE)
  }
}

#' Stratified folds (uses the current random state)
#' @noRd
.make_folds <- function(y, k = 5) {
  folds <- integer(length(y))
  for (cl in unique(y)) {
    idx <- which(y == cl)
    n <- length(idx)
    # full rounds of 1..k, plus randomly chosen folds for the leftovers
    base <- rep(seq_len(k), n %/% k)
    if (n %% k > 0) base <- c(base, sample.int(k, n %% k))
    folds[idx] <- sample(base)
  }
  folds
}


#' Cross-validated penalized logistic regression
#'
#' Fits an elastic-net multinomial logistic regression (\pkg{glmnet}) with
#' stratified k-fold cross-validation: every text is predicted by a model
#' that never saw it. The penalty (lambda) is tuned inside each training
#' fold (with stratified inner folds). Works with any feature matrix: bag-of-words
#' (\code{\link{bow_features}}) or embeddings (then it is a
#' \emph{linear probe}).
#'
#' @param X Feature matrix (texts x features); dense or sparse.
#' @param y Vector of labels.
#' @param k Number of folds. Default \code{5}.
#' @param alpha Elastic-net mixing: \code{0} = ridge, \code{1} = lasso.
#'   Default \code{0.5}.
#' @param seed Seed for the fold assignment and the inner CV. Default
#'   \code{2026}.
#'
#' @return A data.frame with \code{text_id}, \code{truth}, \code{prediction},
#'   \code{fold}, \code{confidence} and one probability column per class
#'   (so it can be passed to \code{\link{calibration_metrics}}).
#'
#' @examples
#' data(minilm_embeddings)
#' data(extraversion)
#' if (requireNamespace("glmnet", quietly = TRUE)) {
#'   cv <- cv_logistic(minilm_embeddings$texts$extraversion, extraversion$subtopic)
#'   evaluate_classification(cv$truth, cv$prediction, "Linear probe (MiniLM)")
#'   calibration_metrics(cv[, sort(unique(extraversion$subtopic))], cv$truth, "Linear probe")
#' }
#'
#' @seealso \code{\link{linear_probe}}, \code{\link{bow_baseline}}
#' @export
cv_logistic <- function(X, y, k = 5, alpha = 0.5, seed = 2026) {
  .need("glmnet")
  .check_labels(y, min_per_class = max(4, ceiling(3 * k / (k - 1))), "cv_logistic()")
  y <- factor(y)
  classes <- levels(y)
  probs <- matrix(NA_real_, length(y), length(classes), dimnames = list(NULL, classes))
  folds <- NULL
  .with_seed(seed, {
    folds <- .make_folds(as.character(y), k)
    for (f in seq_len(k)) {
      train <- folds != f
      y_train <- droplevels(y[train])
      fit <- glmnet::cv.glmnet(X[train, , drop = FALSE], y_train,
                               family = "multinomial", type.measure = "class",
                               alpha = alpha, foldid = .inner_folds(y_train))
      p <- stats::predict(fit, X[!train, , drop = FALSE], s = "lambda.min", type = "response")
      p <- matrix(p[, , 1], nrow = sum(!train), dimnames = list(NULL, dimnames(p)[[2]]))
      probs[!train, colnames(p)] <- p
    }
  })
  probs[is.na(probs)] <- 0
  out <- data.frame(text_id = seq_along(y), truth = as.character(y),
                    prediction = classes[max.col(probs, ties.method = "first")],
                    fold = folds, confidence = apply(probs, 1, max),
                    stringsAsFactors = FALSE)
  cbind(out, as.data.frame(probs, check.names = FALSE))
}


#' Linear probe: how much label information is linearly readable?
#'
#' A cross-validated logistic regression "head" on top of frozen embeddings.
#' EGA asks whether texts form facet communities \emph{without} labels (NMI);
#' the probe asks whether the facet information is \emph{linearly readable}
#' with labels (accuracy).
#'
#' @inheritParams cv_logistic
#' @param method Name for the output row. Default \code{"Linear probe"}.
#'
#' @return A one-row data.frame from \code{\link{evaluate_classification}}.
#'
#' @examples
#' data(minilm_embeddings)
#' data(extraversion)
#' if (requireNamespace("glmnet", quietly = TRUE)) {
#'   linear_probe(minilm_embeddings$texts$extraversion, extraversion$subtopic)
#' }
#'
#' @seealso \code{\link{probe_layers}}, \code{\link{cv_logistic}}
#' @export
linear_probe <- function(X, y, k = 5, alpha = 0.5, seed = 2026, method = "Linear probe") {
  cv <- cv_logistic(X, y, k = k, alpha = alpha, seed = seed)
  evaluate_classification(cv$truth, cv$prediction, method)
}


#' Linear probe (and optionally EGA) for every layer of a transformer
#'
#' For each layer of \code{\link{embed_local}}, fits a cross-validated linear
#' probe and, if \code{ega = TRUE}, runs EGA and computes its NMI with the
#' labels. Typical finding: the probe reads the facets well from almost
#' every layer, while the unsupervised EGA structure changes a lot across
#' layers (and across models).
#'
#' @param layers List of embedding matrices, one per layer (e.g.,
#'   \code{embed_local(...)$layers}).
#' @param labels Vector of true labels.
#' @param ega Logical. Also run EGA on each layer (needs \pkg{EGAnet}).
#'   Default \code{FALSE}.
#' @inheritParams cv_logistic
#'
#' @return A data.frame with \code{layer}, \code{layer_number},
#'   \code{probe_accuracy}, \code{probe_macro_F1} and, with \code{ega = TRUE},
#'   \code{EGA_NMI} and \code{EGA_dimensions}.
#'
#' @examples
#' \dontrun{
#' activate_llm_env()
#' data(extraversion)
#' emb <- embed_local(extraversion$text, model_name = "bert-base-uncased")
#' layer_table <- probe_layers(emb$layers, extraversion$subtopic, ega = TRUE)
#' layer_table
#'
#' plot(layer_table$layer_number, layer_table$probe_accuracy, type = "b",
#'      ylim = c(0, 1), xlab = "Layer", ylab = "Probe accuracy / EGA NMI")
#' lines(layer_table$layer_number, layer_table$EGA_NMI, type = "b", col = 2)
#' }
#'
#' @seealso \code{\link{linear_probe}}, \code{\link{evaluate_layers}},
#'   \code{\link{ega_classify}}
#' @export
probe_layers <- function(layers, labels, ega = FALSE, k = 5, alpha = 0.5, seed = 2026) {
  if (ega) .need("EGAnet")
  n_layers <- length(layers)
  nm <- if (is.null(names(layers))) paste0("layer_", seq(0, n_layers - 1)) else names(layers)
  # Layer numbers from names like "layer_6" (so a subset keeps its numbers)
  num <- suppressWarnings(as.integer(sub("^layer_", "", nm)))
  if (anyNA(num)) num <- seq(0, n_layers - 1)
  rows <- lapply(seq_len(n_layers), function(l) {
    cat("  Layer", num[l], "(", l, "of", n_layers, ")\n")
    ev <- linear_probe(layers[[l]], labels, k = k, alpha = alpha, seed = seed)
    row <- data.frame(layer = nm[l], layer_number = num[l],
                      probe_accuracy = ev$accuracy, probe_macro_F1 = ev$macro_F1,
                      stringsAsFactors = FALSE)
    if (ega) {
      res <- tryCatch({
        transposed <- as.data.frame(t(layers[[l]]))
        colnames(transposed) <- paste0("doc_", seq_len(ncol(transposed)))
        e <- EGAnet::EGA(transposed, model = "tmfg", corr = "pearson",
                         uni.method = "expand", plot.EGA = FALSE)
        c(nmi = compute_nmi(labels, as.vector(e$wc)), n_dim = e$n.dim)
      }, error = function(e) c(nmi = NA_real_, n_dim = NA_real_))
      row$EGA_NMI <- round(unname(res["nmi"]), 3)
      row$EGA_dimensions <- unname(res["n_dim"])
    }
    row
  })
  do.call(rbind, rows)
}


#' Top words per class (interpretable bag-of-words model)
#'
#' Fits one elastic-net multinomial logistic regression on all texts and
#' returns the words with the largest positive weights for each class.
#'
#' @param X Bag-of-words matrix from \code{\link{bow_features}}.
#' @param y Vector of labels.
#' @param n Number of words per class. Default \code{8}.
#' @param alpha Elastic-net mixing. Default \code{0.5}.
#' @param seed Seed for the internal cross-validation. Default \code{2026}.
#'
#' @return A data.frame with \code{class}, \code{term} and \code{weight}.
#'
#' @examples
#' data(extraversion)
#' if (requireNamespace("tm", quietly = TRUE) && requireNamespace("glmnet", quietly = TRUE) &&
#'     requireNamespace("SnowballC", quietly = TRUE)) {
#'   X <- bow_features(extraversion$text)
#'   top_terms(X, extraversion$subtopic, n = 5)
#' }
#' @export
top_terms <- function(X, y, n = 8, alpha = 0.5, seed = 2026) {
  .need("glmnet")
  .check_labels(y, min_per_class = 3, "top_terms()")
  y <- factor(y)
  fit <- .with_seed(seed, glmnet::cv.glmnet(X, y, family = "multinomial",
                                            type.measure = "class", alpha = alpha,
                                            foldid = .inner_folds(y)))
  coefs <- stats::coef(fit, s = "lambda.min")
  out <- do.call(rbind, lapply(names(coefs), function(cl) {
    b <- as.matrix(coefs[[cl]])[-1, 1]   # drop the intercept
    b <- utils::head(sort(b[b > 0], decreasing = TRUE), n)
    if (length(b) == 0) return(NULL)
    data.frame(class = cl, term = names(b), weight = round(unname(b), 4),
               stringsAsFactors = FALSE)
  }))
  rownames(out) <- NULL
  out
}


#' A bag-of-words baseline in one call
#'
#' TF-IDF features + cross-validated elastic-net logistic regression, the
#' baseline every LLM classifier should be compared with. It is fast, free,
#' private (runs locally) and interpretable. On the course extraversion data
#' it reaches about 97\% accuracy, which is a high bar for zero-shot LLMs.
#'
#' @param texts Character vector of texts.
#' @param labels Vector of true labels.
#' @param n_top Number of top words per class to return (\code{0} to skip).
#'   Default \code{8}.
#' @inheritParams cv_logistic
#' @param ... Passed to \code{\link{bow_features}} (e.g., \code{stem = FALSE}).
#'
#' @return A list with \code{evaluation} (from
#'   \code{\link{evaluate_classification}}), \code{predictions} (from
#'   \code{\link{cv_logistic}}), \code{top_terms} and the feature matrix
#'   \code{X}.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @examples
#' data(extraversion)
#' if (requireNamespace("tm", quietly = TRUE) && requireNamespace("glmnet", quietly = TRUE) &&
#'     requireNamespace("SnowballC", quietly = TRUE)) {
#'   bow <- bow_baseline(extraversion$text, extraversion$subtopic, n_top = 5)
#'   bow$evaluation
#'   head(bow$top_terms, 10)
#'   table(True = bow$predictions$truth, Predicted = bow$predictions$prediction)
#' }
#'
#' @seealso \code{\link{learning_curve}}, \code{\link{compare_methods}}
#' @export
bow_baseline <- function(texts, labels, k = 5, alpha = 0.5, seed = 2026, n_top = 8, ...) {
  X <- bow_features(texts, ...)
  cat("Bag-of-words matrix:", nrow(X), "texts x", ncol(X), "terms\n")
  cv <- cv_logistic(X, labels, k = k, alpha = alpha, seed = seed)
  ev <- evaluate_classification(labels, cv$prediction, "TF-IDF + logistic regression")
  print(ev, row.names = FALSE)
  list(evaluation = ev,
       predictions = cv,
       top_terms = if (n_top > 0) top_terms(X, labels, n = n_top, alpha = alpha, seed = seed) else NULL,
       X = X)
}


#' Learning curve: more data or more tuning?
#'
#' Trains a penalized logistic regression on \code{n} labeled texts per
#' class (fixed penalty, no tuning), tests on all remaining texts, and
#' repeats with different random samples. If accuracy is still rising at the
#' largest training size, more labeled texts will probably help more than
#' hyperparameter tuning.
#'
#' @param X Feature matrix (bag-of-words or embeddings).
#' @param y Vector of labels.
#' @param n_per_class Training sizes per class. Default
#'   \code{c(3, 5, 8, 12, 16, 20, 24)}.
#' @param n_repeats Random repetitions per size. Default \code{10}.
#' @param alpha Elastic-net mixing. Default \code{0.5}.
#' @param lambda Fixed penalty (no tuning). Default \code{0.05}.
#' @param seed Seed. Default \code{2026}.
#'
#' @return A data.frame with \code{n_train} (total training texts),
#'   \code{mean_acc}, and the 10th and 90th percentiles (\code{lo},
#'   \code{hi}) across repetitions.
#'
#' @examples
#' data(extraversion)
#' if (requireNamespace("tm", quietly = TRUE) && requireNamespace("glmnet", quietly = TRUE) &&
#'     requireNamespace("SnowballC", quietly = TRUE)) {
#'   X <- bow_features(extraversion$text)
#'   lc <- learning_curve(X, extraversion$subtopic, n_per_class = c(3, 12, 24), n_repeats = 3)
#'   lc
#'   plot(lc$n_train, lc$mean_acc, type = "b", ylim = c(0, 1),
#'        xlab = "Labeled training texts", ylab = "Held-out accuracy")
#' }
#'
#' @seealso \code{\link{bow_baseline}}
#' @export
learning_curve <- function(X, y, n_per_class = c(3, 5, 8, 12, 16, 20, 24),
                           n_repeats = 10, alpha = 0.5, lambda = 0.05, seed = 2026) {
  .need("glmnet")
  .check_labels(y, min_per_class = 2, "learning_curve()")
  y <- factor(y)
  if (max(n_per_class) >= min(table(y))) {
    stop("Every class needs more texts than max(n_per_class), to keep a test set.",
         call. = FALSE)
  }
  .with_seed(seed, {
    rows <- lapply(n_per_class, function(n) {
      accs <- replicate(n_repeats, {
        train <- unlist(lapply(split(seq_along(y), y), function(i) i[sample.int(length(i), n)]))
        test <- setdiff(seq_along(y), train)
        # glmnet warns about classes with fewer than 8 texts: that is the point here
        fit <- suppressWarnings(glmnet::glmnet(X[train, , drop = FALSE], y[train],
                                               family = "multinomial",
                                               alpha = alpha, lambda = lambda))
        mean(stats::predict(fit, X[test, , drop = FALSE], type = "class") == y[test])
      })
      data.frame(n_train = n * nlevels(y), mean_acc = round(mean(accs), 3),
                 lo = round(unname(stats::quantile(accs, 0.10)), 3),
                 hi = round(unname(stats::quantile(accs, 0.90)), 3))
    })
    do.call(rbind, rows)
  })
}
