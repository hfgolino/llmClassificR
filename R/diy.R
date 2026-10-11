# ==============================================================================
# A do-it-yourself "Jev-like" classifier from embeddings
# ==============================================================================
#
# Raschka (2026) sketches the idea behind Jev-like classifiers:
#   1. represent the text AND each candidate option (its description) with a
#      transformer, giving a vector h_i for every candidate i;
#   2. score each candidate with the SAME small head:  s_i = w . h_i
#   3. turn the scores into probabilities with a softmax over the candidates.
# Because w is shared, the number of candidates can change freely.
# Here: h_i = e_text * e_candidate (element-wise) with unit-length sentence
# embeddings. With w = (20, ..., 20), s_i = 20 x cosine, i.e. softmax(cos / 0.05).


#' Sentence embeddings with a small local model
#'
#' Embeds texts with a sentence-transformers model through the Python
#' environment (mean pooling over tokens of the last layer). Much faster than
#' \code{\link{embed_local}}, which extracts every layer. The model is loaded
#' once per session and reused. No API key and no cost.
#'
#' @param texts Character vector of texts.
#' @param model_name HuggingFace model. Default is
#'   \code{"sentence-transformers/all-MiniLM-L6-v2"} (384 dimensions, runs on
#'   a laptop CPU). Other options: \code{"sentence-transformers/all-mpnet-base-v2"}
#'   (768 dimensions, slower, usually better).
#' @param batch_size Texts per forward pass. Default is \code{16}.
#' @param max_length Maximum number of tokens per text. Default is \code{256}.
#' @param normalize Logical. Scale every row to unit length, so that a dot
#'   product equals the cosine similarity. Default \code{TRUE}.
#'
#' @return A numeric matrix (texts x dimensions).
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @examples
#' \dontrun{
#' activate_llm_env()
#' data(extraversion)
#' E <- embed_sentences(extraversion$text)
#' dim(E)   # 120 x 384
#'
#' # Embed the label descriptions too, then classify zero-shot
#' L <- embed_sentences(label_descriptions$extraversion)
#' rownames(L) <- names(label_descriptions$extraversion)
#' res <- classify_by_description(text_embeddings = E, label_embeddings = L)
#' }
#'
#' @seealso \code{\link{classify_by_description}}, \code{\link{minilm_embeddings}}
#' @export
embed_sentences <- function(texts,
                            model_name = "sentence-transformers/all-MiniLM-L6-v2",
                            batch_size = 16L, max_length = 256L, normalize = TRUE) {
  if (is.null(.pkg_env$encoders)) .pkg_env$encoders <- list()
  enc <- .pkg_env$encoders[[model_name]]
  if (is.null(enc)) {
    transformers <- reticulate::import("transformers")
    cat("Loading model:", model_name, "...\n")
    enc <- list(
      torch = reticulate::import("torch"),
      tokenizer = transformers$AutoTokenizer$from_pretrained(model_name),
      model = transformers$AutoModel$from_pretrained(model_name)
    )
    enc$model$eval()
    .pkg_env$encoders[[model_name]] <- enc
  }
  torch <- enc$torch
  texts <- as.character(texts)

  out <- list()
  for (start in seq(1, length(texts), by = batch_size)) {
    batch <- texts[start:min(start + batch_size - 1, length(texts))]
    inputs <- enc$tokenizer(as.list(batch), return_tensors = "pt", truncation = TRUE,
                            padding = TRUE, max_length = as.integer(max_length))
    with(torch$no_grad(), {
      outputs <- enc$model(input_ids = inputs$input_ids,
                           attention_mask = inputs$attention_mask)
    })
    # Mean pooling over real (non-padding) tokens
    mask <- inputs$attention_mask$unsqueeze(-1L)$float()
    pooled <- torch$sum(outputs$last_hidden_state * mask, dim = 1L) /
      torch$sum(mask, dim = 1L)
    out[[length(out) + 1]] <- pooled$numpy()
  }
  E <- do.call(rbind, out)
  if (normalize) E <- E / sqrt(rowSums(E^2))
  rownames(E) <- if (!is.null(names(texts))) names(texts) else paste0("text_", seq_along(texts))
  E
}


#' Cosine similarity between texts and candidate labels
#'
#' @param text_embeddings Matrix of text embeddings (texts x dimensions).
#' @param label_embeddings Matrix of label (description) embeddings
#'   (labels x dimensions), with label names as row names.
#'
#' @return A texts x labels matrix of cosine similarities, with the label
#'   names as column names.
#'
#' @examples
#' data(minilm_embeddings)
#' S <- cosine_scores(minilm_embeddings$texts$extraversion,
#'                    minilm_embeddings$labels$extraversion)
#' round(head(S), 2)
#'
#' @seealso \code{\link{classify_by_description}}
#' @export
cosine_scores <- function(text_embeddings, label_embeddings) {
  if (ncol(text_embeddings) != ncol(label_embeddings)) {
    stop("Text and label embeddings must have the same number of dimensions ",
         "(use the same embedding model for both).", call. = FALSE)
  }
  unit <- function(M) M / sqrt(rowSums(M^2))
  S <- unit(as.matrix(text_embeddings)) %*% t(unit(as.matrix(label_embeddings)))
  colnames(S) <- rownames(label_embeddings)
  S
}


#' Zero-shot classification from label descriptions (a DIY Jev-like classifier)
#'
#' Classifies texts into categories it has never been trained on: each text
#' is compared with a short \emph{description} of each category (like Jev's
#' criteria), and the cosine similarities are turned into probabilities with
#' \code{softmax(cosine / temperature)}. No labeled data are used.
#'
#' Text and descriptions are embedded separately (a "bi-encoder"). Jev reads
#' the text and the options together through one transformer (a
#' "cross-encoder"), so each option representation "knows" the text.
#'
#' @param texts Character vector of texts. Can be \code{NULL} if
#'   \code{text_embeddings} is given.
#' @param descriptions Named character vector: names are the classes, values
#'   the descriptions (see \code{\link{label_descriptions}}). Can be
#'   \code{NULL} if \code{label_embeddings} is given.
#' @param embed_fun Function that embeds a character vector and returns a
#'   matrix. Default is \code{\link{embed_sentences}} (local, free). Also
#'   works with \code{\link{embed_voyage}} or \code{\link{embed_jina}}.
#' @param temperature Softmax temperature. Default \code{0.05} (equivalent to
#'   a head that multiplies cosines by 20). Cosines are small numbers, so
#'   \code{temperature = 1} gives nearly uniform, under-confident
#'   probabilities. Fit a better value with \code{\link{fit_temperature}}.
#' @param text_embeddings,label_embeddings Optional pre-computed embeddings
#'   (label embeddings need the class names as row names).
#'
#' @return A data.frame with \code{text_id}, \code{text} (if \code{texts} was
#'   given), \code{prediction}, \code{confidence} and one probability column
#'   per class. The attribute \code{"scores"} holds the cosine matrix.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @examples
#' # Offline, with the pre-computed MiniLM embeddings shipped with the package
#' data(minilm_embeddings)
#' data(neuroticism)
#' emb <- minilm_embeddings
#'
#' res <- classify_by_description(text_embeddings = emb$texts$neuroticism,
#'                                label_embeddings = emb$labels$emotions)
#' head(res)
#' evaluate_classification(neuroticism$subtopic, res$prediction, "DIY zero-shot")
#'
#' # Probabilities: accuracy is the same for every temperature, confidence is not
#' S <- attr(res, "scores")
#' rbind(calibration_metrics(softmax(S, 1), neuroticism$subtopic, "T = 1"),
#'       calibration_metrics(softmax(S, 0.05), neuroticism$subtopic, "T = 0.05"))
#'
#' \dontrun{
#' # From raw texts (needs the Python environment)
#' activate_llm_env()
#' res <- classify_by_description(extraversion$text, label_descriptions$extraversion)
#'
#' # Try it: rewrite one description and see how accuracy changes
#' }
#'
#' @seealso \code{\link{jev_classify}}, \code{\link{train_shared_head}},
#'   \code{\link{fit_temperature}}
#' @export
classify_by_description <- function(texts = NULL, descriptions = NULL,
                                    embed_fun = embed_sentences,
                                    temperature = 0.05,
                                    text_embeddings = NULL,
                                    label_embeddings = NULL) {
  if (is.null(text_embeddings)) {
    if (is.null(texts)) stop("Give 'texts' or 'text_embeddings'.", call. = FALSE)
    text_embeddings <- embed_fun(texts)
  }
  if (is.null(label_embeddings)) {
    if (is.null(descriptions) || is.null(names(descriptions))) {
      stop("Give a NAMED vector of 'descriptions' (or 'label_embeddings').", call. = FALSE)
    }
    label_embeddings <- embed_fun(unname(descriptions))
    rownames(label_embeddings) <- names(descriptions)
  }
  if (is.null(rownames(label_embeddings))) {
    stop("'label_embeddings' needs the class names as row names.", call. = FALSE)
  }

  S <- cosine_scores(text_embeddings, label_embeddings)
  P <- softmax(S, temperature)
  classes <- colnames(S)
  out <- data.frame(text_id = seq_len(nrow(S)), stringsAsFactors = FALSE)
  if (!is.null(texts)) out$text <- substr(texts, 1, 200)
  out$prediction <- classes[max.col(P, ties.method = "first")]
  out$confidence <- apply(P, 1, max)
  out <- cbind(out, as.data.frame(P, check.names = FALSE))
  rownames(out) <- NULL
  attr(out, "scores") <- S
  out
}


#' Train a shared scoring head (does it transfer to new label sets?)
#'
#' Learns one weight per embedding dimension, \eqn{w}, shared by all
#' candidates of all tasks: the score of candidate \eqn{i} for a text is
#' \eqn{s_i = w \cdot (e_{text} \odot e_i)} and the probabilities are
#' \code{softmax(s)}. Starting from \eqn{w = 20} (the untrained cosine rule)
#' the weights are fitted by penalized maximum likelihood (BFGS) on labeled
#' \emph{tasks}, then applied to tasks whose labels were never seen.
#'
#' With only a few training tasks, the trained head usually does not beat
#' the untrained cosine rule on a new task, and weaker regularization makes
#' transfer worse: the head memorizes the training tasks. A Jev-like model
#' needs many diverse, curated training tasks.
#'
#' @param tasks A list of tasks. Each task is a list with
#'   \code{text_embeddings} (texts x d), \code{label_embeddings} (labels x d,
#'   with class names as row names) and \code{labels} (true class of each
#'   text).
#' @param lambda Ridge penalty pulling \eqn{w} toward \code{w_start}.
#'   Default \code{1e-5}.
#' @param w_start Starting value (and penalty center) for every weight.
#'   Default \code{20}.
#' @param maxit Maximum BFGS iterations. Default \code{500}.
#' @param object A \code{shared_head} object.
#' @param text_embeddings,label_embeddings Embeddings of a (new) task.
#' @param ... Not used.
#'
#' @return \code{train_shared_head()} returns an object of class
#'   \code{shared_head} (the weights \code{w} plus settings).
#'   \code{predict()} returns a texts x labels probability matrix.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @examples
#' data(minilm_embeddings)
#' data(extraversion); data(neuroticism); data(personality_descriptions)
#' emb <- minilm_embeddings
#'
#' tasks <- list(
#'   extraversion = list(text_embeddings = emb$texts$extraversion,
#'                       label_embeddings = emb$labels$extraversion,
#'                       labels = extraversion$subtopic),
#'   emotions = list(text_embeddings = emb$texts$neuroticism,
#'                   label_embeddings = emb$labels$emotions,
#'                   labels = neuroticism$subtopic),
#'   bigfive = list(text_embeddings = emb$texts$personality_descriptions,
#'                  label_embeddings = emb$labels$bigfive,
#'                  labels = personality_descriptions$Personality)
#' )
#'
#' # Leave one TASK out: train on two tasks, test on the third (new labels!)
#' head_fit <- train_shared_head(tasks[c("extraversion", "emotions")], lambda = 1e-3)
#' head_fit
#' P_trained <- predict(head_fit, emb$texts$personality_descriptions, emb$labels$bigfive)
#' P_cosine  <- softmax(cosine_scores(emb$texts$personality_descriptions,
#'                                    emb$labels$bigfive), 0.05)
#' rbind(
#'   calibration_metrics(P_cosine, personality_descriptions$Personality, "Untrained (w = 20)"),
#'   calibration_metrics(P_trained, personality_descriptions$Personality, "Trained head")
#' )
#'
#' @seealso \code{\link{classify_by_description}}
#' @export
train_shared_head <- function(tasks, lambda = 1e-5, w_start = 20, maxit = 500) {
  prepared <- lapply(seq_along(tasks), function(k) {
    tk <- tasks[[k]]
    if (!all(c("text_embeddings", "label_embeddings", "labels") %in% names(tk))) {
      stop("Each task needs 'text_embeddings', 'label_embeddings' and 'labels'.",
           call. = FALSE)
    }
    unit <- function(M) M / sqrt(rowSums(M^2))
    C <- unit(as.matrix(tk$label_embeddings))
    y <- match(normalize_labels(tk$labels, rownames(C)), rownames(C))
    if (anyNA(y)) {
      stop("Task ", k, ": some labels are not row names of 'label_embeddings'.", call. = FALSE)
    }
    list(E = unit(as.matrix(tk$text_embeddings)), C = C, y = y)
  })
  d <- ncol(prepared[[1]]$E)
  w0 <- rep(w_start, d)

  loss_grad <- function(w) {
    loss <- 0
    grad <- 0
    n <- 0
    for (tk in prepared) {
      P <- softmax(tk$E %*% (t(tk$C) * w))                  # texts x candidates
      Y <- outer(tk$y, seq_len(nrow(tk$C)), "==") * 1        # one-hot truth
      loss <- loss - sum(log(pmax(P[cbind(seq_along(tk$y), tk$y)], 1e-12)))
      grad <- grad + colSums(tk$E * ((P - Y) %*% tk$C))     # d loss / d w
      n <- n + length(tk$y)
    }
    list(loss = loss / n + lambda * sum((w - w0)^2) / 2,
         grad = grad / n + lambda * (w - w0))
  }

  fit <- stats::optim(w0,
                      fn = function(w) loss_grad(w)$loss,
                      gr = function(w) loss_grad(w)$grad,
                      method = "BFGS", control = list(maxit = maxit))
  structure(list(w = fit$par, lambda = lambda, w_start = w_start,
                 n_tasks = length(tasks), convergence = fit$convergence,
                 train_loss = fit$value),
            class = "shared_head")
}

#' @rdname train_shared_head
#' @export
predict.shared_head <- function(object, text_embeddings, label_embeddings, ...) {
  unit <- function(M) M / sqrt(rowSums(M^2))
  C <- unit(as.matrix(label_embeddings))
  P <- softmax(unit(as.matrix(text_embeddings)) %*% (t(C) * object$w))
  colnames(P) <- rownames(label_embeddings)
  P
}

#' @rdname train_shared_head
#' @param x A \code{shared_head} object (for printing).
#' @export
print.shared_head <- function(x, ...) {
  cat("Shared scoring head s_i = w . (e_text * e_candidate)\n")
  cat(sprintf("  dimensions: %d | trained on %d task(s) | lambda = %g\n",
              length(x$w), x$n_tasks, x$lambda))
  cat(sprintf("  weights: mean %.2f, range %.2f to %.2f (start: %g)\n",
              mean(x$w), min(x$w), max(x$w), x$w_start))
  cat(sprintf("  training loss: %.4f | converged: %s\n", x$train_loss,
              if (x$convergence == 0) "yes" else "no"))
  invisible(x)
}
